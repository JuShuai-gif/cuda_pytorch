"""
Analyze dataset statistics relevant to LLM pretraining.

All functions operate on synthetic or small in-memory data so the module can
be run without downloading large corpora.

Statistics covered:
    - Token count (word-level and subword-level approximation)
    - Vocabulary diversity (type-token ratio)
    - Sentence length distribution
    - Data quality estimation via n-gram frequency perplexity approximation
"""

from __future__ import annotations

import collections
import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Synthetic data for demonstration
# ---------------------------------------------------------------------------

_SAMPLE_CORPUS: list[str] = [
    "Machine learning models require vast amounts of high-quality text data for pretraining.",
    "The transformer architecture introduced in Attention Is All You Need revolutionized NLP.",
    "Large language models are trained on diverse corpora including web pages, books, and code.",
    "Data quality is often more important than data quantity for downstream task performance.",
    "CommonCrawl provides petabytes of web data but requires extensive filtering and deduplication.",
    "Subword tokenization methods like BPE and SentencePiece help handle out-of-vocabulary words.",
    "Pretraining datasets for modern LLMs can exceed several trillion tokens in total size.",
    "Careful data curation involves language detection, quality filtering, and toxicity removal.",
    "The Pile dataset combines 22 diverse high-quality subsets for language model training.",
    "Repetition and redundancy in training data can lead to memorization and reduced generalization.",
    "Scaling laws suggest that model performance improves predictably with more data and compute.",
    "Data mixing ratios between different sources significantly impact model capabilities.",
    "Deduplication of training data reduces memorization and improves generalization performance.",
    "Multilingual datasets require balanced sampling to avoid performance degradation on low-resource languages.",
    "Code data in pretraining improves reasoning capabilities and structured generation.",
]


# ---------------------------------------------------------------------------
# 1. Token counting (word-level and subword-level approximation)
# ---------------------------------------------------------------------------


def count_tokens(
    texts: Sequence[str],
    method: str = "word",
) -> dict[str, int]:
    """Count tokens in a collection of texts.

    Supports two methods:
        - ``"word"``: split on whitespace (simple word-level tokenization).
        - ``"subword"``: approximate BPE-style tokens by splitting on whitespace
          and further breaking each "word" into character 3-grams. This is NOT
          a real BPE tokenizer but gives a rough estimate of subword token count.

    Args:
        texts: A sequence of text strings.
        method: Tokenization method, either ``"word"`` or ``"subword"``.

    Returns:
        A dictionary with keys ``"total_tokens"``, ``"num_documents"``,
        ``"avg_tokens_per_doc"``, and ``"method"``.
    """
    tokenized_docs: list[list[str]] = []

    for text in texts:
        if method == "word":
            tokens = text.split()
        elif method == "subword":
            tokens = _simulate_subword_tokens(text)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'word' or 'subword'.")
        tokenized_docs.append(tokens)

    total_tokens = sum(len(t) for t in tokenized_docs)
    num_docs = len(texts)

    return {
        "method": method,
        "total_tokens": total_tokens,
        "num_documents": num_docs,
        "avg_tokens_per_doc": round(total_tokens / num_docs, 2) if num_docs > 0 else 0,
    }


def _simulate_subword_tokens(text: str) -> list[str]:
    """Approximate subword tokenization by breaking words into 3-gram chunks.

    This gives a rough upper-bound estimate of how many BPE tokens a word would
    produce, useful for comparing word-level vs subword-level corpus sizes.
    """
    words = text.split()
    subwords: list[str] = []
    for word in words:
        if len(word) <= 3:
            subwords.append(word)
        else:
            # Simulate BPE merges by breaking long words into overlapping trigrams
            for i in range(0, len(word) - 2, 2):
                subwords.append(word[i : i + 3])
            # Add remaining tail
            remainder = len(word) % 2
            if remainder:
                subwords.append(word[-1])
    return subwords


# ---------------------------------------------------------------------------
# 2. Vocabulary diversity (Type-Token Ratio)
# ---------------------------------------------------------------------------


def compute_type_token_ratio(
    texts: Sequence[str],
    lowercase: bool = True,
) -> dict[str, float]:
    """Compute the Type-Token Ratio (TTR) as a measure of lexical diversity.

    TTR = number of unique word types / total number of word tokens.
    Higher TTR = more diverse vocabulary. Low TTR = repetitive.

    Args:
        texts: A sequence of text strings.
        lowercase: If True, normalize all words to lowercase before counting.

    Returns:
        A dictionary with keys ``"num_types"``, ``"num_tokens"``,
        ``"type_token_ratio"``, and ``"lowercase"``.
    """
    all_tokens: list[str] = []
    for text in texts:
        words = text.split()
        if lowercase:
            words = [w.lower() for w in words]
        all_tokens.extend(words)

    types = set(all_tokens)
    num_types = len(types)
    num_tokens = len(all_tokens)

    return {
        "lowercase": lowercase,
        "num_types": num_types,
        "num_tokens": num_tokens,
        "type_token_ratio": round(num_types / num_tokens, 6) if num_tokens > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Sentence length distribution
# ---------------------------------------------------------------------------


def compute_sentence_length_distribution(
    texts: Sequence[str],
    num_buckets: int = 5,
) -> dict[str, object]:
    """Compute sentence length statistics and bucketed distribution.

    Sentences are split by ``.``, ``!``, ``?``.

    Args:
        texts: A sequence of text strings.
        num_buckets: Number of histogram buckets for the distribution.

    Returns:
        A dictionary with ``"num_sentences"``, ``"mean"``, ``"median"``,
        ``"min"``, ``"max"``, ``"std"``, and ``"histogram"`` (list of
        ``(bucket_label, count)`` tuples).
    """
    import re

    lengths: list[int] = []
    for text in texts:
        # Split on sentence-ending punctuation
        sentences = re.split(r"[.!?]+", text)
        for sent in sentences:
            words = sent.strip().split()
            if words:
                lengths.append(len(words))

    if not lengths:
        return {
            "num_sentences": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "std": 0.0,
            "histogram": [],
        }

    sorted_lengths = sorted(lengths)
    n = len(lengths)
    mean = sum(lengths) / n
    median = (
        sorted_lengths[n // 2]
        if n % 2 == 1
        else (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2
    )
    variance = sum((x - mean) ** 2 for x in lengths) / n
    std = math.sqrt(variance)

    # Bucket into num_buckets histogram
    min_len = sorted_lengths[0]
    max_len = sorted_lengths[-1]
    if max_len == min_len:
        bucket_size = 1
    else:
        bucket_size = (max_len - min_len) / num_buckets

    histogram: list[tuple[str, int]] = []
    for b in range(num_buckets):
        low = min_len + b * bucket_size
        high = low + bucket_size
        # Last bucket is inclusive on both ends
        if b == num_buckets - 1:
            count = sum(1 for x in lengths if low <= x <= high)
        else:
            count = sum(1 for x in lengths if low <= x < high)
        label = (
            f"[{low:.0f}, {high:.0f})"
            if b < num_buckets - 1
            else f"[{low:.0f}, {high:.0f}]"
        )
        histogram.append((label, count))

    return {
        "num_sentences": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "min": min_len,
        "max": max_len,
        "std": round(std, 2),
        "histogram": histogram,
    }


# ---------------------------------------------------------------------------
# 4. Data quality: perplexity approximation via n-gram frequencies
# ---------------------------------------------------------------------------


def estimate_perplexity(
    texts: Sequence[str],
    n: int = 2,
) -> dict[str, float]:
    """Estimate a simple n-gram perplexity as a proxy for data quality.

    Low n-gram perplexity on the training corpus means the data is internally
    predictable and may be lower-quality (repetitive, formulaic). High
    perplexity means high diversity – generally desirable for pretraining,
    though extremely high values may indicate noise.

    Uses a simple unsmoothed n-gram language model over words:
        p(w_i | w_{i-n+1} ... w_{i-1}) = count(ngram) / count(context)

    Perplexity = exp(-1/M * sum log p(w_i)), where M is total tokens evaluated.

    Args:
        texts: A sequence of text strings.
        n: The n-gram order (2 = bigram, 3 = trigram, etc.).

    Returns:
        A dictionary with ``"n"``, ``"perplexity"``, ``"vocab_size"``,
        ``"num_ngrams"``, and ``"num_evaluated"``.
    """
    if n < 2:
        raise ValueError("n must be >= 2 for perplexity estimation")

    # Tokenize all texts into words
    all_words: list[str] = []
    for text in texts:
        all_words.extend(text.split())

    # Build n-gram frequency counts
    ngram_counts: dict[tuple[str, ...], int] = collections.Counter()
    context_counts: dict[tuple[str, ...], int] = collections.Counter()

    for i in range(len(all_words) - n + 1):
        ngram = tuple(all_words[i : i + n])
        ngram_counts[ngram] += 1

    for i in range(len(all_words) - n + 1):
        context = tuple(all_words[i : i + n - 1])
        context_counts[context] += 1

    # Evaluate: compute log probabilities for all n-gram positions
    total_log_prob = 0.0
    num_evaluated = 0

    for i in range(len(all_words) - n + 1):
        ngram = tuple(all_words[i : i + n])
        context = tuple(all_words[i : i + n - 1])
        count = ngram_counts.get(ngram, 0)
        ctx_count = context_counts.get(context, 0)

        if ctx_count > 0:
            # Unscented MLE: p = count / context_count
            prob = count / ctx_count
            if prob > 0:
                total_log_prob += math.log(prob)
                num_evaluated += 1

    if num_evaluated == 0:
        return {
            "n": n,
            "perplexity": float("inf"),
            "vocab_size": len(set(all_words)),
            "num_ngrams": len(ngram_counts),
            "num_evaluated": 0,
        }

    avg_log_prob = total_log_prob / num_evaluated
    perplexity = math.exp(-avg_log_prob)

    return {
        "n": n,
        "perplexity": round(perplexity, 2),
        "vocab_size": len(set(all_words)),
        "num_ngrams": len(ngram_counts),
        "num_evaluated": num_evaluated,
    }


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 65)
    print("Dataset Statistics for LLM Pretraining")
    print("=" * 65)
    print(f"Corpus size: {len(_SAMPLE_CORPUS)} documents\n")

    # 1. Token counts
    print("--- 1. Token Counts ---")
    for method in ("word", "subword"):
        stats = count_tokens(_SAMPLE_CORPUS, method=method)
        print(f"  Method:      {stats['method']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Documents:   {stats['num_documents']}")
        print(f"  Avg/Doc:     {stats['avg_tokens_per_doc']}")
        print()

    # 2. Type-Token Ratio
    print("--- 2. Vocabulary Diversity (Type-Token Ratio) ---")
    ttr = compute_type_token_ratio(_SAMPLE_CORPUS, lowercase=True)
    print(f"  Unique types:      {ttr['num_types']}")
    print(f"  Total tokens:      {ttr['num_tokens']}")
    print(f"  Type-Token Ratio:   {ttr['type_token_ratio']}")
    print(f"  (Lowercase: {ttr['lowercase']})")
    print()

    # 3. Sentence length distribution
    print("--- 3. Sentence Length Distribution ---")
    sld = compute_sentence_length_distribution(_SAMPLE_CORPUS, num_buckets=5)
    print(f"  Sentences: {sld['num_sentences']}")
    print(f"  Mean:      {sld['mean']} words")
    print(f"  Median:    {sld['median']} words")
    print(f"  Min:       {sld['min']} words")
    print(f"  Max:       {sld['max']} words")
    print(f"  Std:       {sld['std']} words")
    print("  Histogram (sentence length buckets):")
    for label, count in sld["histogram"]:
        bar = "#" * count
        print(f"    {label:>15s}: {count:2d} {bar}")
    print()

    # 4. Perplexity estimation (bigram)
    print("--- 4. Perplexity Approximation (n-gram based) ---")
    ppl = estimate_perplexity(_SAMPLE_CORPUS, n=2)
    print(f"  n-gram order: {ppl['n']}")
    print(f"  Vocab size:   {ppl['vocab_size']}")
    print(f"  Unique ngrams:{ppl['num_ngrams']}")
    print(f"  Evaluated:    {ppl['num_evaluated']}")
    print(f"  Perplexity:   {ppl['perplexity']}")
    print(
        "  (Higher = more diverse/less predictable = generally better for pretraining)"
    )

    print()
    print("Demo complete. All statistics computed from in-memory data.")


if __name__ == "__main__":
    main()
