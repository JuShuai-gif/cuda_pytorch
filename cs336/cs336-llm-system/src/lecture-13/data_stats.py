"""
分析与 LLM 预训练相关的数据集统计信息。

所有函数均基于合成的小型内存数据运行，因此无需下载大规模语料库即可执行。

涵盖的统计信息：
    - 词元计数（词级和子词级近似）
    - 词汇多样性（type-token ratio）
    - 句子长度分布
    - 通过 n-gram 频率 perplexity 近似进行数据质量估计
"""

from __future__ import annotations

import collections
import math
from typing import Sequence


# ---------------------------------------------------------------------------
# 用于演示的合成数据
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
# 1. 词元计数（词级和子词级近似）
# ---------------------------------------------------------------------------


def count_tokens(
    texts: Sequence[str],
    method: str = "word",
) -> dict[str, int]:
    """统计文本集合中的词元数量。

    支持两种方法：
        - ``"word"``：按空白字符分割（简单的词级 tokenization）。
        - ``"subword"``：通过按空白字符分割后，再将每个"词"拆分为
          字符级 3-gram 来近似 BPE 风格的 tokenization。这并非真正的
          BPE tokenizer，但可以给出子词 token 数量的粗略估计。

    Args:
        texts: 文本字符串序列。
        method: Tokenization 方法，可选 ``"word"`` 或 ``"subword"``。

    Returns:
        一个字典，包含键 ``"total_tokens"``、``"num_documents"``、
        ``"avg_tokens_per_doc"`` 和 ``"method"``。
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
    """通过将单词拆分为 3-gram 片段来近似子词 tokenization。

    这给出了一个词大致会产生多少 BPE token 的粗略上界估计，
    对于比较词级与子词级语料库大小很有用。
    """
    words = text.split()
    subwords: list[str] = []
    for word in words:
        if len(word) <= 3:
            subwords.append(word)
        else:
            # 通过将长单词拆分为重叠的 trigram 来模拟 BPE 合并
            for i in range(0, len(word) - 2, 2):
                subwords.append(word[i : i + 3])
            # 添加剩余尾部
            remainder = len(word) % 2
            if remainder:
                subwords.append(word[-1])
    return subwords


# ---------------------------------------------------------------------------
# 2. 词汇多样性（Type-Token Ratio）
# ---------------------------------------------------------------------------


def compute_type_token_ratio(
    texts: Sequence[str],
    lowercase: bool = True,
) -> dict[str, float]:
    """计算 Type-Token Ratio (TTR) 作为词汇多样性的度量指标。

    TTR = 唯一词类型数 / 总词元数（token 数）。
    TTR 越高 = 词汇越多样。TTR 越低 = 重复度高。

    Args:
        texts: 文本字符串序列。
        lowercase: 如果为 True，计数前将所有词归一化为小写。

    Returns:
        一个字典，包含键 ``"num_types"``、``"num_tokens"``、
        ``"type_token_ratio"`` 和 ``"lowercase"``。
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
# 3. 句子长度分布
# ---------------------------------------------------------------------------


def compute_sentence_length_distribution(
    texts: Sequence[str],
    num_buckets: int = 5,
) -> dict[str, object]:
    """计算句子长度统计和分桶分布。

    句子按 ``.``、``!``、``?`` 进行分割。

    Args:
        texts: 文本字符串序列。
        num_buckets: 分布直方图的桶数。

    Returns:
        一个字典，包含 ``"num_sentences"``、``"mean"``、``"median"``、
        ``"min"``、``"max"``、``"std"`` 和 ``"histogram"``
        （``(bucket_label, count)`` 元组列表）。
    """
    import re

    lengths: list[int] = []
    for text in texts:
        # 按句子结束标点分割
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

    # 分桶到 num_buckets 个直方图桶
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
        # 最后一个桶两端都包含
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
# 4. 数据质量：通过 n-gram 频率进行 perplexity 近似
# ---------------------------------------------------------------------------


def estimate_perplexity(
    texts: Sequence[str],
    n: int = 2,
) -> dict[str, float]:
    """估计简单的 n-gram perplexity 作为数据质量的代理指标。

    训练语料库上的低 n-gram perplexity 意味着数据内部可预测性强，
    可能是低质量数据（重复、公式化）。高 perplexity 意味着高多样性
    —— 这通常是预训练所期望的，但极高的值可能表明存在噪声。

    使用基于词的简单无平滑 n-gram 语言模型：
        p(w_i | w_{i-n+1} ... w_{i-1}) = count(ngram) / count(context)

    Perplexity = exp(-1/M * sum log p(w_i))，其中 M 为评估的总 token 数。

    Args:
        texts: 文本字符串序列。
        n: n-gram 阶数（2 = bigram, 3 = trigram 等）。

    Returns:
        一个字典，包含 ``"n"``、``"perplexity"``、``"vocab_size"``、
        ``"num_ngrams"`` 和 ``"num_evaluated"``。
    """
    if n < 2:
        raise ValueError("n must be >= 2 for perplexity estimation")

    # 将所有文本 tokenize 为词
    all_words: list[str] = []
    for text in texts:
        all_words.extend(text.split())

    # 构建 n-gram 频率计数
    ngram_counts: dict[tuple[str, ...], int] = collections.Counter()
    context_counts: dict[tuple[str, ...], int] = collections.Counter()

    for i in range(len(all_words) - n + 1):
        ngram = tuple(all_words[i : i + n])
        ngram_counts[ngram] += 1

    for i in range(len(all_words) - n + 1):
        context = tuple(all_words[i : i + n - 1])
        context_counts[context] += 1

    # 评估：计算所有 n-gram 位置的对数概率
    total_log_prob = 0.0
    num_evaluated = 0

    for i in range(len(all_words) - n + 1):
        ngram = tuple(all_words[i : i + n])
        context = tuple(all_words[i : i + n - 1])
        count = ngram_counts.get(ngram, 0)
        ctx_count = context_counts.get(context, 0)

        if ctx_count > 0:
            # 无平滑 MLE：p = count / context_count
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
# 演示
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
