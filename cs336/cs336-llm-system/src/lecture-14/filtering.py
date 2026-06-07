"""
Data filtering utilities for LLM pretraining corpora.

Implements three complementary filters:
1. Language detection via character frequency profiles
2. Quality filtering via perplexity ratio (n-gram model)
3. MinHash-based approximate near-duplicate detection
"""

from __future__ import annotations

import collections
import hashlib
import math
import random
import struct
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Language detection via character frequency profiles
# ---------------------------------------------------------------------------

# Character ranges that are indicative of each writing system.
# A more sophisticated approach could use n-gram frequency profiles,
# but character-set heuristics are lightweight and work well in practice.
_LANGUAGE_CHAR_MAPS: Dict[str, str] = {
    "en": "English (Latin script)",
    "zh": "Chinese (CJK ideographs)",
    "ja": "Japanese (Hiragana + Katakana)",
    "ko": "Korean (Hangul syllables)",
    "ar": "Arabic (Arabic script)",
    "ru": "Russian (Cyrillic script)",
}


def _count_script_blocks(text: str) -> Dict[str, int]:
    """Count characters belonging to each script block.

    Returns a dictionary mapping script category names to counts.
    The categories are the same keys used in ``_LANGUAGE_CHAR_MAPS``
    plus an ``"other"`` bucket.

    Notes
    -----
    - CJK Unified Ideographs are counted in a shared ``"cjk"`` bucket
      because they are used by both Chinese and Japanese.  The final
      classification later adjudicates between the two by checking for
      the presence of kana (Japanese) vs pure CJK (Chinese).
    - Latin-script letters (including accented variants) count as ``"en"``.
    """
    counts: Dict[str, int] = collections.defaultdict(int)
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or ch.isdigit():
            continue
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            counts["cjk"] += 1  # CJK Unified Ideographs + Ext-A
        elif 0x3040 <= cp <= 0x309F:  # Hiragana
            counts["ja"] += 1
        elif 0x30A0 <= cp <= 0x30FF:  # Katakana
            counts["ja"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:  # Hangul syllables
            counts["ko"] += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            counts["ar"] += 1  # Arabic + Supplement
        elif 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
            counts["ru"] += 1  # Cyrillic + Supplement
        elif 0x0041 <= cp <= 0x007A:  # Basic Latin letters (A-Z, a-z)
            counts["en"] += 1
        elif 0x00C0 <= cp <= 0x024F:  # Latin-1 Supplement + Extended-A/B
            counts["en"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def _english_ngram_profile() -> Dict[str, float]:
    """Return a rough English bigram frequency profile (log probabilities).

    These are approximate values derived from a small English corpus
    and are sufficient for distinguishing English from noise.
    """
    # Log-probabilities for common English bigrams (character-level).
    # Smoothed with a small constant to avoid -inf.
    common_bigrams: Dict[str, float] = {
        "th": math.log(0.035),
        "he": math.log(0.030),
        "in": math.log(0.022),
        "er": math.log(0.020),
        "an": math.log(0.019),
        "on": math.log(0.019),
        "at": math.log(0.017),
        "en": math.log(0.016),
        "nd": math.log(0.016),
        "ti": math.log(0.016),
        "es": math.log(0.015),
        "or": math.log(0.015),
        "te": math.log(0.014),
        "of": math.log(0.014),
        "ed": math.log(0.013),
        "is": math.log(0.013),
        "it": math.log(0.013),
        "al": math.log(0.012),
        "ar": math.log(0.012),
        "st": math.log(0.012),
        "to": math.log(0.012),
        "nt": math.log(0.012),
        "ng": math.log(0.011),
        "se": math.log(0.011),
        "ha": math.log(0.011),
        "as": math.log(0.010),
        "ou": math.log(0.010),
        "io": math.log(0.010),
        "le": math.log(0.010),
        "ve": math.log(0.010),
        "co": math.log(0.010),
        "me": math.log(0.010),
        "de": math.log(0.009),
        "hi": math.log(0.009),
        "ri": math.log(0.009),
        "ro": math.log(0.009),
        "ic": math.log(0.008),
        "ne": math.log(0.008),
        "ea": math.log(0.008),
        "ra": math.log(0.008),
        "ce": math.log(0.008),
        "li": math.log(0.008),
        "ch": math.log(0.008),
        "ll": math.log(0.008),
        "be": math.log(0.007),
        "ma": math.log(0.007),
        "si": math.log(0.007),
        "om": math.log(0.007),
        "ur": math.log(0.007),
    }
    default_logprob = math.log(0.001)  # smoothing
    return collections.defaultdict(lambda: default_logprob, common_bigrams)


_EN_PROFILE = _english_ngram_profile()


def detect_language(text: str) -> Tuple[str, float]:
    """Detect the most likely language of *text*.

    Uses a two-tier approach:
    1. If non-Latin scripts dominate, classify by character set.
    2. For Latin-script text, compute the average bigram log-probability
       against an English profile and compare with a threshold.

    Returns
    -------
    (language_code, confidence_score)
        ``language_code`` is one of ``"en"``, ``"zh"``, ``"ja"``, ``"ko"``,
        ``"ar"``, ``"ru"``, or ``"unknown"``.  ``confidence_score`` is a
        float between 0 and 1.
    """
    if not text.strip():
        return ("unknown", 0.0)

    script_counts = _count_script_blocks(text)
    total = sum(script_counts.values())
    if total == 0:
        return ("unknown", 0.0)

    # --- Non-Latin script classification ---
    cjk_count = script_counts.get("cjk", 0)
    ja_kana_count = script_counts.get("ja", 0)  # hiragana + katakana
    ko_count = script_counts.get("ko", 0)
    ar_count = script_counts.get("ar", 0)
    ru_count = script_counts.get("ru", 0)

    non_latin_total = cjk_count + ja_kana_count + ko_count + ar_count + ru_count
    if non_latin_total > total * 0.5:
        # Adjudicate between Chinese and Japanese:
        # - Japanese has *both* CJK ideographs and kana.
        # - Chinese primarily uses CJK ideographs with no/minimal kana.
        if ja_kana_count > 0 and cjk_count > 0:
            conf = (cjk_count + ja_kana_count) / total
            return ("ja", conf)
        elif cjk_count > 0:
            conf = cjk_count / total
            return ("zh", conf)
        # Other scripts: pick the dominant one
        dominant = max(
            [("ko", ko_count), ("ar", ar_count), ("ru", ru_count)],
            key=lambda x: x[1],
        )
        if dominant[1] > 0:
            return (dominant[0], dominant[1] / total)

    # Latin-dominant: use bigram profile scoring
    lower = text.lower()
    bigram_count = 0
    total_logprob = 0.0
    for i in range(len(lower) - 1):
        bigram = lower[i : i + 2]
        if bigram[0].isalpha() and bigram[1].isalpha():
            bigram_count += 1
            total_logprob += _EN_PROFILE[bigram]

    if bigram_count == 0:
        return ("unknown", 0.0)

    avg_logprob = total_logprob / bigram_count
    # Convert to a confidence score via sigmoid-like scaling.
    # English text typically averages around -3.5 to -4.0.
    # Random strings are much lower (< -6).
    # Center the sigmoid at -4.5 so that typical English (-3.5..-4.0)
    # scores high confidence and random strings score low.
    confidence = 1.0 / (1.0 + math.exp(-(avg_logprob + 4.5) * 2.5))
    if confidence > 0.5:
        return ("en", confidence)
    else:
        return ("unknown", confidence)


# ---------------------------------------------------------------------------
# 2. Quality filtering via perplexity ratio
# ---------------------------------------------------------------------------


class NGramModel:
    """A simple character-level n-gram language model with add-k smoothing.

    Parameters
    ----------
    n : int
        Order of the n-gram (e.g. 2 for bigram, 3 for trigram).
    k : float
        Add-k smoothing constant (default 0.1).
    """

    def __init__(self, n: int = 3, k: float = 0.1) -> None:
        self._n = n
        self._k = k
        # context (n-1 chars) -> dict of next-char counts
        self._counts: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self._context_totals: Dict[str, int] = collections.defaultdict(int)
        self._vocab: Set[str] = set()

    def train(self, texts: List[str]) -> None:
        """Build n-gram counts from a list of *texts*."""
        for text in texts:
            # Pad with start/end markers
            padded = ("<s>" * (self._n - 1)) + text + "</s>"
            for i in range(len(padded) - self._n + 1):
                ctx = padded[i : i + self._n - 1]
                nxt = padded[i + self._n - 1]
                self._counts[ctx][nxt] += 1
                self._context_totals[ctx] += 1
                self._vocab.add(nxt)

    def logprob(self, context: str, char: str) -> float:
        """Return log-probability of *char* given *context*."""
        total = self._context_totals.get(context, 0)
        if total == 0:
            # Unseen context – uniform over vocabulary
            V = max(len(self._vocab), 1)
            return math.log(self._k / (self._k * V))
        count = self._counts.get(context, {}).get(char, 0)
        V = len(self._vocab)
        return math.log((count + self._k) / (total + self._k * V))

    def perplexity(self, text: str) -> float:
        """Compute perplexity of *text* under this model."""
        if len(text) < self._n:
            return float("inf")
        padded = ("<s>" * (self._n - 1)) + text + "</s>"
        total_logprob = 0.0
        tokens = 0
        for i in range(len(padded) - self._n + 1):
            ctx = padded[i : i + self._n - 1]
            nxt = padded[i + self._n - 1]
            total_logprob += self.logprob(ctx, nxt)
            tokens += 1
        avg_neg_logprob = -total_logprob / tokens
        return math.exp(avg_neg_logprob)


def quality_filter_by_perplexity(
    text: str,
    clean_model: NGramModel,
    baseline_perplexity: float,
    max_ratio: float = 2.0,
) -> Tuple[bool, float]:
    """Filter text by comparing its perplexity against a baseline.

    Text whose perplexity exceeds ``baseline_perplexity * max_ratio`` is
    considered low quality (often repetitive, garbled, or not fluent).

    Returns
    -------
    (keep, perplexity)
        ``keep`` is ``True`` if the text passes the quality filter.
    """
    if len(text.strip()) < 20:
        return (False, float("inf"))
    ppl = clean_model.perplexity(text)
    ratio = ppl / baseline_perplexity if baseline_perplexity > 0 else float("inf")
    return (ratio <= max_ratio, ppl)


# ---------------------------------------------------------------------------
# 3. MinHash-based approximate deduplication
# ---------------------------------------------------------------------------


def _fnv1a_32(data: bytes, seed: int = 0x811C9DC5) -> int:
    """FNV-1a 32-bit hash."""
    h = seed
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class MinHash:
    """MinHash sketch for Jaccard similarity estimation.

    Parameters
    ----------
    num_perm : int
        Number of permutations (i.e. the sketch size).
    """

    def __init__(self, num_perm: int = 128) -> None:
        self._num_perm = num_perm
        # Two random seeds per permutation to create a pseudo-permutation
        rng = random.Random(42)
        self._seeds_a: List[int] = [rng.randint(1, 2**31 - 1) for _ in range(num_perm)]
        self._seeds_b: List[int] = [rng.randint(1, 2**31 - 1) for _ in range(num_perm)]
        # Sketch values initialised to max
        self._hashes: List[int] = [0xFFFFFFFF] * num_perm

    def update(self, token: str) -> None:
        """Add a single token (shingle) to the sketch."""
        raw = token.encode("utf-8")
        for i in range(self._num_perm):
            h = _fnv1a_32(raw, self._seeds_a[i])
            h = ((h ^ self._seeds_b[i]) * 0x01000193) & 0xFFFFFFFF
            if h < self._hashes[i]:
                self._hashes[i] = h

    def update_batch(self, tokens: List[str]) -> None:
        """Add multiple tokens at once."""
        for t in tokens:
            self.update(t)

    def digest(self) -> List[int]:
        """Return the MinHash signature (list of 32-bit integers)."""
        return list(self._hashes)

    def jaccard(self, other: "MinHash") -> float:
        """Estimate Jaccard similarity with *other* MinHash."""
        if self._num_perm != other._num_perm:
            raise ValueError("MinHash sketches must have the same num_perm")
        matches = sum(1 for a, b in zip(self._hashes, other._hashes) if a == b)
        return matches / self._num_perm

    @classmethod
    def from_tokens(cls, tokens: List[str], num_perm: int = 128) -> "MinHash":
        """Create a MinHash sketch from a list of tokens."""
        mh = cls(num_perm)
        mh.update_batch(tokens)
        return mh


def _shingle(text: str, k: int = 5) -> List[str]:
    """Extract character k-shingles from *text*, lowercased."""
    lower = text.lower()
    return [lower[i : i + k] for i in range(len(lower) - k + 1)]


def find_near_duplicates(
    documents: List[str],
    num_perm: int = 128,
    threshold: float = 0.8,
) -> List[Tuple[int, int, float]]:
    """Find near-duplicate pairs among *documents* using MinHash.

    Parameters
    ----------
    documents : List[str]
        List of document strings.
    num_perm : int
        Number of MinHash permutations.
    threshold : float
        Jaccard similarity threshold above which a pair is flagged.

    Returns
    -------
    List of ``(idx_a, idx_b, estimated_jaccard)`` for duplicate pairs.
    """
    sketches: List[MinHash] = []
    for doc in documents:
        tokens = _shingle(doc)
        sketches.append(MinHash.from_tokens(tokens, num_perm))

    pairs: List[Tuple[int, int, float]] = []
    for i in range(len(sketches)):
        for j in range(i + 1, len(sketches)):
            jac = sketches[i].jaccard(sketches[j])
            if jac >= threshold:
                pairs.append((i, j, jac))
    return pairs


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate language detection, perplexity filtering, and MinHash dedup."""
    # ---- Language detection ----
    print("=" * 60)
    print("LANGUAGE DETECTION")
    print("=" * 60)
    samples: Dict[str, str] = {
        "en": "The quick brown fox jumps over the lazy dog. "
        "Natural language processing is a fascinating field of study.",
        "zh": "自然语言处理是人工智能的一个重要分支，它研究如何让计算机理解人类语言。",
        "ja": "自然言語処理は人工知能の一分野であり、コンピュータが人間の言語を理解する方法を研究します。",
        "ko": "자연어 처리는 인공지능의 한 분야로, 컴퓨터가 인간의 언어를 이해하는 방법을 연구합니다.",
        "ar": "معالجة اللغات الطبيعية هي فرع من فروع الذكاء الاصطناعي.",
        "ru": "Обработка естественного языка - это область искусственного интеллекта.",
    }
    for lang, text in samples.items():
        detected, conf = detect_language(text)
        status = "✓" if detected == lang else "✗"
        print(
            f"  [{status}] Expected: {lang:<4}  Detected: {detected:<7}  "
            f"Confidence: {conf:.3f}"
        )

    # ---- Quality filtering (perplexity) ----
    print("\n" + "=" * 60)
    print("QUALITY FILTERING (Perplexity Ratio)")
    print("=" * 60)
    clean_corpus = [
        "the quick brown fox jumps over the lazy dog",
        "natural language processing is a fascinating field of study",
        "machine learning has transformed many industries in recent years",
        "deep neural networks can learn complex patterns from large datasets",
        "the weather today is sunny and warm with a gentle breeze",
    ]
    model = NGramModel(n=3, k=0.1)
    model.train(clean_corpus)

    # Compute baseline perplexity on the training data
    baseline_ppls = [model.perplexity(t) for t in clean_corpus]
    baseline = sum(baseline_ppls) / len(baseline_ppls)
    print(f"  Baseline perplexity (avg of clean corpus): {baseline:.2f}")

    test_texts = [
        ("Clean English", "the quick brown fox runs through the green forest"),
        ("Repetitive", "the the the the the the the the the the the the the"),
        ("Random chars", "asdf qwer zxcv poiuy lkjhg mnbvc xz"),
    ]
    for label, text in test_texts:
        keep, ppl = quality_filter_by_perplexity(text, model, baseline)
        status = "PASS" if keep else "FAIL"
        print(f"  [{status}] {label:<20}  PPL={ppl:.2f}  Ratio={ppl / baseline:.2f}")

    # ---- MinHash deduplication ----
    print("\n" + "=" * 60)
    print("MINHASH DEDUPLICATION")
    print("=" * 60)
    docs = [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "The quick brown fox jumps over the lazy dog near the river bank.",  # exact dup
        "A completely different document about machine learning and AI systems.",
        "The quick brown fox jumps over the lazy dog near the riverbank.",  # near dup
        "Another unique document discussing climate change and global warming.",
    ]
    pairs = find_near_duplicates(docs, num_perm=128, threshold=0.7)
    if pairs:
        for i, j, jac in pairs:
            print(f"  Near-duplicate: doc[{i}] <-> doc[{j}]  (est. Jaccard={jac:.4f})")
    else:
        print("  No near-duplicates found.")
    print(
        f"  (Scanned {len(docs)} documents, {len(docs) * (len(docs) - 1) // 2} pairs)"
    )


if __name__ == "__main__":
    main()
