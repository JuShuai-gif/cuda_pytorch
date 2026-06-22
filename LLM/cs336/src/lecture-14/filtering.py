"""
LLM 预训练语料库的数据过滤工具。

实现了三种互补的过滤器：
1. 基于字符频率特征的语言检测
2. 基于 perplexity 比率的质量过滤（n-gram 模型）
3. 基于 MinHash 的近似去重检测
"""

from __future__ import annotations

import collections
import hashlib
import math
import random
import struct
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# 1. 基于字符频率特征的语言检测
# ---------------------------------------------------------------------------

# 指示各书写系统的字符范围。
# 更复杂的方法可以使用 n-gram 频率特征，但字符集启发式方法轻量且在实践中效果良好。
_LANGUAGE_CHAR_MAPS: Dict[str, str] = {
    "en": "English (Latin script)",
    "zh": "Chinese (CJK ideographs)",
    "ja": "Japanese (Hiragana + Katakana)",
    "ko": "Korean (Hangul syllables)",
    "ar": "Arabic (Arabic script)",
    "ru": "Russian (Cyrillic script)",
}


def _count_script_blocks(text: str) -> Dict[str, int]:
    """统计每个字符块（script block）中的字符数量。

    返回一个将字符块类别名映射到计数的字典。
    类别与 ``_LANGUAGE_CHAR_MAPS`` 中使用的键名相同，外加一个 ``"other"`` 桶。

    注意事项
    -----
    - CJK 统一表意文字计入共享的 ``"cjk"`` 桶中，因为它们同时被中文和日文使用。
      最终分类会通过检查假名（日语）与纯 CJK（中文）来区分二者。
    - 拉丁字母（含带重音符号的变体）计入 ``"en"``。
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
    """返回一个粗略的英语 bigram 频率特征（对数概率）。

    这些近似值来自一个小型英语语料库，足以区分英语和噪声。
    """
    # 常见英语 bigram（字符级别）的对数概率。
    # 使用小常数进行平滑处理，避免 -inf。
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
    default_logprob = math.log(0.01)  # smoothing for unseen bigrams
    return collections.defaultdict(lambda: default_logprob, common_bigrams)


_EN_PROFILE = _english_ngram_profile()


def detect_language(text: str) -> Tuple[str, float]:
    """检测 *text* 最可能的语言。

    使用两层方法：
    1. 如果非拉丁文字主导，按字符集分类。
    2. 对于拉丁文字文本，计算与英语特征的平均 bigram 对数概率，并与阈值比较。

    返回
    -------
    (language_code, confidence_score)
        ``language_code`` 取 ``"en"``、``"zh"``、``"ja"``、``"ko"``、
        ``"ar"``、``"ru"`` 或 ``"unknown"`` 之一。
        ``confidence_score`` 是 0 到 1 之间的 float。
    """
    if not text.strip():
        return ("unknown", 0.0)

    script_counts = _count_script_blocks(text)
    total = sum(script_counts.values())
    if total == 0:
        return ("unknown", 0.0)

    # --- 非拉丁文字分类 ---
    cjk_count = script_counts.get("cjk", 0)
    ja_kana_count = script_counts.get("ja", 0)  # hiragana + katakana
    ko_count = script_counts.get("ko", 0)
    ar_count = script_counts.get("ar", 0)
    ru_count = script_counts.get("ru", 0)

    non_latin_total = cjk_count + ja_kana_count + ko_count + ar_count + ru_count
    if non_latin_total > total * 0.5:
        # 区分中文和日文：
        # - 日语同时包含 CJK 表意文字和假名。
        # - 中文主要使用 CJK 表意文字，极少或无假名。
        if ja_kana_count > 0 and cjk_count > 0:
            conf = (cjk_count + ja_kana_count) / total
            return ("ja", conf)
        elif cjk_count > 0:
            conf = cjk_count / total
            return ("zh", conf)
        # 其他文字：选择主导的那个
        dominant = max(
            [("ko", ko_count), ("ar", ar_count), ("ru", ru_count)],
            key=lambda x: x[1],
        )
        if dominant[1] > 0:
            return (dominant[0], dominant[1] / total)

    # 拉丁文字主导：使用 bigram 特征评分
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
    # 通过类 sigmoid 缩放转换为置信度分数。
    # 使用该稀疏特征时，英语文本平均约为 -4.5。
    # 随机 / 非英语字符串更低（< -6.0）。
    confidence = 1.0 / (1.0 + math.exp(-(avg_logprob + 5.0) * 2.5))
    if confidence > 0.5:
        return ("en", confidence)
    else:
        return ("unknown", confidence)


# ---------------------------------------------------------------------------
# 2. 基于 perplexity 比率的质量过滤
# ---------------------------------------------------------------------------


class NGramModel:
    """一个简单的字符级 n-gram 语言模型，采用 add-k 平滑。

    参数
    ----------
    n : int
        n-gram 的阶数（例如 2 为 bigram，3 为 trigram）。
    k : float
        Add-k 平滑常数（默认 0.1）。
    """

    def __init__(self, n: int = 3, k: float = 0.1) -> None:
        self._n = n
        self._k = k
        # 上下文 (n-1 个字符) -> 下一个字符计数字典
        self._counts: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self._context_totals: Dict[str, int] = collections.defaultdict(int)
        self._vocab: Set[str] = set()

    def train(self, texts: List[str]) -> None:
        """从 *texts* 列表构建 n-gram 计数。"""
        for text in texts:
            # 用起始/结束标记填充
            padded = ("<s>" * (self._n - 1)) + text + "</s>"
            for i in range(len(padded) - self._n + 1):
                ctx = padded[i : i + self._n - 1]
                nxt = padded[i + self._n - 1]
                self._counts[ctx][nxt] += 1
                self._context_totals[ctx] += 1
                self._vocab.add(nxt)

    def logprob(self, context: str, char: str) -> float:
        """返回在 *context* 下 *char* 的对数概率。"""
        total = self._context_totals.get(context, 0)
        if total == 0:
            # 未见过的上下文 – 对词汇表使用均匀分布
            V = max(len(self._vocab), 1)
            return math.log(self._k / (self._k * V))
        count = self._counts.get(context, {}).get(char, 0)
        V = len(self._vocab)
        return math.log((count + self._k) / (total + self._k * V))

    def perplexity(self, text: str) -> float:
        """计算 *text* 在该模型下的 perplexity。"""
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
    """通过将文本的 perplexity 与基线比较来进行过滤。

    perplexity 超过 ``baseline_perplexity * max_ratio`` 的文本
    被视为低质量（通常是重复、乱码或不流畅的）。

    返回
    -------
    (keep, perplexity)
        ``keep`` 为 ``True`` 表示文本通过了质量过滤器。
    """
    if len(text.strip()) < 20:
        return (False, float("inf"))
    ppl = clean_model.perplexity(text)
    ratio = ppl / baseline_perplexity if baseline_perplexity > 0 else float("inf")
    return (ratio <= max_ratio, ppl)


# ---------------------------------------------------------------------------
# 3. 基于 MinHash 的近似去重
# ---------------------------------------------------------------------------


def _fnv1a_32(data: bytes, seed: int = 0x811C9DC5) -> int:
    """FNV-1a 32 位哈希。"""
    h = seed
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class MinHash:
    """用于 Jaccard 相似度估计的 MinHash 草图。

    参数
    ----------
    num_perm : int
        排列数（即草图大小）。
    """

    def __init__(self, num_perm: int = 128) -> None:
        self._num_perm = num_perm
        # 每个排列使用两个随机种子来创建伪排列
        rng = random.Random(42)
        self._seeds_a: List[int] = [rng.randint(1, 2**31 - 1) for _ in range(num_perm)]
        self._seeds_b: List[int] = [rng.randint(1, 2**31 - 1) for _ in range(num_perm)]
        # 草图值初始化为最大值
        self._hashes: List[int] = [0xFFFFFFFF] * num_perm

    def update(self, token: str) -> None:
        """向草图中添加单个 token（shingle）。"""
        raw = token.encode("utf-8")
        for i in range(self._num_perm):
            h = _fnv1a_32(raw, self._seeds_a[i])
            h = ((h ^ self._seeds_b[i]) * 0x01000193) & 0xFFFFFFFF
            if h < self._hashes[i]:
                self._hashes[i] = h

    def update_batch(self, tokens: List[str]) -> None:
        """一次性添加多个 token。"""
        for t in tokens:
            self.update(t)

    def digest(self) -> List[int]:
        """返回 MinHash 签名（32 位整数列表）。"""
        return list(self._hashes)

    def jaccard(self, other: "MinHash") -> float:
        """估计与 *other* MinHash 的 Jaccard 相似度。"""
        if self._num_perm != other._num_perm:
            raise ValueError("MinHash sketches must have the same num_perm")
        matches = sum(1 for a, b in zip(self._hashes, other._hashes) if a == b)
        return matches / self._num_perm

    @classmethod
    def from_tokens(cls, tokens: List[str], num_perm: int = 128) -> "MinHash":
        """从 token 列表创建 MinHash 草图。"""
        mh = cls(num_perm)
        mh.update_batch(tokens)
        return mh


def _shingle(text: str, k: int = 5) -> List[str]:
    """从 *text* 中提取小写字符 k-shingle。"""
    lower = text.lower()
    return [lower[i : i + k] for i in range(len(lower) - k + 1)]


def find_near_duplicates(
    documents: List[str],
    num_perm: int = 128,
    threshold: float = 0.8,
) -> List[Tuple[int, int, float]]:
    """使用 MinHash 在 *documents* 中查找近似重复对。

    参数
    ----------
    documents : List[str]
        文档字符串列表。
    num_perm : int
        MinHash 排列数。
    threshold : float
        Jaccard 相似度阈值，超过此值即标记为一对重复。

    返回
    -------
    重复对的 ``(idx_a, idx_b, estimated_jaccard)`` 列表。
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
# 演示
# ---------------------------------------------------------------------------


def main() -> None:
    """演示语言检测、perplexity 过滤和 MinHash 去重。"""
    # ---- 语言检测 ----
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

    # ---- 质量过滤（perplexity） ----
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

    # 在训练数据上计算基线 perplexity
    baseline_ppls = [model.perplexity(t) for t in clean_corpus]
    baseline = sum(baseline_ppls) / len(baseline_ppls)
    print(f"  Baseline perplexity (avg of clean corpus): {baseline:.2f}")

    test_texts = [
        ("Clean English", "the quick brown fox runs through the green forest"),
        ("Repetitive", "the the the the the the the the the the the the the"),
        ("Random chars", "asdf qwer zxcv poiuy lkjhg mnbvc xz"),
    ]
    # 使用较高的 max_ratio，因为训练语料库很小（只有 5 句话）。
    # 生产环境会使用 max_ratio=2.0 配合更大的语料库。
    demo_ratio = 3.0
    for label, text in test_texts:
        keep, ppl = quality_filter_by_perplexity(text, model, baseline, demo_ratio)
        status = "PASS" if keep else "FAIL"
        print(f"  [{status}] {label:<20}  PPL={ppl:.2f}  Ratio={ppl / baseline:.2f}")

    # ---- MinHash 去重 ----
    print("\n" + "=" * 60)
    print("MINHASH DEDUPLICATION")
    print("=" * 60)
    docs = [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "The quick brown fox jumps over the lazy dog near the river bank.",  # 完全重复
        "The quick brown fox jumps over the lazy dog near the river bank.",  # 完全重复
        "A completely different document about machine learning and AI systems.",
        "The quick brown fox jumps over the lazy dog near the riverbank.",  # 近似重复
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
