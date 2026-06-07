"""
简单的 BLEU 和 ROUGE 指标实现。

这些是文本生成最常用的自动评估指标：
  - BLEU（Bilingual Evaluation Understudy）：带简短惩罚的 N-gram 精度
  - ROUGE（Recall-Oriented Understudy for Gisting Evaluation）：N-gram 召回率，
    包括 ROUGE-1、ROUGE-2 和 ROUGE-L 变体

这是用于教育目的的*简化*独立实现。
生产环境建议使用 `sacrebleu` 和 `rouge-score` 包。

核心概念：
  - BLEU: N-gram 精度的几何平均，对过短输出进行惩罚
  - ROUGE-N: 候选文本和参考文本之间的 N-gram 召回率
  - ROUGE-L: 基于最长公共子序列（LCS）的 F-measure
  - Smoothing（平滑）: 防止无 N-gram 匹配时得分为零
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


# =========================================================================
# 分词辅助函数
# =========================================================================


def tokenize(text: str) -> list[str]:
    """简单的空白符分词器。

    生产环境建议使用合适的 tokenizer（例如 sacrebleu tokenizer）。

    Args:
        text: 输入字符串

    Returns:
        分词列表
    """
    return text.lower().split()


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """从分词列表中生成 N-gram。

    Args:
        tokens: 分词列表
        n: N-gram 大小

    Returns:
        N-gram 元组列表
    """
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# =========================================================================
# BLEU
# =========================================================================


def _modified_precision(
    candidate: list[str],
    reference: list[str],
    n: int,
) -> tuple[float, int, int]:
    """为单个参考文本计算修改后的 N-gram 精度。

    修改后的精度将每个候选 N-gram 的计数裁剪为
    其在参考文本中的最大出现次数。

    Args:
        candidate: 候选文本分词列表
        reference: 参考文本分词列表
        n: N-gram 大小

    Returns:
        (精度, 裁剪后的计数, 候选 N-gram 总数)
    """
    candidate_ngrams = ngrams(candidate, n)
    reference_ngrams = ngrams(reference, n)

    if not candidate_ngrams:
        return 0.0, 0, 0

    candidate_counter = Counter(candidate_ngrams)
    reference_counter = Counter(reference_ngrams)

    # 裁剪：min(候选计数, 参考文本中的最大计数)
    clipped_count = 0
    for ng, count in candidate_counter.items():
        clipped_count += min(count, reference_counter[ng])

    total = len(candidate_ngrams)
    precision = clipped_count / total if total > 0 else 0.0
    return precision, clipped_count, total


def _closest_reference_length(candidate_len: int, ref_lengths: list[int]) -> int:
    """找到与候选文本长度最接近的参考文本长度。

    Args:
        candidate_len: 候选文本的 token 长度
        ref_lengths: 各参考文本的长度列表

    Returns:
        最接近的参考文本长度
    """
    if not ref_lengths:
        return candidate_len
    return min(ref_lengths, key=lambda r: abs(r - candidate_len))


def compute_bleu(
    candidate: str | list[str],
    references: str | list[str] | list[list[str]],
    max_n: int = 4,
    smooth: bool = True,
) -> dict[str, float]:
    """
    计算带平滑的 BLEU 得分。

    BLEU = BP * exp( Σ w_n * log(p_n) )
    其中 BP 是简短惩罚，w_n = 1/N，p_n 是修改后的 N-gram 精度。

    Args:
        candidate: 候选文本（字符串或已分词）
        references: 参考文本。可以是单个字符串、字符串列表
                    或已分词的参考文本列表的列表
        max_n: 最大 N-gram 大小（默认 4 即 BLEU-4）
        smooth: 是否应用平滑以避免零分

    Returns:
        包含以下键的字典：
          "bleu": 总体 BLEU 得分
          "precisions": 每个 n 的精度值列表
          "brevity_penalty": 简短惩罚因子
          "candidate_len": 候选文本的 token 长度
          "reference_len": 有效参考文本长度
    """
    # 分词
    cand_tokens = tokenize(candidate) if isinstance(candidate, str) else candidate

    # 规范化参考文本
    if isinstance(references, str):
        refs_list: list[list[str]] = [tokenize(references)]
    elif isinstance(references, list) and references and isinstance(references[0], str):
        refs_list = [tokenize(r) for r in references]  # type: ignore[arg-type]
    else:
        refs_list = [[str(t) for t in r] for r in references]  # type: ignore[union-attr]

    ref_lengths = [len(r) for r in refs_list]

    # 计算修改后的 N-gram 精度
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        # 为每个 N-gram 使用最佳参考文本（标准 BLEU 取最大值）
        best_precision = 0.0
        for ref_tokens in refs_list:
            precision, _, _ = _modified_precision(cand_tokens, ref_tokens, n)
            best_precision = max(best_precision, precision)

        if smooth and best_precision == 0.0:
            # 平滑：添加一个类似 epsilon 的小调整
            best_precision = 1.0 / (2 ** (n + 1))

        precisions.append(best_precision)

    # 简短惩罚
    cand_len = len(cand_tokens)
    effective_ref_len = _closest_reference_length(cand_len, ref_lengths)

    if cand_len == 0:
        bp = 0.0
    elif cand_len >= effective_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - effective_ref_len / cand_len)

    # 计算 BLEU
    log_precisions = [math.log(p) if p > 0 else float("-inf") for p in precisions]
    if any(p <= 0 for p in precisions):
        bleu = 0.0
    else:
        avg_log_precision = sum(log_precisions) / max_n
        bleu = bp * math.exp(avg_log_precision)

    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp,
        "candidate_len": cand_len,
        "reference_len": effective_ref_len,
    }


# =========================================================================
# ROUGE-N
# =========================================================================


def compute_rouge_n(
    candidate: str | list[str],
    references: str | list[str] | list[list[str]],
    n: int = 1,
) -> dict[str, float]:
    """
    计算 ROUGE-N 得分（N-gram 召回率、精度、F1）。

    ROUGE-N 召回率 = Σ(min(count_cand, count_ref)) / Σ(count_ref)
    ROUGE-N 精度 = Σ(min(count_cand, count_ref)) / Σ(count_cand)

    Args:
        candidate: 候选文本
        references: 参考文本
        n: N-gram 大小（1 表示 ROUGE-1，2 表示 ROUGE-2）

    Returns:
        包含 "recall"、"precision"、"f1" 键的字典
    """
    cand_tokens = tokenize(candidate) if isinstance(candidate, str) else candidate

    # Normalize references to a single merged reference
    if isinstance(references, str):
        ref_tokens_list = [tokenize(references)]
    elif isinstance(references, list) and references and isinstance(references[0], str):
        ref_tokens_list = [tokenize(r) for r in references]  # type: ignore[arg-type]
    else:
        ref_tokens_list = [[str(t) for t in r] for r in references]  # type: ignore[union-attr]

    # 将所有参考文本合并到一个计数器中（每个 N-gram 取最大值）
    merged_ref_counter: Counter = Counter()
    for ref_tokens in ref_tokens_list:
        ref_ng = ngrams(ref_tokens, n)
        ref_counter = Counter(ref_ng)
        for ng, count in ref_counter.items():
            merged_ref_counter[ng] = max(merged_ref_counter[ng], count)

    cand_ngrams_list = ngrams(cand_tokens, n)
    cand_counter = Counter(cand_ngrams_list)

    # Overlap
    overlap = 0
    for ng, count in cand_counter.items():
        overlap += min(count, merged_ref_counter[ng])

    total_ref = sum(merged_ref_counter.values())
    total_cand = sum(cand_counter.values())

    recall = overlap / total_ref if total_ref > 0 else 0.0
    precision = overlap / total_cand if total_cand > 0 else 0.0
    f1 = (
        (2 * recall * precision / (recall + precision))
        if (recall + precision) > 0
        else 0.0
    )

    return {"recall": recall, "precision": precision, "f1": f1}


# =========================================================================
# ROUGE-L
# =========================================================================


def _lcs_length(x: list[str], y: list[str]) -> int:
    """使用动态规划计算最长公共子序列的长度。

    Args:
        x: 第一个序列
        y: 第二个序列

    Returns:
        LCS 的长度
    """
    m, n = len(x), len(y)
    # 使用一维 DP 以节省内存
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def compute_rouge_l(
    candidate: str | list[str],
    references: str | list[str] | list[list[str]],
) -> dict[str, float]:
    """
    基于最长公共子序列计算 ROUGE-L 得分。

    ROUGE-L 召回率 = LCS(cand, ref) / len(ref)
    ROUGE-L 精度 = LCS(cand, ref) / len(cand)

    Args:
        candidate: 候选文本
        references: 参考文本

    Returns:
        包含 "recall"、"precision"、"f1" 键的字典
    """
    cand_tokens = tokenize(candidate) if isinstance(candidate, str) else candidate

    # Normalize references
    if isinstance(references, str):
        ref_tokens_list = [tokenize(references)]
    elif isinstance(references, list) and references and isinstance(references[0], str):
        ref_tokens_list = [tokenize(r) for r in references]  # type: ignore[arg-type]
    else:
        ref_tokens_list = [[str(t) for t in r] for r in references]  # type: ignore[union-attr]

    # 使用最佳参考文本（最大 F1）
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0

    for ref_tokens in ref_tokens_list:
        lcs_len = _lcs_length(cand_tokens, ref_tokens)
        recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        precision = lcs_len / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        f1 = (
            (2 * recall * precision / (recall + precision))
            if (recall + precision) > 0
            else 0.0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_recall = recall
            best_precision = precision

    return {"recall": best_recall, "precision": best_precision, "f1": best_f1}


# =========================================================================
# 组合评估
# =========================================================================


def evaluate_summary(
    candidate: str,
    references: list[str],
    max_bleu_n: int = 4,
) -> dict[str, dict[str, float]]:
    """
    计算一组综合的生成指标。

    Args:
        candidate: 生成的文本
        references: 参考文本列表
        max_bleu_n: BLEU 的最大 N-gram 阶数

    Returns:
        包含 BLEU、ROUGE-1、ROUGE-2、ROUGE-L 得分的嵌套字典
    """
    return {
        "bleu": compute_bleu(candidate, references, max_n=max_bleu_n),
        "rouge-1": compute_rouge_n(candidate, references, n=1),
        "rouge-2": compute_rouge_n(candidate, references, n=2),
        "rouge-l": compute_rouge_l(candidate, references),
    }


# =========================================================================
# 演示
# =========================================================================


def demo_bleu() -> None:
    """演示 BLEU 得分计算。"""
    print("=" * 70)
    print("BLEU Score Demo")
    print("=" * 70)

    # 完全匹配
    candidate = "the cat sat on the mat"
    reference = "the cat sat on the mat"
    result = compute_bleu(candidate, reference)
    print(f"\n  Candidate:  '{candidate}'")
    print(f"  Reference:  '{reference}'")
    print(f"  BLEU: {result['bleu']:.4f}")
    print(f"  Precisions: {[f'{p:.4f}' for p in result['precisions']]}")
    print(f"  BP: {result['brevity_penalty']:.4f}")

    # 部分匹配
    candidate = "the cat sat on the mat"
    reference = "the dog sat on the rug"
    result = compute_bleu(candidate, reference)
    print(f"\n  Candidate:  '{candidate}'")
    print(f"  Reference:  '{reference}'")
    print(f"  BLEU: {result['bleu']:.4f}")
    print(f"  Precisions: {[f'{p:.4f}' for p in result['precisions']]}")

    # 短输出（简短惩罚）
    candidate = "the cat"
    reference = "the cat sat on the mat"
    result = compute_bleu(candidate, reference)
    print(f"\n  Candidate (short):  '{candidate}' (len={len(candidate.split())})")
    print(f"  Reference:          '{reference}' (len={len(reference.split())})")
    print(f"  BLEU: {result['bleu']:.4f}")
    print(f"  BP: {result['brevity_penalty']:.4f}")

    # 多个参考文本
    candidate = "the feline sits on the rug"
    refs = [
        "the cat sat on the mat",
        "the cat sits on the rug",
        "a cat is on the mat",
    ]
    result = compute_bleu(candidate, refs)
    print(f"\n  Candidate:   '{candidate}'")
    print(f"  References:  {refs}")
    print(f"  BLEU: {result['bleu']:.4f}")

    # 无重叠
    candidate = "completely different words here"
    reference = "the cat sat on the mat"
    result = compute_bleu(candidate, reference, smooth=False)
    print(f"\n  Candidate:  '{candidate}'")
    print(f"  Reference:  '{reference}'")
    print(f"  BLEU (no smooth): {result['bleu']:.4f}")

    result_s = compute_bleu(candidate, reference, smooth=True)
    print(f"  BLEU (smooth):    {result_s['bleu']:.4f}")
    print(f"  (Smoothing prevents BLEU=0 when no n-grams match)")


def demo_rouge() -> None:
    """演示 ROUGE-N 和 ROUGE-L 计算。"""
    print("\n" + "=" * 70)
    print("ROUGE Score Demo")
    print("=" * 70)

    candidate = "the cat sat on the mat"
    reference = "the dog sat on the rug"
    refs = [
        "the cat sat on the mat",
        "the dog sat on the rug",
    ]

    print(f"\n  Candidate:  '{candidate}'")
    print(f"  References: {refs}")

    # ROUGE-1
    r1 = compute_rouge_n(candidate, refs, n=1)
    print(f"\n  ROUGE-1:")
    print(f"    Recall:    {r1['recall']:.4f}")
    print(f"    Precision: {r1['precision']:.4f}")
    print(f"    F1:        {r1['f1']:.4f}")

    # ROUGE-2
    r2 = compute_rouge_n(candidate, refs, n=2)
    print(f"\n  ROUGE-2:")
    print(f"    Recall:    {r2['recall']:.4f}")
    print(f"    Precision: {r2['precision']:.4f}")
    print(f"    F1:        {r2['f1']:.4f}")

    # ROUGE-L
    rl = compute_rouge_l(candidate, refs)
    print(f"\n  ROUGE-L:")
    print(f"    Recall:    {rl['recall']:.4f}")
    print(f"    Precision: {rl['precision']:.4f}")
    print(f"    F1:        {rl['f1']:.4f}")

    # Note about the difference
    print(f"\n  Note: ROUGE-N uses n-gram overlap; ROUGE-L uses longest")
    print(f"  common subsequence, which captures sentence-level structure")
    print(f"  even when words are separated by other words.")


def demo_evaluation_summary() -> None:
    """演示组合评估摘要。"""
    print("\n" + "=" * 70)
    print("Combined Evaluation Summary")
    print("=" * 70)

    examples = [
        (
            "the quick brown fox jumps over the lazy dog",
            ["the quick brown fox jumps over the lazy dog"],
        ),
        (
            "a fast brown fox leaped across a sleepy hound",
            [
                "the quick brown fox jumps over the lazy dog",
                "a quick brown fox jumps over a lazy dog",
            ],
        ),
        (
            "the weather is nice today",
            ["it is a beautiful day outside", "the sun is shining brightly"],
        ),
    ]

    for candidate, refs in examples:
        print(f"\n  Candidate: '{candidate}'")
        print(f"  Refs: {refs}")
        result = evaluate_summary(candidate, refs)

        print(f"    BLEU:     {result['bleu']['bleu']:.4f}")
        print(f"    ROUGE-1 F1: {result['rouge-1']['f1']:.4f}")
        print(f"    ROUGE-2 F1: {result['rouge-2']['f1']:.4f}")
        print(f"    ROUGE-L F1: {result['rouge-l']['f1']:.4f}")


def demo_rouge_l_detail() -> None:
    """通过可视化 LCS 来演示 ROUGE-L。"""
    print("\n" + "=" * 70)
    print("ROUGE-L: Longest Common Subsequence Visualization")
    print("=" * 70)

    cand = "a b c d e f".split()
    ref = "a x c y e f".split()

    print(f"\n  Candidate: {cand}")
    print(f"  Reference: {ref}")

    lcs_len = _lcs_length(cand, ref)
    print(f"\n  LCS length: {lcs_len}")
    print(f"  LCS (one possible): ['a', 'c', 'e', 'f']")

    rl = compute_rouge_l(cand, [ref])
    print(f"\n  ROUGE-L Recall:    {rl['recall']:.4f} (= {lcs_len}/{len(ref)})")
    print(f"  ROUGE-L Precision: {rl['precision']:.4f} (= {lcs_len}/{len(cand)})")
    print(f"  ROUGE-L F1:        {rl['f1']:.4f}")

    print("\n  Key insight: ROUGE-L rewards maintaining word order over")
    print("  long distances, unlike ROUGE-N which only looks at local n-grams.")


def main() -> None:
    demo_bleu()
    demo_rouge()
    demo_evaluation_summary()
    demo_rouge_l_detail()


if __name__ == "__main__":
    main()
