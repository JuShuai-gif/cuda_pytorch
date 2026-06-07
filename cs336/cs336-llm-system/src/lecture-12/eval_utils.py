"""
用于 LLM 评估的评估工具函数。

提供评估语言模型时常用的辅助函数：
  - 精确匹配和 token 级别准确率
  - 模型输出的规范化和后处理
  - 统计显著性检验（bootstrap）
  - 结果格式化和比较
  - 多项选择评估辅助
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


# =========================================================================
# 准确率指标
# =========================================================================


def exact_match(prediction: str, reference: str, normalize: bool = True) -> bool:
    """
    检查预测是否与参考文本完全匹配。

    Args:
        prediction: 模型输出字符串
        reference: 真实标签字符串
        normalize: 如果为 True，在比较前先进行规范化

    Returns:
        如果预测与参考文本匹配则为 True
    """
    if normalize:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)
    return prediction == reference


def token_level_accuracy(
    pred_tokens: list[int],
    ref_tokens: list[int],
    ignore_index: int = -100,
) -> float:
    """
    计算 token 级别准确率，忽略 padding token。

    Args:
        pred_tokens: 预测的 token id
        ref_tokens: 参考的 token id
        ignore_index: 在准确率计算中排除的 token id

    Returns:
        准确率，以分数表示（0.0 到 1.0）
    """
    if len(pred_tokens) != len(ref_tokens):
        raise ValueError(
            f"Length mismatch: pred={len(pred_tokens)}, ref={len(ref_tokens)}"
        )

    correct = 0
    total = 0
    for p, r in zip(pred_tokens, ref_tokens):
        if r == ignore_index:
            continue
        total += 1
        if p == r:
            correct += 1

    return correct / total if total > 0 else 0.0


def sequence_accuracy(
    predictions: list[list[int]],
    references: list[list[int]],
    ignore_index: int = -100,
) -> float:
    """
    计算多个示例的序列级准确率。

    如果所有非 padding token 都匹配，则该序列被认为是正确的。

    Args:
        predictions: 预测的 token 序列列表
        references: 参考的 token 序列列表
        ignore_index: 要忽略的 token id

    Returns:
        序列准确率，以分数表示（0.0 到 1.0）
    """
    if len(predictions) != len(references):
        raise ValueError("Mismatched number of predictions and references")

    correct = 0
    for pred, ref in zip(predictions, references):
        if len(pred) != len(ref):
            continue
        match = all(p == r for p, r in zip(pred, ref) if r != ignore_index)
        if match:
            correct += 1

    return correct / len(predictions) if predictions else 0.0


def multiple_choice_accuracy(
    predictions: list[int],
    references: list[int],
) -> float:
    """
    计算多项选择题的准确率。

    Args:
        predictions: 预测的选项索引列表（基于 0）
        references: 正确选项索引列表

    Returns:
        准确率，以分数表示
    """
    if len(predictions) != len(references):
        raise ValueError("Mismatched lengths between predictions and references")

    if not predictions:
        return 0.0

    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


# =========================================================================
# 规范化
# =========================================================================


def normalize_text(text: str) -> str:
    """
    规范化文本以便比较。

    步骤:
      1. 转换为小写
      2. 去除首尾空白字符
      3. 将多个空白字符合并为单个空格
      4. 去除首尾的标点符号

    Args:
        text: 输入字符串

    Returns:
        规范化后的字符串
    """
    text = text.lower().strip()
    # 合并空白字符
    text = " ".join(text.split())
    # 去除可能在输出之间不同的尾部标点
    while text and text[-1] in ".!?,;:\"'":
        text = text[:-1].strip()
    return text


def strip_thinking(text: str) -> str:
    """
    从模型输出中移除 chain-of-thought / 思考内容。

    许多模型（例如 DeepSeek-R1、Claude、o1）在最终答案之前
    会生成内部推理过程。此函数将其移除。

    处理常见模式：
      - <｜end▁of▁thinking｜>...  response
      - <thinking>...</thinking>
      - [THINK]...[/THINK]

    Args:
        text: 原始模型输出

    Returns:
        移除思考内容后的文本，如果没有找到标记则返回原始文本
    """
    import re

    # 尝试  response...  response 模式（DeepSeek-R1）
    think_match = re.search(r" response(.*?) response", text, re.DOTALL)
    if think_match:
        return text[think_match.end() :].strip()

    # Try <thinking>...</thinking>
    think_match = re.search(
        r"<thinking>.*?</thinking>", text, re.DOTALL | re.IGNORECASE
    )
    if think_match:
        return text[think_match.end() :].strip()

    # Try [THINK]...[/THINK]
    think_match = re.search(r"\[THINK\].*?\[/THINK\]", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        return text[think_match.end() :].strip()

    return text


def extract_final_answer(text: str) -> str:
    """
    从模型输出中提取最终答案。

    尝试匹配常见模式如 "Answer: X"、"The answer is X" 等。

    Args:
        text: 原始模型输出

    Returns:
        提取出的答案或原始文本
    """
    import re

    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)?\s*(.+?)(?:\.|\n|$)",
        r"(?:therefore|thus|so|hence),?\s*(.+?)(?:\.|\n|$)",
        r"^\s*(.+?)(?:\.|\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower(), re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    return text


# =========================================================================
# Bootstrap 置信区间
# =========================================================================


@dataclass
class BootstrapResult:
    """Bootstrap 置信区间计算的结果。

    Attributes:
        mean: Bootstrap 分布的均值
        lower: 置信区间的下界
        upper: 置信区间的上界
        confidence: 置信水平（例如 0.95）
        num_samples: Bootstrap 重采样次数
    """

    mean: float
    lower: float
    upper: float
    confidence: float = 0.95
    num_samples: int = 1000

    def __str__(self) -> str:
        ci_pct = self.confidence * 100
        return (
            f"{self.mean:.4f} [{ci_pct:.0f}% CI: {self.lower:.4f} - {self.upper:.4f}]"
        )


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """
    为每个示例得分列表计算 Bootstrap 置信区间。

    用于判断模型之间的差异是否具有统计显著性。

    Args:
        scores: 每个示例的指标值（例如准确率）
        confidence: 置信水平（0.0 到 1.0）
        n_resamples: Bootstrap 重采样次数
        seed: 随机种子以确保可复现性

    Returns:
        包含均值、下界和上界的 BootstrapResult
    """
    if not scores:
        return BootstrapResult(mean=0.0, lower=0.0, upper=0.0)

    rng = random.Random(seed)
    n = len(scores)
    means: list[float] = []

    for _ in range(n_resamples):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()

    alpha = (1 - confidence) / 2
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1 - alpha) * n_resamples)

    mean_of_means = sum(means) / n_resamples

    return BootstrapResult(
        mean=mean_of_means,
        lower=means[lower_idx],
        upper=means[upper_idx - 1],
        confidence=confidence,
        num_samples=n_resamples,
    )


# =========================================================================
# 统计显著性
# =========================================================================


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """
    用于比较两个模型/系统的配对 Bootstrap 检验。

    检验原假设：模型 A 和模型 B 具有相同的期望得分。
    返回双侧检验的 p-value。

    Args:
        scores_a: 模型 A 的每个示例得分
        scores_b: 模型 B 的每个示例得分
        n_resamples: Bootstrap 重采样次数
        seed: 随机种子

    Returns:
        包含 "delta_mean"（A - B）、"p_value"、"significant"（p < 0.05 时为 True）的字典
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    rng = random.Random(seed)
    n = len(scores_a)

    # 观察到的差异
    observed_delta = sum(scores_a) / n - sum(scores_b) / n

    # 配对差异
    diffs = [a - b for a, b in zip(scores_a, scores_b)]

    # Bootstrap: 有放回地采样差异
    delta_dist: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(diffs) for _ in range(n)]
        delta_dist.append(sum(sample) / n)

    # 双侧 p-value: |delta| >= |observed_delta| 的比例
    extreme_count = sum(1 for d in delta_dist if abs(d) >= abs(observed_delta))
    p_value = extreme_count / n_resamples

    return {
        "delta_mean": observed_delta,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


# =========================================================================
# 结果格式化
# =========================================================================


def format_metric_table(
    results: dict[str, float],
    title: str = "Results",
) -> str:
    """
    将指标字典格式化为可读的表格。

    Args:
        results: 指标名称到值的映射字典
        title: 表格标题

    Returns:
        格式化后的字符串
    """
    lines = [f"\n  {title}", "  " + "-" * 50]
    max_name_len = max(len(k) for k in results) if results else 0

    for name, value in results.items():
        if isinstance(value, float):
            lines.append(f"  {name:<{max_name_len}} : {value:.6f}")
        else:
            lines.append(f"  {name:<{max_name_len}} : {value}")

    return "\n".join(lines)


def compare_results(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> str:
    """
    并排比较两组指标结果。

    Args:
        baseline: Baseline 模型的指标
        candidate: 候选模型的指标

    Returns:
        格式化后的比较字符串
    """
    all_keys = list(dict.fromkeys(list(baseline.keys()) + list(candidate.keys())))
    max_name_len = max((len(k) for k in all_keys), default=0)

    lines = [
        f"\n  {'Metric':<{max_name_len}}  {'Baseline':>12}  {'Candidate':>12}  {'Delta':>12}",
        "  " + "-" * (max_name_len + 44),
    ]

    for key in all_keys:
        b = baseline.get(key, float("nan"))
        c = candidate.get(key, float("nan"))
        delta = c - b if isinstance(b, float) and isinstance(c, float) else "---"

        if isinstance(delta, float):
            sign = "+" if delta > 0 else ""
            lines.append(
                f"  {key:<{max_name_len}}  {b:>12.6f}  {c:>12.6f}  {sign}{delta:>11.6f}"
            )
        else:
            lines.append(
                f"  {key:<{max_name_len}}  {str(b):>12}  {str(c):>12}  {str(delta):>12}"
            )

    return "\n".join(lines)


# =========================================================================
# 演示
# =========================================================================


def demo_accuracy_metrics() -> None:
    """演示准确率指标函数。"""
    print("=" * 70)
    print("Evaluation Utils Demo: Accuracy Metrics")
    print("=" * 70)

    # 精确匹配
    print("\n--- Exact Match ---")
    tests = [
        ("The cat sat.", "the cat sat", True, True),  # 规范化后应匹配
        ("The cat sat.", "The dog ran.", True, False),
        ("Hello world!", "hello world!", False, False),  # 区分大小写
    ]
    for pred, ref, norm, expected in tests:
        result = exact_match(pred, ref, normalize=norm)
        status = "✓" if result == expected else "✗"
        print(f"  {status} EM('{pred}', '{ref}', norm={norm}) = {result}")

    # Token 级别准确率
    print("\n--- Token-Level Accuracy ---")
    pred_tokens = [1, 2, 3, 4, 5]
    ref_tokens = [1, 2, 9, 4, 5]
    acc = token_level_accuracy(pred_tokens, ref_tokens)
    print(f"  Pred: {pred_tokens}")
    print(f"  Ref:  {ref_tokens}")
    print(f"  Accuracy: {acc:.4f}")

    # 带 ignore_index
    pred_tokens2 = [1, 2, 3, 4, 5, 6]
    ref_tokens2 = [1, 2, 3, -100, -100, 6]
    acc2 = token_level_accuracy(pred_tokens2, ref_tokens2, ignore_index=-100)
    print(f"\n  Pred (with padding): {pred_tokens2}")
    print(f"  Ref  (with padding): {ref_tokens2}")
    print(f"  Accuracy: {acc2:.4f}")

    # 多项选择
    print("\n--- Multiple Choice Accuracy ---")
    preds = [0, 2, 1, 3, 0]
    refs = [0, 1, 1, 3, 2]
    mc_acc = multiple_choice_accuracy(preds, refs)
    print(f"  Predictions: {preds}")
    print(f"  References:  {refs}")
    print(
        f"  Accuracy: {mc_acc:.4f} ({sum(1 for p, r in zip(preds, refs) if p == r)}/{len(preds)})"
    )


def demo_normalization() -> None:
    """演示文本规范化工具。"""
    print("\n" + "=" * 70)
    print("Evaluation Utils Demo: Text Normalization")
    print("=" * 70)

    # normalize_text
    print("\n--- normalize_text ---")
    tests = [
        "  Hello   World  ",
        "The answer is: 42.",
        "UPPERCASE TEXT!!!",
    ]
    for t in tests:
        print(f"  '{t}' → '{normalize_text(t)}'")

    # strip_thinking
    print("\n--- strip_thinking ---")
    thinking_examples = [
        " responseLet me think... The answer is 42. response42",
        "<thinking>Step 1: First...\nStep 2: Then...</thinking>The answer is Paris",
        "[THINK]reasoning here[/THINK]Final answer: 3.14",
        "Just a normal response without thinking tags.",
    ]
    for t in thinking_examples:
        result = strip_thinking(t)
        print(f"  Input:  {t[:60]}...")
        print(f"  Output: {result[:60]}")
        print()

    # extract_final_answer
    print("\n--- extract_final_answer ---")
    answer_examples = [
        "Therefore, the answer is 42.",
        "The final answer: Paris",
        "Thus we conclude that x = 3.14.",
        "Simple direct answer.",
    ]
    for t in answer_examples:
        result = extract_final_answer(t)
        print(f"  '{t}' → '{result}'")


def demo_bootstrap() -> None:
    """演示 Bootstrap 置信区间。"""
    print("\n" + "=" * 70)
    print("Evaluation Utils Demo: Bootstrap Confidence Intervals")
    print("=" * 70)

    random.seed(42)

    # 模拟模型的每个示例准确率
    n_examples = 100
    scores = [random.random() for _ in range(n_examples)]
    # 使平均准确率约为 70%
    scores = [1.0 if s > 0.3 else 0.0 for s in scores]
    actual_mean = sum(scores) / len(scores)

    result = bootstrap_confidence_interval(scores, confidence=0.95, n_resamples=1000)
    print(f"\n  True mean: {actual_mean:.4f}")
    print(f"  Bootstrap: {result}")
    print(f"\n  Interpretation: We are 95% confident that the true accuracy")
    print(f"  lies between {result.lower:.4f} and {result.upper:.4f}")


def demo_paired_test() -> None:
    """演示配对 Bootstrap 假设检验。"""
    print("\n" + "=" * 70)
    print("Evaluation Utils Demo: Paired Bootstrap Test")
    print("=" * 70)

    random.seed(42)
    n = 100

    # 模型 A: baseline, 约 70% 准确率
    scores_a = [1.0 if random.random() > 0.3 else 0.0 for _ in range(n)]

    # 模型 B: 略好, 约 75% 准确率
    scores_b = [1.0 if random.random() > 0.25 else 0.0 for _ in range(n)]

    result = paired_bootstrap_test(scores_a, scores_b)
    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n

    print(f"\n  Model A accuracy: {mean_a:.4f}")
    print(f"  Model B accuracy: {mean_b:.4f}")
    print(f"  Delta (A - B):    {result['delta_mean']:.4f}")
    print(f"  p-value:          {result['p_value']:.4f}")
    print(f"  Significant:      {result['significant']}")

    # 模型 C: 与 A 相同（没有真正差异）
    scores_c = [1.0 if random.random() > 0.3 else 0.0 for _ in range(n)]
    result_c = paired_bootstrap_test(scores_a, scores_c)
    print(f"\n  --- A vs C (same distribution) ---")
    print(f"  Delta (A - C):    {result_c['delta_mean']:.4f}")
    print(f"  p-value:          {result_c['p_value']:.4f}")
    print(f"  Significant:      {result_c['significant']}")


def demo_formatting() -> None:
    """演示结果格式化工具。"""
    print("\n" + "=" * 70)
    print("Evaluation Utils Demo: Result Formatting")
    print("=" * 70)

    results = {
        "accuracy": 0.7234,
        "precision": 0.7501,
        "recall": 0.6892,
        "f1_score": 0.7183,
        "perplexity": 12.3456,
    }
    print(format_metric_table(results, "Model Evaluation Metrics"))

    baseline = {
        "accuracy": 0.6500,
        "f1_score": 0.6300,
        "perplexity": 18.5000,
    }
    candidate = {
        "accuracy": 0.7234,
        "f1_score": 0.7183,
        "perplexity": 12.3456,
    }
    print(compare_results(baseline, candidate))


def main() -> None:
    demo_accuracy_metrics()
    demo_normalization()
    demo_bootstrap()
    demo_paired_test()
    demo_formatting()


if __name__ == "__main__":
    main()
