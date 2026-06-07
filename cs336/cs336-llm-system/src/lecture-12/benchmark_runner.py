"""
LLM 评估的 Benchmark 运行器。

提供了一个运行标准 benchmark（模拟）、收集结果并生成
摘要报告的框架。

核心概念：
  - Task: 带有输入/输出规范的命名评估任务
  - Benchmark: 一组任务的集合（例如 MMLU、HellaSwag、HumanEval）
  - Runner: 编排模型在 benchmark 上的推理
  - Results: 带有每个任务指标的结构化输出

由于无法在没有训练模型的情况下运行真实的 benchmark，本模块
模拟 benchmark 执行并专注于运行器基础设施。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# =========================================================================
# 数据结构
# =========================================================================


class BenchmarkStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BenchmarkTask:
    """Benchmark 中的单个评估任务。

    Attributes:
        name: 任务标识符（例如 "mmlu_anatomy"、"hellaswag"）
        description: 人类可读的描述
        num_examples: 示例总数
        metric: 主要指标名称（例如 "accuracy"、"f1"）
        category: 任务类别（例如 "knowledge"、"reasoning"）
    """

    name: str
    description: str
    num_examples: int
    metric: str = "accuracy"
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """单个 benchmark 任务的结果。

    Attributes:
        task_name: 任务名称
        status: 完成状态
        score: 主要指标得分
        num_correct: 正确预测的数量
        num_total: 评估的示例数量
        latency_seconds: 总耗时（秒）
        extra_metrics: 附加指标（例如按类别准确率）
        error_message: 失败时的错误详情
    """

    task_name: str
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    score: float = 0.0
    num_correct: int = 0
    num_total: int = 0
    latency_seconds: float = 0.0
    extra_metrics: dict[str, float] = field(default_factory=dict)
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典以便序列化。"""
        return {
            "task_name": self.task_name,
            "status": self.status.value,
            "score": self.score,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "latency_seconds": self.latency_seconds,
            "extra_metrics": self.extra_metrics,
            "error_message": self.error_message,
        }


@dataclass
class BenchmarkResult:
    """整个 benchmark 套件的聚合结果。

    Attributes:
        benchmark_name: Benchmark 套件的名称
        model_name: 被评估的模型名称
        timestamp: 评估运行的时间
        task_results: 每个任务的结果
        aggregate_score: 所有任务的平均得分
    """

    benchmark_name: str
    model_name: str = "unknown"
    timestamp: str = ""
    task_results: list[TaskResult] = field(default_factory=list)
    aggregate_score: float = 0.0

    def summary(self) -> str:
        """生成人类可读的摘要。"""
        lines = [
            "=" * 60,
            f"  Benchmark: {self.benchmark_name}",
            f"  Model:     {self.model_name}",
            f"  Timestamp: {self.timestamp}",
            "=" * 60,
            "",
        ]

        for tr in self.task_results:
            status_icon = "✓" if tr.status == BenchmarkStatus.COMPLETED else "✗"
            lines.append(
                f"  {status_icon} {tr.task_name:<25} "
                f"{tr.score:.4f} ({tr.num_correct}/{tr.num_total}) "
                f"[{tr.latency_seconds:.1f}s]"
            )
            if tr.extra_metrics:
                for k, v in tr.extra_metrics.items():
                    lines.append(f"      {k}: {v:.4f}")

        lines.append("")
        lines.append(f"  Aggregate score: {self.aggregate_score:.4f}")
        lines.append(
            f"  Tasks completed: {self.completed_tasks()}/{self.total_tasks()}"
        )
        return "\n".join(lines)

    def completed_tasks(self) -> int:
        """成功完成的任务数量。"""
        return sum(
            1 for r in self.task_results if r.status == BenchmarkStatus.COMPLETED
        )

    def total_tasks(self) -> int:
        """此结果中的任务总数。"""
        return len(self.task_results)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典以便 JSON 序列化。"""
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "aggregate_score": self.aggregate_score,
            "task_results": [tr.to_dict() for tr in self.task_results],
        }


# =========================================================================
# Task Runner（模拟）
# =========================================================================


class TaskRunner:
    """
    对模型运行单个 benchmark 任务。

    在实际系统中，这会加载数据集、运行模型推理并计算指标。
    这里我们模拟这个过程。
    """

    def __init__(
        self,
        model_fn: Callable[[str], str] | None = None,
        batch_size: int = 8,
        max_examples: int | None = None,
    ):
        """
        Args:
            model_fn: 接受 prompt 字符串并返回 completion 字符串的函数。
                      如果为 None，则使用桩函数。
            batch_size: 推理的 batch 大小
            max_examples: 示例数量上限（None 表示全部）
        """
        self.model_fn = model_fn or self._stub_model_fn
        self.batch_size = batch_size
        self.max_examples = max_examples

    @staticmethod
    def _stub_model_fn(prompt: str) -> str:
        """返回简单 echo 响应的桩模型。"""
        # 模拟一个有时能答对的模型
        import hashlib

        h = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        return f"response_{h % 100}"

    def run(self, task: BenchmarkTask) -> TaskResult:
        """执行一个 benchmark 任务。

        Args:
            task: 要运行的 benchmark 任务

        Returns:
            包含得分和指标的 TaskResult
        """
        start_time = time.perf_counter()

        try:
            # 模拟通过模型运行示例
            total = min(task.num_examples, self.max_examples or task.num_examples)

            # 根据任务属性模拟不同的准确率
            base_accuracy = 0.25  # random baseline
            if "easy" in task.name.lower():
                base_accuracy = 0.85
            elif "hard" in task.name.lower():
                base_accuracy = 0.35
            elif "math" in task.category.lower():
                base_accuracy = 0.40
            elif "reasoning" in task.category.lower():
                base_accuracy = 0.50
            elif "knowledge" in task.category.lower():
                base_accuracy = 0.60

            # 添加噪声
            import random

            random.seed(hash(task.name) % 10000)
            noise = random.uniform(-0.05, 0.05)
            accuracy = min(1.0, max(0.0, base_accuracy + noise))

            num_correct = int(total * accuracy)
            num_correct += random.randint(-2, 2)
            num_correct = max(0, min(total, num_correct))

            # 模拟延迟（与示例数量成正比）
            latency = total * 0.001  # ~1ms per example simulated

            # 附加指标
            extra = {}
            if "mmlu" in task.name.lower():
                extra["macro_accuracy"] = accuracy + random.uniform(-0.02, 0.02)
            if "classification" in task.metadata.get("subtype", ""):
                extra["f1_score"] = accuracy - random.uniform(0.01, 0.05)

            elapsed = time.perf_counter() - start_time

            return TaskResult(
                task_name=task.name,
                status=BenchmarkStatus.COMPLETED,
                score=num_correct / total if total > 0 else 0.0,
                num_correct=num_correct,
                num_total=total,
                latency_seconds=max(elapsed, latency),
                extra_metrics=extra,
            )

        except Exception as e:
            return TaskResult(
                task_name=task.name,
                status=BenchmarkStatus.FAILED,
                error_message=str(e),
            )


# =========================================================================
# Benchmark Suite Runner
# =========================================================================


class BenchmarkRunner:
    """
    编排对模型运行完整的 benchmark 套件。

    加载任务定义，通过 TaskRunner 运行它们，收集结果并生成报告。
    """

    def __init__(
        self,
        model_name: str = "unknown",
        model_fn: Callable[[str], str] | None = None,
        output_dir: str | None = None,
    ):
        self.model_name = model_name
        self.model_fn = model_fn
        self.output_dir = output_dir
        self.task_runner = TaskRunner(model_fn=model_fn)

    def run_benchmark(
        self,
        benchmark_name: str,
        tasks: list[BenchmarkTask],
    ) -> BenchmarkResult:
        """运行 benchmark 套件中的所有任务。

        Args:
            benchmark_name: Benchmark 的名称
            tasks: 要评估的任务列表

        Returns:
            聚合后的 BenchmarkResult
        """
        from datetime import datetime

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            model_name=self.model_name,
            timestamp=datetime.now().isoformat(),
        )

        print(f"\n  Running benchmark: {benchmark_name}")
        print(f"  Tasks: {len(tasks)}")

        scores: list[float] = []

        for i, task in enumerate(tasks):
            print(f"    [{i + 1}/{len(tasks)}] {task.name}...", end=" ")
            task_result = self.task_runner.run(task)
            result.task_results.append(task_result)

            if task_result.status == BenchmarkStatus.COMPLETED:
                scores.append(task_result.score)
                print(f"{task_result.score:.4f}")
            else:
                print(f"FAILED: {task_result.error_message}")

        if scores:
            result.aggregate_score = sum(scores) / len(scores)

        return result

    def save_results(self, result: BenchmarkResult, filepath: str | None = None) -> str:
        """将 benchmark 结果保存到 JSON 文件。

        Args:
            result: 要保存的 benchmark 结果
            filepath: 可选的文件路径。如果为 None，则自动生成。

        Returns:
            结果保存的文件路径
        """
        if filepath is None:
            import os

            safe_name = result.benchmark_name.replace("/", "_").replace(" ", "_")
            output_dir = self.output_dir or "."
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{safe_name}_results.json")

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return filepath

    def load_results(self, filepath: str) -> BenchmarkResult:
        """从 JSON 文件加载 benchmark 结果。

        Args:
            filepath: 结果 JSON 文件的路径

        Returns:
            重建的 BenchmarkResult
        """
        with open(filepath) as f:
            data = json.load(f)

        result = BenchmarkResult(
            benchmark_name=data["benchmark_name"],
            model_name=data.get("model_name", "unknown"),
            timestamp=data.get("timestamp", ""),
            aggregate_score=data.get("aggregate_score", 0.0),
        )

        for task_data in data.get("task_results", []):
            tr = TaskResult(
                task_name=task_data["task_name"],
                status=BenchmarkStatus(task_data.get("status", "pending")),
                score=task_data.get("score", 0.0),
                num_correct=task_data.get("num_correct", 0),
                num_total=task_data.get("num_total", 0),
                latency_seconds=task_data.get("latency_seconds", 0.0),
                extra_metrics=task_data.get("extra_metrics", {}),
                error_message=task_data.get("error_message", ""),
            )
            result.task_results.append(tr)

        return result


# =========================================================================
# 预定义的 benchmark 套件（用于演示）
# =========================================================================


def create_mmlu_tasks() -> list[BenchmarkTask]:
    """创建一组有代表性的类似 MMLU 的任务。"""
    return [
        BenchmarkTask(
            "mmlu_anatomy",
            "High school anatomy",
            num_examples=135,
            metric="accuracy",
            category="knowledge",
        ),
        BenchmarkTask(
            "mmlu_astronomy",
            "High school astronomy",
            num_examples=152,
            metric="accuracy",
            category="knowledge",
        ),
        BenchmarkTask(
            "mmlu_college_math",
            "College mathematics",
            num_examples=100,
            metric="accuracy",
            category="math",
        ),
        BenchmarkTask(
            "mmlu_computer_science",
            "Computer science",
            num_examples=100,
            metric="accuracy",
            category="knowledge",
        ),
        BenchmarkTask(
            "mmlu_econometrics",
            "Econometrics",
            num_examples=114,
            metric="accuracy",
            category="math",
        ),
        BenchmarkTask(
            "mmlu_high_school_math",
            "High school math",
            num_examples=270,
            metric="accuracy",
            category="math",
        ),
        BenchmarkTask(
            "mmlu_moral_reasoning",
            "Moral reasoning",
            num_examples=200,
            metric="accuracy",
            category="reasoning",
        ),
        BenchmarkTask(
            "mmlu_philosophy",
            "Philosophy",
            num_examples=311,
            metric="accuracy",
            category="reasoning",
        ),
    ]


def create_reasoning_tasks() -> list[BenchmarkTask]:
    """创建面向推理的 benchmark 任务。"""
    return [
        BenchmarkTask(
            "hellaswag",
            "Commonsense NLI (hard negatives)",
            num_examples=10042,
            metric="accuracy",
            category="reasoning",
        ),
        BenchmarkTask(
            "arc_easy",
            "AI2 Reasoning Challenge (easy)",
            num_examples=2376,
            metric="accuracy",
            category="reasoning",
        ),
        BenchmarkTask(
            "arc_hard",
            "AI2 Reasoning Challenge (challenge)",
            num_examples=1172,
            metric="accuracy",
            category="reasoning",
        ),
        BenchmarkTask(
            "winogrande",
            "Winograd schema challenge",
            num_examples=1267,
            metric="accuracy",
            category="reasoning",
        ),
        BenchmarkTask(
            "piqa",
            "Physical commonsense QA",
            num_examples=1838,
            metric="accuracy",
            category="reasoning",
        ),
        BenchmarkTask(
            "gsm8k",
            "Grade school math word problems",
            num_examples=1319,
            metric="accuracy",
            category="math",
        ),
    ]


# =========================================================================
# 演示
# =========================================================================


def demo_benchmark_runner() -> None:
    """演示运行一个模拟的 benchmark。"""
    print("=" * 70)
    print("Benchmark Runner Demo")
    print("=" * 70)

    runner = BenchmarkRunner(
        model_name="DemoModel-7B",
        model_fn=None,  # 使用桩模型
    )

    # Run MMLU-like benchmark
    mmlu_tasks = create_mmlu_tasks()
    result = runner.run_benchmark("MMLU (simulated)", mmlu_tasks)

    print("\n" + result.summary())

    # 保存并重新加载
    filepath = runner.save_results(result)
    print(f"\n  Results saved to: {filepath}")

    # 演示加载
    reloaded = runner.load_results(filepath)
    print(
        f"  Reloaded: {reloaded.benchmark_name}, "
        f"{reloaded.total_tasks()} tasks, "
        f"aggregate={reloaded.aggregate_score:.4f}"
    )
    assert reloaded.aggregate_score == result.aggregate_score


def demo_multi_benchmark() -> None:
    """运行多个模拟 benchmark 并进行比较。"""
    print("\n" + "=" * 70)
    print("Multi-Benchmark Comparison")
    print("=" * 70)

    # 模拟不同规模的模型
    models = [
        ("SmallModel-1B", 0.35),
        ("MediumModel-7B", 0.50),
        ("LargeModel-70B", 0.70),
    ]

    reasoning_tasks = create_reasoning_tasks()

    all_results: list[BenchmarkResult] = []

    for model_name, _base_score in models:
        # 具有不同"质量"的自定义 model 函数
        def make_model_fn(base: float) -> Callable[[str], str]:
            import hashlib

            def fn(prompt: str) -> str:
                h = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
                return f"response_{int(h * base) % 100}"

            return fn

        runner = BenchmarkRunner(
            model_name=model_name,
            model_fn=make_model_fn(_base_score),
        )
        result = runner.run_benchmark("Reasoning (simulated)", reasoning_tasks)
        all_results.append(result)

    # 比较表格
    task_names = [t.name for t in reasoning_tasks]
    print(f"\n  {'Task':<20}", end="")
    for m in models:
        print(f"  {m[0]:<18}", end="")
    print()
    print("  " + "-" * 70)

    for task_name in task_names:
        print(f"  {task_name:<20}", end="")
        for result in all_results:
            for tr in result.task_results:
                if tr.task_name == task_name:
                    print(f"  {tr.score:<18.4f}", end="")
                    break
        print()

    # 聚合比较
    print("  " + "-" * 70)
    print(f"  {'AGGREGATE':<20}", end="")
    for result in all_results:
        print(f"  {result.aggregate_score:<18.4f}", end="")
    print()

    print("\n  Key insight: Benchmark runners standardize evaluation across")
    print("  models, enabling fair comparisons. Results are reproducible and")
    print("  can be tracked over model versions.")


def main() -> None:
    demo_benchmark_runner()
    demo_multi_benchmark()


if __name__ == "__main__":
    main()
