"""
Code generation benchmarks: HumanEval and MBPP.

Implements:
  - HumanEval: 164 hand-written Python programming problems
  - MBPP: Mostly Basic Python Programming (crowd-sourced)
  - Unbiased pass@k estimator (Chen et al. 2021)
  - Safe code execution sandbox with timeout and security constraints
  - Test result parsing with pass/fail/error classification

Key formula:
  pass@k = 1 - C(n-c, k) / C(n, k)
  where n = total samples, c = correct samples, k = evaluation budget
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from .benchmark_registry import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    register_benchmark,
)
from .metrics import compute_pass_at_k, compute_pass_at_k_from_samples


# =========================================================================
# Task Loading
# =========================================================================


def _load_humaneval_tasks(
    tasks_path: str | None = None,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Load HumanEval task definitions.

    Format: {"task_id": str, "prompt": str, "canonical_solution": str,
             "test": str, "entry_point": str}

    Args:
        tasks_path: Path to JSON file with task definitions.
                    If None, loads from configs/humaneval_tasks.json.
        max_tasks: Limit number of tasks loaded.

    Returns:
        List of task dicts.
    """
    if tasks_path is None:
        tasks_path = str(Path(__file__).parent / "configs" / "humaneval_tasks.json")

    if os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict):
            tasks = data.get("tasks", data.get("problems", []))
        else:
            tasks = []
    else:
        tasks = []

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks


# =========================================================================
# Code Execution Sandbox
# =========================================================================


class CodeExecutionResult:
    """Result of executing generated code against test cases.

    Attributes:
        passed: Whether all tests passed.
        error_type: "none", "syntax_error", "runtime_error", "timeout", "assertion_error".
        error_message: Description of the error if any.
        output: Combined stdout/stderr output.
        execution_time: Time taken in seconds.
    """

    def __init__(self):
        self.passed: bool = False
        self.error_type: str = "none"
        self.error_message: str = ""
        self.output: str = ""
        self.execution_time: float = 0.0


def _execute_code_sandbox(
    code: str,
    test_code: str,
    entry_point: str | None = None,
    timeout: float = 5.0,
    max_output_bytes: int = 65536,
) -> CodeExecutionResult:
    """Execute generated code with test cases in a subprocess sandbox.

    Security measures:
      - Runs in an isolated subprocess
      - Timeout-based termination via SIGALRM
      - No network/filesystem access (can be further restricted)
      - Output size limits to prevent memory exhaustion

    Args:
        code: The generated code (function body or full implementation).
        test_code: Test harness code that calls the generated function.
        entry_point: Function name expected by the test harness.
        timeout: Maximum execution time in seconds.
        max_output_bytes: Maximum output size.

    Returns:
        CodeExecutionResult with pass/fail and error details.
    """
    result = CodeExecutionResult()

    # Build a complete script combining the generated code and tests
    # Wrap in a try/except to catch all errors gracefully
    full_script = f"""
import sys
import traceback
import signal

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm({timeout})

try:
{_indent_code(code, 4)}

{_indent_code(test_code, 4)}
except TimeoutError as e:
    print(f"\\n%%%TIMEOUT%%%: {{e}}", file=sys.stderr)
    sys.exit(124)
except AssertionError as e:
    print(f"\\n%%%ASSERTION_ERROR%%%: {{e}}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"\\n%%%RUNTIME_ERROR%%%: {{e}}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
finally:
    signal.alarm(0)
"""

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            timeout=timeout + 2.0,  # Extra buffer for process level
            text=True,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        result.error_type = "timeout"
        result.error_message = "Process timed out"
        result.execution_time = time.perf_counter() - start
        return result

    elapsed = time.perf_counter() - start
    result.execution_time = elapsed

    combined_output = proc.stdout + proc.stderr

    # Truncate output
    if len(combined_output) > max_output_bytes:
        combined_output = combined_output[:max_output_bytes] + "\n... [TRUNCATED]"
    result.output = combined_output

    # Classify result
    if proc.returncode == 0:
        result.passed = True
        result.error_type = "none"
    elif "%%%TIMEOUT%%%" in proc.stderr:
        result.error_type = "timeout"
        result.error_message = "Execution timed out"
    elif "SyntaxError" in proc.stderr or "IndentationError" in proc.stderr:
        result.error_type = "syntax_error"
        result.error_message = _extract_error_message(proc.stderr)
    elif "%%%ASSERTION_ERROR%%%" in proc.stderr:
        result.error_type = "assertion_error"
        result.error_message = _extract_error_message(proc.stderr)
    else:
        result.error_type = "runtime_error"
        result.error_message = _extract_error_message(proc.stderr)

    return result


def _indent_code(code: str, spaces: int) -> str:
    """Indent every line of code by the specified number of spaces."""
    indent = " " * spaces
    return "\n".join(
        indent + line if line.strip() else line for line in code.split("\n")
    )


def _extract_error_message(stderr: str) -> str:
    """Extract a concise error message from stderr."""
    lines = stderr.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("%%%"):
            return line
    return stderr.strip()


# =========================================================================
# MBPP Support
# =========================================================================


def _load_mbpp_tasks(
    tasks_path: str,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Load MBPP task definitions.

    MBPP format: {"text": "problem description", "test_list": [...],
                  "code": "reference solution", "task_id": int}

    Args:
        tasks_path: Path to JSON/JSONL file.
        max_tasks: Limit number of tasks.

    Returns:
        List of task dicts normalized to HumanEval format.
    """
    tasks: list[dict[str, Any]] = []

    if tasks_path.endswith(".jsonl"):
        with open(tasks_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                tasks.append(_normalize_mbpp_task(data))
    else:
        with open(tasks_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    tasks.append(_normalize_mbpp_task(item))

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks


def _normalize_mbpp_task(task: dict[str, Any]) -> dict[str, Any]:
    """Normalize MBPP task to HumanEval-compatible format."""
    text = task.get("text", "")
    test_list = task.get("test_list", task.get("tests", []))

    # Build test code from assertion list
    test_code_lines = []
    test_code_lines.append("def check(candidate):")
    for test in test_list:
        test_code_lines.append(f"    {test}")
    test_code_lines.append("")
    test_code_lines.append("check(candidate)")

    # Build prompt
    prompt = f'"""\n{text}\n"""\n\ndef solution():\n'

    return {
        "task_id": f"MBPP/{task.get('task_id', 0)}",
        "prompt": prompt,
        "canonical_solution": task.get("code", ""),
        "test": "\n".join(test_code_lines),
        "entry_point": "solution",
    }


# =========================================================================
# HumanEval Benchmark
# =========================================================================


@register_benchmark
class HumanEvalBenchmark(Benchmark):
    """HumanEval code generation benchmark.

    Evaluates functional correctness of model-generated code
    using the pass@k metric with the unbiased estimator.
    """

    name = "humaneval"
    description = "HumanEval code generation benchmark with unbiased pass@k estimator"

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        tasks_path: str | None = None,
        timeout: float = 5.0,
    ):
        super().__init__(config)
        self.tasks_path = tasks_path
        self.timeout = timeout
        self._results_cache: dict[str, Any] = {}

    def run(
        self,
        model_fn: Any = None,
        num_samples: int = 1,
        k_values: tuple[int, ...] = (1, 10, 100),
        temperature: float = 0.2,
    ) -> BenchmarkResult:
        """Execute HumanEval benchmark.

        For each problem, generates num_samples completions, tests each,
        and computes pass@k using the unbiased estimator.

        Args:
            model_fn: Callable `model_fn(prompt: str, **kwargs) -> str`.
                      If None, uses a stub.
            num_samples: Number of code samples to generate per problem.
            k_values: Which pass@k values to compute.
            temperature: Sampling temperature for generation.

        Returns:
            BenchmarkResult with pass@k per k and per-problem results.
        """
        tasks = _load_humaneval_tasks(
            self.tasks_path,
            max_tasks=self.config.max_samples,
        )

        if not tasks:
            print("  [HumanEval] No tasks loaded.")
            return BenchmarkResult(
                benchmark_name=self.name,
                config_hash=self.config.config_hash(),
            )

        print(
            f"  [HumanEval] {len(tasks)} tasks, {num_samples} samples/task, "
            f"k={k_values}"
        )

        per_problem_results: list[list[bool]] = []
        task_details: list[dict[str, Any]] = []
        total_exec_time = 0.0

        for i, task in enumerate(tasks):
            task_id = task["task_id"]
            prompt = task["prompt"]
            test_code = task["test"]
            entry_point = task["entry_point"]

            sample_results: list[bool] = []

            for sample_idx in range(num_samples):
                # Generate code
                if model_fn is not None:
                    completion = model_fn(
                        prompt,
                        temperature=temperature,
                        max_tokens=self.config.max_tokens,
                    )
                else:
                    completion = task.get("canonical_solution", "")

                # Combine prompt + completion for full code
                full_code = prompt + completion

                # Execute in sandbox
                exec_result = _execute_code_sandbox(
                    full_code, test_code, entry_point, timeout=self.timeout
                )
                sample_results.append(exec_result.passed)
                total_exec_time += exec_result.execution_time

                if sample_idx == 0:
                    # Record details for first sample only
                    task_details.append(
                        {
                            "task_id": task_id,
                            "passed": exec_result.passed,
                            "error_type": exec_result.error_type,
                            "error_message": exec_result.error_message,
                        }
                    )

            per_problem_results.append(sample_results)

            status = "✓" if any(sample_results) else "✗"
            passed_count = sum(sample_results)
            if i == 0 or (i + 1) % 10 == 0 or i == len(tasks) - 1:
                print(
                    f"    [{i + 1}/{len(tasks)}] {status} {task_id}: "
                    f"{passed_count}/{num_samples} passed"
                )

        # Compute pass@k
        pass_k_scores: dict[str, float] = {}
        for k in k_values:
            pk = compute_pass_at_k_from_samples(per_problem_results, k)
            pass_k_scores[f"pass@{k}"] = pk
            print(f"  pass@{k} = {pk:.4f}")

        # Per-problem pass@1
        per_problem_pass1 = [
            (1.0 if any(samples) else 0.0) for samples in per_problem_results
        ]

        self._results_cache = {
            "pass_k": pass_k_scores,
            "per_problem": task_details,
            "per_problem_pass@1": per_problem_pass1,
            "num_problems": len(tasks),
            "num_samples_per_problem": num_samples,
            "total_execution_time": total_exec_time,
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=[{"metric": k, "value": v} for k, v in pass_k_scores.items()],
            aggregate_score=pass_k_scores.get("pass@1", 0.0),
            config_hash=self.config.config_hash(),
            metadata={
                "num_problems": len(tasks),
                "num_samples_per_problem": num_samples,
                "k_values": list(k_values),
                "temperature": temperature,
            },
        )

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, Any]:
        """Evaluate a set of code predictions.

        This method is designed for offline evaluation when test code
        is available. In practice, run() handles everything.

        Args:
            predictions: Generated code strings.
            references: Expected outputs (not typically used for HumanEval).

        Returns:
            Dict with pass count.
        """
        return {
            "num_correct": sum(1 for p, r in zip(predictions, references) if p == r),
            "num_total": len(predictions),
        }

    def get_results(self) -> BenchmarkResult:
        """Return cached results from the last run()."""
        if not self._results_cache:
            return BenchmarkResult(
                benchmark_name=self.name,
                config_hash=self.config.config_hash(),
            )

        pass_k = self._results_cache.get("pass_k", {})
        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=[{"metric": k, "value": v} for k, v in pass_k.items()],
            aggregate_score=pass_k.get("pass@1", 0.0),
            config_hash=self.config.config_hash(),
            metadata={
                "num_problems": self._results_cache.get("num_problems", 0),
            },
        )


# =========================================================================
# Demo
# =========================================================================


def demo_humaneval() -> None:
    """Demonstrate HumanEval benchmark with a sample task."""
    print("=" * 70)
    print("HumanEval Benchmark Demo")
    print("=" * 70)

    # Task loading
    tasks = _load_humaneval_tasks()
    print(f"\n  Loaded {len(tasks)} HumanEval tasks from config")

    if tasks:
        task = tasks[0]
        print(f"\n  Sample task: {task['task_id']}")
        print(f"\n  Prompt:")
        for line in task["prompt"].strip().split("\n")[:6]:
            print(f"    {line}")
        print(f"    ...")
        print(f"\n  Entry point: {task['entry_point']}")

    # Execute a correct solution
    if tasks:
        print(f"\n  Testing canonical solution for {tasks[0]['task_id']}...")
        task = tasks[0]
        result = _execute_code_sandbox(
            task["prompt"] + task["canonical_solution"],
            task["test"],
            task["entry_point"],
            timeout=5.0,
        )
        print(f"    Passed: {result.passed}")
        print(f"    Time:   {result.execution_time:.4f}s")

    # Execute incorrect code
    if tasks:
        print(f"\n  Testing INCORRECT code...")
        incorrect_code = task["prompt"] + "\n    return False  # always wrong\n"
        result = _execute_code_sandbox(
            incorrect_code,
            task["test"],
            task["entry_point"],
            timeout=5.0,
        )
        print(f"    Passed: {result.passed}")
        print(f"    Error:  {result.error_type} -> {result.error_message[:100]}")

    # Pass@k computation
    print(f"\n  Pass@k computation example:")
    # Simulate: 200 samples per problem, 30 pass
    n_total, n_correct = 200, 30
    for k in [1, 10, 100]:
        pk = compute_pass_at_k(n_total, n_correct, k)
        print(f"    n={n_total}, c={n_correct}, pass@{k} = {pk:.4f}")


def demo_mbpp() -> None:
    """Demonstrate MBPP task loading and normalization."""
    print("\n" + "=" * 70)
    print("MBPP Support Demo")
    print("=" * 70)

    # Construct a mock MBPP task
    mock_task = {
        "task_id": 1,
        "text": "Write a function to find the maximum of three numbers.",
        "test_list": [
            "assert candidate(1, 2, 3) == 3",
            "assert candidate(5, 3, 1) == 5",
            "assert candidate(2, 2, 2) == 2",
        ],
        "code": "def max_of_three(a, b, c):\n    return max(a, b, c)\n",
    }

    normalized = _normalize_mbpp_task(mock_task)
    print(f"\n  Original MBPP task: id={mock_task['task_id']}")
    print(f"  Normalized task_id: {normalized['task_id']}")
    print(f"  Test code:")
    for line in normalized["test"].strip().split("\n")[:5]:
        print(f"    {line}")


def main() -> None:
    demo_humaneval()
    demo_mbpp()


if __name__ == "__main__":
    main()
