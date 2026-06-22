"""
Math reasoning benchmarks: GSM8K and MATH.

Evaluates mathematical reasoning capabilities of language models:
  - GSM8K: Grade School Math 8K - grade-school-level word problems
  - MATH: Competition-level mathematics with subject breakdown
  - Chain-of-thought prompting support
  - Answer extraction with robust regex patterns
  - Scoring by final answer correctness
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from .benchmark_registry import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    register_benchmark,
)
from .metrics import bootstrap_confidence_interval


# =========================================================================
# Answer Extraction
# =========================================================================


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from GSM8K model output.

    Looks for patterns like:
      - "The answer is 42"
      - "#### 42" (standard GSM8K format)
      - "= 42" or "=42"
      - Final number before end of text

    Args:
        text: Raw model completion including chain-of-thought.

    Returns:
        Extracted answer string (may include commas, decimal points),
        or None if no answer found.
    """
    # Strip thinking tags first
    text = _strip_think_tags(text)

    # Strategy 1: GSM8K standard format "#### <answer>"
    match = re.search(r"####\s*(-?\$?\s*[\d,]+(?:\.\d+)?%?)", text, re.IGNORECASE)
    if match:
        return _clean_number(match.group(1))

    # Strategy 2: "The answer is X" / "Answer: X"
    for pattern in [
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(-?\$?\s*[\d,]+(?:\.\d+)?%?)",
        r"(?:result|solution)\s*(?:is|:)\s*(-?\$?\s*[\d,]+(?:\.\d+)?%?)",
        r"=\s*(-?\$?\s*[\d,]+(?:\.\d+)?%?)",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_number(match.group(1))

    # Strategy 3: Last number-like token in the text
    numbers = re.findall(r"-?\$?\s*[\d,]+(?:\.\d+)?%?", text)
    if numbers:
        return _clean_number(numbers[-1])

    return None


def extract_math_answer(text: str) -> str | None:
    """Extract the final answer from MATH benchmark model output.

    MATH problems have more varied answer formats:
      - Numeric: "42"
      - Fraction: "\\frac{3}{4}" or "3/4"
      - Expression: "x^2 + 3x + 2"
      - LaTeX boxed: "\\boxed{42}"

    Args:
        text: Raw model completion.

    Returns:
        Extracted answer string, or None.
    """
    text = _strip_think_tags(text)

    # Strategy 1: \boxed{...}  (standard MATH format)
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    # Strategy 2: \boxed{...} in plain text "boxed{...}"
    match = re.search(r"boxed\{([^}]+)\}", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Strategy 3: "The answer is X"
    for pattern in [
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|\n|$)",
        r"=\s*(.+?)(?:\.|\n|$)",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            if cleaned and not cleaned.isspace():
                return cleaned

    # Strategy 4: Last line that looks like an answer
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in reversed(lines):
        if any(
            kw in line.lower()
            for kw in ["answer", "therefore", "thus", "hence", "result", "="]
        ):
            candidate = line.split("=")[-1].strip().rstrip(".")
            if candidate:
                return candidate

    return None


def _strip_think_tags(text: str) -> str:
    """Remove <｜end▁of▁thinking｜>... response style thinking tags from text."""
    # Remove content between  response and  response
    text = re.sub(r" response.*? response", "", text, flags=re.DOTALL)
    # Remove <thinking>...</thinking>
    text = re.sub(
        r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    return text


def _clean_number(text: str) -> str:
    """Clean a numeric string: remove $, commas, extra whitespace."""
    cleaned = text.replace("$", "").replace(",", "").replace("%", "").strip()
    # Remove leading/trailing whitespace within
    cleaned = " ".join(cleaned.split())
    return cleaned


def normalize_answer(extracted: str, reference: str) -> tuple[str, str]:
    """Normalize extracted and reference answers for comparison.

    Args:
        extracted: Answer extracted from model output.
        reference: Ground truth answer.

    Returns:
        Tuple of (normalized_extracted, normalized_reference).
    """

    def norm(s: str) -> str:
        s = s.strip().lower()
        # Remove LaTeX commands
        s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\[a-zA-Z]+", "", s)
        # Remove formatting
        s = s.replace("$", "").replace("%", "").replace(",", "").replace(" ", "")
        # Remove trailing punctuation
        s = s.rstrip(".")
        return s

    return norm(extracted), norm(reference)


# =========================================================================
# Dataset Loading
# =========================================================================


def _load_gsm8k(
    dataset_path: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load GSM8K dataset from JSONL file.

    Format: {"question": str, "answer": str}
    The answer field typically includes chain-of-thought with #### final_answer.

    Args:
        dataset_path: Path to GSM8K JSONL file.
        max_samples: Cap on number of questions.
        seed: Random seed for shuffling.

    Returns:
        List of dicts with "question" and "answer" keys.
    """
    if not os.path.exists(dataset_path):
        return []

    questions: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(
                {
                    "question": data.get("question", ""),
                    "answer": data.get("answer", ""),
                }
            )

    if max_samples is not None and max_samples < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, max_samples)

    return questions


def _load_math(
    dataset_path: str,
    subjects: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load MATH dataset.

    The MATH dataset is organized by subject/level directories with
    individual JSON files per problem.

    Args:
        dataset_path: Root directory of MATH dataset.
        subjects: Specific subjects to load (e.g. ["algebra", "counting"]).
        max_samples: Cap on number of problems.
        seed: Random seed for shuffling.

    Returns:
        List of dicts with "question", "answer", "subject", "level" keys.
    """
    if not os.path.isdir(dataset_path):
        return []

    all_subjects = subjects or [
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    problems: list[dict[str, Any]] = []
    for subject in all_subjects:
        subject_dir = os.path.join(dataset_path, subject)
        if not os.path.isdir(subject_dir):
            continue

        for fname in os.listdir(subject_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(subject_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                problems.append(
                    {
                        "question": data.get("problem", data.get("question", "")),
                        "answer": data.get("solution", data.get("answer", "")),
                        "subject": subject,
                        "level": data.get("level", ""),
                    }
                )

    if max_samples is not None and max_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, max_samples)

    return problems


# =========================================================================
# Prompt Formatting
# =========================================================================


def format_gsm8k_prompt(
    question: str, few_shot_examples: list[dict[str, Any]] | None = None
) -> str:
    """Format a GSM8K prompt with optional few-shot examples.

    Args:
        question: The math word problem question.
        few_shot_examples: List of dicts with "question" and "answer" keys.

    Returns:
        Formatted prompt string.
    """
    prompt_parts: list[str] = []

    if few_shot_examples:
        for ex in few_shot_examples:
            prompt_parts.append(f"Q: {ex['question']}")
            prompt_parts.append(f"A: {ex['answer']}")
            prompt_parts.append("")

    prompt_parts.append(f"Q: {question}")
    prompt_parts.append("A:")

    return "\n".join(prompt_parts)


def format_math_prompt(
    question: str, few_shot_examples: list[dict[str, Any]] | None = None
) -> str:
    """Format a MATH prompt with optional few-shot examples.

    Args:
        question: The math problem.
        few_shot_examples: List of dicts with "question" and "answer" keys.

    Returns:
        Formatted prompt string.
    """
    prompt_parts: list[str] = []

    prompt_parts.append("Solve the following math problem step by step.")
    prompt_parts.append("Put your final answer within \\boxed{}.")
    prompt_parts.append("")

    if few_shot_examples:
        for ex in few_shot_examples:
            prompt_parts.append(f"Problem: {ex['question']}")
            prompt_parts.append(f"Solution: {ex['answer']}")
            prompt_parts.append("")

    prompt_parts.append(f"Problem: {question}")
    prompt_parts.append("Solution:")

    return "\n".join(prompt_parts)


# =========================================================================
# GSM8K Benchmark
# =========================================================================


@register_benchmark
class GSM8KBenchmark(Benchmark):
    """Grade School Math 8K benchmark.

    Evaluates multi-step math reasoning with chain-of-thought prompting.
    """

    name = "gsm8k"
    description = (
        "Grade School Math 8K - multi-step math word problems with chain-of-thought"
    )

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        dataset_path: str = "data/gsm8k",
    ):
        super().__init__(config)
        self.dataset_path = dataset_path
        self._results_cache: dict[str, Any] = {}

    def run(
        self,
        model_fn: Any = None,
        dataset_path: str | None = None,
    ) -> BenchmarkResult:
        """Run GSM8K evaluation.

        Args:
            model_fn: Callable `model_fn(prompt: str) -> str`.
                      If None, uses stub.
            dataset_path: Override dataset path.

        Returns:
            BenchmarkResult with accuracy and per-question results.
        """
        path = dataset_path or self.dataset_path

        # Detect whether path is a file or directory
        if os.path.isdir(path):
            # Look for test.jsonl in directory
            path = os.path.join(path, "test.jsonl")

        questions = _load_gsm8k(
            path,
            max_samples=self.config.max_samples,
            seed=self.config.seed,
        )

        if not questions:
            print(f"  [GSM8K] No questions found at {path}")
            return BenchmarkResult(
                benchmark_name=self.name,
                config_hash=self.config.config_hash(),
            )

        print(f"  [GSM8K] {len(questions)} questions (n_shot={self.config.n_shot})")

        # Load few-shot examples
        few_shot: list[dict[str, Any]] = []
        if self.config.n_shot > 0:
            train_path = os.path.join(os.path.dirname(path), "train.jsonl")
            if os.path.exists(train_path):
                few_shot = _load_gsm8k(
                    train_path,
                    max_samples=self.config.n_shot,
                    seed=self.config.seed,
                )

        correct = 0
        per_sample_scores: list[float] = []
        question_details: list[dict[str, Any]] = []
        start = time.perf_counter()

        for i, q in enumerate(questions):
            prompt = format_gsm8k_prompt(q["question"], few_shot)
            output = model_fn(prompt) if model_fn else self._stub_model(prompt)

            extracted = extract_gsm8k_answer(output)
            reference = extract_gsm8k_answer(q["answer"])

            is_correct = False
            if extracted is not None and reference is not None:
                norm_extracted, norm_ref = normalize_answer(extracted, reference)
                is_correct = norm_extracted == norm_ref

            if is_correct:
                correct += 1
            per_sample_scores.append(1.0 if is_correct else 0.0)

            question_details.append(
                {
                    "index": i,
                    "question": q["question"][:100],
                    "extracted_answer": extracted,
                    "reference_answer": reference,
                    "correct": is_correct,
                }
            )

            if (i + 1) % 50 == 0 or i == len(questions) - 1:
                acc = correct / (i + 1)
                print(f"    [{i + 1}/{len(questions)}] accuracy so far: {acc:.4f}")

        elapsed = time.perf_counter() - start
        accuracy = correct / len(questions) if questions else 0.0
        ci = bootstrap_confidence_interval(per_sample_scores, seed=self.config.seed)

        print(
            f"  [GSM8K] Final accuracy: {accuracy:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]"
        )

        self._results_cache = {
            "accuracy": accuracy,
            "num_correct": correct,
            "num_total": len(questions),
            "ci_lower": ci["lower"],
            "ci_upper": ci["upper"],
            "question_details": question_details,
            "latency_seconds": elapsed,
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=[
                {
                    "metric": "accuracy",
                    "value": accuracy,
                    "ci_lower": ci["lower"],
                    "ci_upper": ci["upper"],
                    "num_correct": correct,
                    "num_total": len(questions),
                }
            ],
            aggregate_score=accuracy,
            config_hash=self.config.config_hash(),
            metadata={
                "n_shot": self.config.n_shot,
                "num_questions": len(questions),
            },
        )

    @staticmethod
    def _stub_model(prompt: str) -> str:
        """Stub model returning random answer."""
        # Simulate chain-of-thought + answer
        number = random.randint(1, 1000)
        return f"Let's think step by step.\nThe answer is {number}\n#### {number}"

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, Any]:
        """Evaluate GSM8K predictions by extracting and comparing answers.

        Args:
            predictions: Raw model outputs.
            references: Ground truth answer strings.

        Returns:
            Dict with "accuracy", "num_correct", "num_total".
        """
        correct = 0
        details: list[dict[str, Any]] = []

        for pred, ref in zip(predictions, references):
            extracted = extract_gsm8k_answer(pred)
            ref_answer = extract_gsm8k_answer(ref)
            is_correct = False
            if extracted is not None and ref_answer is not None:
                norm_e, norm_r = normalize_answer(extracted, ref_answer)
                is_correct = norm_e == norm_r
            if is_correct:
                correct += 1
            details.append(
                {
                    "extracted": extracted,
                    "reference": ref_answer,
                    "correct": is_correct,
                }
            )

        return {
            "accuracy": correct / len(predictions) if predictions else 0.0,
            "num_correct": correct,
            "num_total": len(predictions),
            "details": details,
        }

    def get_results(self) -> BenchmarkResult:
        """Return cached results."""
        if not self._results_cache:
            return BenchmarkResult(
                benchmark_name=self.name,
                config_hash=self.config.config_hash(),
            )

        acc = self._results_cache.get("accuracy", 0.0)
        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=[
                {
                    "metric": "accuracy",
                    "value": acc,
                }
            ],
            aggregate_score=acc,
            config_hash=self.config.config_hash(),
        )


# =========================================================================
# MATH Benchmark
# =========================================================================


@register_benchmark
class MATHBenchmark(Benchmark):
    """MATH benchmark for competition-level mathematics.

    Evaluates models across 7 subjects: algebra, counting & probability,
    geometry, intermediate algebra, number theory, prealgebra, precalculus.
    """

    name = "math"
    description = (
        "MATH benchmark - competition-level mathematics with subject breakdown"
    )

    MATH_SUBJECTS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        dataset_path: str = "data/math",
    ):
        super().__init__(config)
        self.dataset_path = dataset_path
        self._results_cache: dict[str, Any] = {}

    def run(
        self,
        model_fn: Any = None,
        subjects: list[str] | None = None,
    ) -> BenchmarkResult:
        """Run MATH evaluation.

        Args:
            model_fn: Callable `model_fn(prompt: str) -> str`.
            subjects: Specific subjects to evaluate.

        Returns:
            BenchmarkResult with subject-level breakdown.
        """
        problems = _load_math(
            self.dataset_path,
            subjects=subjects,
            max_samples=self.config.max_samples,
            seed=self.config.seed,
        )

        if not problems:
            print(f"  [MATH] No problems found at {self.dataset_path}")
            return BenchmarkResult(
                benchmark_name=self.name,
                config_hash=self.config.config_hash(),
            )

        print(f"  [MATH] {len(problems)} problems (n_shot={self.config.n_shot})")

        # Group by subject
        subject_problems: dict[str, list[dict[str, Any]]] = {}
        for p in problems:
            subj = p.get("subject", "unknown")
            subject_problems.setdefault(subj, []).append(p)

        subject_results: dict[str, dict[str, Any]] = {}
        per_sample_scores: list[float] = []
        total_correct = 0
        start = time.perf_counter()

        for subject, subj_problems in subject_problems.items():
            subj_correct = 0
            subj_scores: list[float] = []

            for p in subj_problems:
                prompt = format_math_prompt(p["question"])
                output = model_fn(prompt) if model_fn else self._stub_model(prompt)

                extracted = extract_math_answer(output)
                ref_answer = p["answer"]
                is_correct = False

                if extracted is not None:
                    norm_e, norm_r = normalize_answer(extracted, ref_answer)
                    is_correct = norm_e == norm_r

                if is_correct:
                    subj_correct += 1
                    total_correct += 1
                subj_scores.append(1.0 if is_correct else 0.0)
                per_sample_scores.append(1.0 if is_correct else 0.0)

            subj_acc = subj_correct / len(subj_problems) if subj_problems else 0.0
            ci = bootstrap_confidence_interval(subj_scores, seed=self.config.seed)
            subject_results[subject] = {
                "accuracy": subj_acc,
                "num_correct": subj_correct,
                "num_total": len(subj_problems),
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
            }
            print(
                f"    {subject}: {subj_acc:.4f} ({subj_correct}/{len(subj_problems)})"
            )

        elapsed = time.perf_counter() - start
        overall_acc = total_correct / len(problems) if problems else 0.0
        overall_ci = bootstrap_confidence_interval(
            per_sample_scores, seed=self.config.seed
        )

        print(
            f"  [MATH] Overall accuracy: {overall_acc:.4f} "
            f"[{overall_ci['lower']:.4f}, {overall_ci['upper']:.4f}]"
        )

        self._results_cache = {
            "accuracy": overall_acc,
            "num_correct": total_correct,
            "num_total": len(problems),
            "ci_lower": overall_ci["lower"],
            "ci_upper": overall_ci["upper"],
            "subject_results": subject_results,
            "latency_seconds": elapsed,
        }

        task_results = [
            {
                "metric": f"accuracy_{subj}",
                "value": r["accuracy"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
            }
            for subj, r in subject_results.items()
        ]
        task_results.append(
            {
                "metric": "accuracy_overall",
                "value": overall_acc,
                "ci_lower": overall_ci["lower"],
                "ci_upper": overall_ci["upper"],
            }
        )

        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=task_results,
            aggregate_score=overall_acc,
            config_hash=self.config.config_hash(),
            metadata={
                "n_shot": self.config.n_shot,
                "subjects": list(subject_results.keys()),
            },
        )

    @staticmethod
    def _stub_model(prompt: str) -> str:
        """Stub model returning random answer."""
        answer = random.randint(1, 100)
        return f"Let's solve this step by step.\\n\\nThe answer is \\boxed{{{answer}}}"

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, Any]:
        """Evaluate MATH predictions by extracting and comparing boxed answers.

        Args:
            predictions: Raw model outputs.
            references: Ground truth answers.

        Returns:
            Dict with "accuracy", "num_correct", "num_total".
        """
        correct = 0
        for pred, ref in zip(predictions, references):
            extracted = extract_math_answer(pred)
            if extracted is not None:
                norm_e, norm_r = normalize_answer(extracted, ref)
                if norm_e == norm_r:
                    correct += 1

        return {
            "accuracy": correct / len(predictions) if predictions else 0.0,
            "num_correct": correct,
            "num_total": len(predictions),
        }

    def get_results(self) -> BenchmarkResult:
        """Return cached results."""
        if not self._results_cache:
            return BenchmarkResult(
                benchmark_name=self.name,
                config_hash=self.config.config_hash(),
            )

        acc = self._results_cache.get("accuracy", 0.0)
        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=[{"metric": "accuracy", "value": acc}],
            aggregate_score=acc,
            config_hash=self.config.config_hash(),
        )


# =========================================================================
# Demo
# =========================================================================


def demo_answer_extraction() -> None:
    """Demonstrate answer extraction for GSM8K and MATH."""
    print("=" * 70)
    print("Math Answer Extraction Demo")
    print("=" * 70)

    # GSM8K extraction
    print("\n  GSM8K Answer Extraction:")
    gsm8k_examples = [
        (
            "Let's solve this.\nJanet has 5 apples. She buys 3 more.\nTotal = 5 + 3 = 8\n#### 8",
            "8",
        ),
        ("The answer is 42", "42"),
        ("After computing, we get #### -15", "-15"),
        ("The final answer is $1,234.56", "1234.56"),
        ("Step 1: ... Step 2: ... The answer is 50%", "50"),
    ]
    for text, expected in gsm8k_examples:
        extracted = extract_gsm8k_answer(text)
        status = "✓" if extracted == expected else "✗"
        print(f"    {status} '{text[:60]}...' -> '{extracted}' (expected '{expected}')")

    # MATH extraction
    print("\n  MATH Answer Extraction:")
    math_examples = [
        ("Let's solve: ... \\boxed{42}", "42"),
        ("The solution is boxed{3/4}", "3/4"),
        ("Therefore, the answer is \\boxed{x^2 + 3x + 2}", "x^2 + 3x + 2"),
    ]
    for text, expected in math_examples:
        extracted = extract_math_answer(text)
        status = "✓" if extracted == expected else "✗"
        print(f"    {status} '{text[:60]}...' -> '{extracted}' (expected '{expected}')")

    # Answer normalization
    print("\n  Answer Normalization:")
    norm_tests = [
        ("$1,234.56", "1234.56"),
        ("50%", "50%"),
        ("\\frac{3}{4}", "3/4"),
        ("x^2+3x+2", "x^2+3x+2"),
    ]
    for a, b in norm_tests:
        na, nb = normalize_answer(a, b)
        status = "✓" if na == nb else "✗"
        print(f"    {status} '{a}' vs '{b}' -> '{na}' == '{nb}'")

    # Prompt formatting
    print("\n  GSM8K Prompt Formatting:")
    few_shot = [
        {"question": "What is 2 + 2?", "answer": "4"},
    ]
    prompt = format_gsm8k_prompt("A store sells apples for $1 each. ...", few_shot)
    for line in prompt.split("\n")[:8]:
        print(f"    {line}")


def demo_gsm8k_e2e() -> None:
    """End-to-end GSM8K demo with mock data."""
    print("\n" + "=" * 70)
    print("GSM8K End-to-End Demo")
    print("=" * 70)

    config = BenchmarkConfig(n_shot=1, max_samples=10, seed=42)
    benchmark = GSM8KBenchmark(config=config)

    # Evaluate some mock predictions
    predictions = [
        "Let's think. 5 + 3 = 8. #### 8",
        "The answer is 42",
        "I'm not sure. Maybe 100?",
        "Step by step: 2 * 3 = 6, 6 + 4 = 10. #### 10",
    ]
    references = ["#### 8", "#### 42", "#### 50", "#### 10"]

    result = benchmark.evaluate(predictions, references)
    print(f"\n  Evaluated {len(predictions)} predictions:")
    print(f"    Accuracy: {result['accuracy']:.4f}")
    print(f"    Correct:  {result['num_correct']}/{result['num_total']}")
    for d in result["details"]:
        status = "✓" if d["correct"] else "✗"
        print(f"    {status} extracted='{d['extracted']}' ref='{d['reference']}'")


def main() -> None:
    demo_answer_extraction()
    demo_gsm8k_e2e()


if __name__ == "__main__":
    main()
