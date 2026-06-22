"""
MMLU (Massive Multitask Language Understanding) benchmark.

Evaluates models across 57 subjects spanning STEM, social sciences,
humanities, and other domains. Supports:
  - All 57 MMLU subjects with category grouping
  - Few-shot prompting with configurable n_shot
  - Multiple choice answer parsing from model outputs
  - Subject-level and macro-average scoring
  - Confidence intervals per subject
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
# Utilities
# =========================================================================


def _load_mmlu_subjects() -> dict[str, list[str]]:
    """Load MMLU subject categories from config file."""
    config_path = Path(__file__).parent / "configs" / "mmlu_subjects.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            data = json.load(f)
        return data.get("subjects", {})
    # Fallback: hard-coded minimal list
    return {
        "STEM": [
            "college_mathematics",
            "college_physics",
            "high_school_mathematics",
            "computer_security",
            "machine_learning",
        ],
        "Social_Sciences": ["econometrics", "sociology"],
        "Humanities": ["philosophy", "moral_scenarios"],
        "Other": ["global_facts", "miscellaneous"],
    }


def _load_mmlu_questions(
    subject: str,
    dataset_dir: str,
    max_samples: int | None = None,
    split: str = "test",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load MMLU questions for a single subject.

    Expects files in the format: {dataset_dir}/{split}/{subject}_{split}.csv
    or {dataset_dir}/{split}/{subject}_{split}.jsonl

    CSV format: question,A,B,C,D,answer
    JSONL format: {"question": ..., "choices": [...], "answer": ...}

    Args:
        subject: Subject name (e.g. "college_mathematics").
        dataset_dir: Root directory containing MMLU data.
        max_samples: Cap on number of questions loaded.
        split: "test" or "dev" (for few-shot examples).
        seed: Random seed for shuffling when max_samples is set.

    Returns:
        List of dicts with "question", "choices", "answer" keys.
    """
    import csv

    base_path = Path(dataset_dir) / split
    questions: list[dict[str, Any]] = []

    # Try JSONL first
    jsonl_path = base_path / f"{subject}_{split}.jsonl"
    csv_path = base_path / f"{subject}_{split}.csv"

    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                questions.append(
                    {
                        "question": data.get("question", ""),
                        "choices": data.get("choices", []),
                        "answer": data.get("answer", ""),
                    }
                )
    elif csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return questions
            for row in reader:
                if len(row) < 6:
                    continue
                questions.append(
                    {
                        "question": row[0],
                        "choices": list(row[1:5]),
                        "answer": row[5].strip(),
                    }
                )
    else:
        # Try direct path
        direct_jsonl = Path(dataset_dir) / f"{subject}.jsonl"
        if direct_jsonl.exists():
            with open(direct_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("split", split) == split:
                        questions.append(
                            {
                                "question": data.get("question", ""),
                                "choices": data.get("choices", []),
                                "answer": data.get("answer", ""),
                            }
                        )

    if max_samples is not None and max_samples < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, max_samples)

    return questions


def _format_mmlu_prompt(
    question: str,
    choices: list[str],
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Format an MMLU multiple-choice prompt.

    Args:
        question: The question text.
        choices: List of answer choices (A, B, C, D, ...).
        few_shot_examples: List of dicts with "question", "choices", "answer"
                           for few-shot in-context examples.

    Returns:
        Formatted prompt string.
    """
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    prompt_parts: list[str] = []

    # Add few-shot examples
    if few_shot_examples:
        for ex in few_shot_examples:
            ex_labels = labels[: len(ex["choices"])]
            prompt_parts.append(f"Question: {ex['question']}")
            for i, choice_text in enumerate(ex["choices"]):
                prompt_parts.append(f"{ex_labels[i]}. {choice_text}")
            prompt_parts.append(f"Answer: {ex['answer']}")
            prompt_parts.append("")

    # Add the actual question
    q_labels = labels[: len(choices)]
    prompt_parts.append(f"Question: {question}")
    for i, choice_text in enumerate(choices):
        prompt_parts.append(f"{q_labels[i]}. {choice_text}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def _parse_mmlu_answer(output: str, num_choices: int = 4) -> str | None:
    """Parse the multiple-choice answer from model output.

    Tries several strategies in order:
      1. Match "Answer: X" or "Answer (X)" pattern.
      2. Find the first occurrence of a valid choice letter.
      3. Match the text of the choice itself.

    Args:
        output: Raw model completion text.
        num_choices: Number of choices available.

    Returns:
        The predicted answer letter (e.g. "A", "B"), or None if unparseable.
    """
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:num_choices]
    output_upper = output.strip().upper()

    # Strategy 1: "Answer: X" or "The answer is X"
    patterns = [
        r"(?:ANSWER|ANSWER IS|ANSWER:\s*)\s*([A-Z])",
        r"(?:CHOICE|OPTION)\s*([A-Z])",
        r"([A-Z])\)",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output_upper)
        for match in matches:
            if match in labels:
                return match

    # Strategy 2: First valid label character appearing alone
    tokens = output_upper.split()
    for token in tokens:
        token = token.strip(".,;:!?\"'()[]{}")
        if len(token) == 1 and token in labels:
            return token

    # Strategy 3: Look for the letter as a single character
    for ch in output_upper:
        if ch in labels:
            return ch

    return None


def _format_answer_letter(letter: str) -> str:
    """Normalize an answer to uppercase letter string."""
    return letter.strip().upper()


# =========================================================================
# MMLU Benchmark
# =========================================================================


@register_benchmark
class MMLUBenchmark(Benchmark):
    """Massive Multitask Language Understanding benchmark.

    Evaluates model knowledge across 57 subjects in 4 categories:
    STEM, Social Sciences, Humanities, Other.
    """

    name = "mmlu"
    description = "Massive Multitask Language Understanding (57 subjects, 4 categories)"

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        dataset_dir: str = "data/mmlu",
    ):
        super().__init__(config)
        self.dataset_dir = dataset_dir
        self._subjects = _load_mmlu_subjects()
        self._results_cache: dict[str, Any] = {}

    def run(
        self,
        model_fn: Any = None,
        subjects: list[str] | None = None,
    ) -> BenchmarkResult:
        """Run MMLU evaluation.

        Args:
            model_fn: Callable `model_fn(prompt: str) -> str` for generation.
                      If None, uses a stub that predicts randomly.
            subjects: Specific subjects to evaluate (default: all).

        Returns:
            BenchmarkResult with subject-level and category-level scores.
        """
        # Collect all subjects to evaluate
        all_subjects: list[tuple[str, str]] = []  # (subject, category)
        for category, subject_list in self._subjects.items():
            for s in subject_list:
                if subjects is None or s in subjects:
                    all_subjects.append((s, category))

        if not all_subjects:
            print("  [MMLU] No subjects found to evaluate.")
            return BenchmarkResult(
                benchmark_name=self.name,
                config_hash=self.config.config_hash(),
            )

        print(
            f"  [MMLU] Evaluating {len(all_subjects)} subjects "
            f"(n_shot={self.config.n_shot})"
        )

        task_results: list[dict[str, Any]] = []
        per_sample_scores: list[float] = []
        category_scores: dict[str, list[float]] = {cat: [] for cat in self._subjects}

        for subject, category in all_subjects:
            print(f"    [{subject}]", end=" ", flush=True)

            try:
                # Load questions
                test_questions = _load_mmlu_questions(
                    subject,
                    self.dataset_dir,
                    max_samples=self.config.max_samples,
                    split="test",
                    seed=self.config.seed,
                )

                if not test_questions:
                    task_results.append(
                        {
                            "subject": subject,
                            "category": category,
                            "accuracy": float("nan"),
                            "num_questions": 0,
                            "error": "No questions loaded",
                        }
                    )
                    print("no data")
                    continue

                # Load few-shot examples from dev set
                few_shot_examples: list[dict[str, Any]] = []
                if self.config.n_shot > 0:
                    few_shot_examples = _load_mmlu_questions(
                        subject,
                        self.dataset_dir,
                        max_samples=self.config.n_shot,
                        split="dev",
                        seed=self.config.seed,
                    )

                # Run evaluation
                correct = 0
                subject_scores: list[float] = []
                start = time.perf_counter()

                for q in test_questions:
                    prompt = _format_mmlu_prompt(
                        q["question"], q["choices"], few_shot_examples
                    )
                    output = model_fn(prompt) if model_fn else self._stub_model(prompt)
                    predicted = _parse_mmlu_answer(output, len(q["choices"]))
                    expected = _format_answer_letter(q["answer"])

                    is_correct = (predicted == expected) if predicted else False
                    if is_correct:
                        correct += 1
                    subject_scores.append(1.0 if is_correct else 0.0)

                elapsed = time.perf_counter() - start

                accuracy = correct / len(test_questions)
                ci = bootstrap_confidence_interval(
                    subject_scores, seed=self.config.seed
                )

                task_results.append(
                    {
                        "subject": subject,
                        "category": category,
                        "accuracy": accuracy,
                        "num_correct": correct,
                        "num_questions": len(test_questions),
                        "ci_lower": ci["lower"],
                        "ci_upper": ci["upper"],
                        "latency_seconds": elapsed,
                    }
                )

                category_scores[category].append(accuracy)
                per_sample_scores.extend(subject_scores)

                print(f"acc={accuracy:.4f}")

            except FileNotFoundError as e:
                task_results.append(
                    {
                        "subject": subject,
                        "category": category,
                        "accuracy": float("nan"),
                        "error": str(e),
                    }
                )
                print(f"SKIP: {e}")
            except Exception as e:
                task_results.append(
                    {
                        "subject": subject,
                        "category": category,
                        "accuracy": float("nan"),
                        "error": str(e),
                    }
                )
                print(f"ERROR: {e}")

        # Category-level aggregation
        category_summary: dict[str, dict[str, float]] = {}
        for cat, scores in category_scores.items():
            if scores:
                category_summary[cat] = {
                    "mean_accuracy": sum(scores) / len(scores),
                    "num_subjects": len(scores),
                    "min_accuracy": min(scores),
                    "max_accuracy": max(scores),
                }
            else:
                category_summary[cat] = {
                    "mean_accuracy": float("nan"),
                    "num_subjects": 0,
                    "min_accuracy": float("nan"),
                    "max_accuracy": float("nan"),
                }

        # Macro-average across all subjects
        valid_accs = [
            r["accuracy"]
            for r in task_results
            if not (
                isinstance(r.get("accuracy"), float)
                and (r["accuracy"] != r["accuracy"])
            )  # nan check
        ]
        macro_avg = sum(valid_accs) / len(valid_accs) if valid_accs else float("nan")

        self._results_cache = {
            "per_subject": task_results,
            "category_summary": category_summary,
            "macro_average": macro_avg,
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=task_results,
            aggregate_score=macro_avg,
            config_hash=self.config.config_hash(),
            metadata={
                "n_shot": self.config.n_shot,
                "subjects_evaluated": len(all_subjects),
                "category_summary": category_summary,
            },
        )

    @staticmethod
    def _stub_model(prompt: str) -> str:
        """Stub model that returns a random choice."""
        labels = "ABCD"
        return f"Answer: {random.choice(labels)}"

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        num_choices: int = 4,
    ) -> dict[str, Any]:
        """Evaluate a batch of MMLU predictions.

        Args:
            predictions: Raw model output strings.
            references: Correct answer letters (e.g. ["A", "C", "B"]).
            num_choices: Number of choices per question.

        Returns:
            Dict with "accuracy", "num_correct", "num_total".
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: preds={len(predictions)}, refs={len(references)}"
            )

        correct = 0
        for pred, ref in zip(predictions, references):
            parsed = _parse_mmlu_answer(pred, num_choices)
            expected = _format_answer_letter(ref)
            if parsed == expected:
                correct += 1

        return {
            "accuracy": correct / len(predictions) if predictions else 0.0,
            "num_correct": correct,
            "num_total": len(predictions),
        }

    def get_results(self) -> BenchmarkResult:
        """Return cached results from previous run()."""
        if not self._results_cache:
            return BenchmarkResult(
                benchmark_name=self.name,
                aggregate_score=float("nan"),
                config_hash=self.config.config_hash(),
            )

        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=self._results_cache.get("per_subject", []),
            aggregate_score=self._results_cache.get("macro_average", float("nan")),
            config_hash=self.config.config_hash(),
            metadata={
                "n_shot": self.config.n_shot,
                "category_summary": self._results_cache.get("category_summary", {}),
            },
        )


# =========================================================================
# Demo
# =========================================================================


def demo_mmlu() -> None:
    """Demonstrate MMLU benchmark with mock data."""
    print("=" * 70)
    print("MMLU Benchmark Demo")
    print("=" * 70)

    config = BenchmarkConfig(n_shot=2, max_samples=5, seed=42)
    benchmark = MMLUBenchmark(config=config)

    # Demonstrate prompt formatting
    question = "What is the capital of France?"
    choices = ["London", "Paris", "Berlin", "Madrid"]
    few_shot = [
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": "B",
        },
        {
            "question": "Which planet is closest to the Sun?",
            "choices": ["Venus", "Earth", "Mercury", "Mars"],
            "answer": "C",
        },
    ]

    prompt = _format_mmlu_prompt(question, choices, few_shot)
    print(f"\n  Formatted prompt:\n  {'-' * 50}")
    for line in prompt.split("\n"):
        print(f"  {line}")
    print(f"  {'-' * 50}")

    # Demonstrate answer parsing
    test_outputs = [
        ("Answer: B", "B"),
        ("The answer is C", "C"),
        ("A. London", "A"),
        ("I think the correct choice is D", "D"),
        ("    A", "A"),
        ("unparseable gibberish", None),
    ]
    print(f"\n  Answer parsing:")
    for output, expected in test_outputs:
        parsed = _parse_mmlu_answer(output)
        status = "✓" if parsed == expected else "✗"
        print(f"    {status} '{output}' -> parsed='{parsed}', expected='{expected}'")

    # Demonstrate evaluation
    predictions = ["Answer: B", "The answer is C", "unparseable", "A. London"]
    references = ["B", "C", "A", "A"]
    result = benchmark.evaluate(predictions, references)
    print(f"\n  Batch evaluation:")
    print(f"    Predictions: {predictions}")
    print(f"    References:  {references}")
    print(
        f"    Accuracy: {result['accuracy']:.4f} ({result['num_correct']}/{result['num_total']})"
    )

    # Subject loading summary
    print(f"\n  MMLU subjects loaded from config:")
    for cat, subjs in benchmark._subjects.items():
        print(f"    {cat}: {len(subjs)} subjects")
        print(f"      {', '.join(subjs[:4])}{'...' if len(subjs) > 4 else ''}")


def main() -> None:
    demo_mmlu()


if __name__ == "__main__":
    main()
