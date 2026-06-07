"""
LLM Evaluation Benchmarks Module.

Provides a production-grade benchmark evaluation framework for language models,
including perplexity, multiple-choice knowledge, code generation, math reasoning,
and unified benchmark orchestration.

Usage:
    from cs336.benchmarks import (
        BenchmarkRegistry,
        BenchmarkConfig,
        UnifiedBenchmarkRunner,
        RunnerConfig,
        quick_eval,
        compute_perplexity,
        compute_pass_at_k,
        compute_elo,
        compute_bleu,
        compute_rouge_l,
        bootstrap_confidence_interval,
        DataContaminationDetector,
    )

    # Run a quick evaluation
    def my_model(prompt: str) -> str:
        return generate_response(prompt)

    summary = quick_eval(my_model, benchmarks=["mmlu", "gsm8k"], model_name="MyModel")

    # Or use the full runner with caching
    runner = UnifiedBenchmarkRunner(
        RunnerConfig(model_name="MyModel", output_dir="results")
    )
    runner.model_fn = my_model
    summary = runner.run_suite("standard")
    runner.export_results(summary)
"""

from .benchmark_registry import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkRegistry,
    BenchmarkResult,
    register_benchmark,
)
from .bench_runner import (
    CacheManager,
    RunSummary,
    RunnerConfig,
    UnifiedBenchmarkRunner,
    quick_eval,
    STANDARD_SUITE,
    CODE_SUITE,
    MATH_SUITE,
    FULL_SUITE,
)
from .metrics import (
    bootstrap_confidence_interval,
    compute_bleu,
    compute_elo,
    compute_elo_from_battles,
    compute_pass_at_k,
    compute_pass_at_k_from_samples,
    compute_perplexity,
    compute_perplexity_from_loss,
    compute_rouge_l,
    DataContaminationDetector,
)
from .perplexity import PerplexityBenchmark
from .mmlu import MMLUBenchmark
from .humaneval import HumanEvalBenchmark
from .gsm8k import GSM8KBenchmark, MATHBenchmark

__all__ = [
    # Registry and config
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "register_benchmark",
    # Runner
    "CacheManager",
    "RunSummary",
    "RunnerConfig",
    "UnifiedBenchmarkRunner",
    "quick_eval",
    # Benchmark suites
    "STANDARD_SUITE",
    "CODE_SUITE",
    "MATH_SUITE",
    "FULL_SUITE",
    # Metrics
    "bootstrap_confidence_interval",
    "compute_bleu",
    "compute_elo",
    "compute_elo_from_battles",
    "compute_pass_at_k",
    "compute_pass_at_k_from_samples",
    "compute_perplexity",
    "compute_perplexity_from_loss",
    "compute_rouge_l",
    "DataContaminationDetector",
    # Benchmark classes
    "PerplexityBenchmark",
    "MMLUBenchmark",
    "HumanEvalBenchmark",
    "GSM8KBenchmark",
    "MATHBenchmark",
]
