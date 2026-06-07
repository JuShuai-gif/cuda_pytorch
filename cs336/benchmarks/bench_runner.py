"""
Unified benchmark runner for orchestrating multiple benchmarks.

Provides:
  - Run multiple benchmarks in sequence with progress tracking
  - Result aggregation and model comparison across benchmarks
  - Export to JSON, CSV, and Markdown formats
  - Caching mechanism with hash-based invalidation to avoid re-running
  - Logging with timestamps and benchmark status

Architecture:
  BenchmarkRunner
    ├── run_benchmarks(): Execute a list of benchmarks
    ├── run_suite(): Execute a pre-defined suite (e.g. "standard", "code")
    ├── aggregate_results(): Merge results across benchmarks
    ├── export_results(): Output to various formats
    └── cache/load: Persist and retrieve results
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .benchmark_registry import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkRegistry,
    BenchmarkResult,
)


# =========================================================================
# Cache Manager
# =========================================================================


class CacheManager:
    """File-based cache for benchmark results with hash-based invalidation.

    Cache key = hash(benchmark_name + config_hash + model_name).
    When any of these change, the cache is invalidated.
    """

    def __init__(self, cache_dir: str = "benchmark_results/.cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        benchmark_name: str,
        config: BenchmarkConfig,
        model_name: str = "unknown",
    ) -> str:
        """Generate a deterministic cache key."""
        raw = json.dumps(
            {
                "benchmark": benchmark_name,
                "config": config.to_dict(),
                "model": model_name,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self,
        benchmark_name: str,
        config: BenchmarkConfig,
        model_name: str = "unknown",
    ) -> BenchmarkResult | None:
        """Retrieve cached benchmark result.

        Returns None if not cached or cache is stale.
        """
        key = self._cache_key(benchmark_name, config, model_name)
        path = self._cache_path(key)

        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            return BenchmarkResult.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def put(
        self,
        result: BenchmarkResult,
        benchmark_name: str,
        config: BenchmarkConfig,
        model_name: str = "unknown",
    ) -> None:
        """Store benchmark result in cache."""
        key = self._cache_key(benchmark_name, config, model_name)
        path = self._cache_path(key)

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def invalidate(
        self,
        benchmark_name: str | None = None,
    ) -> int:
        """Invalidate cached results.

        Args:
            benchmark_name: If provided, only invalidate results for
                            this benchmark. If None, invalidate all.

        Returns:
            Number of cache entries removed.
        """
        removed = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if (
                    benchmark_name is None
                    or data.get("benchmark_name") == benchmark_name
                ):
                    path.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError, OSError):
                path.unlink()
                removed += 1
        return removed

    def list_cached(self) -> list[str]:
        """List cached benchmark identifiers."""
        cached: list[str] = []
        for path in self.cache_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                cached.append(
                    f"{data.get('benchmark_name', 'unknown')} [{path.stem[:8]}]"
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return sorted(cached)


# =========================================================================
# Benchmark Suite Definitions
# =========================================================================


# Pre-defined benchmark suites
STANDARD_SUITE = ["perplexity", "mmlu", "gsm8k"]
CODE_SUITE = ["humaneval"]
MATH_SUITE = ["gsm8k", "math"]
FULL_SUITE = ["perplexity", "mmlu", "gsm8k", "math", "humaneval"]


# =========================================================================
# Unified BenchmarkRunner
# =========================================================================


@dataclass
class RunnerConfig:
    """Configuration for the unified benchmark runner.

    Attributes:
        model_name: Name of the model being evaluated.
        output_dir: Directory for saving results.
        cache_enabled: Whether to use result caching.
        verbose: Print progress output.
        fail_fast: Stop on first benchmark failure.
        export_formats: Formats to export results (json, csv, md).
    """

    model_name: str = "unknown"
    output_dir: str = "benchmark_results"
    cache_enabled: bool = True
    verbose: bool = True
    fail_fast: bool = False
    export_formats: list[str] = field(default_factory=lambda: ["json", "csv", "md"])


@dataclass
class RunSummary:
    """Summary of a multi-benchmark run.

    Attributes:
        model_name: Model evaluated.
        timestamp: When the run started.
        benchmarks_run: List of benchmark names executed.
        results: Mapping from benchmark name to BenchmarkResult.
        failures: Dict of benchmark name -> error message.
        total_time_seconds: Total elapsed time.
    """

    model_name: str
    timestamp: str
    benchmarks_run: list[str] = field(default_factory=list)
    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    total_time_seconds: float = 0.0

    def overall_score(self) -> float:
        """Average aggregate score across all successful benchmarks."""
        scores = [
            r.aggregate_score
            for r in self.results.values()
            if r.aggregate_score is not None
            and r.aggregate_score == r.aggregate_score  # not NaN
        ]
        return sum(scores) / len(scores) if scores else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "benchmarks_run": self.benchmarks_run,
            "results": {name: r.to_dict() for name, r in self.results.items()},
            "failures": self.failures,
            "total_time_seconds": self.total_time_seconds,
            "overall_score": self.overall_score(),
        }


class UnifiedBenchmarkRunner:
    """Run multiple benchmarks in sequence with caching and export.

    Usage:
        runner = UnifiedBenchmarkRunner(
            RunnerConfig(model_name="MyModel-7B")
        )
        runner.model_fn = my_generate_fn
        summaries = runner.run_benchmarks(["perplexity", "mmlu", "gsm8k"])
        runner.export_results(summaries)
    """

    def __init__(
        self,
        config: RunnerConfig | None = None,
        model_fn: Any = None,
    ):
        self.config = config or RunnerConfig()
        self.model_fn = model_fn
        self.cache = CacheManager(
            cache_dir=os.path.join(self.config.output_dir, ".cache")
        )
        os.makedirs(self.config.output_dir, exist_ok=True)

    def run_benchmarks(
        self,
        benchmark_names: list[str],
        benchmark_config: BenchmarkConfig | None = None,
    ) -> RunSummary:
        """Run multiple benchmarks sequentially.

        Args:
            benchmark_names: List of benchmark names to evaluate.
            benchmark_config: Shared config for all benchmarks.

        Returns:
            RunSummary with results and status.
        """
        bm_config = benchmark_config or BenchmarkConfig()
        bm_config.output_dir = self.config.output_dir

        summary = RunSummary(
            model_name=self.config.model_name,
            timestamp=datetime.now().isoformat(),
            benchmarks_run=benchmark_names,
        )

        total_start = time.perf_counter()

        for name in benchmark_names:
            if self.config.verbose:
                print(f"\n{'=' * 60}")
                print(f"  Benchmark: {name}")
                print(f"{'=' * 60}")

            # Check cache
            if self.config.cache_enabled:
                cached = self.cache.get(name, bm_config, self.config.model_name)
                if cached is not None:
                    summary.results[name] = cached
                    if self.config.verbose:
                        print(f"  [CACHE HIT] Using cached result for '{name}'")
                    continue

            # Run benchmark
            try:
                benchmark = BenchmarkRegistry.create(name, config=bm_config)
                result = benchmark.run(model_fn=self.model_fn)

                summary.results[name] = result

                # Cache it
                if self.config.cache_enabled:
                    self.cache.put(result, name, bm_config, self.config.model_name)

            except KeyError as e:
                error_msg = f"Benchmark not found: {name}"
                if self.config.verbose:
                    print(f"  [ERROR] {error_msg}")
                summary.failures[name] = error_msg
                if self.config.fail_fast:
                    raise
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                if self.config.verbose:
                    print(f"  [ERROR] {error_msg}")
                summary.failures[name] = error_msg
                if self.config.fail_fast:
                    raise

        summary.total_time_seconds = time.perf_counter() - total_start

        if self.config.verbose:
            self._print_summary(summary)

        return summary

    def run_suite(
        self,
        suite_name: str = "standard",
        benchmark_config: BenchmarkConfig | None = None,
    ) -> RunSummary:
        """Run a pre-defined benchmark suite.

        Args:
            suite_name: One of "standard", "code", "math", "full".
            benchmark_config: Shared benchmark config.

        Returns:
            RunSummary.
        """
        suites = {
            "standard": STANDARD_SUITE,
            "code": CODE_SUITE,
            "math": MATH_SUITE,
            "full": FULL_SUITE,
        }

        if suite_name not in suites:
            raise ValueError(
                f"Unknown suite '{suite_name}'. Available: {list(suites.keys())}"
            )

        return self.run_benchmarks(suites[suite_name], benchmark_config)

    def compare_models(
        self,
        summaries: list[RunSummary],
    ) -> dict[str, Any]:
        """Compare results across multiple model runs.

        Args:
            summaries: List of RunSummary objects from different models.

        Returns:
            Comparison table with per-benchmark deltas.
        """
        if not summaries:
            return {}

        # Gather all benchmark names
        all_benchmarks: list[str] = []
        for s in summaries:
            for name in s.results:
                if name not in all_benchmarks:
                    all_benchmarks.append(name)

        comparison: dict[str, dict[str, float | None]] = {}

        for bm_name in all_benchmarks:
            row: dict[str, float | None] = {}
            for s in summaries:
                if bm_name in s.results:
                    row[s.model_name] = s.results[bm_name].aggregate_score
                else:
                    row[s.model_name] = None
            comparison[bm_name] = row

        return comparison

    def export_results(
        self,
        summary: RunSummary,
        formats: list[str] | None = None,
    ) -> dict[str, str]:
        """Export run results to specified formats.

        Args:
            summary: RunSummary to export.
            formats: List of "json", "csv", "md". Defaults to config.

        Returns:
            Dict mapping format to output file path.
        """
        fmts = formats or self.config.export_formats
        output_paths: dict[str, str] = {}

        safe_name = summary.model_name.replace("/", "_").replace(" ", "_")
        timestamp = summary.timestamp.replace(":", "-")[:19]
        base_name = f"{safe_name}_{timestamp}"

        os.makedirs(self.config.output_dir, exist_ok=True)

        if "json" in fmts:
            path = os.path.join(self.config.output_dir, f"{base_name}.json")
            with open(path, "w") as f:
                json.dump(summary.to_dict(), f, indent=2, default=str)
            output_paths["json"] = path

        if "csv" in fmts:
            path = os.path.join(self.config.output_dir, f"{base_name}.csv")
            self._export_csv(summary, path)
            output_paths["csv"] = path

        if "md" in fmts:
            path = os.path.join(self.config.output_dir, f"{base_name}.md")
            self._export_markdown(summary, path)
            output_paths["md"] = path

        return output_paths

    def _export_csv(self, summary: RunSummary, path: str) -> None:
        """Export results to CSV format."""
        rows: list[dict[str, str]] = []

        for bm_name, result in summary.results.items():
            for task in result.task_results:
                row = {
                    "model": summary.model_name,
                    "benchmark": bm_name,
                    "timestamp": summary.timestamp,
                }
                # Flatten task result dict
                for k, v in task.items():
                    if isinstance(v, (int, float, str, bool)):
                        row[str(k)] = str(v)
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            row[f"{k}_{sub_k}"] = str(sub_v)
                rows.append(row)

        if not rows:
            return

        # Collect all column names
        columns = sorted(set().union(*(r.keys() for r in rows)))
        # Ensure model/benchmark come first
        priority = ["model", "benchmark", "timestamp"]
        columns = [c for c in priority if c in columns] + [
            c for c in columns if c not in priority
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _export_markdown(self, summary: RunSummary, path: str) -> None:
        """Export results to Markdown table format."""
        lines: list[str] = []

        lines.append(f"# Benchmark Results: {summary.model_name}")
        lines.append(f"")
        lines.append(f"**Timestamp:** {summary.timestamp}")
        lines.append(f"**Benchmarks run:** {', '.join(summary.benchmarks_run)}")
        lines.append(f"**Overall score:** {summary.overall_score():.4f}")
        lines.append(f"**Total time:** {summary.total_time_seconds:.1f}s")
        lines.append(f"")

        if summary.failures:
            lines.append(f"## Failures")
            for name, error in summary.failures.items():
                lines.append(f"- **{name}**: {error}")
            lines.append(f"")

        # Per-benchmark tables
        for bm_name, result in summary.results.items():
            lines.append(f"## {bm_name}")
            lines.append(f"")
            lines.append(f"Aggregate score: **{result.aggregate_score:.4f}**")
            lines.append(f"")

            if result.task_results:
                # Build table
                lines.append(f"| Metric | Value |")
                lines.append(f"|--------|-------|")

                for task in result.task_results:
                    for k, v in task.items():
                        if isinstance(v, (int, float)):
                            lines.append(f"| {k} | {v:.4f} |")
                        elif isinstance(v, str):
                            lines.append(f"| {k} | {v} |")
                        elif isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                if isinstance(sub_v, (int, float)):
                                    lines.append(f"| {k}.{sub_k} | {sub_v:.4f} |")

                lines.append(f"")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _print_summary(self, summary: RunSummary) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'=' * 70}")
        print(f"  Run Summary: {summary.model_name}")
        print(f"{'=' * 70}")
        print(f"  Timestamp:      {summary.timestamp}")
        print(f"  Total time:     {summary.total_time_seconds:.1f}s")
        print(f"  Benchmarks run: {len(summary.benchmarks_run)}")
        print(f"  Successful:     {len(summary.results)}")
        print(f"  Failed:         {len(summary.failures)}")
        print(f"  Overall score:  {summary.overall_score():.4f}")
        print(f"")

        for name, result in summary.results.items():
            status_icon = "✓"
            print(
                f"  {status_icon} {name:<20}  aggregate = {result.aggregate_score:.4f}"
            )

        for name, error in summary.failures.items():
            print(f"  ✗ {name:<20}  {error}")


# =========================================================================
# Convenience Functions
# =========================================================================


def quick_eval(
    model_fn: Callable[[str], str],
    benchmarks: list[str] | None = None,
    model_name: str = "model",
    n_shot: int = 0,
    max_samples: int | None = None,
    seed: int = 42,
    output_dir: str = "benchmark_results",
) -> RunSummary:
    """Quick evaluation function for simple use cases.

    Args:
        model_fn: Function that takes a prompt and returns a completion.
        benchmarks: Benchmark names to run (default: standard suite).
        model_name: Name of the model for display.
        n_shot: Number of few-shot examples.
        max_samples: Max evaluation samples per benchmark.
        seed: Random seed.
        output_dir: Output directory for results.

    Returns:
        RunSummary with all benchmark results.
    """
    runner_config = RunnerConfig(
        model_name=model_name,
        output_dir=output_dir,
    )
    bm_config = BenchmarkConfig(
        n_shot=n_shot,
        max_samples=max_samples,
        seed=seed,
        output_dir=output_dir,
    )
    benchmark_names = benchmarks or STANDARD_SUITE

    runner = UnifiedBenchmarkRunner(
        config=runner_config,
        model_fn=model_fn,
    )
    return runner.run_benchmarks(benchmark_names, bm_config)


# =========================================================================
# Demo
# =========================================================================


def demo_runner() -> None:
    """Demonstrate the unified benchmark runner."""
    print("=" * 70)
    print("Unified Benchmark Runner Demo")
    print("=" * 70)

    config = RunnerConfig(
        model_name="DemoModel-7B",
        output_dir="/tmp/benchmark_results_demo",
        cache_enabled=False,
    )

    runner = UnifiedBenchmarkRunner(config=config)

    # List available benchmarks
    BenchmarkRegistry.discover()
    available = BenchmarkRegistry.list_names()
    print(f"\n  Available benchmarks: {available}")

    # Run a single benchmark with stub model
    bm_config = BenchmarkConfig(
        n_shot=2,
        max_samples=5,
        seed=42,
    )

    # Run MMLU as a quick demo (doesn't need data files for the stub)
    print(f"\n  Running 'mmlu' with stub model...")
    try:
        summary = runner.run_benchmarks(["mmlu"], benchmark_config=bm_config)
    except Exception as e:
        print(f"  (Expected demo error: {e})")
        # Fallback: demo without data
        return

    # Export results
    paths = runner.export_results(summary, formats=["json", "md"])
    print(f"\n  Exported to:")
    for fmt, path in paths.items():
        print(f"    {fmt}: {path}")

    # Print result contents
    if paths:
        json_path = paths.get("json", "")
        if json_path and os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            print(f"\n  JSON output preview:")
            print(f"    model: {data.get('model_name')}")
            print(f"    overall_score: {data.get('overall_score', 'N/A')}")
            results = data.get("results", {})
            for name, r in results.items():
                print(f"    {name}: {r.get('aggregate_score', 'N/A')}")

    print(f"\n  Cache entries: {runner.cache.list_cached()}")


def main() -> None:
    demo_runner()


if __name__ == "__main__":
    main()
