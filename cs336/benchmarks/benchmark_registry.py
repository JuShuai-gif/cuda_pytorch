"""
Benchmark registry system providing the foundation for all evaluation benchmarks.

Provides:
  - Benchmark base class with a consistent run() / evaluate() / get_results() interface
  - Registry for registering and discovering benchmarks
  - BenchmarkConfig dataclass for controlling benchmark execution
  - Automatic benchmark discovery by scanning the package directory
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


# =========================================================================
# BenchmarkConfig
# =========================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        n_shot: Number of few-shot examples to provide in the prompt.
        max_samples: Maximum number of evaluation samples (None = all).
        seed: Random seed for reproducibility.
        batch_size: Batch size for model inference.
        temperature: Sampling temperature for generation tasks.
        max_tokens: Maximum number of tokens to generate.
        context_window: Sliding vs full context for perplexity evaluation.
        output_dir: Directory for caching and saving results.
        extra: Arbitrary extra configuration passed through to benchmarks.
    """

    n_shot: int = 0
    max_samples: int | None = None
    seed: int = 42
    batch_size: int = 8
    temperature: float = 0.0
    max_tokens: int = 256
    context_window: str = "sliding"
    output_dir: str = "benchmark_results"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for hashing and storage."""
        return {
            "n_shot": self.n_shot,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "output_dir": self.output_dir,
            "extra": self.extra,
        }

    def config_hash(self, length: int = 8) -> str:
        """Compute a deterministic hash of the config for cache keys."""
        raw = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:length]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkConfig:
        return cls(
            n_shot=d.get("n_shot", 0),
            max_samples=d.get("max_samples"),
            seed=d.get("seed", 42),
            batch_size=d.get("batch_size", 8),
            temperature=d.get("temperature", 0.0),
            max_tokens=d.get("max_tokens", 256),
            context_window=d.get("context_window", "sliding"),
            output_dir=d.get("output_dir", "benchmark_results"),
            extra=d.get("extra", {}),
        )


# =========================================================================
# BenchmarkResult
# =========================================================================


@dataclass
class BenchmarkResult:
    """Container for results from a single benchmark run.

    Attributes:
        benchmark_name: Name of the benchmark (e.g. "mmlu", "gsm8k").
        task_results: Per-task or per-subject result entries.
        aggregate_score: Overall score across all tasks.
        config_hash: Hash of the config used to produce these results.
        metadata: Arbitrary additional metadata.
    """

    benchmark_name: str
    task_results: list[dict[str, Any]] = field(default_factory=list)
    aggregate_score: float = 0.0
    config_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "aggregate_score": self.aggregate_score,
            "config_hash": self.config_hash,
            "task_results": self.task_results,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkResult:
        return cls(
            benchmark_name=d.get("benchmark_name", "unknown"),
            task_results=d.get("task_results", []),
            aggregate_score=d.get("aggregate_score", 0.0),
            config_hash=d.get("config_hash", ""),
            metadata=d.get("metadata", {}),
        )


# =========================================================================
# Benchmark base class
# =========================================================================


class Benchmark(ABC):
    """Abstract base class for all benchmarks.

    Each benchmark must implement:
      - name (class attribute): unique identifier string.
      - description (class attribute): human-readable description.
      - run(): execute the benchmark and return results.
      - evaluate(): compute metrics from model outputs.
      - get_results(): package results into BenchmarkResult.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()

    @abstractmethod
    def run(self, model_fn: Any = None) -> BenchmarkResult:
        """Execute the benchmark.

        Args:
            model_fn: Callable that takes a prompt and returns completion.
                      Signature: model_fn(prompt: str) -> str
                      If None, benchmarks may use a stub.

        Returns:
            BenchmarkResult with metrics.
        """
        ...

    @abstractmethod
    def evaluate(self, predictions: list[Any], references: list[Any]) -> dict[str, Any]:
        """Evaluate predictions against references.

        Args:
            predictions: Model outputs.
            references: Ground truth.

        Returns:
            Dictionary of metric name -> value.
        """
        ...

    @abstractmethod
    def get_results(self) -> BenchmarkResult:
        """Return accumulated results after run() has been called."""
        ...

    def cache_key(self) -> str:
        """Generate a cache key from benchmark name and config."""
        return f"{self.name}_{self.config.config_hash()}"


# =========================================================================
# Benchmark Registry
# =========================================================================


class BenchmarkRegistry:
    """Registry for discovering and retrieving benchmark classes.

    Benchmarks can be registered manually or discovered automatically
    by scanning the package directory for Benchmark subclasses.
    """

    _registry: ClassVar[dict[str, type[Benchmark]]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def register(cls, benchmark_cls: type[Benchmark]) -> type[Benchmark]:
        """Register a benchmark class.

        Can be used as a decorator or called directly.

        Args:
            benchmark_cls: A class inheriting from Benchmark.

        Returns:
            The registered class (for decorator usage).
        """
        name = benchmark_cls.name
        if not name:
            raise ValueError(
                f"Benchmark class {benchmark_cls.__name__} must define a 'name' class attribute"
            )
        # Idempotent: skip if already registered (handles python -m double-import)
        if name in cls._registry:
            return benchmark_cls
        cls._registry[name] = benchmark_cls
        return benchmark_cls

    @classmethod
    def get(cls, name: str) -> type[Benchmark]:
        """Retrieve a benchmark class by name.

        Args:
            name: The benchmark name (e.g. 'perplexity', 'mmlu').

        Returns:
            The Benchmark subclass.

        Raises:
            KeyError: If the benchmark name is not registered.
        """
        if not cls._initialized:
            cls.discover()
        if name not in cls._registry:
            raise KeyError(
                f"Benchmark '{name}' not found. Available: {cls.list_names()}"
            )
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> list[str]:
        """Return sorted list of all registered benchmark names."""
        if not cls._initialized:
            cls.discover()
        return sorted(cls._registry.keys())

    @classmethod
    def list_benchmarks(cls) -> dict[str, str]:
        """Return mapping of benchmark name -> description."""
        if not cls._initialized:
            cls.discover()
        return {name: bm_cls.description for name, bm_cls in cls._registry.items()}

    @classmethod
    def discover(cls) -> None:
        """Auto-discover benchmarks by scanning the package modules.

        Import every module in the package and look for Benchmark subclasses
        that have not yet been registered.
        """
        if cls._initialized:
            return

        # Dynamically import all modules in the current package
        try:
            import cs336.benchmarks as pkg
        except ImportError:
            cls._initialized = True
            return

        package_path = pkg.__path__
        for _, module_name, _ in pkgutil.iter_modules(package_path):
            if module_name.startswith("_"):
                continue
            try:
                importlib.import_module(f"cs336.benchmarks.{module_name}")
            except ImportError:
                pass

        # Scan for Benchmark subclasses in all imported modules
        for name, obj in inspect.getmembers(pkg):
            if not inspect.ismodule(obj):
                continue
            for _, member in inspect.getmembers(obj, inspect.isclass):
                if (
                    issubclass(member, Benchmark)
                    and member is not Benchmark
                    and member.name
                    and member.name not in cls._registry
                ):
                    cls._registry[member.name] = member

        cls._initialized = True

    @classmethod
    def create(
        cls,
        name: str,
        config: BenchmarkConfig | None = None,
    ) -> Benchmark:
        """Create a benchmark instance by name.

        Args:
            name: The benchmark name.
            config: Optional BenchmarkConfig.

        Returns:
            An instantiated Benchmark.
        """
        benchmark_cls = cls.get(name)
        return benchmark_cls(config=config)

    @classmethod
    def reset(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._registry.clear()
        cls._initialized = False


# =========================================================================
# Utility: register decorator
# =========================================================================


def register_benchmark(cls: type[Benchmark]) -> type[Benchmark]:
    """Decorator to register a benchmark class in the registry."""
    return BenchmarkRegistry.register(cls)


# =========================================================================
# Demo
# =========================================================================


def demo_registry() -> None:
    """Demonstrate the benchmark registry system."""
    print("=" * 70)
    print("Benchmark Registry Demo")
    print("=" * 70)

    # Discover benchmarks
    BenchmarkRegistry.discover()
    available = BenchmarkRegistry.list_names()

    print(f"\n  Registered benchmarks: {available}")
    print(f"  Descriptions:")
    for name, desc in BenchmarkRegistry.list_benchmarks().items():
        print(f"    - {name}: {desc}")

    # Config hashing
    config1 = BenchmarkConfig(n_shot=5, seed=42, max_samples=100)
    config2 = BenchmarkConfig(n_shot=5, seed=42, max_samples=100)
    config3 = BenchmarkConfig(n_shot=0, seed=123, max_samples=200)

    print(
        f"\n  Config hash (same config): {config1.config_hash()} == {config2.config_hash()}: "
        f"{config1.config_hash() == config2.config_hash()}"
    )
    print(
        f"  Config hash (different):   {config1.config_hash()} != {config3.config_hash()}: "
        f"{config1.config_hash() != config3.config_hash()}"
    )


def main() -> None:
    demo_registry()


if __name__ == "__main__":
    main()
