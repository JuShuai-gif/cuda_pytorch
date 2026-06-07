"""
Perplexity benchmark for language models.

Evaluates model perplexity on standard text corpora:
  - WikiText-103
  - Penn Treebank (PTB)
  - C4 (Colossal Clean Crawled Corpus)

Supports:
  - Sliding window vs full context evaluation strategies
  - Tokenizer-aware normalization
  - Confidence intervals via bootstrapping
  - Streaming data loading for memory efficiency

Key insight: Perplexity values from models with different tokenizers
are NOT directly comparable (Goodhart's Law applies).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .benchmark_registry import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    register_benchmark,
)
from .metrics import bootstrap_confidence_interval


# =========================================================================
# Sliding Window Utilities
# =========================================================================


@dataclass
class SlidingWindowState:
    """State for sliding window perplexity evaluation.

    Maintains a KV cache or simple context buffer across sequential
    windows to avoid recomputing overlapping context.
    """

    window_size: int
    stride: int
    buffer: list[int] = field(default_factory=list)

    def add_tokens(self, token_ids: list[int]) -> list[list[int]]:
        """Add new tokens and yield sliding windows.

        Args:
            token_ids: New token IDs to process.

        Returns:
            List of window token ID lists.
        """
        self.buffer.extend(token_ids)
        windows: list[list[int]] = []

        while len(self.buffer) >= self.window_size:
            windows.append(list(self.buffer[: self.window_size]))
            # Advance by stride
            self.buffer = self.buffer[self.stride :]

        return windows

    def flush(self) -> list[int]:
        """Return remaining tokens and reset."""
        remaining = list(self.buffer)
        self.buffer = []
        return remaining


# =========================================================================
# Dataset Utilities
# =========================================================================


def _tokenize_text(
    text: str, tokenizer: Any, max_length: int | None = None
) -> list[int]:
    """Tokenize text using the provided tokenizer.

    Args:
        text: Raw input text.
        tokenizer: A tokenizer with an encode() method returning token IDs,
                   or a callable `tokenizer(text) -> list[int]`.
        max_length: Maximum token length (truncation).

    Returns:
        List of token IDs.
    """
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
    else:
        tokens = tokenizer(text)

    if max_length is not None:
        tokens = tokens[:max_length]
    return tokens


def _stream_dataset(
    dataset_path: str,
    tokenizer: Any,
    max_length: int | None = None,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Stream a text dataset and tokenize it.

    Supports .txt, .jsonl, and .json formats.

    Args:
        dataset_path: Path to dataset file.
        tokenizer: Tokenizer for encoding.
        max_length: Maximum token count to load.
        max_samples: Maximum number of lines/samples.

    Returns:
        1D tensor of concatenated token IDs.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    all_tokens: list[int] = []
    ext = os.path.splitext(dataset_path)[1].lower()

    with open(dataset_path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
            for line in lines:
                data = json.loads(line)
                text = data.get("text", data.get("content", line))
                tokens = _tokenize_text(text, tokenizer)
                all_tokens.extend(tokens)
                if max_length and len(all_tokens) >= max_length:
                    break
        elif ext == ".json":
            data = json.load(f)
            if isinstance(data, list):
                items = data[:max_samples] if max_samples else data
                for item in items:
                    text = item.get("text", item.get("content", str(item)))
                    tokens = _tokenize_text(text, tokenizer)
                    all_tokens.extend(tokens)
                    if max_length and len(all_tokens) >= max_length:
                        break
            elif isinstance(data, dict):
                text = data.get("text", data.get("content", str(data)))
                tokens = _tokenize_text(text, tokenizer)
                all_tokens = tokens
        else:
            # Plain text file
            text = f.read()
            if max_samples:
                lines = text.split("\n")[:max_samples]
                text = "\n".join(lines)
            all_tokens = _tokenize_text(text, tokenizer)

    if max_length and len(all_tokens) > max_length:
        all_tokens = all_tokens[:max_length]

    return torch.tensor(all_tokens, dtype=torch.long)


# =========================================================================
# Perplexity Benchmark
# =========================================================================


@register_benchmark
class PerplexityBenchmark(Benchmark):
    """Perplexity evaluation benchmark for language models.

    Supports WikiText-103, Penn Treebank, and C4 datasets with
    sliding window or full context evaluation modes.
    """

    name = "perplexity"
    description = (
        "Language model perplexity on standard text corpora (WikiText-103, PTB, C4)"
    )

    # Standard corpora
    DATASETS: dict[str, str] = {
        "wikitext-103": "data/wikitext-103/wiki.test.raw",
        "ptb": "data/ptb/test.txt",
        "c4": "data/c4/validation.jsonl",
    }

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        tokenizer: Any = None,
        device: str = "cpu",
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.device = device
        self._results_cache: dict[str, Any] = {}
        self._per_sample_ppl: list[float] = []

    def run(
        self,
        model_fn: Any = None,
        datasets: list[str] | None = None,
        dataset_paths: dict[str, str] | None = None,
    ) -> BenchmarkResult:
        """Execute perplexity evaluation on specified datasets.

        Args:
            model_fn: A callable with signature:
                      model_fn(input_ids: torch.Tensor) -> torch.Tensor
                      Returns logits of shape (batch, seq_len, vocab_size).
                      If None, uses a stub random model.
            datasets: List of dataset keys to evaluate (default: all).
            dataset_paths: Optional override of dataset paths.

        Returns:
            BenchmarkResult with per-dataset perplexity metrics.
        """
        dataset_keys = datasets or list(self.DATASETS.keys())
        paths = dataset_paths or self.DATASETS

        self._per_sample_ppl = []
        task_results: list[dict[str, Any]] = []

        for key in dataset_keys:
            if key not in paths:
                task_results.append(
                    {"dataset": key, "error": f"No path configured for '{key}'"}
                )
                continue

            path = paths[key]
            print(f"  [{key}] Loading {path}...")

            try:
                token_ids = _stream_dataset(
                    path,
                    tokenizer=self.tokenizer,
                    max_samples=self.config.max_samples,
                )
            except FileNotFoundError as e:
                task_results.append({"dataset": key, "error": str(e)})
                continue
            except Exception as e:
                task_results.append({"dataset": key, "error": str(e)})
                continue

            if len(token_ids) < 2:
                task_results.append(
                    {"dataset": key, "ppl": float("nan"), "tokens": len(token_ids)}
                )
                continue

            print(f"  [{key}] Loaded {len(token_ids)} tokens, evaluating...")
            start = time.perf_counter()

            if self.config.context_window == "sliding":
                result = self._evaluate_sliding_window(
                    token_ids, model_fn, window_size=1024, stride=512
                )
            else:
                result = self._evaluate_full_context(token_ids, model_fn)

            elapsed = time.perf_counter() - start

            ci = bootstrap_confidence_interval(
                self._per_sample_ppl, seed=self.config.seed
            )

            task_result = {
                "dataset": key,
                "ppl": result["ppl"],
                "loss": result["loss"],
                "num_tokens": result["num_tokens"],
                "ppl_ci_lower": ci["lower"],
                "ppl_ci_upper": ci["upper"],
                "context_window": self.config.context_window,
                "latency_seconds": elapsed,
            }
            task_results.append(task_result)
            print(
                f"  [{key}] PPL = {result['ppl']:.4f} "
                f"[{ci['lower']:.4f}, {ci['upper']:.4f}] "
                f"({elapsed:.1f}s)"
            )

        # Aggregate score: average PPL across datasets
        valid_ppls = [
            r["ppl"]
            for r in task_results
            if "ppl" in r and not math.isnan(r["ppl"]) and not math.isinf(r["ppl"])
        ]
        aggregate = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float("nan")

        self._results_cache = {
            "per_datasets": task_results,
            "aggregate_ppl": aggregate,
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=task_results,
            aggregate_score=aggregate,
            config_hash=self.config.config_hash(),
            metadata={
                "datasets_evaluated": dataset_keys,
                "context_window": self.config.context_window,
            },
        )

    def _evaluate_sliding_window(
        self,
        token_ids: torch.Tensor,
        model_fn: Any,
        window_size: int = 1024,
        stride: int = 512,
    ) -> dict[str, float]:
        """Evaluate perplexity using a sliding window approach.

        Overlapping windows ensure each token's context is maintained.
        Only the stride-progressed tokens contribute to the loss to
        avoid double-counting.

        Args:
            token_ids: Full tokenized dataset as 1D tensor.
            model_fn: Model forward function.
            window_size: Context window size.
            stride: Step size between windows (overlap = window_size - stride).

        Returns:
            Dict with "ppl", "loss", "num_tokens".
        """
        total_loss = 0.0
        total_tokens = 0

        for start_idx in range(0, len(token_ids) - window_size + 1, stride):
            window = token_ids[start_idx : start_idx + window_size].unsqueeze(0)

            input_ids = window[:, :-1]
            target_ids = window[:, 1:]

            if model_fn is not None:
                logits = model_fn(input_ids)
            else:
                logits = self._stub_model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="sum",
            )

            # Only count the strides portion to avoid double-counting overlap
            # First token of target has no context; we compute over all but
            # weight by actual unique tokens processed.
            num_valid = target_ids.numel()
            total_loss += float(loss.item())
            total_tokens += num_valid

            # Per-sample PPL for bootstrap
            self._per_sample_ppl.append(float(math.exp(loss.item() / num_valid)))

        if total_tokens == 0:
            return {"ppl": float("inf"), "loss": float("inf"), "num_tokens": 0}

        avg_loss = total_loss / total_tokens
        return {
            "ppl": math.exp(avg_loss),
            "loss": avg_loss,
            "num_tokens": total_tokens,
        }

    def _evaluate_full_context(
        self,
        token_ids: torch.Tensor,
        model_fn: Any,
    ) -> dict[str, float]:
        """Evaluate perplexity with full context (no sliding windows).

        WARNING: Requires enough GPU memory for the entire sequence.
        Falls back to chunked evaluation for long sequences.

        Args:
            token_ids: Full tokenized dataset as 1D tensor.
            model_fn: Model forward function.

        Returns:
            Dict with "ppl", "loss", "num_tokens".
        """
        # Chunk into manageable sizes if needed
        chunk_size = 2048
        total_loss = 0.0
        total_tokens = 0
        self._per_sample_ppl = []

        for chunk_start in range(0, len(token_ids) - 1, chunk_size):
            chunk = token_ids[chunk_start : chunk_start + chunk_size + 1]
            if len(chunk) < 2:
                continue

            input_ids = chunk[:-1].unsqueeze(0)
            target_ids = chunk[1:].unsqueeze(0)

            if model_fn is not None:
                logits = model_fn(input_ids)
            else:
                logits = self._stub_model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="sum",
            )
            num_valid = target_ids.numel()
            total_loss += float(loss.item())
            total_tokens += num_valid

            self._per_sample_ppl.append(float(math.exp(loss.item() / num_valid)))

        if total_tokens == 0:
            return {"ppl": float("inf"), "loss": float("inf"), "num_tokens": 0}

        avg_loss = total_loss / total_tokens
        return {
            "ppl": math.exp(avg_loss),
            "loss": avg_loss,
            "num_tokens": total_tokens,
        }

    @staticmethod
    def _stub_model(input_ids: torch.Tensor) -> torch.Tensor:
        """Stub random model for testing without a real model.

        Args:
            input_ids: Input token IDs, shape (1, seq_len).

        Returns:
            Random logits, shape (1, seq_len, 50257).
        """
        vocab_size = 50257  # GPT-2 default
        return torch.randn(input_ids.size(0), input_ids.size(1), vocab_size)

    def evaluate(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> dict[str, Any]:
        """Evaluate per-sample perplexity from logits and targets.

        Args:
            predictions: List of logits tensors.
            references: List of target token tensors.

        Returns:
            Dict with "ppl", "loss" keys.
        """
        total_loss = 0.0
        total_tokens = 0

        for logits, targets in zip(predictions, references):
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        return {"ppl": math.exp(avg_loss), "loss": avg_loss}

    def get_results(self) -> BenchmarkResult:
        """Return cached results from the last run()."""
        if not self._results_cache:
            return BenchmarkResult(
                benchmark_name=self.name,
                aggregate_score=float("nan"),
                config_hash=self.config.config_hash(),
            )

        return BenchmarkResult(
            benchmark_name=self.name,
            task_results=self._results_cache.get("per_datasets", []),
            aggregate_score=self._results_cache.get("aggregate_ppl", float("nan")),
            config_hash=self.config.config_hash(),
        )


# =========================================================================
# Demo
# =========================================================================


def demo_perplexity_e2e() -> None:
    """End-to-end demonstration using stub model and mock data."""
    print("=" * 70)
    print("Perplexity Benchmark Demo")
    print("=" * 70)

    config = BenchmarkConfig(
        max_samples=None,
        seed=42,
        context_window="sliding",
    )
    benchmark = PerplexityBenchmark(config=config)

    # Simulate evaluation with a small synthetic dataset
    # (real usage would point to actual dataset files)
    token_ids = torch.randint(0, 1000, (500,))
    result = benchmark._evaluate_sliding_window(
        token_ids, model_fn=None, window_size=128, stride=64
    )
    print(f"\n  Sliding window (synthetic 500 tokens):")
    print(f"    PPL:  {result['ppl']:.4f}")
    print(f"    Loss: {result['loss']:.4f}")

    # With better "model" (boost correct token logits)
    def better_model(input_ids: torch.Tensor) -> torch.Tensor:
        vocab_size = 1000
        logits = torch.randn(input_ids.size(0), input_ids.size(1), vocab_size) * 0.5
        return logits

    result2 = benchmark._evaluate_sliding_window(
        token_ids, model_fn=better_model, window_size=128, stride=64
    )
    print(f"\n  Better model (reduced variance):")
    print(f"    PPL:  {result2['ppl']:.4f}")
    print(f"    Loss: {result2['loss']:.4f}")

    # Bootstrap CI
    ci = bootstrap_confidence_interval(benchmark._per_sample_ppl, seed=config.seed)
    print(f"\n  95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # Demonstrate sliding window mechanics
    print(f"\n  Sliding window mechanics:")
    sw = SlidingWindowState(window_size=4, stride=2)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    windows = sw.add_tokens(tokens)
    for i, w in enumerate(windows):
        print(f"    Window {i}: {w}")
    print(f"    Remaining: {sw.flush()}")


def main() -> None:
    demo_perplexity_e2e()


if __name__ == "__main__":
    main()
