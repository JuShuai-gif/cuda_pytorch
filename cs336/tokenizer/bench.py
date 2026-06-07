"""
Tokenizer benchmarks and profiling.

Measures throughput, compression ratio, and performs ablation studies
comparing vocabulary size against compression efficiency.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cs336.tokenizer.bpe import BPETokenizer, NUM_SPECIAL, TokenizerStats


# ---------------------------------------------------------------------------
# Benchmark data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str = ""
    text_type: str = ""
    vocab_size: int = 0
    num_texts: int = 0
    total_chars: int = 0
    total_tokens: int = 0
    compression_ratio: float = 0.0
    encode_time_s: float = 0.0
    tokens_per_second: float = 0.0
    mb_per_second: float = 0.0
    num_threads: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic text generators for testing
# ---------------------------------------------------------------------------


def generate_random_text(num_chars: int, alphabet: str | None = None) -> str:
    """Generate random text for benchmarking.

    Args:
        num_chars: Number of characters to generate.
        alphabet: Optional alphabet string. Defaults to printable ASCII.

    Returns:
        Random text string.
    """
    if alphabet is None:
        alphabet = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
            "0123456789.,!?;:'\"()\n\t"
        )
    return "".join(random.choice(alphabet) for _ in range(num_chars))


def generate_repetitive_text(
    num_chars: int, pattern: str = "the quick brown fox jumps over the lazy dog "
) -> str:
    """Generate repetitive text for benchmarking.

    Args:
        num_chars: Target character count.
        pattern: Repetitive pattern string.

    Returns:
        Repetitive text string.
    """
    reps = math.ceil(num_chars / len(pattern))
    return (pattern * reps)[:num_chars]


def generate_natural_text(num_chars: int) -> str:
    """Generate semi-natural text using repeated English templates.

    Args:
        num_chars: Target character count.

    Returns:
        Semi-natural English text.
    """
    templates = [
        "The {adj} {noun} {verb} over the {adj} {noun}. ",
        "When {pronoun} looked at the {noun}, it was very {adj}. ",
        "A {adj} {noun} can {verb} through the {noun} quickly. ",
        "After the {noun} had {verb}, the {adj} {noun} appeared. ",
        "It is a well-known fact that a {adj} {noun} {verb} {adv}. ",
    ]
    adjectives = [
        "quick",
        "lazy",
        "bright",
        "dark",
        "small",
        "large",
        "happy",
        "sad",
        "red",
        "blue",
        "ancient",
        "modern",
        "clever",
        "brave",
    ]
    nouns = [
        "fox",
        "dog",
        "cat",
        "bird",
        "tree",
        "house",
        "car",
        "book",
        "computer",
        "sun",
        "moon",
        "star",
        "river",
        "mountain",
        "city",
    ]
    verbs = ["jumps", "runs", "flies", "swims", "walks", "drives", "reads"]
    adverbs = ["quickly", "slowly", "happily", "sadly", "carefully"]
    pronouns = ["he", "she", "it", "they"]

    result = ""
    while len(result) < num_chars:
        template = random.choice(templates)
        text = template.format(
            adj=random.choice(adjectives),
            noun=random.choice(nouns),
            verb=random.choice(verbs),
            pronoun=random.choice(pronouns),
            adv=random.choice(adverbs),
        )
        result += text
    return result[:num_chars]


# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------


def benchmark_throughput(
    tokenizer: BPETokenizer,
    texts: list[str],
    num_warmups: int = 3,
    num_runs: int = 10,
    num_threads: int = 1,
) -> BenchmarkResult:
    """Benchmark encoding throughput.

    Args:
        tokenizer: Trained BPETokenizer instance.
        texts: Texts to encode for benchmarking.
        num_warmups: Number of warmup iterations.
        num_runs: Number of measurement iterations.
        num_threads: Number of threads for batch encoding.

    Returns:
        BenchmarkResult with throughput metrics.
    """
    # Warmup
    for _ in range(num_warmups):
        if num_threads <= 1:
            for t in texts:
                tokenizer.encode(t, add_special_tokens=False)
        else:
            tokenizer.encode_batch(
                texts, add_special_tokens=False, num_threads=num_threads
            )

    # Measure
    total_chars = sum(len(t) for t in texts)
    total_tokens = 0
    total_time = 0.0

    for _ in range(num_runs):
        start = time.perf_counter()
        if num_threads <= 1:
            results = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        else:
            results = tokenizer.encode_batch(
                texts, add_special_tokens=False, num_threads=num_threads
            )
        elapsed = time.perf_counter() - start

        run_tokens = sum(len(r) for r in results)
        total_tokens += run_tokens
        total_time += elapsed

    avg_time = total_time / num_runs
    avg_tokens = total_tokens / num_runs
    avg_chars = total_chars

    return BenchmarkResult(
        name="throughput",
        total_chars=avg_chars,
        total_tokens=avg_tokens,
        encode_time_s=avg_time,
        tokens_per_second=avg_tokens / max(avg_time, 0.0001),
        mb_per_second=(avg_chars / (1024 * 1024)) / max(avg_time, 0.0001),
        num_threads=num_threads,
        compression_ratio=avg_chars / max(avg_tokens, 1),
        vocab_size=tokenizer.vocab_size,
        num_texts=len(texts),
    )


def benchmark_compression(
    tokenizer: BPETokenizer,
    texts: dict[str, list[str]],
) -> list[BenchmarkResult]:
    """Benchmark compression ratio across different text types.

    Args:
        tokenizer: Trained BPETokenizer instance.
        texts: Dict mapping text type name to list of texts.

    Returns:
        List of BenchmarkResult, one per text type.
    """
    results = []
    for text_type, text_list in texts.items():
        total_chars = sum(len(t) for t in text_list)
        total_tokens = 0
        for t in text_list:
            encoded = tokenizer.encode(t, add_special_tokens=False)
            total_tokens += len(encoded)

        results.append(
            BenchmarkResult(
                name="compression",
                text_type=text_type,
                total_chars=total_chars,
                total_tokens=total_tokens,
                compression_ratio=total_chars / max(total_tokens, 1),
                vocab_size=tokenizer.vocab_size,
                num_texts=len(text_list),
            )
        )

    return results


def benchmark_vocab_ablation(
    texts: list[str],
    vocab_sizes: list[int],
    min_frequency: int = 2,
    num_threads: int = 1,
) -> list[BenchmarkResult]:
    """Benchmark compression ratio vs vocabulary size.

    Trains tokenizers at different vocabulary sizes on the same text
    and measures compression effectiveness.

    Args:
        texts: Training and evaluation texts.
        vocab_sizes: List of vocabulary sizes to test.
        min_frequency: Minimum pair frequency during training.
        num_threads: Number of threads.

    Returns:
        List of BenchmarkResult, one per vocab size.
    """
    results = []
    total_chars = sum(len(t) for t in texts)

    for v_size in vocab_sizes:
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=v_size, min_frequency=min_frequency)

        total_tokens = 0
        for t in texts:
            encoded = tokenizer.encode(t, add_special_tokens=False)
            total_tokens += len(encoded)

        results.append(
            BenchmarkResult(
                name="vocab_ablation",
                vocab_size=tokenizer.vocab_size,
                total_chars=total_chars,
                total_tokens=total_tokens,
                compression_ratio=total_chars / max(total_tokens, 1),
                num_threads=num_threads,
                extra={"target_vocab_size": v_size},
            )
        )

    return results


def benchmark_compare_tokenizers(
    tokenizers: dict[str, BPETokenizer],
    texts: list[str],
) -> list[BenchmarkResult]:
    """Compare multiple tokenizers on the same texts.

    Args:
        tokenizers: Dict mapping tokenizer names to BPETokenizer instances.
        texts: Evaluation texts.

    Returns:
        List of BenchmarkResult, one per tokenizer.
    """
    results = []
    total_chars = sum(len(t) for t in texts)

    for name, tokenizer in tokenizers.items():
        total_tokens = 0
        start = time.perf_counter()
        for t in texts:
            encoded = tokenizer.encode(t, add_special_tokens=False)
            total_tokens += len(encoded)
        elapsed = time.perf_counter() - start

        results.append(
            BenchmarkResult(
                name="compare",
                text_type=name,
                vocab_size=tokenizer.vocab_size,
                total_chars=total_chars,
                total_tokens=total_tokens,
                compression_ratio=total_chars / max(total_tokens, 1),
                encode_time_s=elapsed,
                tokens_per_second=total_tokens / max(elapsed, 0.0001),
                mb_per_second=(total_chars / (1024 * 1024)) / max(elapsed, 0.0001),
                num_texts=len(texts),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def print_benchmark_report(results: list[BenchmarkResult]) -> None:
    """Print a formatted benchmark report.

    Args:
        results: List of benchmark results to display.
    """
    separator = "-" * 80
    print(separator)
    print(f"{'Tokenizer Benchmark Report':^80}")
    print(separator)
    print(
        f"{'Name':<16} {'Type':<14} {'Vocab':>6} {'Chars':>8} {'Tokens':>8} "
        f"{'Comp.R':>8} {'T/s':>10} {'MB/s':>10} {'Thr':>4}"
    )
    print(separator)

    for r in results:
        print(
            f"{r.name:<16} {r.text_type:<14} {r.vocab_size:>6,} "
            f"{r.total_chars:>8,} {r.total_tokens:>8,} "
            f"{r.compression_ratio:>8.2f} {r.tokens_per_second:>10.0f} "
            f"{r.mb_per_second:>10.2f} {r.num_threads:>4}"
        )

    print(separator)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Tokenizer Benchmarks ===\n")

    # Generate test texts (smaller set for demo speed)
    texts_natural = [generate_natural_text(500) for _ in range(10)]
    texts_repetitive = [generate_repetitive_text(500) for _ in range(10)]
    texts_random = [generate_random_text(500) for _ in range(10)]

    all_texts = texts_natural + texts_repetitive + texts_random
    train_texts = all_texts[:20]
    eval_texts = all_texts[20:]

    # Train a tokenizer
    print("Training tokenizer (vocab_size=1000)...")
    tokenizer = BPETokenizer()
    tokenizer.train(train_texts, vocab_size=1000, min_frequency=2, show_progress=False)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Num merges: {len(tokenizer.merges)}")

    # Benchmark throughput
    print("\n--- Throughput Benchmarks ---")
    results: list[BenchmarkResult] = []

    for threads in [1, 2]:
        r = benchmark_throughput(tokenizer, eval_texts, num_threads=threads)
        r.text_type = f"thr={threads}"
        results.append(r)

    # Benchmark compression
    print("\n--- Compression Benchmarks ---")
    text_types = {
        "natural": texts_natural,
        "repetitive": texts_repetitive,
        "random": texts_random,
    }
    comp_results = benchmark_compression(tokenizer, text_types)
    results.extend(comp_results)

    # Vocab size ablation (smaller for speed)
    print("\n--- Vocab Size Ablation ---")
    ablation_sizes = [300, 500, 800]
    ablation_results = benchmark_vocab_ablation(
        train_texts[:10],
        ablation_sizes,
    )
    results.extend(ablation_results)

    print_benchmark_report(results)
    print("\nDone!")
