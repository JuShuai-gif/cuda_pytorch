"""
Vocabulary optimization tools.

Provides utilities for analyzing, extending, merging, and pruning
BPE vocabularies to improve token efficiency for domain-specific use.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cs336.tokenizer.bpe import BPETokenizer, NUM_SPECIAL, SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    """Result of vocabulary coverage analysis."""

    total_tokens: int = 0
    total_chars: int = 0
    oov_tokens: int = 0
    oov_chars: int = 0
    oov_rate: float = 0.0  # fraction of OOV tokens
    compression_ratio: float = 0.0
    token_frequencies: dict[int, int] = field(default_factory=dict)
    most_frequent_oov: list[tuple[str, int]] = field(default_factory=list)
    unused_tokens: list[int] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class MergeResult:
    """Result of merging two vocabularies."""

    merged_vocab_size: int = 0
    num_shared_tokens: int = 0
    num_added_from_a: int = 0
    num_added_from_b: int = 0
    num_conflicts: int = 0


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------


def analyze_coverage(
    tokenizer: BPETokenizer,
    texts: list[str],
    max_suggestions: int = 20,
) -> CoverageReport:
    """Analyze vocabulary coverage on a target corpus.

    Identifies out-of-vocabulary patterns and suggests new tokens
    that would improve coverage.

    Args:
        tokenizer: Trained BPETokenizer instance.
        texts: Target corpus texts to analyze.
        max_suggestions: Maximum number of token suggestions to generate.

    Returns:
        CoverageReport with detailed analysis.
    """
    token_freqs: dict[int, int] = Counter()
    oov_subword_counts: Counter[str] = Counter()
    total_tokens = 0
    total_chars = 0
    oov_tokens = 0

    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(encoded)
        total_chars += len(text)

        for tid in encoded:
            token_freqs[tid] = token_freqs.get(tid, 0) + 1

        # Track OOV occurrences from original byte perspective
        for i, tid in enumerate(encoded):
            token_str = tokenizer.id_to_token(tid)
            if token_str.startswith("<0x"):
                # Byte-level token: extract OOV subword patterns
                if i < len(encoded) - 1:
                    next_str = tokenizer.id_to_token(encoded[i + 1])
                    if not next_str.startswith("<0x"):
                        continue
                oov_tokens += 1
                # Collect the unicode character context
                try:
                    decoded = tokenizer.decode([tid], skip_special_tokens=True)
                    if decoded and decoded != token_str:
                        oov_subword_counts[decoded] += 1
                except Exception:
                    pass

    # Find unused tokens (tokens that appear 0 times in the corpus)
    all_used = set(token_freqs.keys())
    all_known = set(tokenizer.vocab.keys()) - set(tokenizer.special_tokens.values())
    unused = sorted(all_known - all_used)

    # Generate suggestions: find frequent pairs that should be merged
    pair_freqs: Counter[tuple[int, int]] = Counter()
    for text in texts[:1000]:  # Sample first 1000 texts for speed
        encoded = tokenizer.encode(text, add_special_tokens=False)
        for i in range(len(encoded) - 1):
            pair_freqs[(encoded[i], encoded[i + 1])] += 1

    # Filter pairs already in merges, suggest most frequent new pairs
    suggestions: list[str] = []
    existing_merges = set(tokenizer.merges.keys())
    for (a, b), freq in pair_freqs.most_common(max_suggestions):
        if (a, b) not in existing_merges:
            a_str = tokenizer.id_to_token(a)
            b_str = tokenizer.id_to_token(b)
            combined = a_str + b_str
            # Skip already-merged or single-byte patterns
            if len(combined) > 3 and not combined.startswith("<0x"):
                suggestions.append(
                    f"{a_str!r} + {b_str!r} -> {combined!r} (freq={freq})"
                )

    # Most frequent OOV subword patterns
    most_frequent_oov = [
        (char, count) for char, count in oov_subword_counts.most_common(20)
    ]

    oov_rate = oov_tokens / max(total_tokens, 1)

    return CoverageReport(
        total_tokens=total_tokens,
        total_chars=total_chars,
        oov_tokens=oov_tokens,
        oov_chars=sum(len(char) for char, _ in most_frequent_oov),
        oov_rate=oov_rate,
        compression_ratio=total_chars / max(total_tokens, 1),
        token_frequencies=token_freqs,
        most_frequent_oov=most_frequent_oov,
        unused_tokens=unused,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# Vocabulary merging
# ---------------------------------------------------------------------------


def merge_vocabularies(
    base_tokenizer: BPETokenizer,
    other_tokenizer: BPETokenizer,
    conflict_strategy: str = "keep_base",
) -> tuple[BPETokenizer, MergeResult]:
    """Merge two vocabularies into a new tokenizer.

    The base tokenizer's special tokens and byte-level entries are
    preserved. New tokens from other are appended (with re-indexing).

    Args:
        base_tokenizer: The primary tokenizer to extend.
        other_tokenizer: Tokenizer whose vocabulary to merge in.
        conflict_strategy: How to handle overlapping tokens:
            - "keep_base": Keep base tokenizer's ID (default).
            - "keep_other": Use other tokenizer's ID.
            - "skip": Don't include conflicting tokens from other.

    Returns:
        Tuple of (merged BPETokenizer, MergeResult).

    Raises:
        ValueError: If conflict_strategy is invalid.
    """
    if conflict_strategy not in ("keep_base", "keep_other", "skip"):
        raise ValueError(
            f"Invalid conflict_strategy: {conflict_strategy}. "
            f"Use 'keep_base', 'keep_other', or 'skip'."
        )

    merged = BPETokenizer()
    result = MergeResult()

    # Copy base vocabulary (special tokens + bytes + base merges)
    merged.vocab = dict(base_tokenizer.vocab)
    merged.merges = dict(base_tokenizer.merges)
    merged.special_tokens = dict(base_tokenizer.special_tokens)
    merged._vocab_rev = dict(base_tokenizer._vocab_rev)
    merged._merges_rev = dict(base_tokenizer._merges_rev)

    # Build a set of token bytes already in the base vocab
    base_bytes = set(base_tokenizer.vocab.values())
    base_pairs = set(base_tokenizer.merges.keys())

    # Find the next available ID
    next_id = (
        max(base_tokenizer.vocab.keys()) + 1
        if base_tokenizer.vocab
        else NUM_SPECIAL + 256
    )

    # Add new tokens from other
    shared = 0
    added = 0
    conflicts = 0

    for other_tid, other_bytes in sorted(
        other_tokenizer.vocab.items(), key=lambda x: x[0]
    ):
        # Skip special token range (0..NUM_SPECIAL-1) and bytes (NUM_SPECIAL..NUM_SPECIAL+255)
        if other_tid < NUM_SPECIAL + 256:
            shared += 1
            continue

        if other_bytes in base_bytes:
            if conflict_strategy == "keep_base":
                conflicts += 1
            elif conflict_strategy == "keep_other":
                # Update with other's mapping (potential conflict resolution)
                existing_id = merged._vocab_rev[other_bytes]
                if existing_id != other_tid:
                    conflicts += 1
            else:  # skip
                conflicts += 1
            continue

        # Add new token
        merged.vocab[next_id] = other_bytes
        merged._vocab_rev[other_bytes] = next_id
        next_id += 1
        added += 1

    # Add new merges from other (re-indexed to merged IDs)
    merge_next_id = next_id
    for (a, b), merge_tid in sorted(other_tokenizer.merges.items(), key=lambda x: x[1]):
        a_bytes = other_tokenizer.vocab.get(a)
        b_bytes = other_tokenizer.vocab.get(b)
        if a_bytes is None or b_bytes is None:
            continue

        a_new_id = merged._vocab_rev.get(a_bytes)
        b_new_id = merged._vocab_rev.get(b_bytes)
        if a_new_id is None or b_new_id is None:
            continue

        new_pair = (a_new_id, b_new_id)
        if new_pair in base_pairs:
            continue

        merged.merges[new_pair] = merge_next_id
        merged._merges_rev[merge_next_id] = new_pair
        merge_next_id += 1

    result.merged_vocab_size = len(merged.vocab)
    result.num_shared_tokens = shared
    result.num_added_from_a = len(base_tokenizer.vocab) - NUM_SPECIAL - 256
    result.num_added_from_b = added
    result.num_conflicts = conflicts

    return merged, result


# ---------------------------------------------------------------------------
# Vocabulary pruning
# ---------------------------------------------------------------------------


def prune_unused_tokens(
    tokenizer: BPETokenizer,
    texts: list[str],
    min_frequency: int = 1,
    preserve_special: bool = True,
    preserve_bytes: bool = True,
) -> tuple[BPETokenizer, list[int]]:
    """Prune tokens that appear below minimum frequency in a corpus.

    Creates a new tokenizer with only frequently-used tokens.

    Args:
        tokenizer: The original BPETokenizer.
        texts: Corpus to compute token frequencies from.
        min_frequency: Minimum occurrence count to keep a token.
        preserve_special: If True, always keep special tokens.
        preserve_bytes: If True, always keep byte-level tokens (0-255).

    Returns:
        Tuple of (pruned BPETokenizer, list of removed token IDs).
    """
    # Count token occurrences
    token_counts: Counter[int] = Counter()
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        token_counts.update(encoded)

    # Identify tokens to keep
    keep_ids: set[int] = set()
    removed: list[int] = []

    for tid in tokenizer.vocab:
        should_keep = False

        if preserve_special and tid in tokenizer.special_tokens.values():
            should_keep = True
        elif preserve_bytes and NUM_SPECIAL <= tid < NUM_SPECIAL + 256:
            should_keep = True
        elif token_counts.get(tid, 0) >= min_frequency:
            should_keep = True

        if should_keep:
            keep_ids.add(tid)
        else:
            removed.append(tid)

    # Create new tokenizer with only kept tokens
    pruned = BPETokenizer()
    pruned.special_tokens = dict(tokenizer.special_tokens)

    # Copy kept vocab entries
    pruned.vocab = {}
    # Copy merges that only reference kept IDs
    pruned.merges = {}

    # Build reverse mapping: old_id -> new_id
    id_map: dict[int, int] = {}
    new_id = 0

    # Assign new IDs (preserve order: special -> bytes -> rest)
    for tid in sorted(keep_ids):
        id_map[tid] = new_id
        pruned.vocab[new_id] = tokenizer.vocab[tid]
        new_id += 1

    # Remap merges
    for (a, b), merge_id in sorted(tokenizer.merges.items(), key=lambda x: x[1]):
        if a in id_map and b in id_map and merge_id in id_map:
            new_pair = (id_map[a], id_map[b])
            pruned.merges[new_pair] = id_map[merge_id]

    pruned._vocab_rev = {v: k for k, v in pruned.vocab.items()}
    pruned._merges_rev = {v: k for k, v in pruned.merges.items()}

    return pruned, removed


def add_domain_tokens(
    tokenizer: BPETokenizer,
    domain_texts: list[str],
    num_new_tokens: int = 1000,
    min_frequency: int = 5,
) -> BPETokenizer:
    """Add domain-specific tokens to an existing vocabulary.

    Trains additional BPE merges on domain text, then adds new tokens
    to the existing vocabulary.

    Args:
        tokenizer: Existing trained tokenizer.
        domain_texts: Domain-specific texts for new token training.
        num_new_tokens: Maximum number of new tokens to add.
        min_frequency: Minimum frequency for new merges.

    Returns:
        Extended BPETokenizer with domain-specific tokens.
    """
    # Start from the existing vocabulary state
    extended = BPETokenizer()

    # Copy existing vocab and merges
    extended.vocab = dict(tokenizer.vocab)
    extended.merges = dict(tokenizer.merges)
    extended.special_tokens = dict(tokenizer.special_tokens)
    extended._vocab_rev = dict(tokenizer._vocab_rev)
    extended._merges_rev = dict(tokenizer._merges_rev)

    # Train additional merges on domain texts
    # Convert domain texts to byte sequences using existing vocab
    byte_sequences: list[list[int]] = []
    for text in domain_texts:
        byte_sequences.append([NUM_SPECIAL + b for b in text.encode("utf-8")])

    existing_pairs = set(extended.merges.keys())
    next_id = max(extended.vocab.keys()) + 1 if extended.vocab else NUM_SPECIAL + 256

    added = 0
    for _ in range(num_new_tokens):
        # Count pair frequencies across domain sequences
        pair_counts: Counter[tuple[int, int]] = Counter()
        for seq in byte_sequences:
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += 1

        if not pair_counts:
            break

        # Find most frequent pair not already in merges
        best_pair = None
        best_freq = 0
        for pair, freq in pair_counts.items():
            if pair not in existing_pairs and freq > best_freq:
                best_pair = pair
                best_freq = freq

        if best_pair is None or best_freq < min_frequency:
            break

        # Record new merge
        new_token = extended.vocab[best_pair[0]] + extended.vocab[best_pair[1]]
        extended.vocab[next_id] = new_token
        extended._vocab_rev[new_token] = next_id
        extended.merges[best_pair] = next_id
        extended._merges_rev[next_id] = best_pair
        existing_pairs.add(best_pair)

        # Apply merge to byte sequences
        for seq in byte_sequences:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                    seq[i] = next_id
                    del seq[i + 1]
                else:
                    i += 1

        next_id += 1
        added += 1

    return extended


# ---------------------------------------------------------------------------
# Report utilities
# ---------------------------------------------------------------------------


def print_coverage_report(report: CoverageReport) -> None:
    """Print a formatted coverage analysis report.

    Args:
        report: CoverageReport from analyze_coverage().
    """
    print("=" * 60)
    print("VOCABULARY COVERAGE REPORT")
    print("=" * 60)
    print(f"  Total tokens encoded: {report.total_tokens:>10,}")
    print(f"  Total chars encoded:  {report.total_chars:>10,}")
    print(f"  OOV tokens:           {report.oov_tokens:>10,} ({report.oov_rate:.2%})")
    print(f"  Compression ratio:    {report.compression_ratio:>10.2f} chars/token")
    print(f"  Unused tokens:        {len(report.unused_tokens):>10,}")
    print()

    if report.most_frequent_oov:
        print("Most frequent OOV patterns:")
        for char, count in report.most_frequent_oov[:10]:
            print(f"  {char!r:<20} {count:>8,}")

    if report.suggestions:
        print("\nTop suggestions for new merges:")
        for s in report.suggestions[:10]:
            print(f"  {s}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Vocabulary Optimizer Demo ===\n")

    # Train base tokenizer on general text
    general_texts = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test",
        "machine learning is transforming the world",
        "natural language processing with deep learning",
        "the cat sat on the mat",
    ]

    tokenizer = BPETokenizer()
    tokenizer.train(general_texts * 10, vocab_size=300, min_frequency=2)

    # Analyze coverage
    report = analyze_coverage(tokenizer, general_texts * 5)
    print_coverage_report(report)

    # Train another tokenizer on domain text
    domain_texts = [
        "transformer attention mechanism self-attention cross-attention",
        "gradient descent backpropagation optimization loss function",
        "neural network layer normalization dropout regularization",
        "convolutional recurrent LSTM GRU sequence modeling",
    ]

    domain_tokenizer = BPETokenizer()
    domain_tokenizer.train(domain_texts * 10, vocab_size=300, min_frequency=2)

    # Merge vocabularies
    merged, merge_result = merge_vocabularies(tokenizer, domain_tokenizer)
    print(f"\nMerge result: {merge_result}")

    # Add domain tokens to base tokenizer
    extended = add_domain_tokens(tokenizer, domain_texts * 5, num_new_tokens=50)
    print(f"Extended vocab size: {extended.vocab_size}")

    # Prune unused tokens
    pruned, removed = prune_unused_tokens(tokenizer, general_texts, min_frequency=2)
    print(f"Pruned: removed {len(removed)} tokens, kept {pruned.vocab_size}")

    print("\nDone!")
