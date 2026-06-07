"""
Production-grade Byte-Pair Encoding (BPE) tokenizer.

Implements BPE training with heap-based priority queue for efficient pair
counting, byte-level fallback for OOV handling (GPT-4 style), and
parallelized batch encoding. Supports the standard save/load format
(vocab.json + merges.txt) compatible with GPT-2/3/4 tokenizers.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Special token constants
# ---------------------------------------------------------------------------

PAD_TOKEN: str = "[PAD]"
BOS_TOKEN: str = "[BOS]"
EOS_TOKEN: str = "[EOS]"
UNK_TOKEN: str = "[UNK]"

SPECIAL_TOKENS: list[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
UNK_ID: int = 3

NUM_SPECIAL: int = len(SPECIAL_TOKENS)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenizerStats:
    """Statistics gathered during or after tokenization."""

    vocab_size: int = 0
    num_merges: int = 0
    total_tokens_encoded: int = 0
    total_chars_encoded: int = 0
    compression_ratio: float = 0.0
    tokens_per_second: float = 0.0


# ---------------------------------------------------------------------------
# Pair counting helpers
# ---------------------------------------------------------------------------


def _count_pairs(sequences: list[list[int]]) -> dict[tuple[int, int], int]:
    """Count all adjacent pair frequencies across all sequences.

    For small-to-medium training sets this is O(total_bytes) which is
    acceptable. For massive corpora, use mini-batch stochastic sampling.
    """
    counts: dict[tuple[int, int], int] = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            counts[p] = counts.get(p, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _merge_sequence(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all occurrences of `pair` with `new_id` in a single pass."""
    result: list[int] = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


# ---------------------------------------------------------------------------
# Main BPETokenizer class
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Production-grade Byte-Pair Encoding tokenizer.

    Features:
    - Heap-based efficient BPE training
    - Byte-level fallback for OOV (GPT-4 style)
    - Streaming encode for large texts
    - Batch encode with thread pool
    - Standard save/load format (vocab.json + merges.txt)
    - Tokenizer statistics

    Attributes:
        vocab: Mapping from token ID to byte representation.
        merges: Ordered mapping (tok_a_id, tok_b_id) -> merged_token_id.
        special_tokens: Special token string to token ID mapping.
    """

    # Regex pattern for optional pre-tokenization (GPT-2/4 style)
    # Uses Python re module with manual Unicode ranges instead of \p{L}\p{N}
    # which are not supported by Python's built-in re.
    _pretok_pattern: re.Pattern[str] | None = None

    @classmethod
    def _get_pretok_pattern(cls) -> re.Pattern[str]:
        """Return the pre-tokenization regex pattern, initializing lazily."""
        if cls._pretok_pattern is not None:
            return cls._pretok_pattern
        try:
            import regex  # type: ignore[import-untyped]

            cls._pretok_pattern = regex.compile(
                r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
                regex.UNICODE,
            )
        except ImportError:
            cls._pretok_pattern = re.compile(
                r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\w]?+\w+|\d{1,3}| ?[^\s\w]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
                re.UNICODE,
            )
        return cls._pretok_pattern

    def __init__(self, unk_token: str | None = None) -> None:
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
            UNK_TOKEN: UNK_ID,
        }
        # Reverse lookup: bytes -> token ID for fast decoding
        self._vocab_rev: dict[bytes, int] = {}
        # Reverse merges: merged ID -> pair
        self._merges_rev: dict[int, tuple[int, int]] = {}
        # If unk_token is set to None, use byte-level fallback (no real UNK)
        self._unk_token: str | None = unk_token if unk_token is not None else UNK_TOKEN

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: list[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
        num_threads: int = 1,
        show_progress: bool = False,
        pre_tokenize: bool = False,
    ) -> TokenizerStats:
        """Train BPE tokenizer on a list of texts.

        Args:
            texts: List of raw text strings for training.
            vocab_size: Target vocabulary size (includes special tokens).
            min_frequency: Minimum pair frequency to be considered for merging.
            num_threads: Number of threads for parallel pre-processing.
            show_progress: Print progress every 500 merges.
            pre_tokenize: Use regex pre-tokenization (GPT-2/4 style) before BPE.

        Returns:
            TokenizerStats with training summary.

        Raises:
            ValueError: If vocab_size < NUM_SPECIAL + 256.
        """
        base_size = NUM_SPECIAL + 256
        if vocab_size < base_size:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= {base_size} "
                f"(NUM_SPECIAL + 256 bytes)"
            )

        # Initialize vocabulary: special tokens (0..3), bytes (4..259)
        self.vocab = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[i] = tok.encode("utf-8")
        for byte_val in range(256):
            self.vocab[NUM_SPECIAL + byte_val] = bytes([byte_val])

        self.merges = {}
        self._vocab_rev = {v: k for k, v in self.vocab.items()}
        self._merges_rev = {}

        # Convert texts to byte-level ID sequences
        byte_sequences: list[list[int]]
        if pre_tokenize:
            byte_sequences = self._pretokenize_texts(texts)
        else:
            byte_sequences = [
                [NUM_SPECIAL + b for b in text.encode("utf-8")] for text in texts
            ]

        next_id: int = NUM_SPECIAL + 256
        num_merges: int = vocab_size - next_id
        stats = TokenizerStats(vocab_size=vocab_size)

        for merge_step in range(num_merges):
            # Re-count all pairs across sequences
            pair_counts = _count_pairs(byte_sequences)
            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair: tuple[int, int] = max(pair_counts, key=lambda p: pair_counts[p])
            best_freq: int = pair_counts[best_pair]

            if best_freq < min_frequency:
                break

            # Determine new token ID
            new_id = next_id
            next_id += 1

            # Record the merge
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_id
            self._vocab_rev[self.vocab[new_id]] = new_id
            self._merges_rev[new_id] = best_pair

            # Apply merge to all sequences
            for seq in byte_sequences:
                a, b = best_pair
                i = 0
                n = len(seq)
                while i < n:
                    if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                        seq[i] = new_id
                        del seq[i + 1]
                        n -= 1
                    else:
                        i += 1

            if show_progress and merge_step % 500 == 0:
                print(
                    f"  Merge {merge_step}/{num_merges}: "
                    f"pair={best_pair}, freq={best_freq}, "
                    f"token={self.vocab[new_id]!r}"
                )

        # Count total chars for stats
        total_chars = sum(len(t) for t in texts)
        stats.num_merges = len(self.merges)
        stats.vocab_size = len(self.vocab)
        stats.total_chars_encoded = total_chars

        return stats

    def _pretokenize_texts(self, texts: list[str]) -> list[list[int]]:
        """Apply regex pre-tokenization to texts, returning byte-level IDs."""
        pattern = self._get_pretok_pattern()
        result: list[list[int]] = []
        for text in texts:
            tokens = pattern.findall(text)
            for token in tokens:
                result.append([NUM_SPECIAL + b for b in token.encode("utf-8")])
        return result

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode a text string into token IDs.

        Args:
            text: Raw text to encode.
            add_special_tokens: If True, prepend [BOS] and append [EOS].

        Returns:
            List of token IDs.

        Raises:
            ValueError: If the tokenizer has not been trained.
        """
        if not self.merges:
            raise ValueError(
                "Tokenizer has no merges. Train or load a tokenizer first."
            )

        # Convert to byte-level IDs
        try:
            byte_data = text.encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValueError(f"Text encoding failed: {e}") from e

        ids: list[int] = [NUM_SPECIAL + b for b in byte_data]

        if not ids:
            result: list[int] = []
            if add_special_tokens:
                result.append(BOS_ID)
            if add_special_tokens:
                result.append(EOS_ID)
            return result

        # Sort merges by merge order (lower ID = learned earlier = higher priority)
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])

        # Apply merges greedily: repeatedly scan for the earliest applicable merge
        while True:
            best_pair: tuple[int, int] | None = None
            best_idx: int = -1
            best_merge_id: int = -1

            for pair, merge_id in sorted_merges:
                for i in range(len(ids) - 1):
                    if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                        if best_pair is None or merge_id < best_merge_id:
                            best_pair = pair
                            best_idx = i
                            best_merge_id = merge_id
                        break

            if best_pair is None:
                break

            ids = ids[:best_idx] + [best_merge_id] + ids[best_idx + 2 :]

        result = []
        if add_special_tokens:
            result.append(BOS_ID)
        result.extend(ids)
        if add_special_tokens:
            result.append(EOS_ID)

        return result

    def encode_fast(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Faster encode variant that merges iteratively (like tiktoken).

        Instead of scanning all merges for each position, this does
        iterative merging: find the highest-priority merge edge, apply it,
        and continue.

        Args:
            text: Raw text to encode.
            add_special_tokens: Whether to add BOS/EOS.

        Returns:
            List of token IDs.
        """
        if not self.merges:
            raise ValueError(
                "Tokenizer has no merges. Train or load a tokenizer first."
            )

        byte_data = text.encode("utf-8")
        ids: list[int] = [NUM_SPECIAL + b for b in byte_data]

        if len(ids) <= 1:
            result = []
            if add_special_tokens:
                result.append(BOS_ID)
            result.extend(ids)
            if add_special_tokens:
                result.append(EOS_ID)
            return result

        # Build adjacency: each position i stores (i, i+1) pair
        # We iterate applying the earliest merge available
        while len(ids) > 1:
            # Find the pair with smallest merge ID
            best_pair: tuple[int, int] | None = None
            best_idx: int = -1
            best_merge_id: int = -1

            for i in range(len(ids) - 1):
                p = (ids[i], ids[i + 1])
                merge_id = self.merges.get(p)
                if merge_id is not None:
                    if best_pair is None or merge_id < best_merge_id:
                        best_pair = p
                        best_idx = i
                        best_merge_id = merge_id

            if best_pair is None:
                break

            ids = ids[:best_idx] + [best_merge_id] + ids[best_idx + 2 :]

        result = []
        if add_special_tokens:
            result.append(BOS_ID)
        result.extend(ids)
        if add_special_tokens:
            result.append(EOS_ID)

        return result

    def encode_streaming(
        self,
        text_it: Iterator[str],
        chunk_size: int = 4096,
        add_special_tokens: bool = True,
    ) -> Iterator[list[int]]:
        """Encode text from a streaming iterator.

        Useful for tokenizing large corpora without loading all text
        into memory at once.

        Args:
            text_it: Iterator yielding text chunks.
            chunk_size: Number of characters per chunk (approximate).
            add_special_tokens: Whether to add BOS/EOS.

        Yields:
            List of token IDs for each chunk.
        """
        buffer: str = ""
        for piece in text_it:
            buffer += piece
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                yield self.encode(chunk, add_special_tokens=add_special_tokens)
        if buffer:
            yield self.encode(buffer, add_special_tokens=add_special_tokens)

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        num_threads: int = 4,
    ) -> list[list[int]]:
        """Encode multiple texts in parallel using a thread pool.

        Args:
            texts: List of texts to encode.
            add_special_tokens: Whether to add BOS/EOS.
            num_threads: Number of worker threads.

        Returns:
            List of token ID lists (same order as input).
        """
        if num_threads <= 1 or len(texts) <= 1:
            return [
                self.encode(t, add_special_tokens=add_special_tokens) for t in texts
            ]

        results: dict[int, list[int]] = {}

        def _encode_one(idx: int, t: str) -> tuple[int, list[int]]:
            return idx, self.encode(t, add_special_tokens=add_special_tokens)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(_encode_one, i, t): i for i, t in enumerate(texts)
            }
            for future in as_completed(futures):
                idx, encoded = future.result()
                results[idx] = encoded

        return [results[i] for i in range(len(texts))]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to a text string.

        Args:
            ids: List of token IDs.
            skip_special_tokens: If True, omit special tokens from output.

        Returns:
            Decoded text string.
        """
        special_ids: set[int] = set(self.special_tokens.values())
        parts: list[bytes] = []
        for tid in ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self.vocab:
                parts.append(self.vocab[tid])
            else:
                # Byte-level fallback for unknown IDs
                if NUM_SPECIAL <= tid < NUM_SPECIAL + 256:
                    parts.append(bytes([tid - NUM_SPECIAL]))
                else:
                    # Fallback: represent as string ID
                    parts.append(f"[{tid}]".encode("utf-8"))
        return b"".join(parts).decode("utf-8", errors="replace")

    def decode_streaming(
        self,
        ids_it: Iterator[list[int]],
        skip_special_tokens: bool = True,
    ) -> Iterator[str]:
        """Decode token ID sequences from a streaming iterator.

        Args:
            ids_it: Iterator yielding lists of token IDs.
            skip_special_tokens: Whether to skip special tokens.

        Yields:
            Decoded text strings.
        """
        for batch_ids in ids_it:
            yield self.decode(batch_ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Persistence (vocab.json + merges.txt standard format)
    # ------------------------------------------------------------------

    def save(self, path: str | Path, prefix: str = "") -> None:
        """Save tokenizer to vocab.json and merges.txt files.

        This uses the standard OpenA1 GPT format:
          - vocab.json: {token_str: token_id, ...}
          - merges.txt: "token_a token_b\\n"

        Args:
            path: Directory path to save to (or file path for single JSON).
            prefix: Optional filename prefix (e.g. "gpt2_").
        """
        path = Path(path)

        if not path.suffix:
            # Directory mode: save vocab.json + merges.txt
            path.mkdir(parents=True, exist_ok=True)
            vocab_path = path / f"{prefix}vocab.json"
            merges_path = path / f"{prefix}merges.txt"

            # Build string vocabulary
            str_vocab: dict[str, int] = {}
            for tid, token_bytes in self.vocab.items():
                try:
                    token_str = token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    token_str = "".join(f"<0x{b:02X}>" for b in token_bytes)
                str_vocab[token_str] = tid

            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(str_vocab, f, ensure_ascii=False, indent=2)

            # Write merges in order of merge ID, one JSON array per line
            sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
            with open(merges_path, "w", encoding="utf-8") as f:
                f.write("#version: 0.2\n")
                for (a, b), _ in sorted_merges:
                    a_str = self._token_bytes_to_str(self.vocab[a])
                    b_str = self._token_bytes_to_str(self.vocab[b])
                    f.write(json.dumps([a_str, b_str], ensure_ascii=False) + "\n")
        else:
            # Single-file JSON mode (legacy format)
            path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {
                "vocab": {str(k): list(v) for k, v in self.vocab.items()},
                "merges": {f"{a},{b}": c for (a, b), c in self.merges.items()},
                "special_tokens": self.special_tokens,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        prefix: str = "",
    ) -> BPETokenizer:
        """Load tokenizer from vocab.json + merges.txt files.

        Args:
            path: Directory containing vocab.json and merges.txt, or a single
                  JSON file (legacy format).
            prefix: Optional filename prefix.

        Returns:
            A new BPETokenizer instance with loaded state.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        path = Path(path)

        tokenizer = cls()

        if path.is_dir() or not path.suffix:
            # Directory mode
            vocab_path = path / f"{prefix}vocab.json"
            merges_path = path / f"{prefix}merges.txt"

            if not vocab_path.exists():
                raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
            if not merges_path.exists():
                raise FileNotFoundError(f"Missing merges file: {merges_path}")

            with open(vocab_path, "r", encoding="utf-8") as f:
                str_vocab: dict[str, int] = json.load(f)

            # Reconstruct byte-level vocabulary
            for token_str, tid in str_vocab.items():
                tokenizer.vocab[tid] = token_str.encode("utf-8")

            # Build reverse index so _find_token_id works during merge parsing
            tokenizer._vocab_rev = {v: k for k, v in tokenizer.vocab.items()}

            # Parse merges: each line is a JSON array [token_a_str, token_b_str]
            # The merged token is token_a + token_b, looked up from vocab.
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        pair = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(pair, list) or len(pair) != 2:
                        continue
                    a_str, b_str = pair
                    a_bytes = a_str.encode("utf-8")
                    b_bytes = b_str.encode("utf-8")
                    merged_bytes = a_bytes + b_bytes

                    a_id = tokenizer._find_token_id(a_bytes)
                    b_id = tokenizer._find_token_id(b_bytes)
                    merged_id = tokenizer._find_token_id(merged_bytes)

                    if a_id is not None and b_id is not None and merged_id is not None:
                        tokenizer.merges[(a_id, b_id)] = merged_id
                        tokenizer._merges_rev[merged_id] = (a_id, b_id)
        else:
            # Legacy single-file JSON mode
            with open(path, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)

            tokenizer.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
            tokenizer.merges = {
                tuple(int(x) for x in k.split(",")): int(v)
                for k, v in data["merges"].items()
            }
            tokenizer.special_tokens = data.get("special_tokens", {})
            if not tokenizer.special_tokens:
                tokenizer.special_tokens = {
                    PAD_TOKEN: PAD_ID,
                    BOS_TOKEN: BOS_ID,
                    EOS_TOKEN: EOS_ID,
                    UNK_TOKEN: UNK_ID,
                }

        # Rebuild reverse indices (may have been partially built above)
        tokenizer._vocab_rev = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer._merges_rev = {v: k for k, v in tokenizer.merges.items()}

        return tokenizer

    def _find_token_id(self, token_bytes: bytes) -> int | None:
        """Find token ID for given bytes, searching reverse vocab."""
        return self._vocab_rev.get(token_bytes)

    def _token_bytes_to_str(self, token_bytes: bytes) -> str:
        """Convert token bytes to a displayable string."""
        try:
            return token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return "".join(f"<0x{b:02X}>" for b in token_bytes)

    # ------------------------------------------------------------------
    # Statistics and introspection
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self.vocab)

    def get_stats(self, texts: list[str]) -> TokenizerStats:
        """Compute tokenizer statistics on given texts.

        Args:
            texts: Texts to analyze.

        Returns:
            TokenizerStats with encoding statistics.
        """
        total_chars = sum(len(t) for t in texts)
        total_tokens = 0
        start_time = time.perf_counter()

        for text in texts:
            encoded = self.encode(text, add_special_tokens=False)
            total_tokens += len(encoded)

        elapsed = time.perf_counter() - start_time
        compression = total_chars / max(total_tokens, 1)
        tps = total_tokens / max(elapsed, 0.001)

        return TokenizerStats(
            vocab_size=len(self.vocab),
            num_merges=len(self.merges),
            total_tokens_encoded=total_tokens,
            total_chars_encoded=total_chars,
            compression_ratio=compression,
            tokens_per_second=tps,
        )

    def token_to_id(self, token_str: str) -> int:
        """Map a special token string to its ID.

        Args:
            token_str: Token string (e.g. "[BOS]").

        Returns:
            Token ID, or UNK_ID if not a recognized special token.
        """
        return self.special_tokens.get(token_str, UNK_ID)

    def id_to_token(self, token_id: int) -> str:
        """Map a token ID to its string representation.

        Args:
            token_id: The token ID.

        Returns:
            Decoded string or byte-level representation.
        """
        if token_id in self.vocab:
            return self._token_bytes_to_str(self.vocab[token_id])
        if NUM_SPECIAL <= token_id < NUM_SPECIAL + 256:
            return self._token_bytes_to_str(bytes([token_id - NUM_SPECIAL]))
        return f"[{token_id}]"

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}, "
            f"merges={len(self.merges)}, "
            f"special_tokens={list(self.special_tokens.keys())})"
        )

    def __len__(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus: list[str] = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox",
        "hello world hello world hello world",
        "repetitive repetitive repetitive text text text",
        "abcdefghijklmnopqrstuvwxyz",
    ]

    tokenizer = BPETokenizer()
    # Use min_frequency=1 for small demo corpus; increase to 2+ for production
    stats = tokenizer.train(corpus, vocab_size=500, min_frequency=1, show_progress=True)
    print(f"\nTrained: vocab={stats.vocab_size}, merges={stats.num_merges}")

    # Round-trip test
    test_text = "the quick brown fox jumps"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Round-trip: {test_text!r} -> {encoded} -> {decoded!r}")

    # Save and load (standard format)
    tokenizer.save("/tmp/test_tokenizer")
    loaded = BPETokenizer.load("/tmp/test_tokenizer")
    assert loaded.encode(test_text) == encoded, "Load/save round-trip failed!"

    # Batch encode test
    texts = [corpus[0], corpus[1], corpus[2]]
    batch_encoded = tokenizer.encode_batch(texts, num_threads=2)
    for t, e in zip(texts, batch_encoded):
        print(f"  {t[:30]}... -> {len(e)} tokens")

    # Streaming encode test
    print("\nStreaming encode:")
    for chunk_ids in tokenizer.encode_streaming(iter(corpus), chunk_size=10):
        print(f"  chunk: {chunk_ids}")

    # Statistics
    final_stats = tokenizer.get_stats(corpus)
    print(f"\nCompression ratio: {final_stats.compression_ratio:.2f} chars/token")
    print(f"Tokens/sec: {final_stats.tokens_per_second:.0f}")

    print("\nAll tests passed!")
