"""
Byte-Pair Encoding (BPE) tokenizer from scratch.

Implements the BPE algorithm as described in:
  Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units"

The tokenizer works at the byte level:
  - Base vocabulary: 256 single-byte tokens (0-255)
  - Training: iteratively merges the most frequent adjacent pair of tokens
  - Encoding: applies learned merges in order to produce token IDs
  - Decoding: maps token IDs back to bytes, then decodes as UTF-8

Includes self-contained test cases that run when the file is executed directly.

Usage:
    from bpe import BPETokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(corpus_text, vocab_size=300)
    ids = tokenizer.encode("hello world")
    text = tokenizer.decode(ids)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Byte-Pair Encoding tokenizer with byte-level base vocabulary.

    Attributes:
        vocab: Mapping from token ID to its byte representation.
        merges: Ordered mapping from token pair to merged token ID.
                Order matters: earlier merges are applied first during encoding.
    """

    def __init__(self) -> None:
        # Base vocabulary: one entry per byte value (0-255)
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Learned merges: (id_a, id_b) -> new_merged_id
        self.merges: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int) -> None:
        """Train the BPE tokenizer on `text` to reach `vocab_size` tokens.

        Starting from 256 single-byte tokens, performs `vocab_size - 256`
        merge operations. Each iteration:
          1. Count frequencies of all adjacent token pairs in the current sequence.
          2. Select the most frequent pair.
          3. Merge that pair into a new token ID and update vocab/merges.

        Args:
            text: Training corpus (UTF-8 string).
            vocab_size: Target vocabulary size. Must be >= 256.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256, got {vocab_size}")

        # Work with integer IDs (byte values initially)
        ids: list[int] = list(text.encode("utf-8"))
        num_merges: int = vocab_size - 256

        for i in range(num_merges):
            # Count adjacent pair frequencies
            pair_counts = self._count_adjacent_pairs(ids)
            if not pair_counts:
                break  # No more pairs to merge

            # Pick the most frequent pair
            best_pair = max(pair_counts, key=lambda p: pair_counts[p])

            # Create a new token ID for the merged pair
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply the merge to the current ID sequence
            ids = self._merge(ids, best_pair, new_id)

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a UTF-8 string into a list of token IDs.

        Starts from individual bytes, then iteratively applies learned
        merges. At each step, picks the merge with the **lowest** new
        token ID (i.e., the earliest merge learned during training),
        which is canonical in BPE.

        Args:
            text: Input string to tokenize.

        Returns:
            List of token IDs.
        """
        ids: list[int] = list(text.encode("utf-8"))

        if not self.merges:
            return ids

        while len(ids) >= 2:
            # Find all adjacent pairs and pick the one whose merge ID is smallest
            best_pair: tuple[int, int] | None = None
            best_merge_id: int = float("inf")  # type: ignore[assignment]

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                merge_id = self.merges.get(pair)
                if merge_id is not None and merge_id < best_merge_id:
                    best_pair = pair
                    best_merge_id = merge_id

            if best_pair is None:
                break  # No more applicable merges

            ids = self._merge(ids, best_pair, self.merges[best_pair])

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a UTF-8 string.

        Args:
            ids: List of token IDs to decode.

        Returns:
            Decoded UTF-8 string. Uses 'replace' error handling for
            invalid byte sequences.
        """
        tokens: bytes = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_adjacent_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
        """Count occurrences of each adjacent pair in a token ID sequence."""
        counts: dict[tuple[int, int], int] = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of `pair` with `new_id` in the ID sequence."""
        new_ids: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self.vocab)

    def compression_ratio(self, text: str) -> float:
        """Return UTF-8 bytes per token for the given text.

        Higher values mean better compression (fewer tokens per byte).
        """
        ids = self.encode(text)
        num_bytes = len(text.encode("utf-8"))
        num_tokens = len(ids)
        if num_tokens == 0:
            return 0.0
        return num_bytes / num_tokens


# ---------------------------------------------------------------------------
# Test cases (run with: python bpe.py)
# ---------------------------------------------------------------------------


def run_tests() -> None:
    """Run all test cases for the BPETokenizer class."""

    # ------------------------------------------------------------------
    # Test 1: Basic round-trip with small vocab
    # ------------------------------------------------------------------
    print("Test 1: Basic encode/decode round-trip ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = "the cat in the hat the cat on the mat"
    tokenizer.train(corpus, vocab_size=270)  # 256 + 14 merges

    test_strings = [
        "the",
        "the cat",
        "the cat in the hat",
        "hello world",
    ]
    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        assert decoded == s, f"Round-trip failed: {s!r} -> {ids} -> {decoded!r}"
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 2: Empty string
    # ------------------------------------------------------------------
    print("Test 2: Empty string encoding ...", end=" ")
    tokenizer = BPETokenizer()
    tokenizer.train("hello world", vocab_size=260)
    ids = tokenizer.encode("")
    assert ids == [], f"Expected [], got {ids}"
    decoded = tokenizer.decode(ids)
    assert decoded == "", f"Expected '', got {decoded!r}"
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 3: Unicode handling (emojis, Chinese, accented chars)
    # ------------------------------------------------------------------
    print("Test 3: Unicode handling ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = "Hello, 🌍! 你好! こんにちは! Café résumé naïve"
    tokenizer.train(corpus, vocab_size=512)

    unicode_strings = [
        "Hello, 🌍!",
        "你好世界",
        "こんにちは",
        "Café résumé",
        "🌍🌍🌍",
        "αβγδε",
    ]
    for s in unicode_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
    # All should decode without raising exceptions
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 4: Round-trip for Unicode strings
    # ------------------------------------------------------------------
    print("Test 4: Unicode round-trip ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = (
        "Hello, 🌍! 你好世界! こんにちは! Café résumé naïve "
        "the quick brown fox jumps over the lazy dog "
        "the quick brown fox the quick brown fox "
    )
    tokenizer.train(corpus, vocab_size=512)

    original = "Hello, 🌍! 你好世界! Café résumé"
    ids = tokenizer.encode(original)
    decoded = tokenizer.decode(ids)
    assert decoded == original, (
        f"Unicode round-trip failed: {original!r} != {decoded!r}"
    )
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 5: Compression ratio improves with larger vocab
    # ------------------------------------------------------------------
    print("Test 5: Compression ratio improves ...", end=" ")
    corpus = (
        "the cat in the hat the cat on the mat "
        "the quick brown fox jumps over the lazy dog "
        "the cat the hat the mat the fox the dog "
    ) * 10

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=256)  # No merges
    ratio_baseline = tokenizer.compression_ratio(corpus)
    assert ratio_baseline == 1.0, f"Baseline ratio should be 1.0, got {ratio_baseline}"

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300)
    ratio_trained = tokenizer.compression_ratio(corpus)
    assert ratio_trained > 1.0, (
        f"Trained ratio ({ratio_trained}) must exceed baseline ({ratio_baseline})"
    )
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 6: Vocabulary size is correct
    # ------------------------------------------------------------------
    print("Test 6: Vocabulary size ...", end=" ")
    tokenizer = BPETokenizer()
    assert tokenizer.vocab_size == 256, (
        f"Initial vocab should be 256, got {tokenizer.vocab_size}"
    )

    tokenizer.train("hello world", vocab_size=300)
    # Should be >= 256 and <= 300 (may be less if text is too short)
    assert 256 <= tokenizer.vocab_size <= 300, (
        f"Trained vocab should be in [256, 300], got {tokenizer.vocab_size}"
    )
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 7: Repetitive text produces meaningful merges
    # ------------------------------------------------------------------
    print("Test 7: Repetitive text merges ...", end=" ")
    tokenizer = BPETokenizer()
    tokenizer.train("abababab cdcdcdcd abababab cdcdcdcd", vocab_size=260)
    # Should have merged some patterns
    assert tokenizer.vocab_size > 256, "Expected merges to increase vocabulary"
    ids = tokenizer.encode("abab")
    # With ab merged, should produce fewer tokens than bytes
    assert len(ids) < 4, f"Expected <4 tokens for 'abab', got {len(ids)}"
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 8: Special characters (newlines, tabs, etc.)
    # ------------------------------------------------------------------
    print("Test 8: Special characters ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = "line1\nline2\nline3\tindented\ttext"
    tokenizer.train(corpus, vocab_size=270)

    test = "line1\nline2"
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)
    assert decoded == test, f"Special char round-trip failed: {test!r} != {decoded!r}"
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 9: Large vocab on realistic text
    # ------------------------------------------------------------------
    print("Test 9: Large vocab ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = (
        "Byte-Pair Encoding (BPE) is a subword tokenization algorithm "
        "originally developed for data compression and later adapted "
        "for neural machine translation. The key idea is to start with "
        "individual characters (or bytes) and iteratively merge the most "
        "frequent adjacent pair of tokens. This produces a vocabulary "
        "that balances character-level and word-level representations. "
        "BPE was popularized in NLP by Sennrich et al. (2016) and is "
        "used by GPT-2, RoBERTa, and many other language models."
    )
    tokenizer.train(corpus, vocab_size=400)
    assert tokenizer.vocab_size > 256, "Expected merges to increase vocabulary"
    orig_ids = tokenizer.encode(corpus)
    decoded = tokenizer.decode(orig_ids)
    assert decoded == corpus, "Full corpus round-trip failed"
    print("PASSED")

    # ------------------------------------------------------------------
    # Test 10: Deterministic behavior
    # ------------------------------------------------------------------
    print("Test 10: Deterministic encoding ...", end=" ")
    tokenizer = BPETokenizer()
    tokenizer.train("aaaaabbbbb aaaaabbbbb", vocab_size=260)
    ids1 = tokenizer.encode("aaaaabbbbb")
    ids2 = tokenizer.encode("aaaaabbbbb")
    assert ids1 == ids2, f"Encoding should be deterministic: {ids1} != {ids2}"
    print("PASSED")

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
