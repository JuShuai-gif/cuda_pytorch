"""
Demonstration of the Byte-Pair Encoding (BPE) tokenizer.

Covers:
  - Training a tokenizer on sample text
  - Encoding and decoding with round-trip verification
  - Unicode handling (emoji, Chinese, accented characters)
  - Compression ratio analysis across vocabulary sizes
  - Visualizing the learned merge sequence

Usage:
    python demo.py
"""

from __future__ import annotations

from bpe import BPETokenizer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def show_encoding(tokenizer: BPETokenizer, text: str, label: str | None = None) -> None:
    """Show the encoding result for a given text."""
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    tokens = [tokenizer.vocab[idx].decode("utf-8", errors="replace") for idx in ids]

    if label:
        print(f"\n{label}:")
    print(f"  Input:    {text!r}")
    print(f"  Tokens:   {tokens}")
    print(f"  IDs:      {ids}")
    print(f"  Decoded:  {decoded!r}")
    print(f"  #Tokens:  {len(ids)}")
    print(f"  #Bytes:   {len(text.encode('utf-8'))}")


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_basic_training() -> BPETokenizer:
    """Demonstrate basic BPE training on English text."""
    print_section("1. Basic BPE Training")

    corpus = (
        "the cat sat on the mat "
        "the dog sat on the log "
        "the cat and the dog "
        "the cat sat on the mat "
        "the dog sat on the log "
    )

    tokenizer = BPETokenizer()
    print(f"Training corpus: {corpus!r}")
    print(f"Initial vocab size: {tokenizer.vocab_size} (all 256 bytes)")
    tokenizer.train(corpus, vocab_size=280)
    print(f"Trained vocab size: {tokenizer.vocab_size}")

    # Show the first few merges learned
    print("\nFirst 5 merges learned:")
    merge_items = list(tokenizer.merges.items())
    for i, (pair, new_id) in enumerate(merge_items[:5]):
        token_a = tokenizer.vocab[pair[0]].decode("utf-8", errors="replace")
        token_b = tokenizer.vocab[pair[1]].decode("utf-8", errors="replace")
        merged = tokenizer.vocab[new_id].decode("utf-8", errors="replace")
        print(f"  {i + 1}. ({token_a!r} + {token_b!r}) -> {merged!r}  (ID: {new_id})")

    return tokenizer


def demo_encode_decode(tokenizer: BPETokenizer) -> None:
    """Demonstrate encoding and decoding on various texts."""
    print_section("2. Encoding & Decoding")

    texts = [
        "the cat",
        "the dog and the cat",
        "hello world",
        "the mat",
    ]

    for text in texts:
        show_encoding(tokenizer, text)


def demo_unicode_handling() -> None:
    """Demonstrate BPE with Unicode-rich text."""
    print_section("3. Unicode Handling (Emoji, Chinese, Accented)")

    corpus = (
        "Hello, 🌍! 你好世界! こんにちは! "
        "Café résumé naïve façade "
        "αβγδε ηθικλ μνξοπ "
        "Hello, 🌍! 你好世界! Hello, 🌍! "
    )

    tokenizer = BPETokenizer()
    print(f"Training corpus: {corpus!r}")
    tokenizer.train(corpus, vocab_size=400)
    print(f"Trained vocab size: {tokenizer.vocab_size}")

    unicode_texts = [
        "Hello, 🌍!",
        "你好世界",
        "こんにちは",
        "Café résumé",
        "你好世界! Hello, 🌍!",
    ]

    for text in unicode_texts:
        show_encoding(tokenizer, text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text, f"Unicode round-trip failed: {text!r} != {decoded!r}"

    print("\nAll Unicode round-trips verified OK.")


def demo_compression_analysis() -> None:
    """Analyze compression ratio at different vocabulary sizes."""
    print_section("4. Compression Ratio by Vocab Size")

    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox is very quick and very brown. "
        "Machine learning models need good tokenization. "
        "Byte-Pair Encoding is a popular subword tokenization algorithm. "
        "It starts with individual bytes and iteratively merges the most "
        "frequent adjacent pairs of tokens. "
    ) * 5

    vocab_sizes = [256, 270, 300, 350, 400, 500]
    print(f"Corpus length: {len(corpus)} characters")
    print(f"Corpus bytes:  {len(corpus.encode('utf-8'))}")
    print()
    print(f"{'Vocab Size':>12}  {'#Tokens':>10}  {'#Bytes':>10}  {'Ratio':>8}")
    print("-" * 48)

    for vs in vocab_sizes:
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, vocab_size=vs)
        ids = tokenizer.encode(corpus)
        num_tokens = len(ids)
        num_bytes = len(corpus.encode("utf-8"))
        ratio = num_bytes / num_tokens if num_tokens > 0 else 0.0
        print(f"{vs:>12}  {num_tokens:>10}  {num_bytes:>10}  {ratio:>8.3f}")

    print()
    print("Interpretation: Larger vocab → fewer tokens → higher compression ratio.")
    print("This means shorter sequences for the model to process (faster attention).")


def demo_merge_visualization() -> None:
    """Visualize the merge process step by step."""
    print_section("5. Merge Process Visualization")

    text = "abcabcabc"
    tokenizer = BPETokenizer()

    # Manually show merges for demonstration
    ids = list(text.encode("utf-8"))
    print(f"Starting text: {text!r}")
    print(f"Initial bytes: {list(map(tokenizer.vocab.get, ids))}")
    print(f"Initial IDs:   {ids}")
    print()

    for i in range(5):
        # Count pairs
        counts: dict[tuple[int, int], int] = {}
        for j in range(len(ids) - 1):
            pair = (ids[j], ids[j + 1])
            counts[pair] = counts.get(pair, 0) + 1

        if not counts:
            break

        best_pair = max(counts, key=lambda p: counts[p])
        new_id = 256 + i
        tokenizer.vocab[new_id] = (
            tokenizer.vocab[best_pair[0]] + tokenizer.vocab[best_pair[1]]
        )

        ids = BPETokenizer._merge(ids, best_pair, new_id)
        tokenizer.merges[best_pair] = new_id

        a = tokenizer.vocab[best_pair[0]]
        b = tokenizer.vocab[best_pair[1]]
        merged = tokenizer.vocab[new_id]
        print(
            f"Merge {i + 1}: ({a!r} + {b!r}) -> {merged!r} "
            f"(count: {counts[best_pair]}, new ID: {new_id})"
        )
        print(f"  IDs:   {ids}")
        print(f"  Bytes: {list(map(tokenizer.vocab.get, ids))}")
        print()

    # Show final encoding
    final_ids = tokenizer.encode(text)
    print(f"Final encode('{text}'):  {final_ids}")
    print(f"Final decode:           {tokenizer.decode(final_ids)!r}")
    print(
        f"Compression: {len(text.encode('utf-8'))} bytes -> {len(final_ids)} tokens "
        f"(ratio: {tokenizer.compression_ratio(text):.2f})"
    )


def demo_summary() -> None:
    """Print a summary of key takeaways."""
    print_section("6. Summary")

    print("Key takeaways from BPE tokenization:")
    print()
    print("  1. BPE starts from individual bytes (vocab size = 256).")
    print("  2. Training: iteratively merge the most frequent adjacent token pair.")
    print("  3. Encoding: apply learned merges in order to new text.")
    print("  4. Decoding: map token IDs back to bytes, then decode as UTF-8.")
    print("  5. Unicode is natively supported via UTF-8 byte encoding.")
    print("  6. Larger vocabulary → higher compression ratio → shorter sequences.")
    print("  7. BPE adapts to any language/script in the training data.")
    print("  8. Trade-off: vocab size vs. model embedding size and sparsity.")
    print()
    print("This implementation is the foundation for:")
    print("  - GPT-2/GPT-3/GPT-4 tokenizers")
    print("  - RoBERTa tokenizer")
    print(
        "  - Most modern LLM tokenizers (with extensions like regex pre-tokenization)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all BPE demonstrations."""
    print("=" * 60)
    print("  BPE Tokenizer from Scratch - Demonstration")
    print("=" * 60)

    # 1. Basic training
    tokenizer = demo_basic_training()

    # 2. Encoding and decoding
    demo_encode_decode(tokenizer)

    # 3. Unicode handling
    demo_unicode_handling()

    # 4. Compression analysis
    demo_compression_analysis()

    # 5. Merge visualization
    demo_merge_visualization()

    # 6. Summary
    demo_summary()

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
