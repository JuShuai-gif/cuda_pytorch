"""
HuggingFace tokenizer compatibility layer.

Provides conversion between BPETokenizer and HuggingFace tokenizers,
including support for tokenizer.json parsing and PreTrainedTokenizerFast
interoperability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cs336.tokenizer.bpe import BPETokenizer, NUM_SPECIAL, SPECIAL_TOKENS

# ---------------------------------------------------------------------------
# Convert from HuggingFace tokenizer.json to BPETokenizer
# ---------------------------------------------------------------------------


def hf_tokenizer_json_to_bpe(json_path: str | Path) -> BPETokenizer:
    """Parse a HuggingFace tokenizer.json file into a BPETokenizer.

    Handles the standard HuggingFace tokenizer.json format that contains
    model.vocab and model.merges sections.

    Args:
        json_path: Path to a HuggingFace tokenizer.json file.

    Returns:
        A BPETokenizer with the loaded vocabulary and merges.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the file lacks required sections.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    tokenizer = BPETokenizer()

    model = data.get("model", {})
    if not model:
        raise KeyError("tokenizer.json missing 'model' section")

    # Load vocab {token_str: token_id}
    str_vocab: dict[str, int] = model.get("vocab", {})
    if not str_vocab:
        # Try alternative: added_tokens + vocab from decoder
        str_vocab = _extract_vocab_from_decoder(data)

    # Build vocab (bytes-based)
    tokenizer.vocab = {}
    for token_str, token_id in str_vocab.items():
        # Map token_id to bytes
        token_bytes = _hf_token_to_bytes(token_str)
        tokenizer.vocab[int(token_id)] = token_bytes

    # Load special tokens
    tokenizer.special_tokens = {}
    for tok in data.get("added_tokens", []):
        tok_str = tok.get("content", "")
        tok_id = tok.get("id", -1)
        if tok_str in SPECIAL_TOKENS:
            tokenizer.special_tokens[tok_str] = tok_id

    if not tokenizer.special_tokens:
        # Default special tokens
        tokenizer.special_tokens = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

    tokenizer._vocab_rev = {v: k for k, v in tokenizer.vocab.items()}

    # Load merges: each line is either "token_a token_b" (standard HF)
    # or a JSON array [token_a, token_b] (our extended format).
    raw_merges: list[str] = model.get("merges", [])
    tokenizer.merges = {}
    merge_next = NUM_SPECIAL + 256
    for merge_line in raw_merges:
        if merge_line.startswith("["):
            # JSON array format: ["token_a", "token_b"]
            try:
                pair = json.loads(merge_line)
                if isinstance(pair, list) and len(pair) == 2:
                    a_str, b_str = pair
                    a_bytes = _hf_token_to_bytes(a_str)
                    b_bytes = _hf_token_to_bytes(b_str)
                    merged_bytes = a_bytes + b_bytes
                    a_id = tokenizer._vocab_rev.get(a_bytes)
                    b_id = tokenizer._vocab_rev.get(b_bytes)
                    m_id = tokenizer._vocab_rev.get(merged_bytes)
                    if a_id is not None and b_id is not None and m_id is not None:
                        tokenizer.merges[(a_id, b_id)] = m_id
                continue
            except json.JSONDecodeError:
                pass
        # Standard space-separated format
        parts = merge_line.split()
        if len(parts) != 2:
            continue
        a_bytes = _hf_token_to_bytes(parts[0])
        b_bytes = _hf_token_to_bytes(parts[1])
        merged_bytes = a_bytes + b_bytes
        a_id = tokenizer._vocab_rev.get(a_bytes)
        b_id = tokenizer._vocab_rev.get(b_bytes)
        m_id = tokenizer._vocab_rev.get(merged_bytes)
        if a_id is not None and b_id is not None and m_id is not None:
            tokenizer.merges[(a_id, b_id)] = m_id

    tokenizer._merges_rev = {v: k for k, v in tokenizer.merges.items()}

    return tokenizer


def _extract_vocab_from_decoder(data: dict[str, Any]) -> dict[str, int]:
    """Extract vocab from a HuggingFace decoder-style config."""
    added_tokens = data.get("added_tokens", [])
    vocab: dict[str, int] = {}
    for tok_info in added_tokens:
        content = tok_info.get("content", "")
        tok_id = tok_info.get("id", len(vocab))
        vocab[content] = tok_id

    # Try the model-level vocab
    model_vocab = data.get("model", {}).get("vocab", {})
    vocab.update(model_vocab)
    return vocab


def _hf_token_to_bytes(token_str: str) -> bytes:
    """Convert a HuggingFace token string (with possible \\uXXXX escapes) to bytes.

    HuggingFace tokenizers.json represents non-ASCII tokens using JSON
    escape sequences like \\uXXXX. This function handles that conversion.
    """
    if "\\u" in token_str:
        # Use unicode-escape decoding
        try:
            return token_str.encode("ascii").decode("unicode-escape").encode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return token_str.encode("utf-8")


# ---------------------------------------------------------------------------
# Convert from BPETokenizer to HuggingFace tokenizer.json
# ---------------------------------------------------------------------------


def bpe_to_hf_tokenizer_json(
    tokenizer: BPETokenizer,
    output_path: str | Path,
    tokenizer_name: str = "bpe_tokenizer",
) -> None:
    """Write a BPETokenizer to HuggingFace tokenizer.json format.

    Args:
        tokenizer: The BPETokenizer to convert.
        output_path: Path to write tokenizer.json.
        tokenizer_name: Name for the tokenizer in metadata.
    """
    output_path = Path(output_path)

    # Build string vocab
    str_vocab: dict[str, int] = {}
    for tid, token_bytes in sorted(tokenizer.vocab.items()):
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = "".join(
                f"\\u{ord(chr(b)):04x}" if b < 128 else f"<0x{b:02X}>"
                for b in token_bytes
            )
        str_vocab[token_str] = tid

    # Build merges: use JSON array format to handle tokens with spaces
    sorted_merges = sorted(tokenizer.merges.items(), key=lambda x: x[1])
    merge_lines: list[str] = []
    for (a, b), _ in sorted_merges:
        try:
            a_str = tokenizer.vocab[a].decode("utf-8")
            b_str = tokenizer.vocab[b].decode("utf-8")
        except UnicodeDecodeError:
            continue
        merge_lines.append(json.dumps([a_str, b_str], ensure_ascii=False))

    # Build added_tokens for special tokens
    added_tokens: list[dict[str, Any]] = []
    for tok_str, tok_id in tokenizer.special_tokens.items():
        added_tokens.append(
            {
                "id": tok_id,
                "content": tok_str,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )

    # Build tokenizer.json structure
    data: dict[str, Any] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "BOS", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 1}},
                {"SpecialToken": {"id": "EOS", "type_id": 2}},
            ],
            "pair": [
                {"SpecialToken": {"id": "BOS", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 1}},
                {"SpecialToken": {"id": "EOS", "type_id": 2}},
                {"Sequence": {"id": "B", "type_id": 3}},
                {"SpecialToken": {"id": "EOS", "type_id": 4}},
            ],
            "special_tokens": {
                "BOS": {"id": "BOS", "ids": [1], "tokens": ["[BOS]"]},
                "EOS": {"id": "EOS", "ids": [2], "tokens": ["[EOS]"]},
            },
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": str_vocab,
            "merges": merge_lines,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Convert to HuggingFace PreTrainedTokenizerFast
# ---------------------------------------------------------------------------


def to_hf_fast_tokenizer(
    tokenizer: BPETokenizer,
    output_dir: str | Path,
) -> None:
    """Export BPETokenizer to HuggingFace PreTrainedTokenizerFast format.

    This creates a directory with tokenizer.json and special_tokens_map.json
    that can be loaded with ``AutoTokenizer.from_pretrained(output_dir)``.

    Args:
        tokenizer: The BPETokenizer to export.
        output_dir: Directory to write tokenizer files to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write tokenizer.json
    bpe_to_hf_tokenizer_json(tokenizer, output_dir / "tokenizer.json")

    # Write special_tokens_map.json
    special_map: dict[str, str | dict[str, str]] = {
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "additional_special_tokens": [],
    }
    with open(output_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_map, f, ensure_ascii=False, indent=2)

    # Write tokenizer_config.json
    config = {
        "add_prefix_space": False,
        "bos_token": "[BOS]",
        "clean_up_tokenization_spaces": False,
        "eos_token": "[EOS]",
        "model_max_length": 2048,
        "pad_token": "[PAD]",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "[UNK]",
    }
    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Convert from HuggingFace to BPE vocab+merges
# ---------------------------------------------------------------------------


def from_hf_pretrained(
    model_id: str,
    cache_dir: str | None = None,
) -> BPETokenizer:
    """Load a BPETokenizer from a HuggingFace pretrained model ID.

    Downloads (or loads from cache) the tokenizer files and converts
    to a BPETokenizer instance.

    Args:
        model_id: HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf").
        cache_dir: Optional cache directory for downloaded files.

    Returns:
        A BPETokenizer with the loaded vocabulary and merges.

    Raises:
        ImportError: If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError:
        try:
            from transformers.utils.hub import cached_file  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "huggingface_hub or transformers is required to use from_hf_pretrained(). "
                "Install with: pip install huggingface_hub"
            )

    # Try to download tokenizer.json
    try:
        json_path = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer.json",
            cache_dir=cache_dir,
        )
        return hf_tokenizer_json_to_bpe(json_path)
    except Exception:
        pass

    # Fallback: try vocab.json + merges.txt
    try:
        vocab_path = hf_hub_download(
            repo_id=model_id,
            filename="vocab.json",
            cache_dir=cache_dir,
        )
        merges_path = hf_hub_download(
            repo_id=model_id,
            filename="merges.txt",
            cache_dir=cache_dir,
        )
        # Use BPETokenizer.load with the directory
        return BPETokenizer.load(Path(vocab_path).parent, prefix="")
    except Exception:
        pass

    raise ValueError(
        f"Could not load tokenizer from {model_id}. "
        f"Make sure the model has a tokenizer.json or vocab.json+merges.txt."
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== HuggingFace Compatibility Demo ===\n")

    # Build a small tokenizer and export it
    corpus = ["hello world", "the quick brown fox", "test tokenizer"]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300, min_frequency=1)

    # Export to HuggingFace format
    output_dir = Path("/tmp/hf_tokenizer_test")
    to_hf_fast_tokenizer(tokenizer, output_dir)
    print(f"Exported to {output_dir}")

    # List generated files
    for f in output_dir.iterdir():
        print(f"  {f.name}")

    # Round-trip: reload
    loaded = hf_tokenizer_json_to_bpe(output_dir / "tokenizer.json")
    test = "hello world"
    assert tokenizer.encode(test) == loaded.encode(test), "Round-trip mismatch!"
    print(f"\nRound-trip test: {test!r} -> OK")

    print("\nAll HuggingFace compatibility tests passed!")
