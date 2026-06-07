"""
SentencePiece compatibility layer.

Supports loading SentencePiece .model files, converting between formats,
and basic proto parsing for SentencePiece model protobufs.

Note:
    Full proto deserialization requires the `protobuf` package.
    A lightweight parser is provided as a fallback.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lightweight SentencePiece ModelProto parser
# ---------------------------------------------------------------------------

# Protobuf wire types
WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LENGTH_DELIMITED = 2
WIRE_FIXED32 = 5

# SentencePiece ModelProto field numbers (simplified)
FIELD_PIECES = 1  # repeated SentencePiece
FIELD_TRAINER_SPEC = 2  # TrainerSpec
FIELD_NORMALIZER_SPEC = 3  # NormalizerSpec


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read a varint from data starting at offset. Returns (value, new_offset)."""
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return value, offset


def _parse_model_proto(data: bytes) -> dict[str, Any]:
    """Parse a SentencePiece ModelProto binary into a Python dict.

    Args:
        data: Raw protobuf bytes.

    Returns:
        Dict with keys: pieces, trainer_spec, normalizer_spec.
    """
    result: dict[str, Any] = {
        "pieces": [],
        "trainer_spec": {},
        "normalizer_spec": {},
    }

    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == WIRE_VARINT:
            value, offset = _read_varint(data, offset)
            if field_number == FIELD_TRAINER_SPEC:
                result["trainer_spec"]["_raw"] = value
            elif field_number == FIELD_NORMALIZER_SPEC:
                result["normalizer_spec"]["_raw"] = value

        elif wire_type == WIRE_LENGTH_DELIMITED:
            length, offset = _read_varint(data, offset)
            chunk = data[offset : offset + length]
            offset += length

            if field_number == FIELD_PIECES:
                piece = _parse_sentence_piece(chunk)
                if piece:
                    result["pieces"].append(piece)
            elif field_number == FIELD_TRAINER_SPEC:
                result["trainer_spec"] = _parse_trainer_spec(chunk)
            elif field_number == FIELD_NORMALIZER_SPEC:
                result["normalizer_spec"] = _parse_normalizer_spec(chunk)

        else:
            # Skip unknown wire types
            if wire_type == WIRE_FIXED64:
                offset += 8
            elif wire_type == WIRE_FIXED32:
                offset += 4

    return result


# SentencePiece sub-message field numbers
PIECE_PIECE = 1  # string piece
PIECE_SCORE = 2  # float score
PIECE_TYPE = 3  # enum type


def _parse_sentence_piece(data: bytes) -> dict[str, Any] | None:
    """Parse a single SentencePiece sub-message."""
    piece: dict[str, Any] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == WIRE_LENGTH_DELIMITED:
            length, offset = _read_varint(data, offset)
            chunk = data[offset : offset + length]
            offset += length
            if field_number == PIECE_PIECE:
                piece["piece"] = chunk.decode("utf-8", errors="replace")
        elif wire_type == WIRE_VARINT:
            value, offset = _read_varint(data, offset)
            if field_number == PIECE_TYPE:
                piece["type"] = value
        elif wire_type == WIRE_FIXED32:
            # 32-bit float for score
            if field_number == PIECE_SCORE and offset + 4 <= len(data):
                score = struct.unpack("<f", data[offset : offset + 4])[0]
                piece["score"] = score
            offset += 4

    return piece if piece else None


TRAINER_VOCAB_SIZE = 3
TRAINER_CHARACTER_COVERAGE = 10
TRAINER_MODEL_TYPE = 1
TRAINER_BOS_ID = 30
TRAINER_EOS_ID = 31
TRAINER_PAD_ID = 32
TRAINER_UNK_ID = 40


def _parse_trainer_spec(data: bytes) -> dict[str, Any]:
    """Parse TrainerSpec sub-message."""
    spec: dict[str, Any] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == WIRE_VARINT:
            value, offset = _read_varint(data, offset)
            spec[_trainer_field_name(field_number)] = value
        elif wire_type == WIRE_LENGTH_DELIMITED:
            length, offset = _read_varint(data, offset)
            chunk = data[offset : offset + length]
            offset += length
            spec[_trainer_field_name(field_number)] = chunk.decode(
                "utf-8", errors="replace"
            )
        elif wire_type == WIRE_FIXED32:
            if field_number == TRAINER_CHARACTER_COVERAGE:
                spec["character_coverage"] = struct.unpack(
                    "<f", data[offset : offset + 4]
                )[0]
            offset += 4
    return spec


NORMALIZER_NAME = 1
NORMALIZER_PRE_NORMALIZER = 3


def _parse_normalizer_spec(data: bytes) -> dict[str, Any]:
    """Parse NormalizerSpec sub-message."""
    spec: dict[str, Any] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == WIRE_LENGTH_DELIMITED:
            length, offset = _read_varint(data, offset)
            chunk = data[offset : offset + length]
            offset += length
            if field_number == NORMALIZER_NAME:
                spec["name"] = chunk.decode("utf-8", errors="replace")
            elif field_number == NORMALIZER_PRE_NORMALIZER:
                spec["pre_compiled_charsmap"] = chunk
    return spec


def _trainer_field_name(field_number: int) -> str:
    """Map TrainerSpec field numbers to names."""
    mapping: dict[int, str] = {
        1: "model_type",
        3: "vocab_size",
        10: "character_coverage",
        30: "bos_id",
        31: "eos_id",
        32: "pad_id",
        40: "unk_id",
    }
    return mapping.get(field_number, f"field_{field_number}")


# ---------------------------------------------------------------------------
# Python protobuf parser (optional, preferred if protobuf is installed)
# ---------------------------------------------------------------------------


def _parse_with_protobuf(data: bytes) -> dict[str, Any] | None:
    """Parse ModelProto using the official protobuf library if available."""
    try:
        import sentencepiece_model_pb2  # type: ignore[import-untyped]
    except ImportError:
        return None

    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(data)

    pieces: list[dict[str, Any]] = []
    for sp in model.pieces:
        pieces.append(
            {
                "piece": sp.piece,
                "score": sp.score,
                "type": sp.type,
            }
        )

    trainer = model.trainer_spec
    return {
        "pieces": pieces,
        "trainer_spec": {
            "model_type": trainer.model_type,
            "vocab_size": trainer.vocab_size,
            "character_coverage": trainer.character_coverage,
            "bos_id": trainer.bos_id,
            "eos_id": trainer.eos_id,
            "pad_id": trainer.pad_id,
            "unk_id": trainer.unk_id,
        },
        "normalizer_spec": {},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_sentencepiece_model(model_path: str | Path) -> dict[str, Any]:
    """Parse a SentencePiece .model file and return structured data.

    Tries the official protobuf parser first; falls back to a lightweight
    custom parser that handles the core fields.

    Args:
        model_path: Path to the .model file.

    Returns:
        Dict with keys: ``pieces``, ``trainer_spec``, ``normalizer_spec``.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model file cannot be parsed.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    data = path.read_bytes()

    # Try protobuf first
    result = _parse_with_protobuf(data)
    if result is not None:
        return result

    # Fallback to lightweight parser
    result = _parse_model_proto(data)
    if not result.get("pieces"):
        raise ValueError(
            f"Failed to parse SentencePiece model from {path}. "
            f"Install protobuf for full support: pip install protobuf"
        )
    return result


def sp_model_to_bpe_vocab(
    model_path: str | Path,
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Convert a SentencePiece model to BPE-style vocab and merges.

    Args:
        model_path: Path to the .model file.

    Returns:
        Tuple of (vocab_dict, merges_list) where:
        - vocab_dict maps token strings to integer IDs
        - merges_list contains (token_a, token_b) tuples
    """
    sp_data = parse_sentencepiece_model(model_path)
    pieces: list[dict[str, Any]] = sp_data["pieces"]

    vocab: dict[str, int] = {}
    merges: list[tuple[str, str]] = []

    for piece in pieces:
        token_str = piece["piece"]
        token_type = piece.get("type", 1)  # 1 = NORMAL

        # Special tokens
        if token_type in (3, 4):  # USER_DEFINED, UNUSED
            continue
        if token_type == 2:  # CONTROL (BOS, EOS, etc.)
            # Add as vocab entry only
            if token_str not in vocab:
                vocab[token_str] = len(vocab)
            continue

        # Normal token
        if len(token_str) <= 1:
            # Single char: add to vocab
            if token_str not in vocab:
                vocab[token_str] = len(vocab)
        else:
            # Multi-char: it's a merge. Find the merge pair.
            # SentencePiece stores tokens; merges need to be inferred.
            # For BPE, this is approximate.
            if token_str not in vocab:
                vocab[token_str] = len(vocab)

            # Heuristic: split into first two sub-tokens in vocab
            for split_point in range(1, len(token_str)):
                prefix = token_str[:split_point]
                suffix = token_str[split_point:]
                if prefix in vocab and suffix in vocab:
                    merges.append((prefix, suffix))
                    break

    return vocab, merges


def bpe_to_sp_model(
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    output_path: str | Path,
    model_proto_available: bool = False,
) -> None:
    """Write BPE vocab and merges to a SentencePiece-style file.

    Note:
        This writes a simplified format. For a proper .model file,
        ``sentencepiece`` or ``sentencepiece_model_pb2`` is required.

    Args:
        vocab: Token string to ID mapping.
        merges: List of merge pairs.
        output_path: Output file path.
        model_proto_available: If True, write proper protobuf (requires lib).
    """
    output_path = Path(output_path)

    if model_proto_available:
        try:
            import sentencepiece_model_pb2  # type: ignore[import-untyped]

            model = sentencepiece_model_pb2.ModelProto()
            # Write tokens
            for token_str, token_id in sorted(vocab.items(), key=lambda x: x[1]):
                sp = model.pieces.add()
                sp.piece = token_str
                sp.score = 0.0
                sp.type = 1  # NORMAL
                if token_str in ("[PAD]", "[BOS]", "[EOS]", "[UNK]"):
                    sp.type = 2  # CONTROL

            model.trainer_spec.model_type = 1  # BPE
            model.trainer_spec.vocab_size = len(vocab)

            output_path.write_bytes(model.SerializeToString())
            return
        except ImportError:
            pass

    # Fallback: write text-based format
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SentencePiece-compatible vocabulary (text format)\n")
        for token_str, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token_str}\t{token_id}\n")
        if merges:
            f.write("\n# Merges\n")
            for a, b in merges:
                f.write(f"{a} {b}\n")


def load_sentencepiece_vocab(
    model_path: str | Path,
) -> dict[str, int]:
    """Load vocabulary from a SentencePiece .model or .vocab file.

    Args:
        model_path: Path to .model or .vocab file.

    Returns:
        Token string to ID mapping.
    """
    path = Path(model_path)

    # Handle .vocab files (text format)
    if path.suffix == ".vocab":
        vocab: dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    vocab[parts[0]] = int(parts[1])
                else:
                    vocab[parts[0]] = len(vocab)
        return vocab

    # Handle .model files (protobuf)
    sp_data = parse_sentencepiece_model(path)
    vocab = {}
    for piece in sp_data["pieces"]:
        token_str = piece["piece"]
        vocab[token_str] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sentencepiece_compat.py <model_path>")
        print("  Parses a SentencePiece model and prints its contents.")
        sys.exit(1)

    model_path = sys.argv[1]
    data = parse_sentencepiece_model(model_path)
    print(f"Pieces: {len(data['pieces'])}")
    print(f"Trainer spec: {data.get('trainer_spec', {})}")
    for piece in data["pieces"][:10]:
        print(f"  {piece}")
    if len(data["pieces"]) > 10:
        print(f"  ... and {len(data['pieces']) - 10} more pieces")
