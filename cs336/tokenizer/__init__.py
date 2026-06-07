"""
Production-grade BPE tokenizer module.

Provides:
  - BPETokenizer: Core BPE training, encode, decode with parallel support
  - SentencePiece compatibility (load .model files, convert)
  - HuggingFace tokenizer compatibility (convert to/from)
  - Benchmarking (throughput, compression ratio, ablation)
  - Vocabulary optimization (coverage analysis, merge, prune)
"""

from cs336.tokenizer.bpe import (
    BPETokenizer,
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_ID,
    UNK_TOKEN,
    TokenizerStats,
)

__all__ = [
    "BPETokenizer",
    "BOS_ID",
    "BOS_TOKEN",
    "EOS_ID",
    "EOS_TOKEN",
    "PAD_ID",
    "PAD_TOKEN",
    "SPECIAL_TOKENS",
    "UNK_ID",
    "UNK_TOKEN",
    "TokenizerStats",
]
