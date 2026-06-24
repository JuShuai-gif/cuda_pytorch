#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module-level parameter analysis for VLA policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import module_splitter as ms

_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.uint8: 1,
}


@dataclass
class ParamStats:
    total: int
    trainable: int
    size_mb: float
    per_module: dict[str, int]  # top-level submodule -> params
    per_category: dict[str, int]  # vision/language/fusion/action -> params
    category_fraction: dict[str, float]


def _dtype_bytes(dtype: torch.dtype) -> int:
    return _DTYPE_BYTES.get(dtype, 4)


def count_params(
    model: nn.Module,
    config: ms.SplitConfig | None = None,
) -> ParamStats:
    """Count parameters and partition them by submodule and VLA category.

    ``size_mb`` accounts for the real per-tensor dtype (so a bf16 checkpoint
    is not over-reported as if it were fp32).
    """
    total = 0
    trainable = 0
    size_bytes = 0
    per_module: dict[str, int] = {}
    per_name: dict[str, int] = {}

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        size_bytes += n * _dtype_bytes(p.dtype)

        top = name.split(".")[0]
        per_module[top] = per_module.get(top, 0) + n
        per_name[name] = n

    per_category = {c: 0 for c in ms.CATEGORIES}
    for c, v in ms.split_by_category(per_name, config).items():
        per_category[c] = int(v)

    return ParamStats(
        total=total,
        trainable=trainable,
        size_mb=size_bytes / (1024**2),
        per_module=per_module,
        per_category=per_category,
        category_fraction=ms.as_fractions(per_category),
    )
