"""Operator fusion module.

The whole point of fusion is to turn a sequence of small kernels that each
round-trip through global memory into one kernel whose intermediates live in
registers/SRAM.  Each case below pairs an eager multi-kernel implementation
with a fused Triton kernel, and the benchmark measures the difference in
latency, kernel count, and memory traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch

TensorFactory = Callable[[torch.device, torch.dtype], Tuple[torch.Tensor, ...]]


@dataclass
class FusionCase:
    name: str
    unfused: Callable[..., torch.Tensor]
    fused: Callable[..., torch.Tensor]
    inputs: TensorFactory
    kwargs: Dict[str, Any] = None
    # Estimated bytes read+written per call (global-memory traffic).  These
    # are analytical estimates (numel * dtype_size per op), not measured, and
    # are meant to show *why* fused moves less data, not to be a benchmark.
    traffic_unfused_bytes: int = 0
    traffic_fused_bytes: int = 0
    note: str = ""

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
