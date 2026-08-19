"""Triton operator implementations and their PyTorch baselines.

Each module exports a ``build(cfg)`` returning an ``Op`` dataclass with the
triton kernel, the torch reference, input factories, and a correctness check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import torch

TensorFactory = Callable[[torch.device, torch.dtype], Tuple[torch.Tensor, ...]]


@dataclass
class Op:
    name: str
    triton: Callable[..., torch.Tensor]
    reference: Callable[..., torch.Tensor]
    inputs: TensorFactory
    # Extra keyword arguments shared by both implementations (e.g. eps, dim).
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # Descriptions used by the benchmark report.
    note: str = ""

    def check(self, device: torch.device, dtype: torch.dtype, *, atol: float = 1e-3,
              rtol: float = 1e-2) -> Tuple[bool, float]:
        """Return (ok, max_abs_diff) comparing triton vs reference."""
        args = self.inputs(device, dtype)
        with torch.no_grad():
            expected = self.reference(*args, **self.kwargs)
            actual = self.triton(*args, **self.kwargs)
        torch.cuda.synchronize(device)
        diff = (actual - expected).abs().max().item()
        ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        return ok, diff
