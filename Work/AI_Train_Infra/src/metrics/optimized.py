"""Candidate compiled implementation; performance must be measured, not assumed."""

from __future__ import annotations

from typing import Any

from .workload import WorkloadConfig, make_model, require_torch


def build_compiled(
    config: WorkloadConfig,
    *,
    device: str,
    dtype: Any,
    mode: str = "default",
) -> Any:
    """Compile the same model graph with ``torch.compile``.

    Compilation/cache population belongs to warmup.  This is an optimization
    candidate, not a promise: small shapes or graph breaks can make it slower.
    """

    torch = require_torch()
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    model = make_model(config, device=device, dtype=dtype)
    return torch.compile(model, mode=mode, fullgraph=True)
