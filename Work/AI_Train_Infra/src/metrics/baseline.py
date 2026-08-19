"""Eager baseline for the token-MLP workload."""

from __future__ import annotations

from typing import Any

from .workload import WorkloadConfig, make_model


def build_baseline(config: WorkloadConfig, *, device: str, dtype: Any) -> Any:
    """Return an ordinary eager ``torch.nn.Module``."""

    return make_model(config, device=device, dtype=dtype)
