"""Intentionally inefficient or synchronization-heavy reference workloads."""

from __future__ import annotations

import time

import torch


def launch_chain(x: torch.Tensor, iterations: int, increment: float) -> torch.Tensor:
    """Issue one pointwise add per iteration (many tiny CUDA kernels)."""

    y = x
    for _ in range(iterations):
        y = y + increment
    return y


def redundant_copies(x: torch.Tensor, copies: int) -> torch.Tensor:
    """Perform semantically redundant full-tensor copies."""

    y = x
    for _ in range(copies):
        y = y.clone()
    return y


def synchronize_each_iteration(
    x: torch.Tensor, iterations: int, scale: float, bias: float
) -> tuple[torch.Tensor, float]:
    """Force a device-to-host dependency with ``item`` every iteration."""

    y = x
    checksum = 0.0
    for _ in range(iterations):
        y = y * scale + bias
        checksum += float(y.sum().item())
    return y, checksum


def cpu_gap(x: torch.Tensor, delay_seconds: float) -> torch.Tensor:
    """Artificially delay the producer CPU before it submits GPU work."""

    time.sleep(delay_seconds)
    return torch.relu(x)


def gemm_einsum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A high-arithmetic-intensity expression whose backend must be profiled."""

    return torch.einsum("mk,kn->mn", a, b)

