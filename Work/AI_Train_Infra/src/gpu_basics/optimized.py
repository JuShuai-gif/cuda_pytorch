"""Optimization candidates paired with :mod:`gpu_basics.baseline`."""

from __future__ import annotations

import torch


def launch_collapsed(x: torch.Tensor, iterations: int, increment: float) -> torch.Tensor:
    """Algebraically collapse repeated adds into one pointwise launch."""

    return x + increment * iterations


def single_copy(x: torch.Tensor, copies: int) -> torch.Tensor:
    """Keep the observable copy while eliminating redundant memory passes."""

    del copies
    return x.clone()


def synchronize_once(
    x: torch.Tensor, iterations: int, scale: float, bias: float
) -> tuple[torch.Tensor, float]:
    """Keep reductions on device and transfer one final scalar to the host."""

    y = x
    checksums: list[torch.Tensor] = []
    for _ in range(iterations):
        y = y * scale + bias
        checksums.append(y.sum())
    checksum = float(torch.stack(checksums).sum().item())
    return y, checksum


def cpu_ready(x: torch.Tensor, delay_seconds: float) -> torch.Tensor:
    """Reference for a pipeline that has prepared the next input in advance."""

    del delay_seconds
    return torch.relu(x)


def gemm_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Use the direct matrix-multiply API; the backend may equal einsum."""

    return torch.mm(a, b)

