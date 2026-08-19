"""Small, deterministic inference workloads for latency/throughput study.

The models here are deliberately tiny but structurally representative: a stack
of residual ``Linear -> LayerNorm -> GELU`` blocks.  This keeps the benchmark
fast to run, keeps FLOPs and parameters cheap to compute analytically, and
still exposes the launch-bound vs compute-bound boundary that Stage 1 is about.

A real LLM/VLM adds attention, KV cache and token-by-token decode; those are
introduced in later stages.  Here the goal is a *controllable* probe, not a
faithful model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class InferenceConfig:
    hidden: int = 1024
    layers: int = 4
    batch: int = 1
    seq_len: int = 1

    @property
    def label(self) -> str:
        return f"hidden{self.hidden}_layers{self.layers}_b{self.batch}_s{self.seq_len}"


class ResidualBlock(nn.Module):
    """A single residual MLP block: LayerNorm -> Linear -> GELU -> Linear."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))


def make_model(config: InferenceConfig, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
    model = nn.Sequential(*[ResidualBlock(config.hidden) for _ in range(config.layers)])
    return model.to(device=device, dtype=dtype).eval()


def make_input(config: InferenceConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    shape = (config.batch, config.seq_len, config.hidden)
    return torch.randn(*shape, device=device, dtype=dtype)


def parameter_count(config: InferenceConfig) -> int:
    # Per block: norm (2*hidden), fc1 (hidden*hidden + hidden), fc2 (hidden*hidden + hidden).
    per_block = 2 * config.hidden + 2 * (config.hidden * config.hidden + config.hidden)
    return config.layers * per_block


def flops_per_forward(config: InferenceConfig) -> int:
    """Count multiply-add FLOPs for one forward pass (matmuls dominate)."""
    tokens = config.batch * config.seq_len
    h = config.hidden
    # Each block: two h*h matmuls per token => 2 * tokens * h * h * 2 FLOPs.
    return config.layers * 2 * tokens * h * h * 2
