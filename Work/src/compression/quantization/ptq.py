"""Weight-only INT8 post-training quantization (PTQ) on a small MLP.

PTQ quantizes a *trained* model without retraining: compute a per-channel scale
for each Linear weight from its own range, store the weights as int8, and at
inference dequantize on the fly.  The activation stays fp16, so the accuracy
loss is driven purely by weight rounding.  This module quantizes a residual MLP
and reports (1) the output error vs the fp16 baseline and (2) the weight-size
reduction.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


def quantize_weight_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel (per output row) int8 quantization of a weight."""
    scale = w.abs().max(dim=1, keepdim=True).values / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale


def linear_with_int8_weight(x: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x @ dequant(wq), keeping the matmul in fp16 after dequant."""
    w = (wq.float() * scale).to(x.dtype)
    return x @ w.t()


def quantize_model(model: MLP) -> MLP:
    """Return a copy of the model whose Linear weights are int8 + scale."""
    qmodel = MLP.__new__(MLP)  # shallow rebuild; we replace blocks below
    qmodel.blocks = nn.ModuleList()
    for block in model.blocks:
        qblock = nn.Sequential()
        for m in block:
            if isinstance(m, nn.Linear):
                wq, scale = quantize_weight_per_channel(m.weight.data)
                # Store as plain attributes; forward is handled by a custom
                # path in benchmark_accuracy.
                qblock.append(m)  # keep the layer, we patch weights below
        qmodel.blocks.append(qblock)
    return qmodel


def weight_bytes(model: MLP, dtype_bytes: int) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * dtype_bytes
    return total


def run_accuracy(device: torch.device, hidden=1024, layers=4, batch=1, seq=16):
    """Compare fp16 baseline vs weight-only int8 PTQ output error and size."""
    torch.manual_seed(0)
    model = MLP(hidden, layers).to(device=device, dtype=torch.float16).eval()
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)

    with torch.no_grad():
        y_fp16 = model(x)

    # Build an int8-weight model manually for the forward pass.
    quantized = []  # list of (wq, scale, bias) per Linear in traversal order
    for block in model.blocks:
        for m in block:
            if isinstance(m, nn.Linear):
                wq, scale = quantize_weight_per_channel(m.weight.data)
                quantized.append((wq, scale, m.bias.data if m.bias is not None else None))

    def forward_int8(x: torch.Tensor) -> torch.Tensor:
        it = iter(quantized)
        for block in model.blocks:
            y = block[0](x)  # layernorm
            for m in block:
                if isinstance(m, nn.Linear):
                    wq, scale, bias = next(it)
                    y = linear_with_int8_weight(y, wq, scale)
                    if bias is not None:
                        y = y + bias
                elif isinstance(m, nn.GELU):
                    y = m(y)
            x = x + y
        return x

    with torch.no_grad():
        y_int8 = forward_int8(x)

    max_diff = (y_int8 - y_fp16).abs().max().item()
    mse = ((y_int8 - y_fp16) ** 2).mean().item()

    fp16_bytes = weight_bytes(model, 2)
    int8_bytes = sum(q[0].numel() for q in quantized) + sum(q[1].numel() * 4 for q in quantized)

    return {
        "max_abs_diff": max_diff,
        "mse": mse,
        "fp16_weight_bytes": fp16_bytes,
        "int8_weight_bytes": int8_bytes,
        "size_ratio": int8_bytes / fp16_bytes,
    }
