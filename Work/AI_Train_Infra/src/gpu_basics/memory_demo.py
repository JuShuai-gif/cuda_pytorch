"""Measure model-state bytes and CUDA caching-allocator observations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .common import dump_json, environment_metadata, resolve_device, seed_everything, synchronize


class TinyMLP(nn.Module):
    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for _ in range(layers):
            blocks.extend((nn.Linear(hidden, hidden), nn.GELU()))
        self.net = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _tensor_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _cuda_point(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    synchronize(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    points: dict[str, Any] = {"before_model": _cuda_point(device)}

    model = TinyMLP(args.hidden, args.layers).to(device=device, dtype=torch.float32)
    parameters = list(model.parameters())
    points["after_model"] = _cuda_point(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    points["after_optimizer_constructor"] = _cuda_point(device)

    inputs = torch.randn(args.batch, args.hidden, device=device)
    saved_logical_bytes = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved_logical_bytes
        saved_logical_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        outputs = model(inputs)
        loss = outputs.square().mean()
    points["after_forward"] = _cuda_point(device)
    loss.backward()
    points["after_backward"] = _cuda_point(device)
    optimizer.step()
    points["after_first_optimizer_step"] = _cuda_point(device)

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    optimizer_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    structural = {
        "parameter_bytes": _tensor_bytes(parameters),
        "gradient_bytes": _tensor_bytes(gradients),
        "optimizer_state_tensor_bytes": _tensor_bytes(optimizer_tensors),
        "logical_autograd_saved_tensor_bytes": saved_logical_bytes,
        "input_bytes": inputs.numel() * inputs.element_size(),
        "notes": [
            "Saved-tensor bytes are logical numel*element_size and can double-count aliases/shared storage.",
            "Saved tensors include anything autograd retains (including inputs/weights), not only activations.",
            "AdamW state is lazily created on the first optimizer step.",
        ],
    }
    return {
        "schema_version": 1,
        "experiment": "gpu_memory_accounting",
        "environment": environment_metadata(device),
        "config": vars(args),
        "structural_bytes": structural,
        "cuda_allocator_points": points,
        "limitations": (
            []
            if device.type == "cuda"
            else ["CPU run validates state accounting; CUDA allocated/reserved/peak metrics were not measured."]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(dump_json(run(args), args.output))


if __name__ == "__main__":
    main()

