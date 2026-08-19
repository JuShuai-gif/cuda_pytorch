"""Small token-MLP training workload used by the metric examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "PyTorch is required for the workload; pure metric tests do not need it"
        ) from error
    return torch


@dataclass(frozen=True)
class WorkloadConfig:
    batch_size: int = 8
    sequence_length: int = 128
    hidden_size: int = 256
    mlp_size: int = 1024
    learning_rate: float = 1.0e-3
    seed: int = 1234

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.sequence_length


def resolve_device(requested: str) -> str:
    torch = require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested, but torch.cuda.is_available() is false")
    return requested


def resolve_dtype(name: str, device: str) -> Any:
    torch = require_torch()
    if name == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[name]
    if device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU training is not supported by this example")
    return dtype


def make_model(config: WorkloadConfig, *, device: str, dtype: Any) -> Any:
    torch = require_torch()

    class TinyTokenMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = torch.nn.Linear(config.hidden_size, config.mlp_size)
            self.down = torch.nn.Linear(config.mlp_size, config.hidden_size)

        def forward(self, inputs: Any) -> Any:
            return self.down(torch.nn.functional.gelu(self.up(inputs), approximate="tanh"))

    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    return TinyTokenMLP().to(device=device, dtype=dtype)


def make_input(config: WorkloadConfig, *, device: str, dtype: Any) -> Any:
    torch = require_torch()
    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(config.seed + 1)
    return torch.randn(
        config.batch_size,
        config.sequence_length,
        config.hidden_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def train_step(model: Any, optimizer: Any, inputs: Any) -> Any:
    """One full step: zero gradients, forward, loss, backward, optimizer update."""

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    # Compute the scalar reduction in FP32 so BF16/FP16 examples remain stable.
    loss = outputs.float().square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def synchronize(device: str) -> None:
    torch = require_torch()
    if device == "cuda":
        torch.cuda.synchronize()
