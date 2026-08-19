"""A small residual MLP whose layer stack creates observable DDP buckets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("distributed_basics requires PyTorch") from exc
    return torch


@dataclass(frozen=True)
class WorkloadConfig:
    local_batch_size: int = 4
    sequence_length: int = 64
    hidden_size: int = 256
    layers: int = 8
    expansion: int = 4
    model_seed: int = 1234
    data_seed: int = 5678
    learning_rate: float = 1.0e-3

    def validate(self) -> None:
        integer_fields = (
            self.local_batch_size,
            self.sequence_length,
            self.hidden_size,
            self.layers,
            self.expansion,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("all workload dimensions must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

    @property
    def local_tokens(self) -> int:
        return self.local_batch_size * self.sequence_length

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


class ResidualMLPBlock:
    """Factory wrapper keeps torch optional until the workload is actually used."""

    @staticmethod
    def build(hidden_size: int, expansion: int) -> Any:
        torch = require_torch()

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                inner = hidden_size * expansion
                self.norm = torch.nn.LayerNorm(hidden_size)
                self.up = torch.nn.Linear(hidden_size, inner)
                self.down = torch.nn.Linear(inner, hidden_size)

            def forward(self, inputs: Any) -> Any:
                residual = inputs
                hidden = self.norm(inputs)
                hidden = torch.nn.functional.gelu(self.up(hidden))
                return residual + self.down(hidden)

        return Block()


class TinyDDPModel:
    """Factory-compatible model constructor exposed as a callable class."""

    def __new__(cls, config: WorkloadConfig) -> Any:
        torch = require_torch()
        config.validate()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        ResidualMLPBlock.build(config.hidden_size, config.expansion)
                        for _ in range(config.layers)
                    ]
                )
                self.final_norm = torch.nn.LayerNorm(config.hidden_size)

            def forward(self, inputs: Any) -> Any:
                hidden = inputs
                for block in self.blocks:
                    hidden = block(hidden)
                return self.final_norm(hidden)

        return Model()


def seed_model(seed: int, device: str) -> None:
    torch = require_torch()
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_local_input(
    config: WorkloadConfig,
    rank: int,
    *,
    device: str,
    dtype: Any,
) -> Any:
    """Generate the same rank-local shard independently on every process."""

    torch = require_torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data_seed + rank)
    shape = (
        config.local_batch_size,
        config.sequence_length,
        config.hidden_size,
    )
    cpu_input = torch.randn(shape, generator=generator, dtype=torch.float32)
    return cpu_input.to(device=device, dtype=dtype)


def build_global_reference_input(
    config: WorkloadConfig,
    world_size: int,
    *,
    device: str,
    dtype: Any,
) -> Any:
    torch = require_torch()
    shards = [
        make_local_input(config, rank, device=device, dtype=dtype)
        for rank in range(world_size)
    ]
    return torch.cat(shards, dim=0)


def loss_fn(outputs: Any) -> Any:
    # Mean reduction is essential: averaging equal-size rank-local gradients is
    # then identical to the gradient of the concatenated global batch.
    return outputs.float().square().mean()


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def linear_training_flops(config: WorkloadConfig, world_size: int) -> int:
    """Analytical model FLOPs; excludes norm, GELU, loss and optimizer work."""

    tokens = config.local_tokens * world_size
    inner = config.hidden_size * config.expansion
    forward_per_layer = 2 * tokens * config.hidden_size * inner
    forward_per_layer += 2 * tokens * inner * config.hidden_size
    return 3 * config.layers * forward_per_layer
