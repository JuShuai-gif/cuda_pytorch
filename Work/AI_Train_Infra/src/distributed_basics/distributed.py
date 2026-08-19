"""Process-group setup and device placement with explicit environment metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import os
from typing import Any

from .options import DDPOptions
from .workload import require_torch


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: str

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initialize(device_request: str, backend_request: str, timeout_s: int) -> DistributedContext:
    torch = require_torch()
    dist = torch.distributed
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "launch with torchrun; missing environment variables: " + ", ".join(missing)
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 0 or timeout_s <= 0:
        raise ValueError("world_size and timeout_s must be > 0")

    if device_request == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_request
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        torch.cuda.set_device(local_rank)
    backend = backend_request
    if backend == "auto":
        backend = "nccl" if device == "cuda" else "gloo"
    if backend == "nccl" and device != "cuda":
        raise ValueError("NCCL requires --device cuda")

    process_group_device = torch.device("cuda", local_rank) if device == "cuda" else None
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=timeout_s),
        device_id=process_group_device,
    )
    return DistributedContext(rank, local_rank, world_size, backend, device)


def wrap_ddp(model: Any, context: DistributedContext, options: DDPOptions) -> Any:
    torch = require_torch()
    options.validate()
    kwargs = {
        "bucket_cap_mb": options.bucket_cap_mb,
        "gradient_as_bucket_view": options.gradient_as_bucket_view,
        "static_graph": options.static_graph,
        "broadcast_buffers": options.broadcast_buffers,
        "find_unused_parameters": options.find_unused_parameters,
    }
    if context.device == "cuda":
        kwargs.update(device_ids=[context.local_rank], output_device=context.local_rank)
    return torch.nn.parallel.DistributedDataParallel(model, **kwargs)


def synchronize_device(context: DistributedContext) -> None:
    if context.device == "cuda":
        require_torch().cuda.synchronize(context.local_rank)


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
        raise ValueError("float16 CPU training is not a supported lab configuration")
    return dtype


def cleanup() -> None:
    torch = require_torch()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
