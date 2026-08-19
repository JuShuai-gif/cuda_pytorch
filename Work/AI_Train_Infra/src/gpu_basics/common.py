"""Shared utilities for reproducible, honest microbenchmarks.

The helpers in this file deliberately keep timing policy explicit.  In
particular, a CUDA launch is asynchronous, so a host ``perf_counter`` interval
without a terminal synchronization measures enqueue latency rather than device
completion latency.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

import torch


DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and fail loudly for an unavailable requested device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(DTYPES)}")
    dtype = DTYPES[name]
    # Keep the failure close to argument parsing.  Some CPU kernels do support
    # fp16/bf16, but coverage depends on the PyTorch build and ISA.
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU coverage is build-dependent; use float32/float64/bfloat16")
    return dtype


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class TimingSummary:
    unit: str
    samples: int
    mean: float
    stddev: float
    median: float
    p50: float
    p90: float
    p99: float
    minimum: float
    maximum: float
    raw: list[float]


def summarize_ms(values: list[float]) -> TimingSummary:
    return TimingSummary(
        unit="ms_per_call",
        samples=len(values),
        mean=statistics.mean(values),
        stddev=statistics.pstdev(values),
        median=statistics.median(values),
        p50=_percentile(values, 0.50),
        p90=_percentile(values, 0.90),
        p99=_percentile(values, 0.99),
        minimum=min(values),
        maximum=max(values),
        raw=values,
    )


@dataclass(frozen=True)
class TimingResult:
    synchronized_wall: TimingSummary
    cuda_event: Optional[TimingSummary]


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> TimingResult:
    """Time a callable using synchronized wall time and CUDA events.

    Inputs and compilation must be prepared before this function is called.
    Each sample contains ``iterations`` calls and is normalized to one call.
    CUDA is synchronized before each sample and after the end event.  This is
    correct for latency measurement but intentionally removes inter-sample
    overlap; use the profiler entry point for an unperturbed step timeline.
    """

    if warmup < 0 or iterations < 1 or repeats < 1:
        raise ValueError("warmup >= 0, iterations >= 1, and repeats >= 1 are required")

    with torch.no_grad():
        for _ in range(warmup):
            fn()
    synchronize(device)

    wall_samples: list[float] = []
    event_samples: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = perf_counter()
            start.record(torch.cuda.current_stream(device))
            with torch.no_grad():
                for _ in range(iterations):
                    fn()
            end.record(torch.cuda.current_stream(device))
            end.synchronize()
            wall_end = perf_counter()
            event_samples.append(start.elapsed_time(end) / iterations)
        else:
            wall_start = perf_counter()
            with torch.no_grad():
                for _ in range(iterations):
                    fn()
            wall_end = perf_counter()
        wall_samples.append((wall_end - wall_start) * 1_000.0 / iterations)

    return TimingResult(
        synchronized_wall=summarize_ms(wall_samples),
        cuda_event=summarize_ms(event_samples) if event_samples else None,
    )


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device_resolved": str(device),
        "num_cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        metadata["cuda_device"] = {
            "index": index,
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory,
            "multiprocessor_count": props.multi_processor_count,
            "cudnn_version": torch.backends.cudnn.version(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        }
    else:
        metadata["cuda_device"] = None
    return metadata


def json_ready(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return value


def dump_json(payload: dict[str, Any], output: Optional[Path]) -> str:
    """Serialize a result and optionally create a new evidence file.

    Existing files are rejected so a rerun cannot silently overwrite a
    baseline or optimized artifact.
    """

    rendered = json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
    return rendered
