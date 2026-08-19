"""Environment and device metadata collection.

Every benchmark report should record *where and on what hardware* it ran,
because a speedup number without hardware context is not reproducible.  This
module returns a plain JSON-serializable dict plus helpers for resolving the
``--device`` and ``--dtype`` CLI arguments.
"""

from __future__ import annotations

import platform
import socket
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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
        raise ValueError(f"unsupported dtype {name!r}; choose from {sorted(DTYPES)}")
    dtype = DTYPES[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU coverage is build-dependent; use float32/float64/bfloat16")
    return dtype


def device_properties(device: torch.device) -> Optional[Dict[str, Any]]:
    """Return a subset of ``torch.cuda`` device properties or ``None`` on CPU."""
    if device.type != "cuda":
        return None
    p = torch.cuda.get_device_properties(device)
    return {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "multi_processor_count": p.multi_processor_count,
        "total_memory_bytes": p.total_memory,
        "clock_rate_khz": getattr(p, "clock_rate", None),
        "memory_clock_rate_khz": getattr(p, "memory_clock_rate", None),
        "memory_bus_width_bits": getattr(p, "memory_bus_width", None),
        "l2_cache_bytes": getattr(p, "L2_cache_size", None),
        "max_threads_per_sm": getattr(p, "max_threads_per_multi_processor", None),
    }


def collect_environment(device: torch.device) -> Dict[str, Any]:
    """Collect host + device metadata for a benchmark report."""
    cuda_available = torch.cuda.is_available()
    metadata: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda if torch.version.cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "cuda_available": cuda_available,
        "cuda_device_count_visible": torch.cuda.device_count() if cuda_available else 0,
        "selected_device": str(device),
    }
    if cuda_available:
        metadata["device_properties"] = device_properties(device)
    return metadata
