"""Minimal, auditable torch.distributed and DDP training lab."""

from .options import DDPOptions, options_for_variant
from .timeline import Interval, summarize_overlap
from .workload import WorkloadConfig, TinyDDPModel

__all__ = [
    "DDPOptions",
    "Interval",
    "TinyDDPModel",
    "WorkloadConfig",
    "options_for_variant",
    "summarize_overlap",
]
