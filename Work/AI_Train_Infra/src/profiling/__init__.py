"""Small, reproducible workloads for learning training-system profiling."""

from .workloads import WorkloadConfig, make_inputs, run_workload

__all__ = ["WorkloadConfig", "make_inputs", "run_workload"]
