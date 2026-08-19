"""Autoscaling simulation: which metric to scale on?

A GPU inference service has a natural bottleneck (the GPU), and the question
is which signal the autoscaler should watch.  The core lesson: CPU utilization
is a *bad* metric for GPU inference, because the GPU can be saturated while
the CPU is nearly idle (the CPU only launches kernels and copies data).  This
module simulates a load spike and scales workers based on three metrics to show
which one reacts correctly.

Time-step simulation.  Each worker serves at a fixed rate (GPU-bound).  The
autoscaler re-evaluates every ``eval_every`` steps and adjusts the worker count
toward a target derived from one metric:
  cpu     - worker CPU utilization (decorrelated from GPU load)
  queue   - pending request queue length
  latency - mean request latency
"""

from __future__ import annotations


def simulate(metric: str, *, steps: int = 200, base_arrival: float = 100.0,
             spike_arrival: float = 500.0, spike_at: int = 30,
             worker_rate: float = 100.0, min_workers: int = 1, max_workers: int = 8,
             eval_every: int = 5) -> dict:
    workers = 1
    queue = 0
    latencies: list[float] = []
    dropped = 0
    worker_history = []

    for t in range(steps):
        arrival = spike_arrival if t >= spike_at else base_arrival

        # 1. Requests arrive (per step).
        queue += int(arrival)

        # 2. Workers serve.
        served = int(workers * worker_rate)
        served = min(served, queue)
        queue -= served
        if served > 0:
            latencies.append(served / (workers * worker_rate))  # rough per-req latency

        # 3. Autoscaler re-evaluates.
        if t % eval_every == 0:
            target = _target_workers(metric, workers, queue, latencies,
                                     arrival, worker_rate)
            workers = max(min_workers, min(max_workers, target))
        worker_history.append(workers)

        # 4. Shed excess queue (bounded queue depth).
        if queue > 1000:
            dropped += queue - 1000
            queue = 1000

    return {
        "metric": metric,
        "final_workers": workers,
        "mean_workers": sum(worker_history) / len(worker_history),
        "total_dropped": dropped,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0,
        "final_queue": queue,
    }


def _target_workers(metric: str, workers: int, queue: int, latencies: list[float],
                    arrival: float, worker_rate: float) -> int:
    if metric == "cpu":
        # CPU util is decorrelated from GPU saturation: report a fixed ~20%
        # regardless of real load, so the autoscaler never scales up.
        cpu_util = 0.20
        return workers + (1 if cpu_util > 0.8 else 0)
    if metric == "queue":
        # Scale up if the queue is growing (more than one worker's capacity).
        if queue > worker_rate:
            return workers + 1
        if queue < worker_rate // 2 and workers > 1:
            return workers - 1
        return workers
    if metric == "latency":
        # Scale up if latency exceeds the single-worker baseline.
        if latencies:
            if latencies[-1] > 1.2 / worker_rate:
                return workers + 1
        if queue == 0 and workers > 1:
            return workers - 1
        return workers
    return workers
