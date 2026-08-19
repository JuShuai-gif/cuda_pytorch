"""Real-time control loop statistics: jitter, deadline misses, p99.

The defining difference between robot inference and server LLM inference:
a robot runs a *periodic* control loop and cares about the worst-case latency
(p99) and its stability (jitter), not the average or the throughput.  A policy
whose mean latency is 15ms but whose p99 is 200ms misses its 20ms control
deadline 1% of the time - which is exactly when the robot overshoots.

This module runs the sensor-to-action pipeline repeatedly as if it were a
control loop, records every cycle's latency, and reports p50/p90/p95/p99,
jitter, and the deadline-miss rate.
"""

from __future__ import annotations

import time
from time import perf_counter

import torch

from robotics.policy_inference.pipeline import (
    VLAPolicy,
    control_step,
    decode_image,
    make_camera_frame,
    postprocess_action,
    preprocess,
)


def run_control_loop(model: VLAPolicy, device: torch.device, *, cycles: int,
                     deadline_ms: float, inject_cpu_jitter: bool = False) -> dict:
    """Run `cycles` sensor-to-action cycles and collect per-cycle latency."""
    model.eval()
    frame = make_camera_frame()

    # Warm up.
    _one_cycle(model, device, frame)
    torch.cuda.synchronize(device)

    latencies = []
    for i in range(cycles):
        t0 = perf_counter()
        # Simulate a periodic OS/GC hiccup *during* this cycle: a CPU stall
        # that pushes the tail latency up.  This is the "jitter" that kills
        # real-time control.
        if inject_cpu_jitter and i % 50 == 0:
            time.sleep(0.005)  # 5ms CPU stall
        _one_cycle(model, device, frame)
        torch.cuda.synchronize(device)
        lat_ms = (perf_counter() - t0) * 1e3
        latencies.append(lat_ms)

    latencies.sort()
    n = len(latencies)

    def pct(q):
        return latencies[min(n - 1, int(n * q))]

    misses = sum(1 for x in latencies if x > deadline_ms)
    return {
        "cycles": n,
        "deadline_ms": deadline_ms,
        "mean_ms": sum(latencies) / n,
        "p50_ms": pct(0.50),
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": latencies[-1],
        "jitter_ms": pct(0.99) - pct(0.50),  # p99-p50 as a jitter proxy
        "deadline_miss_rate": misses / n,
    }


def _one_cycle(model: VLAPolicy, device: torch.device, frame: bytes) -> torch.Tensor:
    img = decode_image(frame)
    x = preprocess(img).unsqueeze(0).to(device)
    action = model.infer(x)
    action = postprocess_action(action)
    return control_step(action)
