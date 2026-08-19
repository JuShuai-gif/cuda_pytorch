"""A second-order plant under PD control with a variable control latency.

The point is to show *why* jitter matters, not just average latency.  A
constant latency can be compensated (it is just a phase lag); a *jittery*
latency cannot, because the effective control gain varies from cycle to
cycle, which drives overshoot and oscillation.

The plant is a damped mass: m*x'' + c*x' + k*x = u.  The controller is PD:
u = Kp*(target - x) - Kd*x'.  The control output passes through a latency
queue, so a command computed at step t takes ``delay`` steps to reach the
plant - a "delay" that can vary per step.
"""

from __future__ import annotations

from collections import deque


def simulate(target: float, latency_seq: list[int], *, dt: float = 0.01,
             m: float = 1.0, c: float = 1.0, k: float = 10.0,
             kp: float = 15.0, kd: float = 3.0, ki: float = 10.0,
             settle_start: int = 400) -> dict:
    x = 0.0   # position
    v = 0.0   # velocity
    integral = 0.0
    queue: deque = deque()          # delayed control forces
    errors = []

    for i, delay in enumerate(latency_seq):
        # Sample current state, compute the control force (PID).
        u = kp * (target - x) - kd * v + ki * integral
        integral += (target - x) * dt
        # The command is delayed: enqueue, apply the oldest if its delay elapsed.
        queue.append(u)
        applied = queue.popleft() if len(queue) > delay else 0.0

        # Plant dynamics (semi-implicit Euler).
        a = (applied - c * v - k * x) / m
        v = v + a * dt
        x = x + v * dt

        # Track error only after the initial transient settles, so the metric
        # reflects steady-state tracking, not the step response.
        if i >= settle_start:
            errors.append(abs(target - x))

    rms = (sum(e * e for e in errors) / len(errors)) ** 0.5
    return {
        "rms_error": rms,
        "max_error": max(errors),
        "final_error": errors[-1],
        "settled": errors[-1] < 0.01 * target,
    }
