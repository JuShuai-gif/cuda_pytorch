"""A/B test: comparing model variants on business metrics, not just accuracy.

The lesson: a model experiment must not be decided on accuracy alone.  In a
robot control task, a *slower but more accurate* model can have a *lower* task
success rate than a faster, slightly less accurate one, because each action
that misses the control deadline fails regardless of how accurate it was.

A ModelVariant carries the raw metrics (accuracy, latency, failure rate); the
task simulator turns them into the business metric (task success rate) by
applying a per-step deadline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ModelVariant:
    name: str
    accuracy: float       # per-step action correctness probability
    latency_ms: float     # per-step inference latency
    failure_rate: float   # system failure probability (crash/OOM)


def robot_task_success(model: ModelVariant, n_steps: int, deadline_ms: float,
                       seed: int = 0) -> float:
    """Simulate a pick/place task: each step must be on-time AND correct.

    A step succeeds only if (a) no system failure, (b) latency within the
    control deadline, and (c) the action is correct.  This is where a slow but
    accurate model loses: it misses the deadline on every step.
    """
    rng = random.Random(seed)
    success = 0
    attempts = 0
    for _ in range(n_steps):
        attempts += 1
        if rng.random() < model.failure_rate:
            continue                       # system failure
        if model.latency_ms > deadline_ms:
            continue                       # missed control deadline
        if rng.random() < model.accuracy:
            success += 1                   # on-time and correct
    return success / attempts


def summarize(model: ModelVariant, n_steps: int, deadline_ms: float, seed: int = 0) -> dict:
    return {
        "name": model.name,
        "accuracy": model.accuracy,
        "latency_ms": model.latency_ms,
        "failure_rate": model.failure_rate,
        "robot_success_rate": robot_task_success(model, n_steps, deadline_ms, seed),
    }
