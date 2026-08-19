"""The robot data loop (flywheel).

    Robot -> runtime data -> upload -> storage -> analysis -> failure mining
          -> dataset -> training -> model -> deploy -> Robot

The whole point is *failure mining*: from the stream of runtime data, pull out
the failed cases, understand why they failed, and turn them into training data
so the next model does not repeat the same failures.  This is the loop that
makes a robot fleet "get better with use".

A RuntimeData record carries sensor/action/task-result/error/metrics.  The
model here is a simple per-failure-type success probability: each failure mode
(lighting, occlusion, novel object) has a success rate that training improves.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


FAILURE_TYPES = ["low_light", "occlusion", "novel_object"]


@dataclass
class RuntimeData:
    robot_id: str
    task: str
    failure_type: str   # one of FAILURE_TYPES, or "none" if no failure mode present
    success: bool
    error: str = ""
    sensor: dict = field(default_factory=dict)


class Model:
    """A model with a per-failure-type success probability."""

    def __init__(self, success_rates: dict[str, float], seed: int = 0):
        self.success_rates = dict(success_rates)
        self.rng = random.Random(seed)

    def predict(self, failure_type: str) -> bool:
        return self.rng.random() < self.success_rates.get(failure_type, 0.9)


class DataLoop:
    def __init__(self, model: Model):
        self.model = model
        self.storage: list[RuntimeData] = []
        self.dataset: list[RuntimeData] = []

    # --- pipeline stages -------------------------------------------------
    def run_robot_fleet(self, n_tasks: int) -> list[RuntimeData]:
        """Simulate a fleet running tasks and uploading runtime data.

        Each round resets storage (the current round's runtime data) but keeps
        the dataset (training data accumulates across rounds).
        """
        self.storage = []
        records = []
        for i in range(n_tasks):
            ftype = random.Random(i).choice(FAILURE_TYPES)
            ok = self.model.predict(ftype)
            records.append(RuntimeData(
                robot_id=f"robot_{i % 10}", task=f"task_{i}",
                failure_type=ftype, success=ok,
                error="" if ok else f"{ftype}_failure",
            ))
        self.storage.extend(records)
        return records

    def mine_failures(self) -> dict[str, int]:
        """Pull failed cases out of storage and count them by failure type."""
        failures = [r for r in self.storage if not r.success]
        counts = {ft: 0 for ft in FAILURE_TYPES}
        for r in failures:
            counts[r.failure_type] += 1
        self.dataset.extend(failures)  # failures become training data
        return counts

    def train(self, improvement: float = 0.15) -> Model:
        """Train on mined failures: improve success rate for each seen type."""
        new_rates = dict(self.model.success_rates)
        counts = {ft: 0 for ft in FAILURE_TYPES}
        for r in self.dataset:
            if not r.success:
                counts[r.failure_type] += 1
        for ft, n in counts.items():
            if n > 0:
                new_rates[ft] = min(0.99, new_rates[ft] + improvement)
        self.model = Model(new_rates, seed=self.model.rng.randint(0, 1 << 30))
        return self.model

    def failure_rate(self) -> float:
        if not self.storage:
            return 0.0
        return sum(1 for r in self.storage if not r.success) / len(self.storage)
