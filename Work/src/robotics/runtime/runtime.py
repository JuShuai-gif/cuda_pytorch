"""Robot runtime data flow: sensors -> sync -> model -> action -> controller.

The core difficulty of a robot runtime is that sensors run at *different
rates* (camera 30Hz, IMU 200Hz, joints 100Hz) but the model needs one
consistent observation.  This module models that pipeline and the two classic
synchronization strategies:

  latest  - use each sensor's most recent reading (never waits, but the
            observations are from slightly different instants)
  exact   - wait until all sensors have a reading at the same timestamp

Then a control loop runs sensor -> sync -> model -> action -> controller and
reports how many control cycles each strategy completes and how fresh the
observations were.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sensor:
    name: str
    rate_hz: float

    def sample_at(self, t: float) -> float:
        """The latest timestamp <= t at which this sensor produced a reading."""
        period = 1.0 / self.rate_hz
        return (t // period) * period


class RobotRuntime:
    def __init__(self, sensors: list[Sensor]):
        self.sensors = sensors
        self.latest: dict[str, float] = {s.name: -1.0 for s in sensors}

    def sync_latest(self, t: float) -> dict[str, float]:
        """Use the most recent reading of each sensor (no waiting)."""
        for s in self.sensors:
            self.latest[s.name] = s.sample_at(t)
        return dict(self.latest)

    def sync_exact(self, t: float) -> dict[str, float] | None:
        """Wait for all sensors to share a timestamp; else return None."""
        stamps = {s.name: s.sample_at(t) for s in self.sensors}
        if len(set(stamps.values())) == 1:
            return stamps
        return None

    def control_loop(self, duration: float, control_hz: float,
                     strategy: str = "latest") -> dict:
        dt = 1.0 / control_hz
        t = 0.0
        cycles = 0
        exact_skips = 0
        staleness: list[float] = []  # max age of an observation at use time

        while t < duration:
            if strategy == "exact":
                obs = self.sync_exact(t)
                if obs is None:
                    exact_skips += 1
                    t += dt
                    continue
                stale = max(t - v for v in obs.values())
            else:
                obs = self.sync_latest(t)
                stale = max(t - v for v in obs.values())
            staleness.append(stale)
            cycles += 1
            t += dt

        return {
            "strategy": strategy,
            "control_hz": control_hz,
            "cycles": cycles,
            "exact_skips": exact_skips,
            "mean_staleness_s": sum(staleness) / len(staleness) if staleness else 0,
            "max_staleness_s": max(staleness) if staleness else 0,
        }
