"""Canary (gray) release of model versions with monitoring and rollback.

Deploying a new model version straight to 100% of traffic is how accuracy or
latency regressions become outages.  Canary release ramps the new version up
through small traffic fractions (1% -> 10% -> 50% -> 100%) while monitoring it
against the stable version, and rolls back automatically when a regression is
detected.

A ModelVersion has an error rate and a latency; the canary controller routes
traffic, evaluates the new version's metrics against the baseline at each
stage, and either advances or rolls back.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ModelVersion:
    name: str
    error_rate: float    # probability a request fails
    latency_ms: float    # per-request latency

    def serve(self, rng: random.Random) -> tuple[bool, float]:
        """Return (ok, latency_ms)."""
        ok = rng.random() > self.error_rate
        return ok, self.latency_ms


class CanaryController:
    def __init__(self, stable: ModelVersion, candidate: ModelVersion,
                 stages=(0.01, 0.10, 0.50, 1.0),
                 err_tolerance=2.0, lat_tolerance=1.5, eval_requests=10000):
        self.stable = stable
        self.candidate = candidate
        self.stages = stages
        self.err_tolerance = err_tolerance   # candidate/stable error ratio limit
        self.lat_tolerance = lat_tolerance   # candidate/stable latency ratio limit
        self.eval_requests = eval_requests
        self.share = 0.0                     # current candidate traffic share
        self.stage_idx = 0
        self.rolled_back = False

    def run(self, rng: random.Random) -> list[dict]:
        """Advance through stages, rolling back on regression."""
        trace = []
        self.share = self.stages[0]
        while not self.rolled_back and self.stage_idx < len(self.stages):
            self.share = self.stages[self.stage_idx]
            metrics = self._evaluate(rng)
            regress = self._is_regression(metrics)
            trace.append({
                "stage_share": self.share,
                "stable_err": metrics["stable_err"],
                "candidate_err": metrics["candidate_err"],
                "stable_lat_ms": metrics["stable_lat_ms"],
                "candidate_lat_ms": metrics["candidate_lat_ms"],
                "action": "rollback" if regress else "advance",
            })
            if regress:
                self.rolled_back = True
                self.share = 0.0
                break
            self.stage_idx += 1
        return trace

    def _evaluate(self, rng: random.Random) -> dict:
        """Serve eval_requests at the current share, collecting per-version stats."""
        n_candidate = int(self.eval_requests * self.share)
        n_stable = self.eval_requests - n_candidate

        def measure(v, n):
            errs = 0
            lat = 0.0
            for _ in range(n):
                ok, l = v.serve(rng)
                if not ok:
                    errs += 1
                lat += l
            return errs / n if n else 0.0, lat / n if n else 0.0

        s_err, s_lat = measure(self.stable, n_stable)
        c_err, c_lat = measure(self.candidate, n_candidate)
        return {"stable_err": s_err, "candidate_err": c_err,
                "stable_lat_ms": s_lat, "candidate_lat_ms": c_lat}

    def _is_regression(self, m: dict) -> bool:
        # Error regression: candidate error rate much worse than stable.
        if m["stable_err"] > 0 and m["candidate_err"] / m["stable_err"] > self.err_tolerance:
            return True
        # Latency regression.
        if m["stable_lat_ms"] > 0 and m["candidate_lat_ms"] / m["stable_lat_ms"] > self.lat_tolerance:
            return True
        return False
