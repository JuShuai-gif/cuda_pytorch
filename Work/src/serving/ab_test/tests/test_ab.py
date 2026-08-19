"""Correctness tests for A/B test.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/serving/ab_test/tests -v
"""

from __future__ import annotations

import unittest

from serving.ab_test.ab_test import ModelVariant, robot_task_success


class TestAB(unittest.TestCase):
    def test_slow_accurate_model_loses_on_deadline(self):
        slow = ModelVariant("slow", accuracy=0.99, latency_ms=100.0, failure_rate=0.0)
        fast = ModelVariant("fast", accuracy=0.80, latency_ms=5.0, failure_rate=0.0)
        # 50ms deadline: slow always misses, fast always makes it.
        slow_rate = robot_task_success(slow, 1000, deadline_ms=50.0, seed=0)
        fast_rate = robot_task_success(fast, 1000, deadline_ms=50.0, seed=0)
        self.assertAlmostEqual(slow_rate, 0.0)
        self.assertAlmostEqual(fast_rate, 0.80, delta=0.05)
        # accuracy favors slow, but robot success favors fast.
        self.assertGreater(slow.accuracy, fast.accuracy)
        self.assertGreater(fast_rate, slow_rate)

    def test_failure_rate_reduces_success(self):
        ok = ModelVariant("ok", accuracy=1.0, latency_ms=5.0, failure_rate=0.0)
        flaky = ModelVariant("flaky", accuracy=1.0, latency_ms=5.0, failure_rate=0.5)
        self.assertGreater(robot_task_success(ok, 1000, 50.0, seed=0),
                           robot_task_success(flaky, 1000, 50.0, seed=0))


if __name__ == "__main__":
    unittest.main()
