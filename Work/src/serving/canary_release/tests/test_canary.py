"""Correctness tests for canary release.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/serving/canary_release/tests -v
"""

from __future__ import annotations

import random
import unittest

from serving.canary_release.canary import CanaryController, ModelVersion


class TestCanary(unittest.TestCase):
    def test_rolls_back_on_accuracy_regression(self):
        stable = ModelVersion("V1", 0.01, 10.0)
        bad = ModelVersion("V2", 0.08, 10.0)
        ctrl = CanaryController(stable, bad)
        trace = ctrl.run(random.Random(0))
        self.assertTrue(ctrl.rolled_back)
        self.assertEqual(trace[-1]["action"], "rollback")

    def test_rolls_back_on_latency_regression(self):
        stable = ModelVersion("V1", 0.01, 10.0)
        slow = ModelVersion("V2", 0.01, 50.0)
        ctrl = CanaryController(stable, slow)
        ctrl.run(random.Random(1))
        self.assertTrue(ctrl.rolled_back)

    def test_healthy_reaches_100(self):
        stable = ModelVersion("V1", 0.01, 10.0)
        good = ModelVersion("V2", 0.01, 9.0)
        ctrl = CanaryController(stable, good)
        ctrl.run(random.Random(2))
        self.assertFalse(ctrl.rolled_back)
        self.assertEqual(ctrl.share, 1.0)


if __name__ == "__main__":
    unittest.main()
