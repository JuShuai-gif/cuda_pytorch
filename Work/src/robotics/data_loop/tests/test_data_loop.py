"""Correctness tests for the data loop.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/robotics/data_loop/tests -v
"""

from __future__ import annotations

import unittest

from robotics.data_loop.data_loop import FAILURE_TYPES, DataLoop, Model


class TestDataLoop(unittest.TestCase):
    def test_failure_mining_collects_failures(self):
        loop = DataLoop(Model({ft: 0.0 for ft in FAILURE_TYPES}))  # always fails
        loop.run_robot_fleet(100)
        counts = loop.mine_failures()
        self.assertEqual(sum(counts.values()), 100)

    def test_training_improves_success_rates(self):
        loop = DataLoop(Model({ft: 0.5 for ft in FAILURE_TYPES}))
        loop.run_robot_fleet(300)
        loop.mine_failures()
        old = dict(loop.model.success_rates)
        loop.train()
        for ft in FAILURE_TYPES:
            self.assertGreater(loop.model.success_rates[ft], old[ft])

    def test_flywheel_reduces_failure_rate(self):
        loop = DataLoop(Model({ft: 0.5 for ft in FAILURE_TYPES}))
        rates = []
        for _ in range(3):
            loop.run_robot_fleet(300)
            loop.mine_failures()
            rates.append(loop.failure_rate())
            loop.train()
        self.assertLess(rates[-1], rates[0])


if __name__ == "__main__":
    unittest.main()
