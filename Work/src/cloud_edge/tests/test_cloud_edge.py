"""Correctness tests for cloud-edge simulation.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/cloud_edge/tests -v
"""

from __future__ import annotations

import unittest

from cloud_edge.architecture import Cloud, EdgeGateway, Robot, Task
from cloud_edge.simulate import simulate


class TestCloudEdge(unittest.TestCase):
    def test_offline_robot_fails_task(self):
        edge = EdgeGateway()
        robot = Robot("r0", online=False)
        edge.register(robot)
        t = Task("t1", "r0", "x")
        self.assertEqual(edge.forward(t), "offline")
        self.assertEqual(t.status, "failed")

    def test_model_rollout(self):
        edge = EdgeGateway()
        robots = [Robot("r0"), Robot("r1")]
        for r in robots:
            edge.register(r)
        for r in robots:
            r.apply_model("v2")
        self.assertTrue(all(r.model_version == "v2" for r in robots))

    def test_fault_recovery_reschedules(self):
        result = simulate()
        # robot_1 went offline, its task should have been rescheduled and the
        # failure recorded.
        self.assertIn("robot_1:offline", result["failures"])
        self.assertIn("rescheduled", result["events"][-1])


if __name__ == "__main__":
    unittest.main()
