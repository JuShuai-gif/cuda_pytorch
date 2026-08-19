"""Correctness tests for cloud-edge infra.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/projects/cloud_edge_infra/tests -v
"""

from __future__ import annotations

import unittest

from cloud_edge.architecture import Task
from projects.cloud_edge_infra.benchmark import CloudEdgeInfra


class TestCloudEdgeInfra(unittest.TestCase):
    def test_ota_updates_all_robots(self):
        infra = CloudEdgeInfra(n_robots=3)
        infra.publish_and_ota("v2", b"w")
        self.assertTrue(all(r.model_version == "v2" for r in infra.robots))

    def test_task_records_metrics(self):
        infra = CloudEdgeInfra(n_robots=2)
        infra.run_task(Task("t1", "robot_0", "x"))
        self.assertEqual(infra.metrics.summary("task_success")["count"], 1)

    def test_fault_recovery_records_failure(self):
        infra = CloudEdgeInfra(n_robots=2)
        result = infra.inject_offline("robot_1")
        self.assertEqual(result, "offline")
        self.assertIn("robot_1:offline", infra.cloud.failures)


if __name__ == "__main__":
    unittest.main()
