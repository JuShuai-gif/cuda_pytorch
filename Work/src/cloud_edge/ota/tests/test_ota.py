"""Correctness tests for OTA.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/cloud_edge/ota/tests -v
"""

from __future__ import annotations

import unittest

from cloud_edge.ota.ota import ModelArtifact, ModelRegistry, RobotOTA


class TestOTA(unittest.TestCase):
    def setUp(self):
        self.v2 = ModelArtifact.make("v2", b"model_v2_weights")
        self.registry = ModelRegistry([self.v2])

    def _robot(self):
        return RobotOTA("v1", b"model_v1_weights", disk_capacity=1024)

    def test_healthy_upgrade(self):
        r = self._robot()
        self.assertEqual(r.update(self.registry, "v2"), "ok")
        self.assertEqual(r.current_version, "v2")

    def test_corruption_rejected(self):
        r = self._robot()
        self.assertEqual(r.update(self.registry, "v2", corrupt=True), "corrupted")
        self.assertEqual(r.current_version, "v1")  # still on old version

    def test_load_failure_rolls_back(self):
        r = self._robot()
        self.assertEqual(r.update(self.registry, "v2", load_fails=True),
                         "health_check_failed")
        self.assertEqual(r.current_version, "v1")

    def test_disk_full_keeps_old(self):
        r = self._robot()
        self.assertEqual(r.update(self.registry, "v2", disk_too_small=True),
                         "disk_full")
        self.assertEqual(r.current_version, "v1")


if __name__ == "__main__":
    unittest.main()
