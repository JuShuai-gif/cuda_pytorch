"""Correctness tests for the robot policy inference module.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/robotics/policy_inference/tests -v
"""

from __future__ import annotations

import unittest

import torch

from robotics.policy_inference.pipeline import VLAPolicy, postprocess_action


class TestVLAPolicy(unittest.TestCase):
    def test_action_shape_and_range(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        model = VLAPolicy(action_dim=7).cuda().eval()
        x = torch.randn(1, 3, 224, 224, device="cuda")
        action = model.infer(x)
        self.assertEqual(action.shape, (1, 7))
        clamped = postprocess_action(action)
        self.assertTrue((clamped.abs() <= 1.0).all())


class TestRealtime(unittest.TestCase):
    def test_deadline_miss_rate_bounded(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        from robotics.policy_inference.realtime import run_control_loop
        model = VLAPolicy().cuda().eval()
        r = run_control_loop(model, torch.device("cuda"), cycles=20, deadline_ms=100,
                             inject_cpu_jitter=False)
        self.assertGreaterEqual(r["deadline_miss_rate"], 0.0)
        self.assertLessEqual(r["deadline_miss_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
