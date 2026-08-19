"""Correctness tests for robot runtime.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/projects/robot_runtime/tests -v
"""

from __future__ import annotations

import unittest

import torch

from inference.vlm.pipeline import make_image_bytes
from projects.robot_runtime.runtime import NaiveRuntime, OptimizedRuntime


class TestRobotRuntime(unittest.TestCase):
    def test_naive_returns_tensor(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        rt = NaiveRuntime(torch.device("cuda"))
        out = rt.infer(make_image_bytes(seed=0))
        self.assertEqual(out.shape[0], 1)

    def test_optimized_returns_tensor(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        rt = OptimizedRuntime(torch.device("cuda"))
        out = rt.infer(make_image_bytes(seed=0))
        rt.sync()
        self.assertEqual(out.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
