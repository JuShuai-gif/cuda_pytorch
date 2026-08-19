"""Correctness tests: fused output matches the eager reference.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/kernel/fusion/tests -v
"""

from __future__ import annotations

import unittest

import torch

import kernel.triton  # noqa: F401
from kernel.fusion.benchmark import build_cases


class TestFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def test_all_cases_fp16(self):
        device = torch.device("cuda")
        for case in build_cases(device, torch.float16):
            inputs = case.inputs(device, torch.float16)
            with torch.no_grad():
                expected = case.unfused(*inputs, **case.kwargs)
                actual = case.fused(*inputs, **case.kwargs)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.allclose(actual, expected, atol=1e-1, rtol=1e-1),
                f"{case.name} fused/unfused mismatch",
            )


if __name__ == "__main__":
    unittest.main()
