from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from workloads import CASES, WorkloadConfig, make_inputs, run_workload  # noqa: E402


class WorkloadCorrectnessTest(unittest.TestCase):
    def assert_variants_match(self, case: str, device: torch.device) -> None:
        config = WorkloadConfig(numel=257, matrix_size=17, repeats=4, cpu_gap_ms=0.0)
        inputs = make_inputs(case, config, device)
        baseline = run_workload(case, "baseline", inputs, config)
        optimized = run_workload(case, "optimized", inputs, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        torch.testing.assert_close(baseline, optimized, rtol=2e-5, atol=2e-5)

    def test_all_cpu_variants_match(self) -> None:
        for case in CASES:
            with self.subTest(case=case):
                self.assert_variants_match(case, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_all_cuda_variants_match(self) -> None:
        for case in CASES:
            with self.subTest(case=case):
                self.assert_variants_match(case, torch.device("cuda"))

    def test_validation_rejects_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            WorkloadConfig(numel=0).validate()

    def test_unknown_case_is_rejected(self) -> None:
        config = WorkloadConfig(numel=8, matrix_size=4, repeats=2, cpu_gap_ms=0)
        with self.assertRaises(ValueError):
            make_inputs("not-a-case", config, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
