from __future__ import annotations

import unittest

import torch

from gpu_basics.common import benchmark_callable
from gpu_basics.workloads import WORKLOAD_NAMES, compare_outputs, prepare_workload


class WorkloadCorrectnessTest(unittest.TestCase):
    def test_all_cpu_pairs_match(self) -> None:
        torch.manual_seed(7)
        for name in WORKLOAD_NAMES:
            with self.subTest(workload=name):
                prepared = prepare_workload(
                    name,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    vector_elements=257,
                    inner_iterations=3,
                    matrix_size=17,
                    cpu_delay_ms=0.0,
                )
                result = compare_outputs(
                    prepared.baseline(), prepared.optimized(), rtol=1e-4, atol=1e-5
                )
                self.assertTrue(result["passed"], result)

    def test_cpu_timing_has_no_cuda_event(self) -> None:
        result = benchmark_callable(
            lambda: torch.ones(16).relu(),
            device=torch.device("cpu"),
            warmup=1,
            iterations=2,
            repeats=2,
        )
        self.assertIsNone(result.cuda_event)
        self.assertGreaterEqual(result.synchronized_wall.median, 0.0)
        self.assertEqual(result.synchronized_wall.samples, 2)

    def test_structure_mismatch_fails(self) -> None:
        result = compare_outputs((torch.ones(1), 1.0), torch.ones(1), rtol=0, atol=0)
        self.assertFalse(result["passed"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaTimingTest(unittest.TestCase):
    def test_all_cuda_pairs_match(self) -> None:
        device = torch.device("cuda")
        torch.manual_seed(11)
        for name in WORKLOAD_NAMES:
            with self.subTest(workload=name):
                prepared = prepare_workload(
                    name,
                    device=device,
                    dtype=torch.float32,
                    vector_elements=257,
                    inner_iterations=3,
                    matrix_size=17,
                    cpu_delay_ms=0.0,
                )
                result = compare_outputs(
                    prepared.baseline(), prepared.optimized(), rtol=1e-4, atol=1e-5
                )
                self.assertTrue(result["passed"], result)

    def test_cuda_event_is_reported(self) -> None:
        device = torch.device("cuda")
        x = torch.randn(1024, device=device)
        result = benchmark_callable(
            lambda: x + 1,
            device=device,
            warmup=1,
            iterations=2,
            repeats=2,
        )
        self.assertIsNotNone(result.cuda_event)
        assert result.cuda_event is not None
        self.assertGreater(result.cuda_event.median, 0.0)


if __name__ == "__main__":
    unittest.main()
