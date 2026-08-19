"""Correctness tests for cuda_async workloads.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/kernel/cuda_async/tests -v
"""

from __future__ import annotations

import unittest

import torch

from kernel.cuda_async.workloads import benchmark_h2d, benchmark_streams


class TestH2D(unittest.TestCase):
    def test_h2d_copies_correctly(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        n = 1 << 20
        src = torch.randn(n, dtype=torch.float32, pin_memory=True)
        dst = torch.empty(n, dtype=torch.float32, device=device)
        dst.copy_(src)
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(src, dst.cpu()))

    def test_h2d_benchmark_reports(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        r = benchmark_h2d(4 * 1024 * 1024, device=torch.device("cuda"),
                          pinned=True, non_blocking=True, iterations=5)
        self.assertGreater(r.bytes, 0)
        self.assertGreaterEqual(r.event_ms, 0.0)


class TestStreams(unittest.TestCase):
    def test_multi_stream_runs(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        result = benchmark_streams(
            device=torch.device("cuda"),
            n_streams=2,
            mat_size=128,
            work_per_stream=2,
            warmup=1,
            iterations=3,
        )
        self.assertGreater(result["single_stream_ms"], 0)
        self.assertGreater(result["multi_stream_ms"], 0)


if __name__ == "__main__":
    unittest.main()
