"""Correctness tests for cuda_graph workloads.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/kernel/cuda_graph/tests -v
"""

from __future__ import annotations

import unittest

import torch

from kernel.cuda_graph.workloads import (
    build_graph,
    make_chain_input,
    run_chain_normal,
)


class TestCudaGraph(unittest.TestCase):
    def test_graph_matches_normal(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        n_ops = 8
        scalar = 1.0001

        base = make_chain_input(128, device=device)

        x_normal = base.clone()
        run_chain_normal(x_normal, n_ops=n_ops, scalar=scalar)
        torch.cuda.synchronize(device)

        x_graph = base.clone()
        graph = build_graph(x_graph, n_ops=n_ops, scalar=scalar)
        # build_graph warmup dirties x_graph; reset before replay so both
        # paths start from identical data.
        x_graph.copy_(base)
        graph.replay()
        torch.cuda.synchronize(device)

        self.assertTrue(torch.allclose(x_normal, x_graph, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
