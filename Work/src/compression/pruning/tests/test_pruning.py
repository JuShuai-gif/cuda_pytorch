"""Correctness tests for pruning methods.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/compression/pruning/tests -v
"""

from __future__ import annotations

import unittest

import torch

from compression.pruning.prune import (
    magnitude_prune_unstructured,
    sparsity,
    structured_row_prune,
)


class TestPrune(unittest.TestCase):
    def test_unstructured_sparsity_achieved(self):
        w = torch.randn(512, 512)
        w_pruned, mask = magnitude_prune_unstructured(w, 0.5)
        self.assertAlmostEqual(sparsity(w_pruned), 0.5, delta=0.01)
        self.assertEqual(w_pruned.shape, w.shape)

    def test_structured_shrinks_dimension(self):
        w = torch.randn(512, 512)
        w_pruned, kept = structured_row_prune(w, 0.5)
        self.assertEqual(w_pruned.shape, (256, 512))
        self.assertEqual(len(kept), 256)

    def test_structured_keeps_largest_norm_rows(self):
        w = torch.zeros(4, 8)
        w[0, 0] = 100.0  # row 0 has huge norm
        w[3, 1] = 50.0   # row 3 second
        w_pruned, kept = structured_row_prune(w, 0.5)
        self.assertIn(0, kept.tolist())
        self.assertIn(3, kept.tolist())


if __name__ == "__main__":
    unittest.main()
