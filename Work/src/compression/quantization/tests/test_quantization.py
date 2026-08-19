"""Correctness tests for quantization fundamentals.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/compression/quantization/tests -v
"""

from __future__ import annotations

import unittest

import torch

from compression.quantization.awq import awq_experiment, make_outlier_weight
from compression.quantization.quantize import (
    dequantize_symmetric,
    granularity_error,
    make_outlier_weight,
    quantize_symmetric,
)
from compression.quantization.ptq import quantize_weight_per_channel
from compression.quantization.smoothquant import (
    make_outlier_activation,
    smooth_scale,
    smoothquant_experiment,
)


class TestQuantize(unittest.TestCase):
    def test_roundtrip_exact_for_small(self):
        # Values already on the int8 grid round-trip exactly.
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        scale = torch.tensor(1.0)
        q = quantize_symmetric(x, scale)
        x_hat = dequantize_symmetric(q, scale)
        self.assertTrue(torch.equal(x_hat, x))

    def test_per_channel_scale_shape(self):
        w = torch.randn(64, 32)
        q, scale = quantize_weight_per_channel(w)
        self.assertEqual(q.shape, w.shape)
        self.assertEqual(tuple(scale.shape), (64, 1))
        self.assertEqual(q.dtype, torch.int8)


class TestGranularity(unittest.TestCase):
    def test_finer_granularity_reduces_error(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        w = make_outlier_weight()(torch.device("cuda"), torch.float32)
        results = {r.granularity: r for r in granularity_error(w)}
        # per-channel must beat per-tensor on the column-level outlier weight:
        # the outlier columns get their own scale instead of dominating a
        # single tensor-wide scale.
        self.assertLess(results["per-channel"].max_abs_err,
                        results["per-tensor"].max_abs_err)
        # per-token (row-wise) also beats per-tensor here.
        self.assertLess(results["per-token"].mse, results["per-tensor"].mse)
        # All granularities produce bounded error (quantization is sane).
        for r in results.values():
            self.assertGreater(r.max_abs_err, 0.0)


class TestSmoothQuant(unittest.TestCase):
    def test_migration_preserves_product(self):
        # X @ W == (X * s) @ (W / s), so the migration does not change the result.
        x = torch.randn(128, 64, dtype=torch.float32)
        w = torch.randn(64, 64, dtype=torch.float32)
        s = smooth_scale(x, w)
        x_hat = x * s[None, :]
        w_hat = w / s[:, None]
        self.assertTrue(torch.allclose(x @ w, x_hat @ w_hat, rtol=1e-4, atol=1e-4))

    def test_smoothquant_reduces_error(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        d = torch.device("cuda")
        x = make_outlier_activation(4096, 1024)(d, torch.float32)
        w = torch.randn(1024, 1024, device=d, dtype=torch.float32) * 0.05
        r = smoothquant_experiment(x, w)
        self.assertLess(r["smooth_max_abs_err"], r["direct_max_abs_err"])


class TestAWQ(unittest.TestCase):
    def test_awq_reduces_weighted_error(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        d = torch.device("cuda")
        x = torch.randn(4096, 1024, device=d, dtype=torch.float32)
        w = make_outlier_weight()(d, torch.float32)
        r = awq_experiment(x, w)
        self.assertLess(r["awq_weighted_error"], r["naive_weighted_error"])


if __name__ == "__main__":
    unittest.main()
