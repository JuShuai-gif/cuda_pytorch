"""Unit tests for common measurement helpers and inference workloads.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/inference/tests -v
"""

from __future__ import annotations

import unittest

import torch

from common.measure import percentile, summarize
from inference.workloads import (
    InferenceConfig,
    flops_per_forward,
    make_input,
    make_model,
    parameter_count,
)


class TestPercentile(unittest.TestCase):
    def test_known_median(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.50), 3.0)

    def test_p99_interpolation(self):
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 0.99), 2.98)

    def test_single_value(self):
        self.assertEqual(percentile([7.0], 0.90), 7.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            percentile([], 0.5)


class TestSummarize(unittest.TestCase):
    def test_basic_stats(self):
        s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(s.samples, 5)
        self.assertAlmostEqual(s.mean, 3.0)
        self.assertEqual(s.minimum, 1.0)
        self.assertEqual(s.maximum, 5.0)


class TestWorkload(unittest.TestCase):
    def _device_dtype(self):
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float32
        return torch.device("cpu"), torch.float32

    def test_forward_shape(self):
        device, dtype = self._device_dtype()
        config = InferenceConfig(hidden=64, layers=2, batch=2, seq_len=3)
        model = make_model(config, device=device, dtype=dtype)
        x = make_input(config, device=device, dtype=dtype)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 3, 64))

    def test_parameter_count_positive(self):
        config = InferenceConfig(hidden=64, layers=2)
        self.assertGreater(parameter_count(config), 0)

    def test_flops_scales_with_batch(self):
        a = flops_per_forward(InferenceConfig(hidden=64, layers=2, batch=1))
        b = flops_per_forward(InferenceConfig(hidden=64, layers=2, batch=4))
        self.assertAlmostEqual(b, 4 * a)


if __name__ == "__main__":
    unittest.main()
