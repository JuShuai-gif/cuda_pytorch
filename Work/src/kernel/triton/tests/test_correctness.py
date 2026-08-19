"""Correctness tests for Triton operators against PyTorch references.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/kernel/triton/tests -v
"""

from __future__ import annotations

import unittest

import torch

import kernel.triton  # noqa: F401
from kernel.triton.operators import (
    attention,
    gemm,
    layernorm,
    quantize,
    reduction,
    rmsnorm,
    softmax,
    vector_add,
)


class TestTritonOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def _check(self, op, dtype, atol=1e-3, rtol=1e-2):
        ok, diff = op.check(torch.device("cuda"), dtype, atol=atol, rtol=rtol)
        self.assertTrue(ok, f"{op.name} mismatch, max_abs_diff={diff}")

    def test_vector_add(self):
        self._check(vector_add.build(), torch.float32)

    def test_reduction(self):
        self._check(reduction.build(), torch.float32)

    def test_softmax(self):
        self._check(softmax.build(), torch.float32)

    def test_layernorm(self):
        self._check(layernorm.build(), torch.float32)

    def test_rmsnorm(self):
        self._check(rmsnorm.build(), torch.float32)

    def test_gemm_fp16(self):
        self._check(gemm.build(), torch.float16, atol=1e-1, rtol=1e-1)

    def test_flash_attention_fp16(self):
        self._check(attention.build(), torch.float16, atol=1e-1, rtol=1e-1)

    def test_int8_quant_dequant(self):
        self._check(quantize.build(), torch.float32, atol=0.03, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
