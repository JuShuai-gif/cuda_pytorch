"""Correctness tests for the LLM inference module.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/inference/llm/tests -v
"""

from __future__ import annotations

import unittest

import torch

from inference.llm.model import TransformerLayer
from inference.llm.roofline import decode_metrics, prefill_metrics, sweep


class TestModel(unittest.TestCase):
    def test_decode_matches_prefill_tail(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        d = 256
        layer = TransformerLayer(d).cuda().eval()
        x = torch.randn(1, 8, d, device="cuda")
        with torch.no_grad():
            out_prefill, k, v = layer.prefill(x)
            # Decode the last token against the first 7 as cache; result for
            # position 8 should match the prefill output at position 8.
            k_cache, v_cache = k[:, :7], v[:, :7]
            out_decode, _, _ = layer.decode(x[:, 7:8], k_cache, v_cache)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(out_decode[0], out_prefill[0, 7], atol=1e-4, rtol=1e-4))


class TestRoofline(unittest.TestCase):
    def test_prefill_higher_ai_than_decode(self):
        p = prefill_metrics(32, 4096, 512, 8, 32000)
        q = decode_metrics(32, 4096, 512, 8, 32000)
        self.assertGreater(p.arithmetic_intensity, q.arithmetic_intensity)

    def test_decode_kv_cache_dominates_at_long_seq(self):
        # At long sequence, KV cache bytes exceed weight bytes in decode.
        q = decode_metrics(32, 4096, 8192, 8, 32000)
        self.assertGreater(q.kv_cache_bytes, q.weight_bytes)

    def test_sweep_classifies(self):
        out = sweep(1e15, 2000, seqs=(512, 8192))
        self.assertEqual(len(out), 2)
        # Decode at long seq is memory-bound.
        self.assertEqual(out[-1]["decode_bound"], "memory-bound")


if __name__ == "__main__":
    unittest.main()
