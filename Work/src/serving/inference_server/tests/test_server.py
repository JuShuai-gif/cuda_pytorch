"""Correctness tests for the inference server.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/serving/inference_server/tests -v
"""

from __future__ import annotations

import unittest

import torch

from serving.inference_server.server import InferenceServer, make_model


class TestInferenceServer(unittest.TestCase):
    def test_infer_returns_correct_shape(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        server = InferenceServer(make_model(hidden=64), torch.device("cuda"),
                                 strategy="no_batch")
        x = torch.randn(64)
        out = server.infer(x)
        self.assertEqual(out.shape, (64,))
        server.stop()

    def test_batching_matches_no_batch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        model = make_model(hidden=64)
        server = InferenceServer(model, torch.device("cuda"), strategy="dynamic",
                                 max_batch=4, max_wait=0.01)
        x = torch.randn(64)
        out = server.infer(x)
        with torch.no_grad():
            expected = model(x.unsqueeze(0).cuda())[0].cpu()
        self.assertTrue(torch.allclose(out, expected, atol=1e-4))
        server.stop()

    def test_static_batch_waits_then_times_out(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        # static strategy waits for a full batch; a single request never fills
        # it, so a short client timeout must fire.
        server = InferenceServer(make_model(hidden=64), torch.device("cuda"),
                                 strategy="static", max_batch=8)
        with self.assertRaises(TimeoutError):
            server.infer(torch.randn(64), timeout=0.05)
        server.stop()


if __name__ == "__main__":
    unittest.main()
