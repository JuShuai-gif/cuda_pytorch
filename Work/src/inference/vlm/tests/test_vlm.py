"""Correctness tests for the VLM pipeline.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/inference/vlm/tests -v
"""

from __future__ import annotations

import unittest

import torch

from inference.vlm.pipeline import VLM, decode_image, make_image_bytes, preprocess


class TestVLMPipeline(unittest.TestCase):
    def test_image_roundtrip(self):
        data = make_image_bytes()
        img = decode_image(data)
        self.assertEqual(img.size, (224, 224))
        self.assertEqual(img.mode, "RGB")

    def test_preprocess_shape(self):
        img = decode_image(make_image_bytes())
        x = preprocess(img)
        self.assertEqual(tuple(x.shape), (3, 224, 224))

    def test_vlm_forward_shapes(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        model = VLM().cuda().eval()
        x = preprocess(decode_image(make_image_bytes())).unsqueeze(0).cuda()
        vt = model.vision_encode(x)
        # (1, 1 + n_patches, llm_hidden)
        self.assertEqual(vt.shape[0], 1)
        self.assertEqual(vt.shape[1], 1 + (224 // 16) ** 2)
        self.assertEqual(vt.shape[2], 512)
        out = model.llm_forward(vt)
        self.assertEqual(out.shape, vt.shape)


if __name__ == "__main__":
    unittest.main()
