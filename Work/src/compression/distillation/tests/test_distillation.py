"""Correctness tests for distillation.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/compression/distillation/tests -v
"""

from __future__ import annotations

import unittest

import torch

from compression.distillation.distill import MLP, accuracy, run_distillation, train_model


class TestDistillation(unittest.TestCase):
    def test_model_shapes(self):
        m = MLP(10, 32, 2, 4)
        out = m(torch.randn(8, 10))
        self.assertEqual(out.shape, (8, 4))

    def test_teacher_beats_student_direct(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        r = run_distillation(torch.device("cuda"), seed=0)
        self.assertGreater(r["teacher_acc"], r["student_direct_acc"])

    def test_distillation_helps(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        r = run_distillation(torch.device("cuda"), seed=0)
        # Distilled student should beat the directly-trained student.
        self.assertGreater(r["student_distilled_acc"], r["student_direct_acc"])


if __name__ == "__main__":
    unittest.main()
