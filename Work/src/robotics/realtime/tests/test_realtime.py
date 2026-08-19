"""Correctness tests for realtime control simulation.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/robotics/realtime/tests -v
"""

from __future__ import annotations

import unittest

from robotics.realtime.control import simulate


class TestRealtime(unittest.TestCase):
    def test_constant_low_settles(self):
        r = simulate(1.0, [1] * 2000, settle_start=400)
        self.assertTrue(r["settled"])

    def test_jitter_worse_than_constant_same_mean(self):
        n = 2000
        constant = simulate(1.0, [1] * n, settle_start=400)  # 10ms constant
        post = [1] * (n - 400)
        for i in range(40, len(post), 40):
            for kk in range(min(5, len(post) - i)):
                post[i + kk] = 20
        jitter = simulate(1.0, [1] * 400 + post, settle_start=400)
        # jitter's worst-case tracking error is much larger than constant 10ms.
        self.assertGreater(jitter["max_error"], constant["max_error"] * 3)


if __name__ == "__main__":
    unittest.main()
