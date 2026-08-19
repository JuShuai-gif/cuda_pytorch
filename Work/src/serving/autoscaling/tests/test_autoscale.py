"""Correctness tests for the autoscaling simulation.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/serving/autoscaling/tests -v
"""

from __future__ import annotations

import unittest

from serving.autoscaling.autoscaler import simulate


class TestAutoscaling(unittest.TestCase):
    def test_queue_scales_up_under_spike(self):
        r = simulate("queue")
        self.assertGreater(r["final_workers"], 1)
        # queue-based scaling reacts, but with a lag -> some shed is expected;
        # far less than the cpu metric which never scales.
        self.assertLess(r["total_dropped"], 10000)

    def test_latency_scales_up_under_spike(self):
        r = simulate("latency")
        self.assertGreater(r["final_workers"], 1)

    def test_cpu_does_not_scale(self):
        # CPU metric stays decorrelated -> never scales past 1 worker.
        r = simulate("cpu")
        self.assertEqual(r["final_workers"], 1)
        # and therefore drops a lot under the spike.
        self.assertGreater(r["total_dropped"], 0)


if __name__ == "__main__":
    unittest.main()
