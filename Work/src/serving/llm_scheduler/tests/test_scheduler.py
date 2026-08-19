"""Correctness tests for the LLM scheduler simulation.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/serving/llm_scheduler/tests -v
"""

from __future__ import annotations

import unittest

from serving.llm_scheduler.paged_kv import simulate_contiguous, simulate_paged
from serving.llm_scheduler.scheduler import (
    gen_requests,
    simulate_continuous,
    simulate_static,
)


class TestScheduler(unittest.TestCase):
    def test_continuous_serves_all(self):
        reqs = gen_requests(100, rate=2.0, seed=0)
        r = simulate_continuous(reqs, max_batch=32)
        self.assertEqual(r["n_requests"], 100)

    def test_static_serves_all(self):
        reqs = gen_requests(100, rate=2.0, seed=0)
        r = simulate_static(reqs, batch_size=8)
        self.assertEqual(r["n_requests"], 100)

    def test_continuous_beats_static_throughput(self):
        reqs = gen_requests(200, rate=2.0, seed=1)
        s = simulate_static(list(reqs), batch_size=8)
        c = simulate_continuous(list(reqs), max_batch=32)
        self.assertGreater(c["throughput_tokens_per_s"], s["throughput_tokens_per_s"])


class TestPagedKV(unittest.TestCase):
    def test_paged_serves_more_than_contiguous(self):
        reqs = gen_requests(200, rate=2.0, seed=0)
        c = simulate_contiguous(reqs, 16, 4096, max_len=512)
        p = simulate_paged(reqs, 16, 4096)
        self.assertGreater(p["served"], c["served"])

    def test_paged_waste_lower(self):
        reqs = gen_requests(200, rate=2.0, seed=0)
        c = simulate_contiguous(reqs, 16, 4096, max_len=512)
        p = simulate_paged(reqs, 16, 4096)
        self.assertLess(p["waste_ratio"], c["waste_ratio"])


if __name__ == "__main__":
    unittest.main()
