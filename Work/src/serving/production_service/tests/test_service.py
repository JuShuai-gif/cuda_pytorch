"""Correctness tests for reliability primitives.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/serving/production_service/tests -v
"""

from __future__ import annotations

import unittest

from serving.production_service import CircuitBreaker, LoadShedder, TokenBucket


class TestTokenBucket(unittest.TestCase):
    def test_capacity_burst_then_empty(self):
        tb = TokenBucket(rate=1000, capacity=3)
        self.assertTrue(tb.allow())
        self.assertTrue(tb.allow())
        self.assertTrue(tb.allow())
        self.assertFalse(tb.allow())  # bucket empty


class TestCircuitBreaker(unittest.TestCase):
    def test_trips_after_failures(self):
        cb = CircuitBreaker(fail_threshold=2, reset_timeout=10.0)
        self.assertTrue(cb.allow())
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, "open")
        self.assertFalse(cb.allow())

    def test_recovers_on_success(self):
        cb = CircuitBreaker(fail_threshold=1, reset_timeout=10.0)
        cb.record_failure()
        self.assertEqual(cb.state, "open")
        cb.record_success()
        self.assertEqual(cb.state, "closed")


class TestLoadShedder(unittest.TestCase):
    def test_drops_when_full(self):
        ls = LoadShedder(capacity=2)
        self.assertTrue(ls.try_admit("a"))
        self.assertTrue(ls.try_admit("b"))
        self.assertFalse(ls.try_admit("c"))
        self.assertEqual(ls.dropped, 1)


if __name__ == "__main__":
    unittest.main()
