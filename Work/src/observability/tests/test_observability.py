"""Correctness tests for observability primitives.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/observability/tests -v
"""

from __future__ import annotations

import unittest

from observability.observability import Metrics, StructuredLogger, Tracer


class TestObservability(unittest.TestCase):
    def test_tracer_records_spans(self):
        t = Tracer()
        with t.span("req_1", "a"):
            pass
        with t.span("req_1", "b"):
            pass
        self.assertEqual(len(t.trace("req_1")), 2)

    def test_metrics_percentiles(self):
        m = Metrics()
        for v in [1, 2, 3, 4, 5]:
            m.record("latency", v)
        s = m.summary("latency")
        self.assertEqual(s["count"], 5)
        self.assertEqual(s["p50"], 3)
        self.assertEqual(s["max"], 5)

    def test_logger_carries_request_id(self):
        l = StructuredLogger()
        l.log("req_1", "INFO", "hi", robot_id="r0")
        self.assertEqual(l.entries[0]["request_id"], "req_1")
        self.assertEqual(l.entries[0]["robot_id"], "r0")


if __name__ == "__main__":
    unittest.main()
