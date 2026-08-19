from __future__ import annotations

import unittest

from distributed_basics.options import options_for_variant
from distributed_basics.timeline import Interval, merge_intervals, summarize_overlap
from distributed_basics.workload import WorkloadConfig, linear_training_flops


class TimelineTests(unittest.TestCase):
    def test_union_does_not_double_count_streams(self) -> None:
        merged = merge_intervals([Interval(0, 10), Interval(4, 12), Interval(20, 25)])
        self.assertEqual(merged, [Interval(0, 12), Interval(20, 25)])

    def test_exposed_communication_is_unoverlapped_union(self) -> None:
        summary = summarize_overlap(
            [Interval(0, 10), Interval(12, 20)],
            [Interval(6, 14), Interval(18, 24)],
        )
        self.assertEqual(summary.compute_total_ns, 18)
        self.assertEqual(summary.communication_total_ns, 14)
        self.assertEqual(summary.compute_communication_overlap_ns, 8)
        self.assertEqual(summary.exposed_communication_ns, 6)
        self.assertAlmostEqual(summary.communication_overlap_fraction or 0.0, 8 / 14)

    def test_no_communication_is_explicitly_unavailable_overlap(self) -> None:
        summary = summarize_overlap([Interval(10, 20)], [])
        self.assertEqual(summary.communication_total_ns, 0)
        self.assertEqual(summary.exposed_communication_ns, 0)
        self.assertIsNone(summary.communication_overlap_fraction)

    def test_invalid_interval_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Interval(4, 4)


class ConfigurationTests(unittest.TestCase):
    def test_variants_have_expected_bucket_policy(self) -> None:
        baseline = options_for_variant("baseline")
        candidate = options_for_variant("optimized")
        self.assertGreater(baseline.bucket_cap_mb, candidate.bucket_cap_mb)
        self.assertFalse(baseline.gradient_as_bucket_view)
        self.assertTrue(candidate.gradient_as_bucket_view)

    def test_flop_convention_is_explicit(self) -> None:
        config = WorkloadConfig(
            local_batch_size=1,
            sequence_length=2,
            hidden_size=4,
            layers=1,
            expansion=2,
        )
        # two linears, FMA=2, backward=2*forward, two ranks
        self.assertEqual(linear_training_flops(config, world_size=2), 1536)


if __name__ == "__main__":
    unittest.main()
