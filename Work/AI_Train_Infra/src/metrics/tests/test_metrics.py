from __future__ import annotations

import math
import unittest

from metrics.metrics import (
    analyze_timeline,
    format_bytes,
    interval_union_duration,
    linear_forward_flops,
    percentile,
    scaling_from_throughput,
    summarize_latencies,
    throughput_per_second,
    training_state_memory,
    transformer_parameter_flop_estimate,
    utilization_from_step,
)


class LatencyTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_latency_units_and_throughput(self) -> None:
        summary = summarize_latencies([0.001, 0.002, 0.003])
        self.assertEqual(summary.p50_ms, 2.0)
        self.assertEqual(summary.count, 3)
        self.assertEqual(throughput_per_second(16, 0.002), 8_000.0)

    def test_invalid_latency_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_latencies([0.0])


class FlopConventionTests(unittest.TestCase):
    def test_dense_transformer_without_recompute_is_six_pt(self) -> None:
        estimate = transformer_parameter_flop_estimate(100, 10)
        self.assertEqual(estimate.forward_flops, 2_000.0)
        self.assertEqual(estimate.model_flops, 6_000.0)
        self.assertEqual(estimate.hardware_flops, 6_000.0)

    def test_full_activation_recompute_changes_hfu_not_mfu_numerator(self) -> None:
        estimate = transformer_parameter_flop_estimate(
            100, 10, recompute_forward_fraction=1.0
        )
        self.assertEqual(estimate.model_flops, 6_000.0)
        self.assertEqual(estimate.hardware_flops, 8_000.0)

    def test_linear_count(self) -> None:
        self.assertEqual(linear_forward_flops(2, ((3, 5), (5, 3))), 120.0)

    def test_missing_peak_produces_null_utilization(self) -> None:
        estimate = transformer_parameter_flop_estimate(100, 10)
        report = utilization_from_step(
            estimate, 2.0, peak_flops_per_device_per_second=None
        )
        self.assertIsNone(report.mfu)
        self.assertIsNone(report.hfu)
        self.assertIn("unavailable", report.status)

    def test_explicit_peak_and_device_count(self) -> None:
        estimate = transformer_parameter_flop_estimate(
            100, 10, recompute_forward_fraction=1.0
        )
        report = utilization_from_step(
            estimate,
            1.0,
            peak_flops_per_device_per_second=10_000.0,
            device_count=2,
        )
        self.assertTrue(math.isclose(report.mfu or 0.0, 0.3))
        self.assertTrue(math.isclose(report.hfu or 0.0, 0.4))


class MemoryTests(unittest.TestCase):
    def test_explicit_mixed_precision_adam_example(self) -> None:
        memory = training_state_memory(
            1_000,
            parameter_bytes_per_parameter=2,
            gradient_bytes_per_parameter=2,
            optimizer_state_bytes_per_parameter=8,
            master_parameter_bytes_per_parameter=4,
            activation_bytes=100,
            temporary_bytes=200,
        )
        self.assertEqual(memory.total_live_bytes, 16_300)
        self.assertEqual(format_bytes(1024), "1.00 KiB")


class TimelineTests(unittest.TestCase):
    def test_union_does_not_double_count_streams(self) -> None:
        self.assertEqual(interval_union_duration([(0, 3), (1, 2), (3, 4)]), 4.0)

    def test_overlap_bubble_and_unhidden_communication(self) -> None:
        report = analyze_timeline(
            0,
            10,
            compute_intervals=[(1, 5), (4, 7)],
            communication_intervals=[(6, 9)],
        )
        self.assertEqual(report.compute_active_union_s, 6.0)
        self.assertEqual(report.communication_active_union_s, 3.0)
        self.assertEqual(report.compute_communication_overlap_s, 1.0)
        self.assertEqual(report.any_gpu_active_union_s, 8.0)
        self.assertEqual(report.gpu_bubble_s, 2.0)
        self.assertEqual(report.unhidden_communication_s, 2.0)

    def test_out_of_window_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_timeline(
                0,
                1,
                compute_intervals=[(-1, 0.5)],
                communication_intervals=[],
            )


class ScalingTests(unittest.TestCase):
    def test_throughput_efficiency(self) -> None:
        report = scaling_from_throughput(100.0, 1, 720.0, 8)
        self.assertAlmostEqual(report.throughput_speedup, 7.2)
        self.assertAlmostEqual(report.scaling_efficiency, 0.9)


if __name__ == "__main__":
    unittest.main()
