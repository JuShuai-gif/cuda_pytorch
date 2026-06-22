"""
Inference metrics collection and analysis.

Tracks request-level metrics (TTFT, TPOT, latency) and system-level
metrics (QPS, GPU utilization, memory usage) with percentile tracking
and timeline visualization for request lifecycles.

Key metrics:
  - TTFT (Time To First Token): User-perceived responsiveness, target <500ms
  - TPOT (Time Per Output Token): Decode latency per token
  - QPS (Queries Per Second): System throughput
  - GPU utilization: Compute and memory bandwidth usage
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Optional, Sequence


class MetricPhase(Enum):
    """Phase of request lifecycle for timeline tracking."""

    QUEUED = auto()
    PREFILL = auto()
    DECODE = auto()
    FINISHED = auto()


@dataclass
class RequestMetrics:
    """Per-request inference metrics.

    Attributes:
        request_id: Unique request identifier.
        prompt_tokens: Number of tokens in the prompt.
        output_tokens: Number of generated tokens.
        ttft_s: Time to first token in seconds.
        tpot_s: Average time per output token during decode (seconds).
        prefill_time_s: Time spent in prefill phase.
        decode_time_s: Time spent in decode phase.
        total_latency_s: End-to-end request latency.
        peak_memory_mb: Peak GPU memory allocated during request.
        finish_reason: Why the request finished ("stop", "length", "abort").
        timestamps: Per-phase timestamps for timeline visualization.
    """

    request_id: int
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    total_latency_s: float = 0.0
    peak_memory_mb: float = 0.0
    finish_reason: str = "unknown"
    timestamps: dict[MetricPhase, float] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-level inference metrics for a time window.

    Attributes:
        qps: Queries per second processed.
        total_requests: Total requests completed in window.
        total_input_tokens: Sum of prompt tokens across requests.
        total_output_tokens: Sum of generated tokens across requests.
        avg_ttft_s: Average time to first token.
        p50_ttft_s: Median time to first token.
        p95_ttft_s: 95th percentile time to first token.
        p99_ttft_s: 99th percentile time to first token.
        avg_tpot_s: Average time per output token.
        throughput_tokens_per_s: Output tokens per second.
        gpu_utilization_pct: Estimated GPU compute utilization.
        memory_usage_mb: Current GPU memory usage.
        memory_capacity_mb: Total GPU memory capacity.
        active_requests: Number of requests currently in-flight.
    """

    qps: float = 0.0
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_ttft_s: float = 0.0
    p50_ttft_s: float = 0.0
    p95_ttft_s: float = 0.0
    p99_ttft_s: float = 0.0
    avg_tpot_s: float = 0.0
    throughput_tokens_per_s: float = 0.0
    gpu_utilization_pct: float = 0.0
    memory_usage_mb: float = 0.0
    memory_capacity_mb: float = 0.0
    active_requests: int = 0


class MetricsCollector:
    """Collects and aggregates inference metrics.

    Maintains per-request metrics and computes system-level aggregates
    with percentile tracking using reservoir sampling for memory efficiency.

    Args:
        max_history: Maximum number of completed requests to retain.
        percentile_window: Sliding window size for percentile computation.
    """

    def __init__(
        self,
        max_history: int = 10000,
        percentile_window: int = 1000,
    ) -> None:
        self._max_history = max_history
        self._percentile_window = percentile_window

        self._active_requests: dict[int, RequestMetrics] = {}
        self._completed_requests: Deque[RequestMetrics] = deque(maxlen=max_history)

        self._ttft_samples: Deque[float] = deque(maxlen=percentile_window)
        self._tpot_samples: Deque[float] = deque(maxlen=percentile_window)
        self._latency_samples: Deque[float] = deque(maxlen=percentile_window)

        self._window_start_time: float = time.perf_counter()
        self._window_request_count: int = 0
        self._window_token_count: int = 0

    def start_request(
        self, request_id: int, prompt_tokens: int, timestamp: Optional[float] = None
    ) -> None:
        """Register the start of a new request.

        Args:
            request_id: Unique request identifier.
            prompt_tokens: Number of tokens in the prompt.
            timestamp: Start time (defaults to now).
        """
        ts = timestamp if timestamp is not None else time.perf_counter()
        self._active_requests[request_id] = RequestMetrics(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            timestamps={MetricPhase.QUEUED: ts},
        )

    def record_prefill_start(
        self, request_id: int, timestamp: Optional[float] = None
    ) -> None:
        """Mark the start of the prefill phase for a request."""
        ts = timestamp if timestamp is not None else time.perf_counter()
        if request_id in self._active_requests:
            self._active_requests[request_id].timestamps[MetricPhase.PREFILL] = ts

    def record_first_token(
        self, request_id: int, timestamp: Optional[float] = None
    ) -> None:
        """Record the time when the first token was generated.

        Computes TTFT from request start to first token.
        """
        ts = timestamp if timestamp is not None else time.perf_counter()
        metrics = self._active_requests.get(request_id)
        if metrics is None:
            return

        prefill_start = metrics.timestamps.get(
            MetricPhase.PREFILL, metrics.timestamps.get(MetricPhase.QUEUED, ts)
        )
        metrics.ttft_s = ts - prefill_start
        metrics.prefill_time_s = ts - prefill_start
        metrics.timestamps[MetricPhase.DECODE] = ts

    def record_decode_step(self, request_id: int, num_tokens: int = 1) -> None:
        """Record that a decode step produced num_tokens for a request.

        Increments the output token count for later TPOT calculation.
        """
        if request_id in self._active_requests:
            self._active_requests[request_id].output_tokens += num_tokens

    def finish_request(
        self,
        request_id: int,
        finish_reason: str = "stop",
        peak_memory_mb: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Optional[RequestMetrics]:
        """Complete a request and compute final metrics.

        Args:
            request_id: Request to finish.
            finish_reason: Why the request finished.
            peak_memory_mb: Peak GPU memory during this request.
            timestamp: Finish time (defaults to now).

        Returns:
            The finalized RequestMetrics, or None if request not found.
        """
        ts = timestamp if timestamp is not None else time.perf_counter()
        metrics = self._active_requests.pop(request_id, None)
        if metrics is None:
            return None

        start_ts = metrics.timestamps.get(
            MetricPhase.PREFILL, metrics.timestamps.get(MetricPhase.QUEUED, ts)
        )
        decode_start = metrics.timestamps.get(MetricPhase.DECODE, ts)

        metrics.total_latency_s = ts - start_ts
        metrics.decode_time_s = ts - decode_start

        if metrics.output_tokens > 0 and metrics.decode_time_s > 0:
            metrics.tpot_s = metrics.decode_time_s / metrics.output_tokens

        metrics.finish_reason = finish_reason
        metrics.peak_memory_mb = peak_memory_mb
        metrics.timestamps[MetricPhase.FINISHED] = ts

        self._completed_requests.append(metrics)
        self._ttft_samples.append(metrics.ttft_s)
        self._tpot_samples.append(metrics.tpot_s)
        self._latency_samples.append(metrics.total_latency_s)

        self._window_request_count += 1
        self._window_token_count += metrics.output_tokens

        return metrics

    def get_request(self, request_id: int) -> Optional[RequestMetrics]:
        """Get current metrics for an active request."""
        return self._active_requests.get(request_id)

    def get_completed(self, n: Optional[int] = None) -> list[RequestMetrics]:
        """Get completed request metrics.

        Args:
            n: Number of most recent requests to return (default: all).
        """
        items = list(self._completed_requests)
        if n is not None:
            items = items[-n:]
        return items

    def compute_system_metrics(
        self,
        gpu_memory_usage_mb: float = 0.0,
        gpu_memory_capacity_mb: float = 0.0,
        gpu_utilization_pct: float = 0.0,
    ) -> SystemMetrics:
        """Compute aggregate system-level metrics.

        Args:
            gpu_memory_usage_mb: Current GPU memory usage.
            gpu_memory_capacity_mb: Total GPU memory capacity.
            gpu_utilization_pct: Estimated GPU utilization percentage.

        Returns:
            SystemMetrics with aggregated statistics.
        """
        now = time.perf_counter()
        elapsed = max(now - self._window_start_time, 1e-6)

        qps = self._window_request_count / elapsed
        throughput = self._window_token_count / elapsed

        # Reset window
        self._window_start_time = now
        self._window_request_count = 0
        self._window_token_count = 0

        ttft_values = list(self._ttft_samples)
        tpot_values = list(self._tpot_samples)

        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0
        avg_tpot = sum(tpot_values) / len(tpot_values) if tpot_values else 0.0

        total_input = sum(r.prompt_tokens for r in self._completed_requests)
        total_output = sum(r.output_tokens for r in self._completed_requests)
        total_requests = len(self._completed_requests)

        return SystemMetrics(
            qps=qps,
            total_requests=total_requests,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            avg_ttft_s=avg_ttft,
            p50_ttft_s=self._percentile(ttft_values, 50),
            p95_ttft_s=self._percentile(ttft_values, 95),
            p99_ttft_s=self._percentile(ttft_values, 99),
            avg_tpot_s=avg_tpot,
            throughput_tokens_per_s=throughput,
            gpu_utilization_pct=gpu_utilization_pct,
            memory_usage_mb=gpu_memory_usage_mb,
            memory_capacity_mb=gpu_memory_capacity_mb,
            active_requests=len(self._active_requests),
        )

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._active_requests.clear()
        self._completed_requests.clear()
        self._ttft_samples.clear()
        self._tpot_samples.clear()
        self._latency_samples.clear()
        self._window_start_time = time.perf_counter()
        self._window_request_count = 0
        self._window_token_count = 0

    @property
    def active_count(self) -> int:
        """Number of currently active requests."""
        return len(self._active_requests)

    @property
    def completed_count(self) -> int:
        """Number of completed requests."""
        return len(self._completed_requests)

    @staticmethod
    def _percentile(values: Sequence[float], pct: float) -> float:
        """Compute the pct-th percentile (e.g., 95 for p95).

        Uses linear interpolation between nearest ranks.

        Args:
            values: Sorted or unsorted list of values.
            pct: Percentile to compute (0-100).

        Returns:
            The pct-th percentile value.
        """
        if not values:
            return 0.0
        if pct <= 0:
            return min(values)
        if pct >= 100:
            return max(values)

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        rank = (pct / 100.0) * (n - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return sorted_vals[lower]
        frac = rank - lower
        return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac


def visualize_request_lifecycle(
    metrics: RequestMetrics,
    width: int = 60,
) -> str:
    """Generate an ASCII timeline of a request lifecycle.

    Args:
        metrics: Completed request metrics with timestamps.
        width: Display width in characters.

    Returns:
        Multi-line string showing the request timeline.
    """
    phases = [
        (MetricPhase.QUEUED, "Queued"),
        (MetricPhase.PREFILL, "Prefill"),
        (MetricPhase.DECODE, "Decode"),
    ]
    total = metrics.total_latency_s

    if total <= 0:
        return "No timing data available"

    lines: list[str] = [
        f"Request {metrics.request_id} Timeline ({total:.3f}s total):",
        f"  Prompt: {metrics.prompt_tokens} tokens, Output: {metrics.output_tokens} tokens",
        f"  TTFT: {metrics.ttft_s * 1000:.1f}ms, TPOT: {metrics.tpot_s * 1000:.1f}ms",
        f"  Finish: {metrics.finish_reason}",
        "",
    ]

    start = metrics.timestamps.get(MetricPhase.QUEUED, 0.0)
    for phase, label in phases:
        t_start = metrics.timestamps.get(phase, start)
        if phase != MetricPhase.DECODE:
            t_end = metrics.timestamps.get(
                MetricPhase.PREFILL
                if phase == MetricPhase.QUEUED
                else MetricPhase.DECODE,
                start + total,
            )
        else:
            t_end = metrics.timestamps.get(MetricPhase.FINISHED, start + total)

        pct = ((t_end - t_start) / total * 100) if total > 0 else 0
        bar_len = max(int(width * (t_end - t_start) / total), 3)
        bar = "=" * bar_len
        lines.append(f"  {label:8s} [{bar:<{width}}] {pct:5.1f}%")

    return "\n".join(lines)
