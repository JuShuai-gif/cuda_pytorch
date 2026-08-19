"""Pure-Python metric primitives for training-system experiments.

This module intentionally does not import PyTorch.  It makes measurement units and
FLOP conventions explicit so the same formulas can be tested on a laptop and used
by CUDA benchmarks without silently assuming a GPU model or a peak FLOP number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Iterable, Sequence


def _require_finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return value


def _require_finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated quantile for a non-empty finite sequence."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile!r}")
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("values must contain only finite numbers")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class LatencySummary:
    """Latency statistics; every numeric latency field is in milliseconds."""

    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def summarize_latencies(step_times_s: Iterable[float]) -> LatencySummary:
    """Summarize positive, synchronized end-to-end step times in seconds."""

    seconds = [
        _require_finite_positive("step_time_s", value) for value in step_times_s
    ]
    if not seconds:
        raise ValueError("step_times_s must not be empty")
    milliseconds = [value * 1_000.0 for value in seconds]
    return LatencySummary(
        count=len(milliseconds),
        min_ms=min(milliseconds),
        max_ms=max(milliseconds),
        mean_ms=statistics.fmean(milliseconds),
        std_ms=statistics.pstdev(milliseconds),
        p50_ms=percentile(milliseconds, 0.50),
        p90_ms=percentile(milliseconds, 0.90),
        p99_ms=percentile(milliseconds, 0.99),
    )


def throughput_per_second(items_per_step: float, step_time_s: float) -> float:
    """Compute global (not per-device) throughput from the same step boundary."""

    items = _require_finite_nonnegative("items_per_step", items_per_step)
    seconds = _require_finite_positive("step_time_s", step_time_s)
    return items / seconds


@dataclass(frozen=True)
class FlopEstimate:
    """A declared model/HFU numerator, never a claim of measured instructions.

    ``model_flops`` excludes activation recomputation. ``hardware_flops`` adds the
    declared recomputation but still excludes unmodelled kernels such as optimizer
    updates unless the caller includes them in ``forward_flops``/``backward_flops``.
    """

    forward_flops: float
    backward_flops: float
    recompute_flops: float
    model_flops: float
    hardware_flops: float
    convention: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def training_flop_estimate(
    forward_flops: float,
    *,
    backward_to_forward_ratio: float = 2.0,
    recompute_forward_fraction: float = 0.0,
    convention: str = "FMA=2; model=forward+backward; HFU adds declared recompute",
) -> FlopEstimate:
    """Build an explicit training FLOP estimate.

    A dense matmul-dominated network commonly uses backward/forward ~= 2.0, but
    callers must treat it as a convention, not a hardware measurement.  A full
    activation recomputation has ``recompute_forward_fraction=1``; selective
    checkpointing should pass the fraction of *forward FLOPs*, not layer count.
    """

    forward = _require_finite_nonnegative("forward_flops", forward_flops)
    backward_ratio = _require_finite_nonnegative(
        "backward_to_forward_ratio", backward_to_forward_ratio
    )
    recompute_fraction = _require_finite_nonnegative(
        "recompute_forward_fraction", recompute_forward_fraction
    )
    backward = forward * backward_ratio
    recompute = forward * recompute_fraction
    return FlopEstimate(
        forward_flops=forward,
        backward_flops=backward,
        recompute_flops=recompute,
        model_flops=forward + backward,
        hardware_flops=forward + backward + recompute,
        convention=convention,
    )


def transformer_parameter_flop_estimate(
    parameter_count: int,
    token_count: int,
    *,
    extra_forward_flops: float = 0.0,
    backward_to_forward_ratio: float = 2.0,
    recompute_forward_fraction: float = 0.0,
) -> FlopEstimate:
    """Approximate dense-Transformer training FLOPs using the parameter term.

    Forward parameter FLOPs are ``2 * P * T`` (FMA=2).  This misses work that is
    not proportional to parameter count, notably the quadratic QK^T and AV terms,
    softmax, normalization, elementwise work, loss, and optimizer.  Architecture-
    specific forward work can be supplied as ``extra_forward_flops``.
    """

    parameters = _require_positive_int("parameter_count", parameter_count)
    tokens = _require_positive_int("token_count", token_count)
    extra = _require_finite_nonnegative("extra_forward_flops", extra_forward_flops)
    forward = 2.0 * parameters * tokens + extra
    return training_flop_estimate(
        forward,
        backward_to_forward_ratio=backward_to_forward_ratio,
        recompute_forward_fraction=recompute_forward_fraction,
        convention=(
            "FMA=2; forward=2*P*T+declared_extra; model=forward+backward; "
            "HFU adds declared activation recompute; optimizer excluded"
        ),
    )


def linear_forward_flops(
    token_count: int,
    dimensions: Sequence[tuple[int, int]],
    *,
    include_bias_add: bool = False,
) -> float:
    """Count forward FLOPs for token-wise dense linear layers (FMA=2)."""

    tokens = _require_positive_int("token_count", token_count)
    total = 0
    for index, (input_features, output_features) in enumerate(dimensions):
        in_features = _require_positive_int(
            f"dimensions[{index}].input_features", input_features
        )
        out_features = _require_positive_int(
            f"dimensions[{index}].output_features", output_features
        )
        total += 2 * tokens * in_features * out_features
        if include_bias_add:
            total += tokens * out_features
    return float(total)


@dataclass(frozen=True)
class UtilizationReport:
    achieved_model_flops_per_second: float
    estimated_hardware_flops_per_second: float
    aggregate_peak_flops_per_second: float | None
    mfu: float | None
    hfu: float | None
    status: str

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def utilization_from_step(
    flop_estimate: FlopEstimate,
    step_time_s: float,
    *,
    peak_flops_per_device_per_second: float | None,
    device_count: int = 1,
) -> UtilizationReport:
    """Compute MFU/HFU only when a matching explicit peak is supplied.

    The peak must match the measured dtype, dense-vs-sparse mode, Tensor Core path,
    and device clocks.  Device name discovery is intentionally insufficient: this
    function never maps a name such as "A100" to a default peak.
    """

    seconds = _require_finite_positive("step_time_s", step_time_s)
    devices = _require_positive_int("device_count", device_count)
    model_rate = flop_estimate.model_flops / seconds
    hardware_rate = flop_estimate.hardware_flops / seconds
    if peak_flops_per_device_per_second is None:
        return UtilizationReport(
            achieved_model_flops_per_second=model_rate,
            estimated_hardware_flops_per_second=hardware_rate,
            aggregate_peak_flops_per_second=None,
            mfu=None,
            hfu=None,
            status=(
                "unavailable: provide a verified per-device peak matching dtype, "
                "Tensor Core/sparsity mode, and clocks"
            ),
        )
    per_device_peak = _require_finite_positive(
        "peak_flops_per_device_per_second", peak_flops_per_device_per_second
    )
    aggregate_peak = per_device_peak * devices
    return UtilizationReport(
        achieved_model_flops_per_second=model_rate,
        estimated_hardware_flops_per_second=hardware_rate,
        aggregate_peak_flops_per_second=aggregate_peak,
        mfu=model_rate / aggregate_peak,
        hfu=hardware_rate / aggregate_peak,
        status="ok: arithmetic estimate divided by caller-declared aggregate peak",
    )


@dataclass(frozen=True)
class MemoryBreakdown:
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    master_parameter_bytes: int
    activation_bytes: int
    temporary_bytes: int
    total_live_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def training_state_memory(
    parameter_count: int,
    *,
    parameter_bytes_per_parameter: int,
    gradient_bytes_per_parameter: int,
    optimizer_state_bytes_per_parameter: int,
    master_parameter_bytes_per_parameter: int,
    activation_bytes: int = 0,
    temporary_bytes: int = 0,
) -> MemoryBreakdown:
    """Estimate live tensor bytes from an explicitly declared precision policy.

    There are deliberately no dtype defaults: optimizers/frameworks differ on
    whether gradients and master weights are FP32, BF16, fused, or aliased.
    Allocator reserved memory and non-PyTorch CUDA allocations are observations,
    not part of this tensor-storage estimate.
    """

    parameters = _require_positive_int("parameter_count", parameter_count)
    byte_fields = {
        "parameter_bytes_per_parameter": parameter_bytes_per_parameter,
        "gradient_bytes_per_parameter": gradient_bytes_per_parameter,
        "optimizer_state_bytes_per_parameter": optimizer_state_bytes_per_parameter,
        "master_parameter_bytes_per_parameter": master_parameter_bytes_per_parameter,
        "activation_bytes": activation_bytes,
        "temporary_bytes": temporary_bytes,
    }
    for name, value in byte_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    parameter_bytes = parameters * parameter_bytes_per_parameter
    gradient_bytes = parameters * gradient_bytes_per_parameter
    optimizer_bytes = parameters * optimizer_state_bytes_per_parameter
    master_bytes = parameters * master_parameter_bytes_per_parameter
    total = (
        parameter_bytes
        + gradient_bytes
        + optimizer_bytes
        + master_bytes
        + activation_bytes
        + temporary_bytes
    )
    return MemoryBreakdown(
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_bytes,
        master_parameter_bytes=master_bytes,
        activation_bytes=activation_bytes,
        temporary_bytes=temporary_bytes,
        total_live_bytes=total,
    )


def format_bytes(byte_count: int) -> str:
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError(f"byte_count must be an integer >= 0, got {byte_count!r}")
    value = float(byte_count)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units[:-1]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} {units[-1]}"


Interval = tuple[float, float]


def _normalize_intervals(
    intervals: Iterable[Interval], *, start: float | None = None, end: float | None = None
) -> list[Interval]:
    normalized: list[Interval] = []
    for index, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError(f"interval[{index}] must contain (start, end)")
        left, right = (float(interval[0]), float(interval[1]))
        if not math.isfinite(left) or not math.isfinite(right) or right < left:
            raise ValueError(f"invalid interval[{index}]={interval!r}")
        if start is not None and left < start:
            raise ValueError(f"interval[{index}] starts before measurement window")
        if end is not None and right > end:
            raise ValueError(f"interval[{index}] ends after measurement window")
        if right > left:
            normalized.append((left, right))
    normalized.sort()
    return normalized


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = _normalize_intervals(intervals)
    merged: list[list[float]] = []
    for left, right in ordered:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def interval_union_duration(intervals: Iterable[Interval]) -> float:
    """Return union duration, avoiding double-counting concurrent streams/kernels."""

    return sum(right - left for left, right in _merge_intervals(intervals))


def interval_intersection_duration(
    first: Iterable[Interval], second: Iterable[Interval]
) -> float:
    left_intervals = _merge_intervals(first)
    right_intervals = _merge_intervals(second)
    left_index = right_index = 0
    duration = 0.0
    while left_index < len(left_intervals) and right_index < len(right_intervals):
        left_start, left_end = left_intervals[left_index]
        right_start, right_end = right_intervals[right_index]
        duration += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return duration


@dataclass(frozen=True)
class TimelineBreakdown:
    step_time_s: float
    compute_active_union_s: float
    communication_active_union_s: float
    compute_communication_overlap_s: float
    any_gpu_active_union_s: float
    gpu_bubble_s: float
    gpu_bubble_fraction: float
    unhidden_communication_s: float
    communication_overlap_fraction: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def analyze_timeline(
    step_start_s: float,
    step_end_s: float,
    *,
    compute_intervals: Iterable[Interval],
    communication_intervals: Iterable[Interval],
) -> TimelineBreakdown:
    """Analyze interval unions on one device timeline.

    ``unhidden_communication_s`` means communication time outside the supplied
    compute intervals.  It is a useful upper-bound-like timeline diagnostic, not a
    causal proof that removing communication shortens the step by the same amount.
    Host work, dependencies, and other GPU streams can still be on the critical path.
    """

    start = float(step_start_s)
    end = float(step_end_s)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("measurement window must be finite with end > start")
    compute = _normalize_intervals(compute_intervals, start=start, end=end)
    communication = _normalize_intervals(
        communication_intervals, start=start, end=end
    )
    compute_duration = interval_union_duration(compute)
    communication_duration = interval_union_duration(communication)
    overlap = interval_intersection_duration(compute, communication)
    active = interval_union_duration([*compute, *communication])
    step = end - start
    bubble = max(0.0, step - active)
    unhidden_communication = max(0.0, communication_duration - overlap)
    overlap_fraction = (
        overlap / communication_duration if communication_duration > 0.0 else None
    )
    return TimelineBreakdown(
        step_time_s=step,
        compute_active_union_s=compute_duration,
        communication_active_union_s=communication_duration,
        compute_communication_overlap_s=overlap,
        any_gpu_active_union_s=active,
        gpu_bubble_s=bubble,
        gpu_bubble_fraction=bubble / step,
        unhidden_communication_s=unhidden_communication,
        communication_overlap_fraction=overlap_fraction,
    )


@dataclass(frozen=True)
class ScalingMetrics:
    reference_device_count: int
    device_count: int
    throughput_speedup: float
    ideal_linear_speedup: float
    scaling_efficiency: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def scaling_from_throughput(
    reference_throughput: float,
    reference_device_count: int,
    throughput: float,
    device_count: int,
) -> ScalingMetrics:
    """Normalize global throughput gain by device-count gain.

    Comparisons are valid only when model, precision, convergence semantics, and
    workload definition are held constant.  Record whether global or per-device
    batch is fixed; otherwise "scaling efficiency" is ambiguous.
    """

    reference_rate = _require_finite_positive(
        "reference_throughput", reference_throughput
    )
    measured_rate = _require_finite_positive("throughput", throughput)
    reference_devices = _require_positive_int(
        "reference_device_count", reference_device_count
    )
    devices = _require_positive_int("device_count", device_count)
    if devices < reference_devices:
        raise ValueError("device_count must be >= reference_device_count")
    speedup = measured_rate / reference_rate
    ideal = devices / reference_devices
    return ScalingMetrics(
        reference_device_count=reference_devices,
        device_count=devices,
        throughput_speedup=speedup,
        ideal_linear_speedup=ideal,
        scaling_efficiency=speedup / ideal,
    )
