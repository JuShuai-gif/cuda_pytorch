#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a ProfileResult into a text report and a markdown file."""

from __future__ import annotations

from .profiler import ProfileResult


def _fmt_count(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}G"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def render_text(result: ProfileResult) -> str:
    p, m, lat, roof = result.params, result.macs, result.latency, result.roofline
    cfg = result.config
    pf, mf = p.category_fraction, m.category_fraction

    lines: list[str] = []
    add = lines.append

    add("================ VLA PROFILER REPORT ================")
    add("")
    add("[Model Overview]")
    add(f"Total Params: {_fmt_count(p.total)}")
    add(f"Total MACs:   {_fmt_count(m.total_macs)}")
    add(f"Model Size:   {p.size_mb:.1f} MB ({cfg.precision})")
    add(f"MACs Backend: {m.backend}")
    add("")

    add("[Module Breakdown]")
    labels = {
        "vision": "Vision Encoder    ",
        "language": "Language Encoder  ",
        "fusion": "Fusion Transformer",
        "action": "Action Head       ",
    }
    for c in ("vision", "language", "fusion", "action"):
        add(f"{labels[c]} : {pf[c] * 100:4.0f}% params | {mf[c] * 100:4.0f}% MACs")
    add("")

    add("[Latency Estimate]")
    add(f"GPU Peak ({cfg.gpu_name} {cfg.precision}): {cfg.gpu_tflops:.0f} TFLOPs")
    add(f"Theoretical Latency : {lat.theoretical_ms:.2f} ms")
    if lat.measured_ms is not None:
        add(
            f"Measured Latency    : {lat.measured_ms:.2f} ms"
            f"  (p50 {lat.p50_ms:.2f} / p99 {lat.p99_ms:.2f})"
        )
        add(f"Throughput          : {lat.throughput_sps:.1f} infer/s")
        add(f"Efficiency          : {lat.efficiency * 100:.1f}%")
    else:
        add("Measured Latency    : (skipped)")
    add("")

    add("[Bottleneck Analysis]")
    for b in result.bottlenecks:
        tag = (
            "FIRE"
            if b.startswith("Primary")
            else ("FIRE" if b.startswith("Secondary") else "NOTE")
        )
        add(f"[{tag}] {b}")
    add("")

    add("[Roofline Status]")
    add(f"Arithmetic Intensity : {roof.arithmetic_intensity:.2f} FLOP/byte")
    add(f"Ridge Point          : {roof.ridge_point:.2f} FLOP/byte")
    add(
        f"Attainable           : {roof.attainable_tflops:.1f} TFLOPs"
        f" / {cfg.gpu_tflops:.0f} peak"
    )
    add(f"Compute-bound ratio  : {roof.compute_bound_ratio:.2f}")
    add(f"Memory-bound ratio   : {roof.memory_bound_ratio:.2f}")
    add(f"Regime               : {roof.regime}")
    add("")

    if result.chunk is not None:
        ch = result.chunk
        add("[Action Chunk Rollout]")
        add(f"Chunk Steps          : {ch.chunk_steps}")
        add(f"MACs / chunk         : {_fmt_count(ch.macs_per_chunk)}")
        add(f"MACs / control step  : {_fmt_count(ch.macs_per_control_step)}")
        add(f"Amortization vs replan: {ch.amortization:.1f}x cheaper")
        add("")

    if result.kv_cache is not None:
        kv = result.kv_cache
        add("[KV Cache]")
        add(f"Size                 : {kv.bytes_mb:.1f} MB")
        add(f"Stream-once time     : {kv.read_time_ms:.3f} ms")
        add(f"Bandwidth-bound      : {kv.is_bandwidth_bound}")
        add("")

    if result.multi_camera is not None and result.multi_camera.num_cameras > 1:
        mc = result.multi_camera
        add("[Multi-Camera]")
        add(f"Cameras              : {mc.num_cameras}")
        add(f"Resolution scale     : {mc.resolution_scale:.2f}x")
        add(f"Vision MACs (total)  : {_fmt_count(mc.vision_macs_total)}")
        add("")

    if result.ros is not None:
        ros = result.ros
        add("[ROS Latency Coupling]")
        add(f"Compute              : {ros.compute_ms:.2f} ms")
        add(f"End-to-end           : {ros.end_to_end_ms:.2f} ms")
        add(f"Required control rate : {ros.control_hz_required:.0f} Hz")
        add(f"Achievable rate      : {ros.control_hz_achievable:.1f} Hz")
        add(f"Meets real-time      : {ros.meets_realtime}")
        add(f"Chunk hides latency  : {ros.chunk_covers_latency}")
        add("")

    if result.macs.unsupported_ops:
        add("[Unsupported Ops] (MACs may be under-counted)")
        for op, cnt in sorted(result.macs.unsupported_ops.items(), key=lambda x: -x[1])[
            :8
        ]:
            add(f"  {op}: {cnt}")
        add("")

    add("====================================================")
    return "\n".join(lines)


def render_markdown(result: ProfileResult) -> str:
    p, m, lat, roof = result.params, result.macs, result.latency, result.roofline
    cfg = result.config
    pf, mf = p.category_fraction, m.category_fraction

    md: list[str] = []
    md.append("# VLA Profiler Report\n")
    md.append("## Model Overview\n")
    md.append(f"- **Total Params**: {_fmt_count(p.total)}")
    md.append(f"- **Total MACs**: {_fmt_count(m.total_macs)}")
    md.append(f"- **Model Size**: {p.size_mb:.1f} MB ({cfg.precision})")
    md.append(f"- **MACs Backend**: {m.backend}\n")

    md.append("## Module Breakdown\n")
    md.append("| Module | Params % | MACs % |")
    md.append("|--------|----------|--------|")
    names = {
        "vision": "Vision Encoder",
        "language": "Language Encoder",
        "fusion": "Fusion Transformer",
        "action": "Action Head",
    }
    for c in ("vision", "language", "fusion", "action"):
        md.append(f"| {names[c]} | {pf[c] * 100:.0f}% | {mf[c] * 100:.0f}% |")
    md.append("")

    md.append("## Latency\n")
    md.append(
        f"- GPU peak ({cfg.gpu_name} {cfg.precision}): {cfg.gpu_tflops:.0f} TFLOPs"
    )
    md.append(f"- Theoretical: {lat.theoretical_ms:.2f} ms")
    if lat.measured_ms is not None:
        md.append(
            f"- Measured: {lat.measured_ms:.2f} ms "
            f"(p50 {lat.p50_ms:.2f} / p99 {lat.p99_ms:.2f})"
        )
        md.append(f"- Throughput: {lat.throughput_sps:.1f} infer/s")
        md.append(f"- Efficiency: {lat.efficiency * 100:.1f}%")
    md.append("")

    md.append("## Roofline\n")
    md.append(f"- Arithmetic intensity: {roof.arithmetic_intensity:.2f} FLOP/byte")
    md.append(f"- Ridge point: {roof.ridge_point:.2f} FLOP/byte")
    md.append(f"- Compute-bound ratio: {roof.compute_bound_ratio:.2f}")
    md.append(f"- Memory-bound ratio: {roof.memory_bound_ratio:.2f}")
    md.append(f"- **Regime: {roof.regime}**\n")

    md.append("## Bottlenecks\n")
    for b in result.bottlenecks:
        md.append(f"- {b}")
    md.append("")
    return "\n".join(md)


def save_markdown(result: ProfileResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
