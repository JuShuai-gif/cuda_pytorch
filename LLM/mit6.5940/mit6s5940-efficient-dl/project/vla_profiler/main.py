#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for the VLA Profiler.

Examples
--------
Profile the bundled synthetic ~705M VLA on an A100 (FP16):

    python -m vla_profiler.main --gpu a100 --precision fp16 --chunk-steps 50

Skip on-device measurement (theoretical only, no GPU required):

    python -m vla_profiler.main --no-measure --device cpu

Save a markdown report:

    python -m vla_profiler.main --markdown report.md
"""

from __future__ import annotations

import argparse

import torch

from .models.synthetic_vla import build_synthetic_vla
from .profiler import ProfilerConfig, VLAProfiler
from .report import render_text, save_markdown

# gpu -> (display, bandwidth GB/s, {precision: TFLOPs})
GPU_TABLE = {
    "a100": (
        "A100",
        2039.0,
        {"fp32": 19.5, "tf32": 156.0, "fp16": 312.0, "bf16": 312.0},
    ),
    "h100": (
        "H100",
        3350.0,
        {"fp32": 67.0, "tf32": 495.0, "fp16": 989.0, "fp8": 1979.0},
    ),
    "rtx4090": ("RTX4090", 1008.0, {"fp32": 82.6, "fp16": 165.0, "bf16": 165.0}),
    "jetson_orin": ("Jetson-Orin", 204.0, {"fp16": 137.0, "int8": 275.0}),
    "jetson_nano": ("Jetson-Nano", 25.6, {"fp16": 0.47}),
}


def resolve_gpu(gpu: str, precision: str) -> tuple[str, float, float]:
    if gpu not in GPU_TABLE:
        raise SystemExit(f"Unknown gpu '{gpu}'. Choose from {list(GPU_TABLE)}")
    name, bw, tflops_by_prec = GPU_TABLE[gpu]
    if precision not in tflops_by_prec:
        raise SystemExit(
            f"{name} has no preset for '{precision}'. Available: {list(tflops_by_prec)}"
        )
    return name, tflops_by_prec[precision], bw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vla_profiler",
        description="Industrial profiler for Vision-Language-Action policies.",
    )
    p.add_argument(
        "--model", default="synthetic", help="model id (only 'synthetic' is bundled)"
    )
    p.add_argument("--preset", default="705M", help="synthetic model preset")
    p.add_argument("--gpu", default="a100", help="target gpu preset")
    p.add_argument(
        "--precision",
        default="fp16",
        choices=["fp32", "tf32", "fp16", "bf16", "fp8", "int8"],
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "fvcore", "torchprofile", "ptflops", "thop", "hook"],
    )
    p.add_argument("--chunk-steps", type=int, default=50)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--num-cameras", type=int, default=1)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument(
        "--no-measure", action="store_true", help="skip on-device latency measurement"
    )
    p.add_argument("--markdown", default=None, help="path to save markdown report")
    p.add_argument(
        "--plot", default=None, help="path to save roofline + module breakdown PNG"
    )
    p.add_argument(
        "--kernels",
        action="store_true",
        help="run torch.profiler kernel-level breakdown",
    )
    p.add_argument(
        "--kernel-top", type=int, default=15, help="number of hottest kernels to show"
    )
    p.add_argument("--kernel-steps", type=int, default=20)
    p.add_argument(
        "--trace", default=None, help="export Chrome/Perfetto trace (implies --kernels)"
    )
    p.add_argument(
        "--print-ncu",
        action="store_true",
        help="print ready-to-run ncu / nsys commands",
    )
    p.add_argument(
        "--run-ncu",
        action="store_true",
        help="launch Nsight Compute on this profiler (slow, needs perms)",
    )
    p.add_argument(
        "--run-nsys", action="store_true", help="launch Nsight Systems on this profiler"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # External profilers re-launch this CLI in a child process, then exit.
    if args.run_ncu or args.run_nsys:
        from . import kernel_profiler as kp

        cmd = kp.build_ncu_command() if args.run_ncu else kp.build_nsys_command()
        print("[launch]", " ".join(cmd))
        return kp.run_external(cmd)

    if args.model != "synthetic":
        raise SystemExit(
            "Only the bundled 'synthetic' model is supported here. "
            "Import VLAProfiler directly to profile a real model."
        )

    model = build_synthetic_vla(args.preset)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> using CPU")
        device = "cpu"

    name, tflops, bw = resolve_gpu(args.gpu, args.precision)

    # Derive KV-cache geometry from the synthetic fusion transformer.
    c = model.c
    seq_len = (c.img_size // c.patch) ** 2 + c.txt_len

    cfg = ProfilerConfig(
        gpu_name=name,
        precision=args.precision,
        gpu_tflops=tflops,
        bandwidth_gbps=bw,
        device=device,
        measure_latency=not args.no_measure,
        warmup=args.warmup,
        repeat=args.repeat,
        macs_backend=args.backend,
        chunk_steps=args.chunk_steps,
        control_hz=args.control_hz,
        num_cameras=args.num_cameras,
        resolution=args.resolution,
        base_resolution=c.img_size,
        kv_layers=c.fus_depth,
        kv_heads=c.heads,
        kv_head_dim=c.fus_dim // c.heads,
        kv_seq_len=seq_len,
    )

    dummy = model.dummy_inputs(batch=args.batch)
    result = VLAProfiler(model, cfg).run(dummy)

    print(render_text(result))

    if args.markdown:
        save_markdown(result, args.markdown)
        print(f"\n[saved] markdown report -> {args.markdown}")

    if args.plot:
        from .plot import save_roofline_plot

        save_roofline_plot(result, args.plot)
        print(f"[saved] roofline plot -> {args.plot}")

    if args.kernels or args.trace:
        from . import kernel_profiler as kp

        prof = kp.profile_kernels(
            model,
            dummy,
            device=device,
            steps=args.kernel_steps,
            export_trace=args.trace,
            top=args.kernel_top,
        )
        print()
        print(kp.render_kernel_table(prof))

    if args.print_ncu:
        from . import kernel_profiler as kp

        print("\n[Nsight commands]")
        print("  ncu  available:", kp.has_ncu())
        print("  nsys available:", kp.has_nsys())
        print("  " + " ".join(kp.build_ncu_command()))
        print("  " + " ".join(kp.build_nsys_command()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
