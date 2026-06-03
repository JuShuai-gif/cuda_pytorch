"""
Memory bandwidth analysis for GPU kernels.

Covers:
  - Peak bandwidth estimation from GPU specs
  - Contiguous vs strided memory access patterns
  - Vectorized vs uncoalesced access in Triton
  - Memory-bound vs compute-bound scaling demonstration
  - Roofline model sketch data generation

Run: python 03_memory_bandwidth/analysis.py
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU spec database (peak bandwidth in GB/s, roughly)
# ---------------------------------------------------------------------------

GPU_BANDWIDTH_DB: dict[str, float] = {
    "NVIDIA GeForce RTX 4090": 1008.0,
    "NVIDIA GeForce RTX 4080": 717.0,
    "NVIDIA GeForce RTX 3090": 936.0,
    "NVIDIA GeForce RTX 3080": 760.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA A10": 600.0,
    "NVIDIA T4": 320.0,
    "NVIDIA H100": 3352.0,
    "NVIDIA L40S": 864.0,
    "NVIDIA L4": 300.0,
}


def get_peak_bandwidth() -> tuple[str, float]:
    """Estimate peak memory bandwidth from GPU specs.

    Returns:
        (device_name, peak_bandwidth_gb_s)
    """
    if not torch.cuda.is_available():
        return ("CPU", 0.0)

    name = torch.cuda.get_device_name(0)
    bw = GPU_BANDWIDTH_DB.get(name)

    if bw is None:
        props = torch.cuda.get_device_properties(0)
        # Estimate from memory clock and bus width
        # memory_clock is in kHz, memory_bus_width in bits
        mem_clock_ghz = props.memory_clock / 1e6  # kHz -> GHz
        bus_width_bytes = props.memory_bus_width / 8  # bits -> bytes
        # GDDR6X is double data rate, so effective BW = clock * bus_width * 2
        bw = mem_clock_ghz * bus_width_bytes * 2.0

    return name, bw


# ---------------------------------------------------------------------------
# Contiguous vs strided bandwidth measurement
# ---------------------------------------------------------------------------


def measure_contiguous_bandwidth(tensor_size: int) -> float:
    """Measure bandwidth with contiguous tensor access.

    Creates a contiguous tensor and performs an elementwise multiply,
    measuring achieved bandwidth.

    Args:
        tensor_size: Number of float32 elements.

    Returns:
        Achieved bandwidth in GB/s.
    """
    x = torch.randn(tensor_size, device="cuda", dtype=torch.float32)
    y = torch.randn(tensor_size, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = x * y
    torch.cuda.synchronize()

    # Timed measurement
    n_iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(n_iters):
        start.record()
        _ = x * y
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_ms = statistics.mean(times_ms)
    # Bytes moved: 3 tensors * numel * 4 bytes (read x, read y, write result)
    total_bytes = 3 * tensor_size * 4
    bw_gb_s = (total_bytes / (avg_ms / 1000.0)) / 1e9
    return bw_gb_s


def measure_strided_bandwidth(tensor_size: int) -> float:
    """Measure bandwidth with strided tensor access.

    Creates a tensor, takes a strided slice (every other element or
    transposed view for 2D), and performs the same elementwise multiply,
    measuring the bandwidth drop due to non-contiguous access.

    Args:
        tensor_size: Number of float32 elements.

    Returns:
        Achieved bandwidth in GB/s for strided access.
    """
    # Create a 2D tensor and take strided slices
    side = int(math.sqrt(tensor_size))
    if side < 2:
        side = 2
    side = min(side, 16384)

    x_full = torch.randn(side, side, device="cuda", dtype=torch.float32)
    y_full = torch.randn(side, side, device="cuda", dtype=torch.float32)

    # Transposed view - strided access
    x_strided = x_full.t()
    y_strided = y_full.t()

    # Warmup
    for _ in range(10):
        _ = x_strided * y_strided
    torch.cuda.synchronize()

    n_iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(n_iters):
        start.record()
        _ = x_strided * y_strided
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_ms = statistics.mean(times_ms)
    total_bytes = 3 * x_strided.numel() * 4
    bw_gb_s = (total_bytes / (avg_ms / 1000.0)) / 1e9
    return bw_gb_s


# ---------------------------------------------------------------------------
# Vectorized load benchmark in Triton
# ---------------------------------------------------------------------------


@triton.jit
def _coalesced_load_kernel(
    x_ptr,
    out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Coalesced memory access: contiguous elements per thread."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _strided_load_kernel(
    x_ptr,
    out_ptr,
    n_elements: int,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided (uncoalesced) memory access: elements separated by STRIDE."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Multiply by stride to create non-contiguous access pattern
    strided_offsets = offsets * STRIDE
    mask = (offsets * STRIDE) < n_elements

    x = tl.load(x_ptr + strided_offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def vectorized_load_benchmark(size: int, vec_width: int) -> dict[str, float]:
    """Benchmark coalesced vs strided load patterns in Triton.

    Args:
        size: Total number of elements.
        vec_width: Stride factor for non-coalesced access.

    Returns:
        Dictionary with coalesced and strided bandwidths in GB/s.
    """
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    out_coalesced = torch.empty_like(x)

    # Coalesced
    n_elements = x.numel()
    block_size = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Warmup
    for _ in range(5):
        _coalesced_load_kernel[grid](x, out_coalesced, n_elements, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    n_iters = 30
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(n_iters):
        start.record()
        _coalesced_load_kernel[grid](x, out_coalesced, n_elements, BLOCK_SIZE=block_size)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_coalesced_ms = statistics.mean(times_ms)
    read_write_bytes = 2 * n_elements * 4  # read + write
    coalesced_bw = (read_write_bytes / (avg_coalesced_ms / 1000.0)) / 1e9

    # Strided - use a subset so the tensor is large enough for the stride
    effective_size = size // vec_width
    if effective_size < block_size:
        effective_size = block_size
    out_strided = torch.empty(effective_size, device="cuda", dtype=torch.float32)
    total_elements = effective_size * vec_width
    if total_elements > size:
        x_large = torch.randn(total_elements, device="cuda", dtype=torch.float32)
    else:
        x_large = x

    grid_strided = lambda meta: (triton.cdiv(effective_size, meta["BLOCK_SIZE"]),)

    for _ in range(5):
        _strided_load_kernel[grid_strided](
            x_large,
            out_strided,
            total_elements,
            STRIDE=vec_width,
            BLOCK_SIZE=block_size,
        )
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_iters):
        start.record()
        _strided_load_kernel[grid_strided](
            x_large,
            out_strided,
            total_elements,
            STRIDE=vec_width,
            BLOCK_SIZE=block_size,
        )
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_strided_ms = statistics.mean(times_ms)
    read_write_bytes_strided = 2 * effective_size * 4
    strided_bw = (read_write_bytes_strided / (avg_strided_ms / 1000.0)) / 1e9

    return {
        "coalesced_gb_s": coalesced_bw,
        "strided_gb_s": strided_bw,
        "vec_width": float(vec_width),
        "slowdown": coalesced_bw / strided_bw if strided_bw > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Memory-bound vs compute-bound demonstration
# ---------------------------------------------------------------------------


def memory_bound_vs_compute_bound() -> dict[str, float]:
    """Demonstrate memory-bound vs compute-bound scaling.

    Memory-bound op: elementwise multiply (O(n) compute, O(n) memory)
    Compute-bound op: matmul (O(n^3) compute, O(n^2) memory)

    Shows how each scales differently with size.

    Returns:
        Dictionary with sizes and achieved bandwidth/GFLOPS.
    """
    results = {}

    sizes = [1024, 4096, 16384, 65536]
    results["sizes"] = sizes

    # Memory-bound: elementwise multiply
    elem_bw_list = []
    for n in sizes:
        x = torch.ones(n, device="cuda", dtype=torch.float32)
        y = torch.ones(n, device="cuda", dtype=torch.float32)

        for _ in range(5):
            _ = x * y
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times_ms = []
        for _ in range(20):
            start.record()
            _ = x * y
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))

        avg_ms = statistics.mean(times_ms)
        total_bytes = 3 * n * 4
        bw = (total_bytes / (avg_ms / 1000.0)) / 1e9
        elem_bw_list.append(bw)

    results["elem_multiply_bw_gb_s"] = elem_bw_list

    # Compute-bound: matmul
    matmul_gflops_list = []
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)

        for _ in range(5):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times_ms = []
        for _ in range(10):
            start.record()
            _ = torch.matmul(a, b)
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))

        avg_ms = statistics.mean(times_ms)
        total_flops = 2.0 * (n**3)  # multiply-add
        gflops = (total_flops / (avg_ms / 1000.0)) / 1e9
        matmul_gflops_list.append(gflops)

    results["matmul_gflops"] = matmul_gflops_list

    return results


# ---------------------------------------------------------------------------
# Roofline model sketch
# ---------------------------------------------------------------------------


def roofline_analysis_sketch(output_dir: str | None = None) -> Path:
    """Generate CSV data for bandwidth vs arithmetic intensity for various ops.

    The roofline model plots performance (GFLOPS) against arithmetic intensity
    (FLOPs/byte). Each operation falls somewhere on this curve:
      - Low arithmetic intensity: memory-bound (below the bandwidth ceiling)
      - High arithmetic intensity: compute-bound (hits the compute ceiling)

    Args:
        output_dir: Directory to save the CSV file. Defaults to current dir.

    Returns:
        Path to the generated CSV file.
    """
    if output_dir is None:
        output_dir = "."

    name, peak_bw = get_peak_bandwidth()

    # Compute ceiling
    if "H100" in name:
        peak_compute_fp32 = 67.0  # TFLOPS
        peak_compute_fp16 = 989.0
    elif "A100" in name:
        peak_compute_fp32 = 19.5
        peak_compute_fp16 = 312.0
    elif "4090" in name:
        peak_compute_fp32 = 82.6
        peak_compute_fp16 = 330.3
    elif "3090" in name:
        peak_compute_fp32 = 35.6
        peak_compute_fp16 = 71.0
    else:
        peak_compute_fp32 = 13.0
        peak_compute_fp16 = 26.0

    # Define ops with estimated arithmetic intensity and actual performance
    ops = []

    # Elementwise multiply: 1 FLOP per element, 12 bytes (2 reads + 1 write)
    # Arithmetic intensity = 1/12 ~ 0.083
    ops.append(
        {
            "operation": "Elementwise Multiply",
            "arithmetic_intensity": 1.0 / 12.0,
            "achieved_gflops": min(peak_bw * (1.0 / 12.0) * 0.8, peak_compute_fp32 * 0.9),
            "bound_type": "memory",
        }
    )

    # ReLU: 1 FLOP per element, 12 bytes -> AI ~ 0.083
    ops.append(
        {
            "operation": "ReLU",
            "arithmetic_intensity": 1.0 / 12.0,
            "achieved_gflops": min(peak_bw * (1.0 / 12.0) * 0.8, peak_compute_fp32 * 0.9),
            "bound_type": "memory",
        }
    )

    # LayerNorm: ~10 FLOPs/element, ~12 bytes -> AI ~ 0.83
    ops.append(
        {
            "operation": "LayerNorm",
            "arithmetic_intensity": 10.0 / 12.0,
            "achieved_gflops": min(peak_bw * (10.0 / 12.0) * 0.7, peak_compute_fp32 * 0.8),
            "bound_type": "memory",
        }
    )

    # Softmax: ~5 FLOPs/element, ~8 bytes -> AI ~ 0.625
    ops.append(
        {
            "operation": "Softmax",
            "arithmetic_intensity": 5.0 / 8.0,
            "achieved_gflops": min(peak_bw * (5.0 / 8.0) * 0.7, peak_compute_fp32 * 0.8),
            "bound_type": "memory",
        }
    )

    # Matmul (1024x1024): 2*1024^3 FLOPs, (1024^2 * 12) bytes -> AI ~ 170
    ops.append(
        {
            "operation": "Matmul (1024^2)",
            "arithmetic_intensity": 170.0,
            "achieved_gflops": min(peak_bw * 170 * 0.7, peak_compute_fp32 * 0.5),
            "bound_type": "compute",
        }
    )

    # Matmul (4096x4096): 2*4096^3 FLOPs, (4096^2 * 12) bytes -> AI ~ 680
    ops.append(
        {
            "operation": "Matmul (4096^2)",
            "arithmetic_intensity": 680.0,
            "achieved_gflops": min(peak_bw * 680 * 0.7, peak_compute_fp32 * 0.7),
            "bound_type": "compute",
        }
    )

    # Roofline curve points
    bw_roofline = []
    for ai in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        bw_roofline.append(
            {
                "operation": f"BW_ceiling_ai={ai:.4f}",
                "arithmetic_intensity": ai,
                "achieved_gflops": ai * peak_bw,
                "bound_type": "roofline_bw",
            }
        )

    compute_roofline = []
    for ai in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        compute_roofline.append(
            {
                "operation": f"Compute_ceiling_ai={ai:.4f}",
                "arithmetic_intensity": ai,
                "achieved_gflops": peak_compute_fp32 * 1000,  # TFLOPS -> GFLOPS
                "bound_type": "roofline_compute",
            }
        )

    all_rows = bw_roofline + compute_roofline + ops

    # Sort by arithmetic intensity
    all_rows.sort(key=lambda r: r["arithmetic_intensity"])

    filepath = Path(output_dir) / "roofline_data.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["operation", "arithmetic_intensity", "achieved_gflops", "bound_type"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    device_name, peak_bw = get_peak_bandwidth()
    print("=" * 70)
    print("  MEMORY BANDWIDTH ANALYSIS")
    print("=" * 70)
    print(f"\n  Device: {device_name}")
    print(f"  Estimated Peak Bandwidth: {peak_bw:.1f} GB/s")

    # Contiguous vs strided
    print(f"\n{'=' * 70}")
    print("  Contiguous vs Strided Access")
    print(f"{'=' * 70}")

    for size in [2**20, 2**22, 2**24]:
        c_bw = measure_contiguous_bandwidth(size)
        s_bw = measure_strided_bandwidth(size)
        print(f"\n  Size: {size:>10,} elements")
        print(f"    Contiguous: {c_bw:>8.1f} GB/s ({c_bw / peak_bw * 100:>5.1f}% peak)")
        print(f"    Strided:    {s_bw:>8.1f} GB/s ({s_bw / peak_bw * 100:>5.1f}% peak)")
        print(f"    Slowdown:   {c_bw / s_bw if s_bw > 0 else 0:>8.1f}x")

    # Vectorized load benchmark
    print(f"\n{'=' * 70}")
    print("  Vectorized vs Strided Load (Triton)")
    print(f"{'=' * 70}")

    for vec_width in [1, 4, 16, 64]:
        result = vectorized_load_benchmark(2**22, vec_width)
        print(f"\n  Vector width: {vec_width}")
        print(f"    Coalesced: {result['coalesced_gb_s']:.1f} GB/s")
        print(f"    Strided (stride={vec_width}): {result['strided_gb_s']:.1f} GB/s")
        print(f"    Slowdown: {result['slowdown']:.1f}x")

    # Memory-bound vs compute-bound
    print(f"\n{'=' * 70}")
    print("  Memory-Bound vs Compute-Bound Scaling")
    print(f"{'=' * 70}")

    mb_vs_cb = memory_bound_vs_compute_bound()
    print(f"\n  {'Size':>10}  {'Element BW (GB/s)':>18}  {'Matmul GFLOPS':>15}")
    print(f"  {'-' * 10}  {'-' * 18}  {'-' * 15}")
    for i, size in enumerate(mb_vs_cb["sizes"]):
        print(
            f"  {size:>10}  {mb_vs_cb['elem_multiply_bw_gb_s'][i]:>18.1f}"
            f"  {mb_vs_cb['matmul_gflops'][i]:>15.1f}"
        )

    # Roofline sketch
    print(f"\n{'=' * 70}")
    print("  Roofline Analysis Sketch")
    print(f"{'=' * 70}")

    csv_path = roofline_analysis_sketch()
    print(f"\n  Roofline data saved to: {csv_path}")

    # Read back and display
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        print(f"\n  {'Operation':<25} {'AI (FLOP/byte)':>16} {'GFLOPS':>12} {'Bound'}")
        print(f"  {'-' * 25} {'-' * 16} {'-' * 12} {'-' * 10}")
        for row in reader:
            ai_val = float(row["arithmetic_intensity"])
            gflops_val = float(row["achieved_gflops"])
            bound = row["bound_type"]
            if bound not in ("roofline_bw", "roofline_compute"):
                print(f"  {row['operation']:<25} {ai_val:>16.4f} {gflops_val:>12.1f} {bound}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  KEY TAKEAWAYS")
    print(f"{'=' * 70}")
    print(f"""
  1. Memory-bound ops (elementwise, norm, activation) are limited by
     memory bandwidth, not compute. Their performance ceiling is
     {peak_bw:.0f} GB/s on {device_name}.

  2. Non-contiguous (strided, transposed) memory access can reduce
     bandwidth by 2-10x due to cache line underutilization.

  3. In transformer inference (decode phase), most operations between
     matmuls are memory-bound. Optimizing these with fusion kernels
     is critical for end-to-end latency.

  4. The roofline model visualizes this: operations with low arithmetic
     intensity (< ~10 FLOP/byte) are memory-bound; operations with
     high arithmetic intensity are compute-bound.
""")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
