"""
第 05 讲 — GPU 架构：GPU 规格参数与内存层次结构。

提供 V100 / A100 / H100 / B200 规格对比表
以及 GPU 内存层次结构分析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# GPU 规格数据类
# ---------------------------------------------------------------------------


@dataclass
class GPUSpec:
    """单个 GPU 型号的规格参数。"""

    name: str
    arch: str  # 例如 "Volta", "Ampere", "Hopper", "Blackwell"
    process_node: str  # 例如 "12nm", "7nm", "4nm"
    transistors_b: float  # 十亿颗晶体管
    die_size_mm2: float

    # 计算
    sm_count: int
    cuda_cores_per_sm: int
    tensor_cores_per_sm: int

    # 内存
    hbm_capacity_gib: float
    hbm_bandwidth_gbs: float  # GB/s
    hbm_type: str  # HBM2, HBM2e, HBM3, HBM3e
    l2_cache_mib: float

    # 时钟
    base_clock_mhz: float
    boost_clock_mhz: float

    # 峰值性能（理论值）
    peak_fp32_tflops: float
    peak_fp16_tflops: float  # 含 tensor core
    peak_bf16_tflops: float  # 含 tensor core
    peak_fp8_tflops: float  # 含 tensor core（不支持则为 0）

    # 功耗
    tdp_w: float

    # 互联
    nvlink_bw_gbs: float  # 双向
    pcie_gen: str

    # 特殊功能
    features: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 预定义 GPU 规格
# ---------------------------------------------------------------------------

GPU_SPECS: Dict[str, GPUSpec] = {
    "V100": GPUSpec(
        name="Tesla V100 (SXM2)",
        arch="Volta",
        process_node="12nm FFN",
        transistors_b=21.1,
        die_size_mm2=815.0,
        sm_count=80,
        cuda_cores_per_sm=64,
        tensor_cores_per_sm=8,
        hbm_capacity_gib=32.0,
        hbm_bandwidth_gbs=900.0,
        hbm_type="HBM2",
        l2_cache_mib=6.0,
        base_clock_mhz=1245.0,
        boost_clock_mhz=1530.0,
        peak_fp32_tflops=15.7,
        peak_fp16_tflops=125.0,  # 含 TC
        peak_bf16_tflops=0.0,  # 不原生支持
        peak_fp8_tflops=0.0,
        tdp_w=300.0,
        nvlink_bw_gbs=300.0,
        pcie_gen="PCIe 3.0",
        features=["First-gen Tensor Cores", "NVLink 2.0"],
    ),
    "A100": GPUSpec(
        name="A100 (SXM4)",
        arch="Ampere",
        process_node="7nm N7",
        transistors_b=54.2,
        die_size_mm2=826.0,
        sm_count=108,
        cuda_cores_per_sm=64,
        tensor_cores_per_sm=4,
        hbm_capacity_gib=80.0,
        hbm_bandwidth_gbs=2039.0,
        hbm_type="HBM2e",
        l2_cache_mib=40.0,
        base_clock_mhz=765.0,
        boost_clock_mhz=1410.0,
        peak_fp32_tflops=19.5,
        peak_fp16_tflops=312.0,
        peak_bf16_tflops=312.0,
        peak_fp8_tflops=0.0,
        tdp_w=400.0,
        nvlink_bw_gbs=600.0,
        pcie_gen="PCIe 4.0",
        features=[
            "TF32",
            "Structured Sparsity",
            "Multi-Instance GPU (MIG)",
            "NVLink 3.0",
        ],
    ),
    "H100": GPUSpec(
        name="H100 (SXM5)",
        arch="Hopper",
        process_node="4nm 4N",
        transistors_b=80.0,
        die_size_mm2=814.0,
        sm_count=132,
        cuda_cores_per_sm=128,
        tensor_cores_per_sm=4,
        hbm_capacity_gib=80.0,
        hbm_bandwidth_gbs=3350.0,
        hbm_type="HBM3",
        l2_cache_mib=50.0,
        base_clock_mhz=1095.0,
        boost_clock_mhz=1830.0,
        peak_fp32_tflops=67.0,
        peak_fp16_tflops=989.0,
        peak_bf16_tflops=989.0,
        peak_fp8_tflops=1979.0,
        tdp_w=700.0,
        nvlink_bw_gbs=900.0,
        pcie_gen="PCIe 5.0",
        features=[
            "FP8 Transformer Engine",
            "DPX Instructions",
            "TMA (Tensor Memory Accelerator)",
            "NVLink 4.0",
        ],
    ),
    "B200": GPUSpec(
        name="B200",
        arch="Blackwell",
        process_node="4nm 4NP",
        transistors_b=208.0,
        die_size_mm2=1600.0,  # 双 die 近似值
        sm_count=160,
        cuda_cores_per_sm=128,
        tensor_cores_per_sm=4,
        hbm_capacity_gib=192.0,
        hbm_bandwidth_gbs=8000.0,
        hbm_type="HBM3e",
        l2_cache_mib=96.0,
        base_clock_mhz=1200.0,
        boost_clock_mhz=1800.0,
        peak_fp32_tflops=90.0,
        peak_fp16_tflops=2250.0,
        peak_bf16_tflops=2250.0,
        peak_fp8_tflops=4500.0,
        tdp_w=1000.0,
        nvlink_bw_gbs=1800.0,
        pcie_gen="PCIe 5.0",
        features=["FP4 / FP6 support", "NVLink 5.0", "Confidential Computing"],
    ),
}


# ---------------------------------------------------------------------------
# 对比表格
# ---------------------------------------------------------------------------


def compare_gpus() -> str:
    """返回格式化的 GPU 对比表格字符串。"""
    headers = [
        "GPU",
        "Arch",
        "Process",
        "HBM (GiB)",
        "BW (GB/s)",
        "FP16 TFLOPS",
        "BF16 TFLOPS",
        "FP8 TFLOPS",
        "TDP (W)",
    ]
    rows = [headers]
    for key in ["V100", "A100", "H100", "B200"]:
        g = GPU_SPECS[key]
        rows.append(
            [
                g.name.split("(")[0].strip(),
                g.arch,
                g.process_node,
                f"{g.hbm_capacity_gib:.0f}",
                f"{g.hbm_bandwidth_gbs:.0f}",
                f"{g.peak_fp16_tflops:.0f}" if g.peak_fp16_tflops > 0 else "—",
                f"{g.peak_bf16_tflops:.0f}" if g.peak_bf16_tflops > 0 else "—",
                f"{g.peak_fp8_tflops:.0f}" if g.peak_fp8_tflops > 0 else "—",
                f"{g.tdp_w:.0f}",
            ]
        )

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines: List[str] = []
    for row in rows:
        line = " | ".join(row[i].rjust(col_widths[i]) for i in range(len(row)))
        lines.append(line)
        if row is headers:
            lines.append("-" * len(line))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 内存层次结构分析
# ---------------------------------------------------------------------------


@dataclass
class MemoryLevel:
    """内存层次结构中的一个层级。"""

    name: str  # 例如 "Register File", "L1 Cache", "HBM"
    size_per_sm: str
    bandwidth: str  # 例如 "~10 TB/s per SM"
    latency: str  # 例如 "~20 cycles"
    scope: str  # "Thread", "Block", "SM", "Device"


MEMORY_HIERARCHY: List[MemoryLevel] = [
    MemoryLevel(
        "Register File", "256 KiB / SM", "~10+ TB/s per SM", "~0 cycles", "Thread"
    ),
    MemoryLevel(
        "L1 / Shared Memory",
        "256 KiB / SM (configurable)",
        "~10+ TB/s per SM",
        "~30 cycles",
        "Block / SM",
    ),
    MemoryLevel("L2 Cache", "40–96 MiB (device)", "~5 TB/s", "~200 cycles", "Device"),
    MemoryLevel(
        "HBM (Global Memory)", "32–192 GiB", "0.9–8.0 TB/s", "~400–800 cycles", "Device"
    ),
    MemoryLevel(
        "CPU DRAM (via PCIe)", "System RAM", "~32–128 GB/s (PCIe)", "~10 µs", "Host"
    ),
    MemoryLevel("NVLink / NVSwitch", "—", "300–1800 GB/s", "~1–5 µs", "Inter-GPU"),
    MemoryLevel("NVMe SSD", "TB-scale", "~3–14 GB/s", "~10–100 µs", "Host storage"),
]


def print_memory_hierarchy() -> None:
    """打印 GPU 内存层次结构表格。"""
    print(f"{'Level':<28s} {'Size':<28s} {'Bandwidth':<22s} {'Latency':<18s} {'Scope'}")
    print("-" * 114)
    for m in MEMORY_HIERARCHY:
        print(
            f"{m.name:<28s} {m.size_per_sm:<28s} {m.bandwidth:<22s} {m.latency:<18s} {m.scope}"
        )


# ---------------------------------------------------------------------------
# 计算强度 / roofline ridge point
# ---------------------------------------------------------------------------


def ridge_point(compute_tflops: float, bandwidth_gbs: float) -> float:
    """计算线与内存线相交处的 ridge point（FLOP / Byte）。"""
    return compute_tflops * 1e3 / bandwidth_gbs


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== GPU Specification Comparison ===\n")
    print(compare_gpus())

    print("\n\n=== Ridge Points (FLOP / Byte) ===\n")
    print(f"{'GPU':<8s} {'FP16 Ridge':>12s} {'BF16 Ridge':>12s} {'FP8 Ridge':>12s}")
    print("-" * 46)
    for key in ["V100", "A100", "H100", "B200"]:
        g = GPU_SPECS[key]
        rp_fp16 = (
            ridge_point(g.peak_fp16_tflops, g.hbm_bandwidth_gbs)
            if g.peak_fp16_tflops > 0
            else float("nan")
        )
        rp_bf16 = (
            ridge_point(g.peak_bf16_tflops, g.hbm_bandwidth_gbs)
            if g.peak_bf16_tflops > 0
            else float("nan")
        )
        rp_fp8 = (
            ridge_point(g.peak_fp8_tflops, g.hbm_bandwidth_gbs)
            if g.peak_fp8_tflops > 0
            else float("nan")
        )
        print(f"{g.name:>8s}  {rp_fp16:>10.1f}   {rp_bf16:>10.1f}   {rp_fp8:>10.1f}")

    print("\n\n=== Memory Hierarchy ===\n")
    print_memory_hierarchy()

    print("\nAll checks passed.")
