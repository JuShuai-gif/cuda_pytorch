"""
Arithmetic Intensity and Roofline Model (Lecture 02)
算术强度与 Roofline 模型（第 02 讲）

Analyses the arithmetic intensity (FLOPs per byte of data movement) for
key deep-learning operations and produces a roofline plot.
分析关键深度学习操作的算术强度（每字节数据传输对应的 FLOPs）并生成 Roofline 图。

  - compute_matmul_intensity: matrix multiply (GEMM)
    compute_matmul_intensity: 矩阵乘法（GEMM）的算术强度
  - compute_attention_intensity: scaled dot-product attention
    compute_attention_intensity: 缩放点积注意力的算术强度
  - compute_elementwise_intensity: pointwise ops (add, ReLU, etc.)
    compute_elementwise_intensity: 逐元素操作（加法、ReLU 等）的算术强度
  - plot_roofline: matplotlib roofline with memory-bound vs compute-bound regions
    plot_roofline: 使用 matplotlib 绘制包含内存受限和计算受限区域的 Roofline 图

All computations are CPU-only; requires torch, numpy, and matplotlib.
所有计算均在 CPU 上运行；需要 torch、numpy 和 matplotlib。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend -- works on headless servers
# 使用非交互式后端 -- 可在无显示器的服务器上运行

# ---------------------------------------------------------------------------
# 常量定义
# Constants
# ---------------------------------------------------------------------------

BYTES_PER_FP32: int = 4  # FP32 每个标量的字节数
BYTES_PER_FP16: int = 2  # FP16 每个标量的字节数
BYTES_PER_BF16: int = 2  # BF16 每个标量的字节数
BYTES_PER_INT8: int = 1  # INT8 每个标量的字节数

# Typical hardware peaks (for reference -- adjust to your device)
# 典型硬件峰值（仅供参考 -- 可根据你的设备调整）
# A100: 312 TFLOPS (fp16 tensor core), 2.0 TB/s HBM bandwidth
# A100：312 TFLOPS（fp16 张量核心），2.0 TB/s HBM 带宽
A100_PEAK_FP16_TFLOPS: float = 312.0
A100_PEAK_BANDWIDTH_GBPS: float = 2039.0  # ~2.0 TB/s

# V100: 125 TFLOPS (fp16 tensor core), 900 GB/s HBM bandwidth
# V100：125 TFLOPS（fp16 张量核心），900 GB/s HBM 带宽
V100_PEAK_FP16_TFLOPS: float = 125.0
V100_PEAK_BANDWIDTH_GBPS: float = 900.0


# ===========================================================================
# 算术强度计算
# Arithmetic intensity computations
# ===========================================================================


def compute_matmul_intensity(
    M: int, N: int, K: int, bytes_per_element: float = BYTES_PER_FP16
) -> Dict[str, float]:
    """Compute arithmetic intensity for a GEMM  C = A @ B.
    计算矩阵乘法 GEMM  C = A @ B 的算术强度。

    A: (M, K), B: (K, N), C: (M, N).

    FLOPs  = 2 * M * N * K          (each MAC = 1 add + 1 mult)
    FLOPs  = 2 × M × N × K          （每个 MAC = 1 次加法 + 1 次乘法）
    Bytes  = (M*K + K*N + M*N) * bytes_per_element
    字节数 = (M×K + K×N + M×N) × 每元素字节数
    AI     = FLOPs / Bytes           (FLOPs per byte)
    AI     = FLOPs / 字节数          （每字节的 FLOPs）

    Args:
        M: Rows of A / C.
           M: A / C 的行数。
        N: Columns of B / C.
           N: B / C 的列数。
        K: Inner dimension.
           K: 内积维度。
        bytes_per_element: Bytes per scalar (default 2 for fp16).
                           bytes_per_element: 每个标量的字节数（fp16 默认为 2）。

    Returns:
        dict with keys flops, bytes_read, bytes_written, arithmetic_intensity.
        返回包含 flops、bytes_read、bytes_written、arithmetic_intensity 的字典。
    """
    # 计算总 FLOPs
    flops: float = 2.0 * M * N * K
    # 计算各矩阵占用的字节数
    a_bytes = M * K * bytes_per_element
    b_bytes = K * N * bytes_per_element
    c_bytes = M * N * bytes_per_element
    total_bytes = a_bytes + b_bytes + c_bytes
    # 算术强度 = FLOPs / 总字节数
    ai = flops / total_bytes if total_bytes > 0 else 0.0

    return {
        "flops": flops,
        "bytes_read": a_bytes + b_bytes,
        "bytes_written": c_bytes,
        "total_bytes": total_bytes,
        "arithmetic_intensity": ai,
    }


def compute_attention_intensity(
    seq_len: int,
    d_model: int,
    bytes_per_element: float = BYTES_PER_FP16,
) -> Dict[str, float]:
    """Arithmetic intensity for a single-head scaled dot-product attention.
    计算单头缩放点积注意力的算术强度。

    Breakdown:
    分解如下：
      Phase 1 -- Q @ K^T:  (S,D) x (D,S) → (S,S)
      阶段 1 -- Q @ K^T：(S,D) × (D,S) → (S,S)
      Phase 2 -- softmax:  row-wise exp + normalise (negligible FLOPs,
                           but significant data movement)
      阶段 2 -- softmax：逐行 exp + 归一化（FLOPs 可忽略，但数据移动显著）
      Phase 3 -- Attn @ V: (S,S) x (S,D) → (S,D)
      阶段 3 -- Attn @ V：(S,S) × (S,D) → (S,D)

    Args:
        seq_len: Sequence length (S).
                 seq_len: 序列长度 (S)。
        d_model: Model / head dimension (D).
                 d_model: 模型/头维度 (D)。
        bytes_per_element: Bytes per scalar.
                           bytes_per_element: 每个标量的字节数。

    Returns:
        dict with flops, bytes, arithmetic_intensity, and per-phase breakdown.
        返回包含 flops、bytes、arithmetic_intensity 及各阶段分解的字典。
    """
    S = seq_len
    D = d_model

    # Phase 1: Q@K^T
    # 阶段 1：Q@K^T 的 FLOPs 和字节数
    qk_flops = 2.0 * S * S * D
    qk_bytes = (S * D + D * S + S * S) * bytes_per_element

    # Phase 2: softmax (approximate -- 5 ops per element)
    # 阶段 2：softmax 的 FLOPs 和字节数（近似 -- 每元素约 5 次操作）
    sm_flops = 5.0 * S * S
    sm_bytes = 2 * S * S * bytes_per_element  # read probs + write probs
    # 读取概率 + 写入概率

    # Phase 3: Attn@V
    # 阶段 3：Attn@V 的 FLOPs 和字节数
    av_flops = 2.0 * S * S * D
    av_bytes = (S * S + S * D + S * D) * bytes_per_element

    # 汇总各阶段
    total_flops = qk_flops + sm_flops + av_flops
    total_bytes = qk_bytes + sm_bytes + av_bytes
    ai = total_flops / total_bytes if total_bytes > 0 else 0.0

    return {
        "seq_len": S,
        "d_model": D,
        "flops": total_flops,
        "total_bytes": total_bytes,
        "arithmetic_intensity": ai,
        "phase_qk_flops": qk_flops,
        "phase_softmax_flops": sm_flops,
        "phase_av_flops": av_flops,
        "phase_qk_bytes": qk_bytes,
        "phase_softmax_bytes": sm_bytes,
        "phase_av_bytes": av_bytes,
    }


def compute_elementwise_intensity(
    N: int, bytes_per_element: float = BYTES_PER_FP32
) -> Dict[str, float]:
    """Arithmetic intensity for an elementwise binary operation  y = f(x1, x2).
    计算逐元素二元操作 y = f(x1, x2) 的算术强度。

    For a simple add or multiply:
    对于简单的加法或乘法：
        FLOPs  = N         (1 operation per element)
        FLOPs  = N         （每个元素 1 次操作）
        Bytes  = (2*N + N) * bytes_per_element = 3*N * bpw
        字节数 = (2×N + N) × 每元素字节数 = 3×N × bpw
        AI     = N / (3*N*bpw) = 1/(3*bpw)
        AI     = N / (3×N×bpw) = 1/(3×bpw)

    This is extremely low, meaning elementwise ops are almost always
    **memory-bound**.
    这个值极低，意味着逐元素操作几乎总是**内存受限**的。

    Args:
        N: Number of elements in the tensor.
           N: 张量中的元素数量。
        bytes_per_element: Bytes per scalar.
                           bytes_per_element: 每个标量的字节数。

    Returns:
        dict with flops, bytes, arithmetic_intensity.
        返回包含 flops、bytes、arithmetic_intensity 的字典。
    """
    flops: float = float(N)  # 每个元素一次操作
    bytes_read = 2 * N * bytes_per_element  # read two inputs：读取两个输入
    bytes_written = N * bytes_per_element  # 写入一个输出
    total_bytes = bytes_read + bytes_written
    ai = flops / total_bytes if total_bytes > 0 else 0.0

    return {
        "flops": flops,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "total_bytes": total_bytes,
        "arithmetic_intensity": ai,
    }


# ===========================================================================
# Roofline 模型
# Roofline model
# ===========================================================================


def plot_roofline(
    peak_flops_tflops: float,
    peak_bandwidth_gbps: float,
    operations: Optional[List[Dict[str, object]]] = None,
    save_path: Optional[str] = None,
    title: str = "Roofline Model",
) -> plt.Figure:
    """Plot a roofline model with peak compute and memory bandwidth.
    使用峰值计算能力和内存带宽绘制 Roofline 模型。

    The roofline consists of two regions:
    Roofline 包含两个区域：
      - **Memory-bound** (left of the ridge point):
        **内存受限**（岭点左侧）：
          Attainable FLOPs = AI × peak_bandwidth
          可达到的 FLOPs = AI × 峰值带宽
      - **Compute-bound** (right of the ridge point):
        **计算受限**（岭点右侧）：
          Attainable FLOPs = peak compute
          可达到的 FLOPs = 峰值计算能力

    The ridge point is at:
    岭点位于：
        AI_ridge = peak_flops / peak_bandwidth
        AI_ridge = 峰值 FLOPs / 峰值带宽

    Args:
        peak_flops_tflops: Peak compute throughput in TFLOPS (e.g. 312 for A100 fp16).
                           peak_flops_tflops: 峰值计算吞吐量，单位为 TFLOPS（如 A100 fp16 为 312）。
        peak_bandwidth_gbps: Peak memory bandwidth in GB/s (e.g. 2039 for A100 HBM).
                             peak_bandwidth_gbps: 峰值内存带宽，单位为 GB/s（如 A100 HBM 为 2039）。
        operations: Optional list of dicts with keys ``label``, ``ai`` (arithmetic
            intensity), and ``tflops`` (achieved TFLOPS).  Each is plotted as a
            scatter point on the roofline chart.
            operations: 可选的字典列表，包含 ``label``（标签）、``ai``（算术强度）
            和 ``tflops``（达到的 TFLOPS）键。每个操作将作为散点绘制在 Roofline 图上。
        save_path: If provided, save the figure to this path.
                   save_path: 如果提供，将图表保存到此路径。
        title: Chart title.
               title: 图表标题。

    Returns:
        The matplotlib Figure.
        返回 matplotlib Figure 对象。
    """
    # Convert units: TFLOPS -> FLOPS, GB/s -> B/s
    # 单位转换：TFLOPS -> FLOPS，GB/s -> B/s
    peak_flops = peak_flops_tflops * 1e12
    peak_bandwidth = peak_bandwidth_gbps * 1e9
    ridge_ai = peak_flops / peak_bandwidth  # FLOPs / byte at the ridge
    # 岭点处的 FLOPs / 字节

    # Generate AI axis (log scale, wide range)
    # 生成 AI 轴（对数刻度，宽范围）
    ai_min = 0.01
    ai_max = max(ridge_ai * 100, 10000)
    ai_range = np.logspace(math.log10(ai_min), math.log10(ai_max), 500)

    # Compute roofline: bounded by min(peak_flops, peak_bandwidth * AI)
    # 计算 roofline：取 min(峰值 FLOPs, 峰值带宽 × AI) 的上限
    roofline_flops = np.minimum(peak_flops, peak_bandwidth * ai_range)

    # -- Plot --
    # -- 绘图 --
    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline curve：Roofline 曲线
    ax.loglog(ai_range, roofline_flops, "b-", linewidth=2, label="Roofline")

    # Ridge point：岭点（内存受限与计算受限的分界线）
    ax.axvline(ridge_ai, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"Ridge = {ridge_ai:.1f} FLOPs/byte",
        xy=(ridge_ai, peak_flops * 0.5),
        xytext=(ridge_ai * 2, peak_flops * 0.15),
        arrowprops={"arrowstyle": "->", "color": "gray"},
        fontsize=9,
        color="gray",
        ha="left",
    )

    # Shaded regions：着色区域
    # 内存受限区域（岭点左侧）
    ax.fill_between(
        ai_range,
        roofline_flops,
        1,
        where=(ai_range <= ridge_ai),
        color="orange",
        alpha=0.1,
        label="Memory-bound",
    )
    # 计算受限区域（岭点右侧）
    ax.fill_between(
        ai_range,
        roofline_flops,
        1,
        where=(ai_range >= ridge_ai),
        color="green",
        alpha=0.1,
        label="Compute-bound",
    )

    # Operations scatter：操作散点图
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if operations:
        for i, op in enumerate(operations):
            label = str(op.get("label", f"op_{i}"))
            ai = float(op["ai"])  # type: ignore[arg-type]
            perf = float(op.get("tflops", 0)) * 1e12  # type: ignore[arg-type]
            color = op.get("color", colors[i % len(colors)])
            marker = op.get("marker", "o")
            ax.loglog(
                ai,
                min(perf, peak_flops),  # 实际性能不能超过峰值
                marker=marker,
                color=color,
                markersize=10,
                label=label,
                zorder=5,
            )

    # Formatting：格式设置
    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=12)
    ax.set_ylabel("Performance (FLOPs / s)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    # Add CPU/GPU reference annotations
    # 添加 CPU/GPU 参考注释
    ref_text = (
        f"Peak compute: {peak_flops_tflops:.0f} TFLOPS\n"
        f"Peak bandwidth: {peak_bandwidth_gbps:.0f} GB/s\n"
        f"Ridge point: {ridge_ai:.1f} FLOPs/byte"
    )
    ax.text(
        0.02,
        0.02,
        ref_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [roofline] figure saved to {save_path}")

    return fig


# ===========================================================================
# 独立演示
# Standalone demo
# ===========================================================================


def _demo() -> None:
    """Demonstrate arithmetic intensity calculations and roofline plot.
    演示算术强度计算和 Roofline 图绘制。
    """
    print("\n" + "=" * 68)
    print("  ARITHMETIC_INTENSITY.PY  --  DEMO")
    print("=" * 68)

    # ------------------------------------------------------------------
    # 1. Matmul intensity for various sizes
    # 1. 不同尺寸矩阵乘法的算术强度
    # ------------------------------------------------------------------
    print("\n  --- Matrix Multiply Arithmetic Intensity ---")
    matmul_configs = [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (128, 128, 4096),
    ]
    print(f"  {'(M,N,K)':<20} {'FLOPs':>16} {'Bytes':>16} {'AI (FLOP/byte)':>18}")
    print("  " + "-" * 68)
    for M, N, K in matmul_configs:
        info = compute_matmul_intensity(M, N, K, bytes_per_element=BYTES_PER_FP16)
        print(
            f"  ({M},{N},{K}){'':<{20 - len(f'({M},{N},{K})')}}"
            f" {info['flops']:>16,.0f}"
            f" {info['total_bytes']:>16,.0f}"
            f" {info['arithmetic_intensity']:>18.1f}"
        )
    print()

    # Key insight print：关键洞察输出
    large = compute_matmul_intensity(4096, 4096, 4096, BYTES_PER_FP16)
    print(
        f"  Large matmul AI: {large['arithmetic_intensity']:.1f} FLOP/byte"
        f" -- typically compute-bound on GPU"
        f"  -- 在 GPU 上通常为计算受限"
    )

    # ------------------------------------------------------------------
    # 2. Attention intensity at different sequence lengths
    # 2. 不同序列长度下的注意力算术强度
    # ------------------------------------------------------------------
    print("\n  --- Attention Arithmetic Intensity ---")
    print(f"  {'S':>6} {'FLOPs':>16} {'Bytes':>16} {'AI':>10}")
    print("  " + "-" * 48)
    for seq_len in [64, 128, 256, 512, 1024, 2048]:
        info = compute_attention_intensity(
            seq_len, d_model=64, bytes_per_element=BYTES_PER_FP16
        )
        print(
            f"  {seq_len:>6}"
            f" {info['flops']:>16,.0f}"
            f" {info['total_bytes']:>16,.0f}"
            f" {info['arithmetic_intensity']:>10.1f}"
        )
    print()

    # ------------------------------------------------------------------
    # 3. Elementwise intensity
    # 3. 逐元素操作的算术强度
    # ------------------------------------------------------------------
    print("  --- Elementwise Arithmetic Intensity (fp32) ---")
    for N in [1024, 65536, 1_000_000]:
        info = compute_elementwise_intensity(N, BYTES_PER_FP32)
        print(
            f"  N={N:>10,}  ->  AI = {info['arithmetic_intensity']:.4f} FLOP/byte"
            f"  (memory-bound)"
            f"  （内存受限）"
        )
    print()

    # ------------------------------------------------------------------
    # 4. Roofline plot
    # 4. Roofline 图
    # ------------------------------------------------------------------
    print("  --- Roofline Plot ---")
    operations = [
        {
            "label": f"Matmul {m}x{n}x{k}",
            "ai": compute_matmul_intensity(m, n, k, BYTES_PER_FP16)[
                "arithmetic_intensity"
            ],
            "tflops": compute_matmul_intensity(m, n, k, BYTES_PER_FP16)[
                "arithmetic_intensity"
            ]
            * A100_PEAK_BANDWIDTH_GBPS
            * 1e9
            / 1e12,
        }
        for (m, n, k) in [
            (64, 64, 64),
            (256, 256, 256),
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (128, 128, 4096),
        ]
    ]

    # 添加逐元素操作到操作列表中
    for label, ai in [
        (
            "Elementwise (fp32)",
            compute_elementwise_intensity(65536, BYTES_PER_FP32)[
                "arithmetic_intensity"
            ],
        ),
        (
            "Elementwise (fp16)",
            compute_elementwise_intensity(65536, BYTES_PER_FP16)[
                "arithmetic_intensity"
            ],
        ),
    ]:
        operations.append(
            {
                "label": label,
                "ai": ai,
                "tflops": ai * A100_PEAK_BANDWIDTH_GBPS * 1e9 / 1e12,
            }
        )

    # 添加注意力操作到操作列表中
    for S in [128, 512, 2048]:
        ops_info = compute_attention_intensity(S, 64, BYTES_PER_FP16)
        operations.append(
            {
                "label": f"Attention S={S} D=64",
                "ai": ops_info["arithmetic_intensity"],
                "tflops": ops_info["arithmetic_intensity"]
                * A100_PEAK_BANDWIDTH_GBPS
                * 1e9
                / 1e12,
            }
        )

    # Colour-code matmuls vs elementwise vs attention
    # 按操作类型进行颜色编码：矩阵乘法、逐元素操作、注意力
    for op in operations:
        lbl: str = str(op["label"])
        if lbl.startswith("Matmul"):
            op["color"] = "blue"
            op["marker"] = "s"
        elif lbl.startswith("Elementwise"):
            op["color"] = "red"
            op["marker"] = "o"
        elif lbl.startswith("Attention"):
            op["color"] = "purple"
            op["marker"] = "^"

    save_path = "/tmp/lecture_02_roofline.png"
    _fig = plot_roofline(
        peak_flops_tflops=A100_PEAK_FP16_TFLOPS,
        peak_bandwidth_gbps=A100_PEAK_BANDWIDTH_GBPS,
        operations=operations,
        save_path=save_path,
        title="Roofline Model -- NVIDIA A100 (fp16 tensor cores)",
    )

    print(f"\n  Roofline plot saved to: {save_path}")
    print("  (close any matplotlib windows to continue)")

    # ------------------------------------------------------------------
    # 5. Key takeaways
    # 5. 关键要点
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  KEY TAKEAWAYS")
    print("=" * 68)
    print(
        """
  1. Elementwise ops (ReLU, Add, LayerNorm):
  1. 逐元素操作（ReLU、Add、LayerNorm）：
     - AI ≈ 0.08 FLOP/byte (fp32)
     - Almost *always* memory-bound.
     - 几乎*总是*内存受限。
     - Optimisation: fuse them together (e.g. Conv→BN→ReLU fusion).
     - 优化策略：将它们融合在一起（例如 Conv→BN→ReLU 融合）。

  2. Small matmuls (M=N=K=64):
  2. 小矩阵乘法（M=N=K=64）：
     - AI ≈ 10-20 FLOP/byte (fp16)
     - Usually memory-bound; batching helps.
     - 通常内存受限；增加 batch 大小有帮助。

  3. Large matmuls (M=N=K=4096):
  3. 大矩阵乘法（M=N=K=4096）：
     - AI ≈ 1000+ FLOP/byte (fp16)
     - Compute-bound; limited by GPU FLOPs, not bandwidth.
     - 计算受限；受限于 GPU 的 FLOPs 而非带宽。

  4. Attention (Q@K^T):
  4. 注意力机制（Q@K^T）：
     - AI grows with sequence length (~O(S)), not O(S²)!
     - AI 随序列长度增长（~O(S)），而非 O(S²)！
     - Short sequences (<128): memory-bound.
     - 短序列（<128）：内存受限。
     - Long sequences (>1024): compute-bound.
     - 长序列（>1024）：计算受限。

  5. The roofline ridge point tells you the minimum AI needed
     to become compute-bound on your hardware.
  5. Roofline 岭点告诉你，在特定硬件上达到计算受限所需的最小 AI。
     - A100 fp16: 312 TFLOPS / 2.0 TB/s ≈ 153 FLOP/byte."""
    )
    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    _demo()
