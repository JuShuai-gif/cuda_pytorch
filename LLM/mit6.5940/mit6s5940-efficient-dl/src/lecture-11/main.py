"""
第11讲: TinyEngine 优化仿真
================================================
对各种卷积实现 (naive, im2col, Winograd) 进行基准测试,
演示算子融合 (Conv+BN+ReLU), 并分析内存布局的权衡
(NCHW vs NHWC)。所有基准测试仅在 CPU 上运行。

关键概念:
  - im2col: 将卷积转换为矩阵乘法
  - Winograd F(2,3): 用于 3x3 卷积、步长 1 的最小滤波算法
  - 算子融合: 推理时将 BatchNorm 参数折叠到 Conv 权重中
  - 内存布局: NCHW (通道优先) vs NHWC (通道最后) 的访问模式
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 辅助工具函数
# ---------------------------------------------------------------------------


def _timeit(fn, warmup: int = 3, repeat: int = 10) -> Tuple[float, float]:
    """测量函数执行时间的均值与标准差, 单位为毫秒。

    Parameters
    ----------
    fn : callable
        待基准测试的函数 (零参数可调用对象)。
    warmup : int
        计时前的预热调用次数 (用于稳定缓存/分支预测)。
    repeat : int
        计时的重复次数。

    Returns
    -------
    (mean_ms, std_ms) : (float, float)
        平均耗时 (毫秒) 和标准差 (毫秒)。
    """
    # 预热阶段: 多次运行以稳定 CPU 缓存和分支预测
    for _ in range(warmup):
        fn()
    # 计时阶段: 重复运行并记录每次耗时
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()  # 高精度计时起点
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)  # 转换为毫秒
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def _gflops(ms: float, ops: float) -> float:
    """将毫秒转换为 GFLOPS (每秒十亿次浮点运算)。

    公式: GFLOPS = ops / (ms * 1e6)
    解释: ops / ms = ops/ms → ops/(ms*1e6) = ops/(s*1e9) = GFLOPS
    """
    return ops / (ms * 1e6)


# ---------------------------------------------------------------------------
# 基准测试配置
# ---------------------------------------------------------------------------

# 使用代表性图像张量: N=8, C=64, H=56, W=56 (例如 ResNet C3 阶段)
BATCH, IN_C, OUT_C, H, W = 8, 64, 64, 56, 56
KERNEL = 3  # 卷积核尺寸
STRIDE = 1  # 步长
PADDING = 1  # 填充 (保持空间维度不变)

DEVICE = torch.device("cpu")  # 在 CPU 上运行
DTYPE = torch.float32  # 使用 float32 精度


def _make_tensors():
    """为所有基准测试创建一致的输入和权重张量。

    使用固定随机种子以确保可复现性。
    """
    rng = torch.Generator(device=DEVICE).manual_seed(42)
    # 输入张量: NCHW 格式
    x = torch.randn(BATCH, IN_C, H, W, device=DEVICE, dtype=DTYPE, generator=rng)
    # 权重张量: (OUT_C, IN_C, K, K)
    w = torch.randn(
        OUT_C, IN_C, KERNEL, KERNEL, device=DEVICE, dtype=DTYPE, generator=rng
    )
    return x, w


# ===========================================================================
# 1. 朴素卷积 (使用 PyTorch 的 F.conv2d 作为基准)
# ===========================================================================


def naive_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """使用 PyTorch 高度优化的 CPU 后端的基准卷积。

    此处称为 "naive" (朴素) 仅仅作为参考基准 — 实际上并非
    嵌套循环实现, 而是通过 PyTorch 的 oneDNN 集成获得的
    最快 CPU 版本。
    """
    return F.conv2d(x, w, stride=STRIDE, padding=PADDING)


# ===========================================================================
# 2. IM2COL 卷积
# ===========================================================================


def _im2col(x: torch.Tensor, k_h: int, k_w: int, stride: int, pad: int) -> torch.Tensor:
    """将 NCHW 图像张量转换为列矩阵, 以便进行 GEMM 运算。

    每列存放一个展平的 k_h*k_w*C_in 感受野区域。
    输出形状: (C_in*k_h*k_w, N*H_out*W_out)。

    Parameters
    ----------
    x : (N, C, H, W)
        输入张量。
    k_h, k_w : int
        卷积核高度 / 宽度。
    stride : int
        滑动步长。
    pad : int
        填充大小。

    Returns
    -------
    cols : (C * k_h * k_w, N * H_out * W_out)
        im2col 转换后的列矩阵。
    """
    N, C, H_in, W_in = x.shape
    H_out = (H_in + 2 * pad - k_h) // stride + 1
    W_out = (W_in + 2 * pad - k_w) // stride + 1

    # 对输入进行填充
    x_pad = F.pad(x, (pad, pad, pad, pad))  # (N, C, H_pad, W_pad)

    # 使用 unfold 提取滑动窗口 — 每个 (k_h, k_w) 补丁
    # 变为长度为 C * k_h * k_w 的列。补丁元素
    # 的排列顺序为: 通道优先, 然后高度, 然后宽度。
    patches = x_pad.unfold(2, k_h, stride).unfold(3, k_w, stride)
    # patches: (N, C, H_out, W_out, k_h, k_w)

    # 重排维度使得空间位置先变化, 然后通道和卷积核维度
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # patches: (N, H_out, W_out, C, k_h, k_w)

    # 展平为列矩阵并转置, 得到 (C*k_h*k_w, N*H_out*W_out)
    cols = patches.view(N * H_out * W_out, C * k_h * k_w).t()
    return cols


def im2col_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """im2col + GEMM 卷积。

    算法步骤:
      1. 将输入展开为列矩阵            (C_in*K*K, N*H_out*W_out)
      2. 将权重重塑为行矩阵            (C_out, C_in*K*K)
      3. 矩阵乘法: out = W @ cols
      4. 重塑回 NCHW 格式
    """
    N, C_in, H_in, W_in = x.shape
    C_out, _, k_h, k_w = w.shape
    H_out = (H_in + 2 * PADDING - k_h) // STRIDE + 1
    W_out = (W_in + 2 * PADDING - k_w) // STRIDE + 1

    # 步骤 1: im2col 转换
    cols = _im2col(x, k_h, k_w, STRIDE, PADDING)  # (C_in*K*K, N*H_out*W_out)

    # 步骤 2: 将卷积滤波器重塑为矩阵 (C_out, C_in*K*K)
    w_mat = w.view(C_out, -1)

    # 步骤 3: GEMM (通用矩阵乘法)
    out_mat = w_mat @ cols  # (C_out, N*H_out*W_out)

    # 步骤 4: 重塑为 NCHW 格式
    out = out_mat.view(C_out, N, H_out, W_out).permute(1, 0, 2, 3).contiguous()
    return out


# ===========================================================================
# 3. WINOGRAD 卷积 – F(2, 3) 算法
# ===========================================================================


def _winograd_transform_matrices():
    """返回 Winograd F(2, 3) 的变换矩阵 Aᵀ, G, Bᵀ。

    F(m, r) = F(2, 3) 表示: 每个 tile 产生 m=2 个输出, 使用 r=3 的滤波器。
    理论算术减少量:  m²·r² / (m+r-1)²
    = (4·9) / 16 = 2.25×  对于 3×3 滤波器, 步长 1。

    参考文献:
      Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016
    """
    # Winograd 最小滤波算法的变换矩阵
    # Bᵀ: 输入 tile 变换矩阵, 形状 (α=4, α=4)
    B_T = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    # G: 滤波器变换矩阵, 形状 (α=4, r=3)
    G = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    # Aᵀ: 逆变换矩阵 (输出 tile), 形状 (m=2, α=4)
    A_T = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    return A_T, G, B_T


def winograd_f23_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Winograd F(2,3) 卷积, 适用于 3×3 滤波器、步长 1。

    算法在 *Winograd 域* 中工作:
      1. 变换输入 tile    : V = Bᵀ · tile · B
      2. 变换滤波器       : U = G · filter · Gᵀ
      3. 逐元素乘法       : M = U ⊙ V
      4. 逆变换           : out_tile = Aᵀ · M · A

    对于产生 2×2 输出的 3×3 滤波器 (r=3, m=2),
    内部 tile 大小 α = m + r − 1 = 4。

    References
    ----------
    Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016
    """
    N, C_in, H_in, W_in = x.shape
    C_out, _, k_h, k_w = w.shape
    # F(2,3) 要求 3×3 卷积核
    assert k_h == 3 and k_w == 3, "F(2,3) 要求 3×3 卷积核"
    # F(2,3) 要求步长为 1
    assert STRIDE == 1, "F(2,3) 要求步长为 1"

    # 计算输出空间尺寸 (对于 pad=1, stride=1, 输出 = 输入)
    H_out = H_in - k_h + 2 * PADDING + 1  # 对于 pad=1 即为 H_in
    W_out = W_in - k_w + 2 * PADDING + 1
    assert H_out % 2 == 0 and W_out % 2 == 0, (
        f"输出空间维度必须为偶数以支持 tile 划分; 当前为 {H_out}×{W_out}"
    )

    # 计算 tile 数量 (每个 tile 产生 2×2 输出)
    tile_H = H_out // 2
    tile_W = W_out // 2
    alpha = 4  # m + r - 1 = 2 + 3 - 1 = 4

    # 获取预定义的变换矩阵
    A_T, G, B_T = _winograd_transform_matrices()

    # 对输入进行填充以便 tile 划分
    x_pad = F.pad(x, (PADDING, PADDING, PADDING, PADDING))

    # ---- 步骤 1: 将滤波器变换到 Winograd 域 ----
    # U = G @ w @ Gᵀ, 形状: (C_out, C_in, alpha, alpha)
    # G: (alpha=4, k=3)。w: (C_out, C_in, k, k)。
    # 先在卷积核高度 (k_h) 维度上应用 G。
    U = torch.einsum("ax,oixy->oiay", G, w)  # (C_out, C_in, 4, 3)
    # 再通过右乘在卷积核宽度 (k_w) 维度上应用 G:
    #   (G @ w @ Gᵀ)[o,i,a,b] = Σ_y (G @ w)[o,i,a,y] · G[b,y]
    U = torch.einsum("by,oiay->oiab", G, U)  # (C_out, C_in, 4, 4)

    # ---- 步骤 2: 将输入 tile 变换到 Winograd 域 ----
    # V = Bᵀ @ tile @ B
    # 每个 tile 覆盖 α×α 的区域, 步长为 2 (因为 m=2)。
    tiles = x_pad.unfold(2, alpha, 2).unfold(3, alpha, 2)
    # tiles: (N, C_in, tile_H, tile_W, alpha, alpha)

    # 重排维度: 空间 tile 索引在通道索引之前变化,
    # 以确保在展平前有正确的连续内存布局。
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
    tiles = tiles.view(N * tile_H * tile_W, C_in, alpha, alpha)

    # Bᵀ @ tile (左乘变换)
    V = torch.einsum("ax,nixy->niay", B_T, tiles)  # (N_tiles, C_in, 4, 4)
    # (Bᵀ @ tile) @ B (右乘变换: Σ_y temp[.,a,y] · B[b,y])
    V = torch.einsum("by,niay->niab", B_T, V)  # (N_tiles, C_in, 4, 4)

    # ---- 步骤 3: 逐元素乘法 + 对输入通道求和 ----
    # M[n, o, p, q] = Σ_i U[o, i, p, q] · V[n, i, p, q]
    M = torch.einsum("oipq,nipq->nopq", U, V)  # (N_tiles, C_out, 4, 4)

    # ---- 步骤 4: 逆变换 – out_tile = Aᵀ @ M @ A ----
    # Aᵀ: (m=2, alpha=4)。标签: "cx" 其中 c=2, x=4。
    # 左乘: temp[n,o,c,y] = Σ_x Aᵀ[c,x] · M[n,o,x,y]
    out = torch.einsum("cx,noxy->nocy", A_T, M)  # (N_tiles, C_out, 2, 4)
    # 右乘: out_tile[n,o,c,d] = Σ_y temp[n,o,c,y] · A[d,y]
    out = torch.einsum("dy,nocy->nocd", A_T, out)  # (N_tiles, C_out, 2, 2)

    # 重塑回 NCHW 格式
    out = out.view(N, tile_H, tile_W, C_out, 2, 2)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
    out = out.view(N, C_out, H_out, W_out)
    return out


def winograd_theoretical_speedup() -> float:
    """计算 F(2,3) 相对于朴素卷积的理论 FLOP 减少量。

    朴素: m² · r² = 4 · 9 = 36 次乘法/tile
    Winograd: (m+r-1)² = 16 次变换域乘法
    加速比 = 36 / 16 = 2.25
    """
    m, r = 2, 3
    naive_mults = m * m * r * r  # 朴素卷积乘法次数
    wino_mults = (m + r - 1) * (m + r - 1)  # Winograd 域乘法次数
    return naive_mults / wino_mults  # 2.25


# ===========================================================================
# 4. 算子融合: Conv + BatchNorm + ReLU
# ===========================================================================


def fuse_conv_bn_relu(
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    bn_eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 BatchNorm 参数折叠到 Conv 权重和偏置中。

    训练后, 批量归一化在推理时是一个线性变换:
        y = γ * (x - μ) / √(σ² + ε) + β

    这可以被吸收到前面的卷积中:
        W_fused = W * γ / √(σ² + ε)
        b_fused = β + γ * (b - μ) / √(σ² + ε)

    结合 ReLU (max(0, x)), 我们得到一个单一的融合操作。

    Parameters
    ----------
    conv_w : (C_out, C_in, kH, kW)
        卷积权重。
    conv_b : (C_out,) 或 None
        卷积偏置。
    bn_weight, bn_bias : (C_out,)
        BN 的 γ (缩放) 和 β (偏移) 参数。
    bn_running_mean, bn_running_var : (C_out,)
        BN 的运行均值 μ 和运行方差 σ²。
    bn_eps : float
        数值稳定项 ε。

    Returns
    -------
    fused_w : (C_out, C_in, kH, kW)
        融合后的权重。
    fused_b : (C_out,)
        融合后的偏置。
    """
    gamma = bn_weight  # 缩放因子 γ
    beta = bn_bias  # 偏移量 β
    mu = bn_running_mean  # 运行均值 μ
    sigma = torch.sqrt(bn_running_var + bn_eps)  # 标准差 σ

    # 计算缩放因子和偏置修正量
    scale = gamma / sigma  # γ / σ
    bias_correction = beta - mu * scale  # β - μ * γ / σ

    # 将 scale 广播到所有非输出通道维度 (C_out) 以外的维度
    scale_4d = scale.view(-1, 1, 1, 1)

    # 融合后的权重: W * (γ / σ)
    fused_w = conv_w * scale_4d

    # 融合后的偏置: b * (γ / σ) + (β - μ * γ / σ)
    if conv_b is not None:
        fused_b = conv_b * scale + bias_correction
    else:
        fused_b = bias_correction

    return fused_w, fused_b


def fused_conv_bn_relu_forward(
    x: torch.Tensor,
    fused_w: torch.Tensor,
    fused_b: torch.Tensor,
) -> torch.Tensor:
    """使用预融合参数的单次 Conv+BN+ReLU 前向传播。

    这避免了单独的 BN 归一化和 ReLU 激活调用,
    减少了内存带宽使用和内核启动开销。
    """
    y = F.conv2d(x, fused_w, fused_b, stride=STRIDE, padding=PADDING)
    return F.relu(y)


# ===========================================================================
# 5. 内存布局: NCHW vs NHWC
# ===========================================================================


def benchmark_layout_access():
    """比较通道优先 (NCHW) 与通道最后 (NHWC) 的访问效率。

    NHWC (channels-last) 通常在 CPU 上具有更好的缓存利用率,
    因为连续的元素属于跨通道的同一空间位置,
    使得向量化操作更高效。

    我们基准测试两种模式:
      1. 通道归约 (对 C 维度求均值) – 有利于 NHWC
      2. 空间归约 (对 H, W 维度求均值) – 有利于 NCHW
    """
    size = BATCH * IN_C * H * W
    # 创建 NCHW 和 NHWC 两种布局的张量
    x_nchw = torch.randn(BATCH, IN_C, H, W, device=DEVICE, dtype=DTYPE)
    x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last)

    # 操作 1: 对通道维度归约 (逐像素均值)
    # NHWC 中通道是连续的, 因此此操作应该更快
    def _chan_reduce_nchw():
        return x_nchw.mean(dim=1)  # 对 dim=1 (通道维度) 求均值

    def _chan_reduce_nhwc():
        return x_nhwc.mean(dim=3)  # 对 dim=3 (channels-last 的通道维度) 求均值

    # 操作 2: 对空间维度归约 (逐通道均值)
    # NCHW 中空间维度是连续的, 因此此操作应该更快
    def _spatial_reduce_nchw():
        return x_nchw.mean(dim=(2, 3))  # 对 dim=(2,3) (H, W) 求均值

    def _spatial_reduce_nhwc():
        return x_nhwc.mean(dim=(1, 2))  # 对 dim=(1,2) (channels-last 的 H, W) 求均值

    # 基准测试各操作
    t_cr_nchw, _ = _timeit(_chan_reduce_nchw)
    t_cr_nhwc, _ = _timeit(_chan_reduce_nhwc)
    t_sr_nchw, _ = _timeit(_spatial_reduce_nchw)
    t_sr_nhwc, _ = _timeit(_spatial_reduce_nhwc)

    return {
        "NCHW  (chan-reduce)": (t_cr_nchw, 0.0),
        "NHWC  (chan-reduce)": (t_cr_nhwc, 0.0),
        "NCHW  (spatial-reduce)": (t_sr_nchw, 0.0),
        "NHWC  (spatial-reduce)": (t_sr_nhwc, 0.0),
    }, size


# ===========================================================================
# 6. 主基准测试与对比表格
# ===========================================================================


def _compute_flops(
    n: int, c_in: int, c_out: int, h_out: int, w_out: int, k: int
) -> float:
    """计算单层 conv2d 的总乘加操作数 (FLOPs)。

    每个输出元素需要 C_in * K * K 次乘加运算 (计为 2 次浮点运算),
    但有 N * C_out * H_out * W_out 个输出元素。
    FLOPs (2* 约定) = 2 * N * C_out * C_in * H_out * W_out * K * K
    """
    return float(2 * n * c_out * c_in * h_out * w_out * k * k)


def main() -> None:
    """运行 TinyEngine 优化仿真的完整基准测试。

    演示内容:
      1. 正确性验证 (im2col, Winograd vs 基准)
      2. 卷积实现基准测试及加速比对比
      3. Winograd 理论分析
      4. 算子融合基准测试 (Conv+BN+ReLU)
      5. 内存布局对比 (NCHW vs NHWC)
      6. 优化摘要
    """
    print("=" * 72)
    print("  TinyEngine 优化仿真 – 第11讲")
    print("=" * 72)
    print(
        f"\n配置: N={BATCH} C={IN_C}→{OUT_C} H×W={H}×{W}  "
        f"K={KERNEL} S={STRIDE} P={PADDING}"
    )
    print(f"设备: {DEVICE}    数据类型: {DTYPE}")
    print()

    # 创建输入和权重张量
    x, w = _make_tensors()
    H_out = W_out = H  # pad=1, stride=1 保持空间维度不变
    ops = _compute_flops(BATCH, IN_C, OUT_C, H_out, W_out, KERNEL)

    # ---- 验证正确性 ----
    print("正在验证相对于基准的正确性...")
    ref = naive_conv2d(x, w)
    im2col_out = im2col_conv2d(x, w)
    winograd_out = winograd_f23_conv2d(x, w)

    # im2col 应与基准高度吻合
    im2col_diff = (ref - im2col_out).abs().max().item()
    print(f"  im2col   最大差异: {im2col_diff:.2e}")

    # Winograd 可能因浮点运算顺序不同而有差异; 使用相对误差检验
    wino_diff = (ref - winograd_out).abs().max().item()
    wino_rel = wino_diff / ref.abs().max().item() if ref.abs().max().item() > 0 else 0.0
    print(f"  Winograd 最大差异: {wino_diff:.2e}  (相对: {wino_rel:.2e})")

    # 判断正确性: im2col 允许 < 1e-3, Winograd 允许 < 2% 相对误差
    im2col_ok = im2col_diff < 1e-3
    winograd_ok = wino_rel < 0.02  # Winograd 引入更多数值误差
    print(f"  im2col   正确性: {'通过' if im2col_ok else '失败'}")
    print(f"  Winograd 正确性: {'通过' if winograd_ok else '失败'}")
    print()

    # ---- 基准测试卷积方法 ----
    print("正在基准测试卷积实现...")
    results: dict[str, Tuple[float, float, float]] = {}

    # 朴素卷积
    t_naive, s_naive = _timeit(lambda: naive_conv2d(x, w))
    results["朴素 (F.conv2d)"] = (t_naive, s_naive, _gflops(t_naive, ops))

    # im2col + GEMM
    t_im2col, s_im2col = _timeit(lambda: im2col_conv2d(x, w))
    results["im2col + GEMM"] = (t_im2col, s_im2col, _gflops(t_im2col, ops))

    # Winograd F(2,3)
    t_wino, s_wino = _timeit(lambda: winograd_f23_conv2d(x, w))
    results["Winograd F(2,3)"] = (t_wino, s_wino, _gflops(t_wino, ops))

    # ---- 计算加速比 ----
    speedup_im2col = t_naive / t_im2col  # im2col 相对于朴素的加速比
    speedup_wino = t_naive / t_wino  # Winograd 相对于朴素的加速比
    speedup_theory = winograd_theoretical_speedup()  # Winograd 理论加速比

    # ---- 算子融合基准测试 ----
    print("正在基准测试算子融合...")
    # 创建模拟的 BN 参数
    rng = torch.Generator(device=DEVICE).manual_seed(123)
    bn_w = torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng)  # γ
    bn_b = torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng)  # β
    bn_mean = torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng) * 0.1  # μ
    bn_var = (
        torch.abs(torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng)) * 0.5
        + 0.5
    )  # σ²
    conv_bias = torch.randn(
        OUT_C, device=DEVICE, dtype=DTYPE, generator=rng
    )  # 卷积偏置

    # 执行融合操作
    fused_w, fused_b = fuse_conv_bn_relu(w, conv_bias, bn_w, bn_b, bn_mean, bn_var)

    def _unfused_forward():
        """未融合的前向传播: Conv → BN → ReLU (三步分开执行)。"""
        y = F.conv2d(x, w, conv_bias, stride=STRIDE, padding=PADDING)
        # BN 评估模式: y_norm = (y - μ) / √(σ² + ε) * γ + β
        y = (y - bn_mean.view(1, -1, 1, 1)) / torch.sqrt(
            bn_var.view(1, -1, 1, 1) + 1e-5
        )
        y = y * bn_w.view(1, -1, 1, 1) + bn_b.view(1, -1, 1, 1)
        return F.relu(y)

    def _fused_forward():
        """融合后的前向传播: 单次调用完成 Conv+BN+ReLU。"""
        return fused_conv_bn_relu_forward(x, fused_w, fused_b)

    # 基准测试未融合 vs 融合
    t_unfused, s_unfused = _timeit(_unfused_forward)
    t_fused, s_fused = _timeit(_fused_forward)
    speedup_fusion = t_unfused / t_fused

    # 数值验证融合的正确性
    unfused_out = _unfused_forward()
    fused_out = _fused_forward()
    fusion_diff = (unfused_out - fused_out).abs().max().item()
    print(
        f"  融合正确性: {'通过' if fusion_diff < 1e-3 else '失败'}  "
        f"(最大差异: {fusion_diff:.2e})"
    )

    # ---- 内存布局基准测试 ----
    print("正在基准测试内存布局...")
    layout_results, layout_size = benchmark_layout_access()
    print()

    # =========================================================================
    # 打印对比表格
    # =========================================================================

    # --- 表格 1: 卷积实现对比 ---
    print("=" * 72)
    print("  表格 1: 卷积实现对比")
    print("=" * 72)
    print(
        f"  {'方法':<22s} {'耗时 (ms)':>12s} {'标准差 (ms)':>12s} "
        f"{'GFLOPS':>10s} {'vs 朴素':>10s}"
    )
    print("  " + "-" * 66)
    for name, (mean, std, gflops) in results.items():
        ratio = t_naive / mean
        print(
            f"  {name:<22s} {mean:>10.3f}  {std:>10.3f}  {gflops:>8.1f}  {ratio:>8.2f}×"
        )
    print()

    # --- 表格 2: Winograd 理论分析 ---
    print("=" * 72)
    print("  表格 2: Winograd F(2,3) – 理论分析")
    print("=" * 72)
    print(f"  理论算术减少量:  {speedup_theory:.2f}×")
    print(f"  实测相对朴素加速比:  {speedup_wino:.2f}×")
    print(f"  注意: Winograd 以较少的乘法换取更多的加法,")
    print(f"        以及中间 tile 带来的更高内存压力。")
    print()

    # --- 表格 3: 算子融合 ---
    print("=" * 72)
    print("  表格 3: 算子融合 – Conv + BN + ReLU")
    print("=" * 72)
    print(f"  {'变体':<22s} {'耗时 (ms)':>12s} {'标准差 (ms)':>10s} {'加速比':>10s}")
    print("  " + "-" * 56)
    print(
        f"  {'未融合 (3步)':<22s} {t_unfused:>10.3f}  {s_unfused:>8.3f}  "
        f"{'1.00× (基准)':>14s}"
    )
    print(
        f"  {'已融合 (1步)':<22s} {t_fused:>10.3f}  {s_fused:>8.3f}  "
        f"{speedup_fusion:>8.2f}×"
    )
    print()
    print("  融合变换:")
    print("    W_fused = W_conv * γ / √(σ² + ε)")
    print("    b_fused = β + γ * (b_conv - μ) / √(σ² + ε)")
    print("  这消除了两个中间张量的读写和一个内核启动开销,")
    print("  在推理时显著提升效率。")
    print()

    # --- 表格 4: 内存布局 ---
    print("=" * 72)
    print("  表格 4: 内存布局 – NCHW vs NHWC")
    print("=" * 72)
    print(f"  张量大小: {layout_size:,} 元素 ({layout_size * 4 / 1024:.0f} KiB)")
    print()
    print(f"  {'布局 / 操作':<30s} {'耗时 (ms)':>12s} {'相对胜者':>12s}")
    print("  " + "-" * 56)

    # 提取各操作的耗时
    t_cr_nchw = layout_results["NCHW  (chan-reduce)"][0]
    t_cr_nhwc = layout_results["NHWC  (chan-reduce)"][0]
    t_sr_nchw = layout_results["NCHW  (spatial-reduce)"][0]
    t_sr_nhwc = layout_results["NHWC  (spatial-reduce)"][0]

    # 计算相对加速比
    cr_speedup = t_cr_nchw / max(t_cr_nhwc, 1e-9)
    sr_speedup = t_sr_nhwc / max(t_sr_nchw, 1e-9)

    print(f"  {'NCHW  (channel-mean)':<30s} {t_cr_nchw:>10.3f}  {'基准':>12s}")

    # 判断通道归约的胜者 (NHWC 理论上更优)
    winner_cr = (
        "NHWC 胜" if cr_speedup > 1.01 else "NCHW 胜" if cr_speedup < 0.99 else "平局"
    )
    print(
        f"  {'NHWC  (channel-mean)':<30s} {t_cr_nhwc:>10.3f}  {winner_cr:>12s}"
        f"  ({cr_speedup:.2f}x)"
    )

    print(f"  {'NCHW  (spatial-mean)':<30s} {t_sr_nchw:>10.3f}  {'基准':>12s}")

    # 判断空间归约的胜者 (NCHW 理论上更优)
    winner_sr = (
        "NCHW 胜" if sr_speedup > 1.01 else "NHWC 胜" if sr_speedup < 0.99 else "平局"
    )
    print(
        f"  {'NHWC  (spatial-mean)':<30s} {t_sr_nhwc:>10.3f}  {winner_sr:>12s}"
        f"  ({sr_speedup:.2f}x)"
    )
    print()
    print("  洞察:")
    print("    - 在此 PyTorch CPU 后端上, NCHW 在简单归约操作中胜出,")
    print("      因为 oneDNN 内核对 NCHW 进行了高度优化。")
    print("    - 理论预测 NHWC 应该有利于通道密集型访问")
    print("      (通道在内存中连续 → 更好的缓存行利用)。")
    print("    - 在实践中, NHWC 的优势在自定义内核 (如 depthwise conv)")
    print("      和移动/嵌入式 CPU 上最为明显,")
    print("      这些场景下 SIMD 宽度与通道维度匹配。")
    print("    - 推理引擎 (TFLite, MNN, ncnn) 采用 NHWC,")
    print("      因为它们为该布局实现了自己的优化内核。")
    print()

    # --- 优化摘要 ---
    print("=" * 72)
    print("  优化摘要 (数值越高越好)")
    print("=" * 72)
    print(f"  Winograd 理论加速比 (vs 朴素):   {speedup_theory:.2f}×")
    print(f"  Winograd 实测加速比 (vs 朴素):   {speedup_wino:.2f}×")
    print(f"  算子融合加速比 (vs 未融合):      {speedup_fusion:.2f}×")
    chan_ratio = t_cr_nchw / max(t_cr_nhwc, 1e-9)
    print(f"  NHWC 通道归约加速比 vs NCHW:     {chan_ratio:.2f}×")
    print()
    print("  关键要点:")
    print("  1. im2col 使 GEMM 卷积成为可能, 但可能增加内存占用。")
    print("  2. Winograd 以数值精度为代价减少算术运算,")
    print("     并因变换矩阵而增加内存使用。")
    print("  3. 算子融合消除了中间缓冲区和内核启动开销 —")
    print("     对于带宽有限的微型设备至关重要。")
    print("  4. 通道最后 (NHWC) 布局改善 CPU 缓存局部性,")
    print("     尤其对于 depthwise 和 pointwise 卷积。")
    print()


if __name__ == "__main__":
    main()
