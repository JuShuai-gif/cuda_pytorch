"""
线性量化 int8/int4/int2 与 K-Means 量化（第 05 讲）
=========================================================

实现可配置位宽的线性（仿射）量化、基于 K-means 的非线性量化，
以及量化级别与原始权重分布的可视化对比。

核心概念：
  - linear_quantize: 非对称仿射量化为 b 比特
  - dequantize: 从量化值重建近似的浮点值
  - kmeans_quantize: 通过 K-means 对权重聚类，将码本存储为量化值
  - 比较不同位宽（int8, int4, int2）下的误差（MSE, MAE, 余弦相似度）
  - 绘制权重重方图并叠加量化网格线

所有计算均在 CPU 上运行；无需 GPU。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# 使用非交互式后端，以便在没有显示器的情况下保存图像
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Chinese font configuration - fix missing Chinese characters in output images
# Noto Sans CJK JP covers all CJK characters (including Simplified Chinese)
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "AR PL UMing CN",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False  # prevent minus sign rendering issues

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

BITS_LIST: List[int] = [8, 4, 2]  # 待测试的位宽列表
SEED: int = 42  # 随机种子，保证可复现性
NUM_WEIGHTS: int = 5000  # 要生成的合成权重数量

# ---------------------------------------------------------------------------
# 线性（仿射）量化
# ---------------------------------------------------------------------------


def linear_quantize(
    tensor: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, float, int, float, float]:
    """使用非对称量化将浮点张量量化为 `bits` 比特的整数。

    仿射映射公式为：

        scale  = (x_max - x_min) / (2^bits - 1)
        zp     = round(-x_min / scale)     [截断到 [0, 2^bits - 1]]
        q      = round(x / scale + zp)     [截断到 [0, 2^bits - 1]]

    参数:
        tensor: 任意形状的 float32 张量。
        bits:   位宽（例如 8, 4, 2）。

    返回:
        (quantized_tensor_int, scale, zero_point, x_min, x_max) 元组。
    """
    if bits <= 0:
        raise ValueError(f"bits 必须为正数；当前值为 {bits}")

    qmin: int = 0  # 量化范围最小值
    qmax: int = int(2**bits - 1)  # 量化范围最大值

    x_min = tensor.min().item()
    x_max = tensor.max().item()

    # 当所有值相同时，避免除以零
    if x_max == x_min:
        scale = 1.0
        zp = 0
        q = torch.zeros_like(tensor, dtype=torch.float32).round().long()
        return q, scale, zp, x_min, x_max

    # 计算缩放因子: 将浮点范围映射到整数范围
    scale = (x_max - x_min) / (qmax - qmin)

    # 计算零点: 浮点值 0.0 对应的量化整数值
    zp_f = -x_min / scale
    zp = int(round(zp_f))
    zp = max(qmin, min(qmax, zp))  # 将零点截断到有效范围内

    # 量化: 将浮点值 x 映射到整数 q
    q = torch.round(tensor / scale + zp)
    q = torch.clamp(q, qmin, qmax).long()

    return q, scale, zp, x_min, x_max


def dequantize(q: torch.Tensor, scale: float, zp: int) -> torch.Tensor:
    """从量化整数重建近似的浮点值。

    参数:
        q:     量化后的整数张量 (int32/int64)。
        scale: 量化缩放因子（浮点数）。
        zp:    零点（整数）。

    返回:
        重建的 float32 张量。
    """
    return (q.float() - zp) * scale


def compute_quantization_error(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> Dict[str, float]:
    """计算原始张量与重建张量之间的误差指标。

    参数:
        original:      原始浮点张量。
        reconstructed: 反量化后的浮点张量。

    返回:
        包含以下键的字典: 'mse', 'mae', 'cosine_sim', 'max_abs_err'。
    """
    # 展平张量以便统一计算
    orig_flat = original.view(-1).float()
    recon_flat = reconstructed.view(-1).float()

    # 均方误差 (MSE)
    mse = torch.mean((orig_flat - recon_flat) ** 2).item()
    # 平均绝对误差 (MAE)
    mae = torch.mean(torch.abs(orig_flat - recon_flat)).item()
    # 最大绝对误差
    max_abs_err = torch.max(torch.abs(orig_flat - recon_flat)).item()

    # 余弦相似度: dot(a, b) / (||a|| * ||b||)
    dot = torch.dot(orig_flat, recon_flat).item()
    norm_orig = orig_flat.norm(p=2).item()
    norm_recon = recon_flat.norm(p=2).item()
    if norm_orig > 1e-12 and norm_recon > 1e-12:
        cosine_sim = dot / (norm_orig * norm_recon)
    else:
        cosine_sim = 1.0

    return {
        "mse": mse,
        "mae": mae,
        "max_abs_err": max_abs_err,
        "cosine_sim": cosine_sim,
    }


# ---------------------------------------------------------------------------
# K-Means 量化
# ---------------------------------------------------------------------------


def kmeans_quantize(
    tensor: torch.Tensor,
    bits: int,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用 K-means 聚类对权重进行量化。

    每个权重被分配到 k = 2^bits 个聚类之一。聚类中心构成码本（codebook）；
    分配索引作为量化表示存储。这是一种非均匀量化方案。

    如果 scipy.cluster.vq.kmeans2 可用则使用它；否则回退到纯 PyTorch 实现的
    Lloyd 算法。

    参数:
        tensor:   任意形状的 float32 张量。
        bits:     位宽（聚类数 = 2^bits）。
        max_iter: K-means 的最大迭代次数。
        tol:      收敛容差。

    返回:
        (assignments, centroids, reconstructed) 元组。
          assignments:   与原始形状相同的 int32 张量，值域为 [0, 2^bits - 1]。
          centroids:     形状为 (2^bits,) 的 float32 张量——码本。
          reconstructed: 与原始形状相同的反量化 float32 张量。
    """
    num_clusters = int(2**bits)
    # 将数据重塑为 (N, 1) 以便聚类
    data = tensor.view(-1, 1).float()

    try:
        from scipy.cluster.vq import kmeans2

        data_np = data.numpy()
        centroids_np, assignments_np = kmeans2(
            data_np[:, 0], num_clusters, iter=max_iter, thresh=tol, minit="points"
        )
        # kmeans2 返回一维 centroids；转换为 torch 张量
        centroids = torch.from_numpy(centroids_np).float()
        assignments = torch.from_numpy(assignments_np.astype(np.int32)).long()
    except ImportError:
        # 回退: 纯 PyTorch 实现的 K-means（Lloyd 算法）
        centroids = _kmeans_fallback(data.squeeze(-1), num_clusters, max_iter, tol)
        diffs = (data.squeeze(-1).unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = diffs.argmin(dim=1).long()

    # 重建: 将每个权重替换为其所属聚类的中心值
    reconstructed = centroids[assignments].view(tensor.shape)
    assignments = assignments.view(tensor.shape)

    return assignments, centroids, reconstructed


def _kmeans_fallback(
    data: torch.Tensor,
    num_clusters: int,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> torch.Tensor:
    """纯 PyTorch 实现的 K-means（Lloyd 算法），用于一维聚类。

    参数:
        data:         一维 float 张量 (N,)。
        num_clusters: 聚类数量。
        max_iter:     最大迭代次数。
        tol:          聚类中心移动的收敛容差。

    返回:
        形状为 (num_clusters,) 的聚类中心张量，按升序排列。
    """
    n = data.numel()
    # 通过均匀间隔的百分位数来初始化聚类中心
    sorted_data = torch.sort(data).values
    indices = torch.linspace(0, n - 1, num_clusters).long()
    centroids = sorted_data[indices].clone()

    for _iter in range(max_iter):
        # 将每个点分配到最近的聚类中心
        diffs = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = diffs.argmin(dim=1)

        # 更新聚类中心为各聚类内所有点的均值
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            mask = assignments == k
            if mask.sum() > 0:
                new_centroids[k] = data[mask].mean()
            else:
                new_centroids[k] = centroids[k]  # 如果聚类为空，保留旧中心

        # 检查聚类中心的变化是否小于容差
        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids

        if shift < tol:
            break

    return torch.sort(centroids).values


# ---------------------------------------------------------------------------
# 权重生成
# ---------------------------------------------------------------------------


def generate_synthetic_weights(
    num_weights: int = NUM_WEIGHTS,
    seed: int = SEED,
) -> torch.Tensor:
    """生成模拟真实权重分布的合成权重。

    分布由以下混合组成:
      - 高斯分布 N(0, 0.5) 构成权重的主体部分
      - 少量来自 N(0, 2.0) 的离群值以模拟厚尾分布
      - 负偏态分量使分布变得不对称

    参数:
        num_weights: 要生成的标量权重数量。
        seed:        随机种子，保证可复现性。

    返回:
        长度为 `num_weights` 的一维 float 张量。
    """
    torch.manual_seed(seed)

    # 主体: 以 0 为中心的正态分布
    bulk = torch.randn(int(num_weights * 0.85)) * 0.5

    # 离群值 / 厚尾
    tails = torch.randn(int(num_weights * 0.10)) * 2.0

    # 轻微正偏斜
    skewed = torch.randn(int(num_weights * 0.05)) * 0.8 + 1.5

    weights = torch.cat([bulk, tails, skewed])

    # 修整到精确数量
    if weights.numel() > num_weights:
        weights = weights[:num_weights]
    elif weights.numel() < num_weights:
        extra = torch.randn(num_weights - weights.numel()) * 0.5
        weights = torch.cat([weights, extra])

    return weights


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def plot_weight_histogram(
    weights: torch.Tensor,
    quant_info: Dict[int, Dict[str, object]],
    save_path: str = "quantization_histogram.png",
) -> None:
    """绘制原始权重直方图，并叠加量化网格线。

    对于每个位宽，量化级别（反量化后的值）以垂直虚线的形式绘制在
    直方图之上。

    参数:
        weights:     原始一维 float 张量。
        quant_info:  以位宽为键的字典:
                     {bits: {"levels": List[float], "scale": float, "zp": int}}
        save_path:   保存图像的路径 (PNG)。
    """
    w_np = weights.numpy()

    # 为每个位宽创建一个子图
    fig, axes = plt.subplots(len(BITS_LIST), 1, figsize=(10, 4 * len(BITS_LIST)))

    for idx, bits in enumerate(BITS_LIST):
        ax = axes[idx]
        # 绘制权重直方图
        ax.hist(w_np, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
        ax.set_title(f"权重直方图 + int{bits} 量化级别", fontsize=13)
        ax.set_xlabel("权重值", fontsize=11)
        ax.set_ylabel("频率", fontsize=11)

        # 将量化级别绘制为垂直线
        if bits in quant_info:
            levels = quant_info[bits].get("levels", [])
            if levels:
                for lv in levels:
                    ax.axvline(
                        x=lv,
                        color="red",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.6,
                    )
                # 添加一个代理元素用于图例
                ax.axvline(
                    x=levels[0],
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                    label=f"量化级别 ({len(levels)})",
                )
                ax.legend(loc="upper right", fontsize=9)

        # 添加注释文字框
        if bits in quant_info:
            scale = quant_info[bits].get("scale", 0.0)
            text = f"int{bits}: {len(quant_info[bits].get('levels', []))} 个级别, scale={scale:.4f}"
            ax.text(
                0.98,
                0.95,
                text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n权重直方图已保存至: {save_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    """运行完整的量化演示流水线。"""
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  第 05 讲: 线性量化 int8 / int4 / int2")
    print("=" * 70)

    # ---- 1. 生成合成权重 ------------------------------------------------
    print(f"\n[1] 正在生成 {NUM_WEIGHTS} 个合成权重 ...")
    weights = generate_synthetic_weights(NUM_WEIGHTS, SEED)
    print(f"  形状: {tuple(weights.shape)}")
    print(f"  最小值: {weights.min().item():.4f},  最大值: {weights.max().item():.4f}")
    print(f"  均值: {weights.mean().item():.4f},  标准差: {weights.std().item():.4f}")

    # ---- 2. 线性（仿射）量化 --------------------------------------------
    print("\n[2] 各种位宽下的线性（仿射）量化 ...")
    linear_results: Dict[int, Dict[str, object]] = {}
    quant_level_map: Dict[int, Dict[str, object]] = {}

    for bits in BITS_LIST:
        # 执行线性量化
        q, scale, zp, x_min, x_max = linear_quantize(weights, bits)
        # 反量化以评估误差
        reconstructed = dequantize(q, scale, zp)
        errors = compute_quantization_error(weights, reconstructed)

        # 计算唯一的量化级别（反量化后的值）
        unique_q = torch.unique(q).long()
        levels = ((unique_q.float() - zp) * scale).tolist()

        linear_results[bits] = {
            "q": q,
            "scale": scale,
            "zp": zp,
            "reconstructed": reconstructed,
            "errors": errors,
        }
        quant_level_map[bits] = {"levels": levels, "scale": scale, "zp": zp}

        print(
            f"  int{bits:>2d}:  "
            f"范围=[{x_min:.4f}, {x_max:.4f}], "
            f"scale={scale:.6f}, "
            f"zp={zp}, "
            f"级别数={len(levels)}, "
            f"MSE={errors['mse']:.6f}, "
            f"MAE={errors['mae']:.6f}, "
            f"max_err={errors['max_abs_err']:.6f}, "
            f"cos_sim={errors['cosine_sim']:.6f}"
        )

    # ---- 3. 误差对比总结 -----------------------------------------------
    print("\n[3] 量化误差对比:")
    print(
        f"  {'位宽':<12} {'MSE':>12} {'MAE':>12} {'最大绝对误差':>14} {'余弦相似度':>10}"
    )
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 14} {'-' * 10}")
    for bits in BITS_LIST:
        e = linear_results[bits]["errors"]
        print(
            f"  int{bits:<9d} {e['mse']:>12.6f} {e['mae']:>12.6f} "
            f"{e['max_abs_err']:>14.6f} {e['cosine_sim']:>10.6f}"
        )

    # ---- 4. K-Means 量化 -----------------------------------------------
    print("\n[4] K-Means 量化（非均匀） ...")
    kmeans_rec: Dict[int, torch.Tensor] = {}

    for bits in BITS_LIST:
        assignments, centroids, reconstructed = kmeans_quantize(weights, bits)
        errors = compute_quantization_error(weights, reconstructed)
        kmeans_rec[bits] = reconstructed

        print(
            f"  int{bits:>2d}:  "
            f"聚类数={centroids.numel()}, "
            f"MSE={errors['mse']:.6f}, "
            f"MAE={errors['mae']:.6f}, "
            f"max_err={errors['max_abs_err']:.6f}, "
            f"cos_sim={errors['cosine_sim']:.6f}"
        )

    # ---- 5. 线性量化 vs K-Means 量化对比 -------------------------------
    print("\n[5] 线性量化 vs K-Means 量化对比:")
    print(f"  {'位宽':<12} {'线性 MSE':>12} {'K-Means MSE':>12} {'改进':>14}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 14}")
    for bits in BITS_LIST:
        linear_mse = linear_results[bits]["errors"]["mse"]
        kmeans_mse = compute_quantization_error(weights, kmeans_rec[bits])["mse"]
        # K-Means 相对于线性量化的 MSE 改进百分比
        improvement = (
            (1.0 - kmeans_mse / linear_mse) * 100 if linear_mse > 1e-12 else 0.0
        )
        print(
            f"  int{bits:<9d} {linear_mse:>12.6f} {kmeans_mse:>12.6f} "
            f"{improvement:>13.2f}%"
        )

    # ---- 6. 可视化权重直方图 -------------------------------------------
    print("\n[6] 正在绘制带量化级别的权重直方图 ...")
    plot_weight_histogram(
        weights, quant_level_map, save_path="quantization_histogram.png"
    )

    # ---- 7. 总结 -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    print(f"  合成权重数量: {NUM_WEIGHTS}")
    print(f"  测试的位宽: {BITS_LIST}")
    print(f"  量化方法: 非对称仿射（线性）+ K-means（非均匀）")
    print(f"  图像已保存至: quantization_histogram.png")
    print("=" * 70)

    print("\n第 05 讲完成。")


if __name__ == "__main__":
    main()
