"""
第04讲 — 训练：cross-entropy loss 的实现与验证。

提供一个数值稳定的 cross-entropy loss，并与 PyTorch 内置实现进行对比。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cross-entropy loss
# ---------------------------------------------------------------------------


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """数值稳定的 cross-entropy loss。

    参数
    ----------
    logits : (B, V) 或 (B, S, V) 浮点张量。
        未归一化的 logits。
    targets : (B,) 或 (B, S) 长整型张量。
        真实类别索引。
    ignore_index : int
        在 loss 计算中需要忽略的目标值。
    reduction : str
        'none'、'mean' 或 'sum'。
    label_smoothing : float
        label smoothing 因子，取值范围 [0, 1)。

    返回
    -------
    loss : 标量或逐元素张量。
    """
    # 如果是 3D 则展平
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits = logits.reshape(-1, V)
        targets = targets.reshape(-1)

    V = logits.size(-1)

    # 使用 log-softmax 以保证数值稳定
    log_probs = F.log_softmax(logits, dim=-1)  # (N, V)

    # 收集目标类别的 log-prob。
    # 将超出范围的目标（例如 ignore_index = -100）临时 clamp 到 0，
    # 以避免 gather 报错；这些值稍后会被 mask 掉。
    safe_targets = targets.clamp(min=0, max=V - 1)
    nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)  # (N,)

    # Label smoothing
    if label_smoothing > 0.0:
        smooth = -log_probs.mean(dim=-1)  # 对词表取平均
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth

    # 将需要忽略的索引 mask 掉。当指定了 ignore_index（通常为 -100，即 < 0）
    # 时，始终应用 mask。
    mask = (targets != ignore_index).float()
    nll = nll * mask

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    # reduction == "mean"
    return nll.sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def verify_cross_entropy() -> None:
    """将自定义 cross_entropy 与 torch.nn.functional.cross_entropy 进行比较。"""
    torch.manual_seed(42)

    B, V = 32, 1000
    logits = torch.randn(B, V)
    targets = torch.randint(0, V, (B,))

    # 无 label smoothing
    loss_custom = cross_entropy(logits, targets, ignore_index=-100)
    loss_torch = F.cross_entropy(logits, targets, reduction="mean")
    assert torch.allclose(loss_custom, loss_torch, atol=1e-5), (
        f"Custom: {loss_custom:.6f}, Torch: {loss_torch:.6f}"
    )
    print(f"No smoothing — custom: {loss_custom:.6f}, torch: {loss_torch:.6f}  ✓")

    # 带 label smoothing
    smoothing = 0.1
    loss_custom_s = cross_entropy(logits, targets, label_smoothing=smoothing)
    loss_torch_s = F.cross_entropy(
        logits, targets, label_smoothing=smoothing, reduction="mean"
    )
    assert torch.allclose(loss_custom_s, loss_torch_s, atol=1e-5), (
        f"Custom smoothed: {loss_custom_s:.6f}, Torch: {loss_torch_s:.6f}"
    )
    print(
        f"Label smoothing 0.1 — custom: {loss_custom_s:.6f}, torch: {loss_torch_s:.6f}  ✓"
    )

    # 带 ignore_index
    targets[0] = -100
    loss_ignored_c = cross_entropy(logits, targets, ignore_index=-100)
    loss_ignored_t = F.cross_entropy(
        logits, targets, ignore_index=-100, reduction="mean"
    )
    assert torch.allclose(loss_ignored_c, loss_ignored_t, atol=1e-5), (
        f"Custom ignored: {loss_ignored_c:.6f}, Torch: {loss_ignored_t:.6f}"
    )
    print(
        f"Ignore index −100 — custom: {loss_ignored_c:.6f}, torch: {loss_ignored_t:.6f}  ✓"
    )

    # 3D logits (B, S, V)
    BS, S, V = 4, 16, 100
    logits_3d = torch.randn(BS, S, V)
    targets_3d = torch.randint(0, V, (BS, S))
    loss_3d_c = cross_entropy(logits_3d, targets_3d)
    loss_3d_t = F.cross_entropy(
        logits_3d.reshape(-1, V), targets_3d.reshape(-1), reduction="mean"
    )
    assert torch.allclose(loss_3d_c, loss_3d_t, atol=1e-5)
    print(f"3D input — custom: {loss_3d_c:.6f}, torch: {loss_3d_t:.6f}  ✓")


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    verify_cross_entropy()
    print("\nAll checks passed.")
