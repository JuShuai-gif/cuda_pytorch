"""
语言模型训练的损失函数。

实现 cross-entropy 损失，支持可选的 label smoothing。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    计算语言建模的 cross-entropy 损失。

    在语言建模中，对于序列 [t1, t2, ..., tn]：
    - 输入: [t1, t2, ..., t_{n-1}]
    - 目标: [t2, t3, ..., tn]
    模型根据所有前置 token 预测下一个 token。

    Args:
        logits: 模型预测输出 [batch, seq_len, vocab_size]。
        targets: 真实标签 token ID [batch, seq_len]。
        ignore_index: 在损失计算中忽略的 token ID（例如 padding）。
        label_smoothing: 平滑因子（0 = 不平滑, 0.1 = 10% 平滑）。

    Returns:
        标量损失值。
    """
    # 展平为 [batch * seq_len, vocab_size] 和 [batch * seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat: torch.Tensor = logits.reshape(-1, vocab_size)
    targets_flat: torch.Tensor = targets.reshape(-1)

    if label_smoothing > 0:
        log_probs: torch.Tensor = F.log_softmax(logits_flat, dim=-1)
        nll_loss: torch.Tensor = F.nll_loss(
            log_probs, targets_flat, ignore_index=ignore_index, reduction="sum"
        )
        # 平滑损失：应用 ignore_index 掩码，使 padding token 不参与计算
        non_ignore_mask: torch.Tensor = (targets_flat != ignore_index).float()
        num_tokens: int = non_ignore_mask.sum().item()
        if num_tokens == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        smooth_loss: torch.Tensor = (-log_probs * non_ignore_mask.unsqueeze(-1)).sum()
        loss = (
            1.0 - label_smoothing
        ) * nll_loss / num_tokens + label_smoothing * smooth_loss / num_tokens
    else:
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            reduction="mean",
        )

    return loss


class CrossEntropyLoss(nn.Module):
    """
    用于语言建模的 cross-entropy 损失模块。

    Args:
        ignore_index: 在损失计算中忽略的 token ID。
        label_smoothing: label smoothing 因子。
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ignore_index: int = ignore_index
        self.label_smoothing: float = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失。详见 cross_entropy_loss()。"""
        return cross_entropy_loss(
            logits,
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


# 快速测试
if __name__ == "__main__":
    batch, seq, vocab = 2, 16, 100
    logits = torch.randn(batch, seq, vocab)
    targets = torch.randint(0, vocab, (batch, seq))

    loss_no_smooth = cross_entropy_loss(logits, targets, label_smoothing=0.0)
    loss_smooth = cross_entropy_loss(logits, targets, label_smoothing=0.1)

    assert loss_no_smooth > 0, "Loss should be positive"
    assert loss_smooth > 0, "Smoothed loss should be positive"

    print(f"Cross-entropy test passed!")
    print(f"  Loss (no smoothing): {loss_no_smooth:.4f}")
    print(f"  Loss (0.1 smoothing): {loss_smooth:.4f}")
