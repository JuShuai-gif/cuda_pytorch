"""
带有 Bradley-Terry 成对比较损失的 reward model。

实现一个微型 transformer 版 reward model，将 token 序列映射为
标量奖励，然后使用 Bradley-Terry 目标在合成偏好 pair 上训练：

    L = -log sigma(r_chosen - r_rejected)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 微型 Transformer Reward Model
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """注入到 embedding 流中的正弦位置编码。"""

    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)  # (max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将位置编码添加到形状为 (batch, seq, embed_dim) 的 *x* 上。"""
        return x + self.pe[:, : x.size(1)]


class RewardModel(nn.Module):
    """为 token 序列输出标量奖励的微型 transformer。

    架构：Embedding -> PositionalEncoding -> N 层 transformer
    encoder -> mean pool -> linear -> scalar。
    """

    def __init__(
        self,
        vocab_size: int = 200,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """返回每个序列的标量奖励。

        Args:
            input_ids: 形状为 (batch, seq_len) 的 LongTensor。

        Returns:
            形状为 (batch,) 的 FloatTensor，包含奖励分数。
        """
        # Embedding 并添加位置信息
        emb = self.embedding(input_ids)  # (B, S, E)
        emb = self.pos_encoder(emb)
        # 通过 transformer encoder
        enc_out = self.encoder(emb)  # (B, S, E)
        # 在序列维度上做 mean pool
        pooled = enc_out.mean(dim=1)  # (B, E)
        # 投影到标量奖励
        reward = self.head(pooled).squeeze(-1)  # (B,)
        return reward


# ---------------------------------------------------------------------------
# Bradley-Terry 成对损失
# ---------------------------------------------------------------------------


def bradley_terry_loss(
    r_chosen: torch.Tensor,
    r_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry 成对比较损失。

    L = -log sigma(r_chosen - r_rejected)

    Args:
        r_chosen:  对偏好（被选中）序列的预测奖励。
        r_rejected: 对非偏好（未被选中）序列的预测奖励。

    Returns:
        对 batch 取平均后的标量损失。
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def pairwise_accuracy(
    r_chosen: torch.Tensor,
    r_rejected: torch.Tensor,
) -> float:
    """r_chosen > r_rejected 的 pair 占比。"""
    return (r_chosen > r_rejected).float().mean().item()


# ---------------------------------------------------------------------------
# 合成偏好数据生成
# ---------------------------------------------------------------------------


def generate_synthetic_pairs(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """创建合成的偏好 pair。

    "chosen" 序列在其 token id 的一个子集上添加了一个小的正向偏置，
    使得训练良好的模型能够学会将它们与 "rejected" 序列区分开。

    Returns:
        (chosen_ids, rejected_ids, dummy_rewards)，其中 dummy_rewards
        是仅用于合理性检查的 ground-truth 标量值。
    """
    # 两个序列都是随机采样的，但 chosen 序列在其 token 分布中
    # 具有系统性的更高"质量信号"。
    chosen_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    # rejected 序列平均使用词汇表中下半部分的 token，
    # 使它们可以被区分。
    rejected_ids = torch.randint(1, max(2, vocab_size // 3), (batch_size, seq_len))

    # 合成的"真实"奖励（不用于训练，仅用于合理性检查）
    true_r_c = chosen_ids.float().mean(dim=1)  # 平均 id 越高 -> 奖励越高
    true_r_r = rejected_ids.float().mean(dim=1)
    return chosen_ids, rejected_ids, true_r_c, true_r_r


# ---------------------------------------------------------------------------
# 训练演示
# ---------------------------------------------------------------------------


def main() -> None:
    """在合成偏好 pair 上训练 reward model 并展示损失。"""
    torch.manual_seed(42)

    # 超参数（微型模型，快速训练）
    vocab_size = 200
    embed_dim = 64
    num_layers = 2
    num_heads = 4
    seq_len = 32
    batch_size = 64
    num_steps = 200
    lr = 3e-4

    model = RewardModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    print(
        f"{'step':>6s}  {'loss':>10s}  {'acc':>8s}  {'r_c mean':>10s}  {'r_r mean':>10s}"
    )
    print("-" * 55)

    for step in range(1, num_steps + 1):
        chosen, rejected, _, _ = generate_synthetic_pairs(
            batch_size, seq_len, vocab_size
        )

        r_c = model(chosen)  # (batch,)
        r_r = model(rejected)  # (batch,)

        loss = bradley_terry_loss(r_c, r_r)
        acc = pairwise_accuracy(r_c, r_r)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            print(
                f"{step:>6d}  {loss.item():>10.4f}  {acc:>8.3f}  "
                f"{r_c.mean().item():>10.4f}  {r_r.mean().item():>10.4f}"
            )

    # 最终评估
    print("\n--- Final evaluation ---")
    model.eval()
    with torch.no_grad():
        chosen, rejected, true_r_c, true_r_r = generate_synthetic_pairs(
            256, seq_len, vocab_size
        )
        r_c = model(chosen)
        r_r = model(rejected)
        acc = pairwise_accuracy(r_c, r_r)
        loss = bradley_terry_loss(r_c, r_r)
        print(f"Test loss : {loss.item():.4f}")
        print(f"Test acc  : {acc:.3f}")
        print(f"r_chosen  : mean={r_c.mean().item():.4f}, std={r_c.std().item():.4f}")
        print(f"r_rejected: mean={r_r.mean().item():.4f}, std={r_r.std().item():.4f}")
        print("(r_chosen 应显著高于 r_rejected)")


if __name__ == "__main__":
    main()
