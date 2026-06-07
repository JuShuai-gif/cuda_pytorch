"""
语言模型的 Perplexity（困惑度）计算。

Perplexity（困惑度，PPL）是语言模型的标准内在评估指标。
它衡量模型对给定文本的"惊讶"程度。困惑度越低，表示性能越好。

核心概念：
  - Perplexity = exp(cross-entropy loss)
  - PPL 为 N 表示模型如同在每一步都必须从 N 个等可能选项中
    均匀选择一样困惑
  - PPL 在留出的测试集上计算
  - 子词分词会影响 PPL（不同分词器之间不能直接比较）

公式：
  PPL = exp( -1/N * Σ log P(token_i | context) )
     = exp( cross_entropy_loss )
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# =========================================================================
# 核心 Perplexity 计算
# =========================================================================


def compute_perplexity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> float:
    """
    根据 logits 和目标 token id 计算 perplexity。

    Args:
        logits: 模型输出的 logits，形状为 (batch, seq_len, vocab_size)
        target_ids: 真实 token id，形状为 (batch, seq_len)
        ignore_index: 在 loss 计算中忽略的 token id（例如 padding）
        reduction: "mean" 表示对所有 token 取平均，
                   "sum" 表示总的负对数似然，
                   "none" 表示逐 token 的值

    Returns:
        Perplexity 值（若 reduction 为 "mean" 或 "sum" 则为标量，
        否则为逐 token 的值）
    """
    # Cross-entropy loss: -log P(target | context)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )

    if reduction == "none":
        return torch.exp(loss).tolist()

    return float(torch.exp(loss).item())


def compute_perplexity_from_loss(loss: float) -> float:
    """
    将 cross-entropy loss 转换为 perplexity。

    Args:
        loss: 平均 cross-entropy loss

    Returns:
        Perplexity 值
    """
    return float(math.exp(loss))


def compute_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    计算平均 cross-entropy loss（不做指数运算）。

    Args:
        logits: 模型输出的 logits，形状为 (batch, seq_len, vocab_size)
        target_ids: 真实 token id，形状为 (batch, seq_len)
        ignore_index: 要忽略的 token id

    Returns:
        平均 cross-entropy loss
    """
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
    )
    return float(loss.item())


# =========================================================================
# 数据集级别 Perplexity
# =========================================================================


def evaluate_perplexity_on_dataset(
    logits_list: list[torch.Tensor],
    target_list: list[torch.Tensor],
    ignore_index: int = -100,
) -> dict[str, float]:
    """
    从多个 batch 计算整体 perplexity。

    在做指数运算前正确地在数据集的所有 token 上累积 cross-entropy，
    从而正确处理变长序列。

    Args:
        logits_list: logits 张量列表，每个形状为 (batch, seq_len, vocab_size)
        target_list: 目标张量列表，每个形状为 (batch, seq_len)
        ignore_index: 要忽略的 token id

    Returns:
        包含 "loss"、"ppl"、"total_tokens" 键的字典
    """
    total_loss = 0.0
    total_tokens = 0

    for logits, targets in zip(logits_list, target_list):
        # 统计有效（未被忽略）的 token 数量
        valid_mask = targets != ignore_index
        num_valid = int(valid_mask.sum().item())
        if num_valid == 0:
            continue

        # 计算该 batch 的负对数似然之和
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss += float(loss_sum.item())
        total_tokens += num_valid

    if total_tokens == 0:
        return {"loss": float("inf"), "ppl": float("inf"), "total_tokens": 0}

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "total_tokens": total_tokens,
    }


def compute_per_token_perplexity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> list[float]:
    """
    计算逐 token 的 perplexity 值。

    可用于分析模型对哪些 token 最感惊讶。

    Args:
        logits: (batch, seq_len, vocab_size)
        target_ids: (batch, seq_len)
        ignore_index: 要忽略的 token id

    Returns:
        逐 token 的 perplexity 值列表（仅包含有效 token）
    """
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = target_ids.reshape(-1) != ignore_index
    return torch.exp(nll[valid]).tolist()


# =========================================================================
# 演示
# =========================================================================


def demo_basic_perplexity() -> None:
    """演示基础的 perplexity 计算。"""
    print("=" * 70)
    print("Perplexity Calculation Demo")
    print("=" * 70)

    torch.manual_seed(42)

    # 模拟一个小 batch 的模型输出
    batch_size = 2
    seq_len = 8
    vocab_size = 100

    # 随机 logits（模拟模型输出）
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # 目标 token（真实标签）
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(
        f"\nInput shapes: logits={list(logits.shape)}, targets={list(target_ids.shape)}"
    )
    print(f"Vocab size: {vocab_size}")

    # 计算 perplexity
    ppl = compute_perplexity(logits, target_ids)
    loss = compute_loss(logits, target_ids)

    print(f"\n  Cross-entropy loss: {loss:.4f}")
    print(f"  Perplexity:          {ppl:.4f}")
    print(f"  exp(loss):           {math.exp(loss):.4f}  (verification)")

    # 带 padding（ignore_index）
    print("\n--- With Padding ---")
    target_with_pad = target_ids.clone()
    target_with_pad[:, -2:] = -100  # 忽略最后 2 个 token
    ppl_pad = compute_perplexity(logits, target_with_pad, ignore_index=-100)
    loss_pad = compute_loss(logits, target_with_pad, ignore_index=-100)
    print(f"  Loss (ignoring last 2): {loss_pad:.4f}")
    print(f"  PPL (ignoring last 2):  {ppl_pad:.4f}")


def demo_perplexity_comparison() -> None:
    """比较不同模拟模型质量下的 perplexity。"""
    print("\n" + "=" * 70)
    print("Perplexity Comparison: Model Quality")
    print("=" * 70)

    torch.manual_seed(123)

    vocab_size = 100
    seq_len = 20
    target_ids = torch.randint(0, vocab_size, (1, seq_len))

    # 通过缩放 logit 尖锐度来模拟不同质量的模型
    print(f"\n  Target tokens: {target_ids[0, :8].tolist()}...")

    qualities = [
        ("Random (uniform)", 0.0),  # PPL ≈ vocab_size
        ("Poor (scale=0.5)", 0.5),
        ("Medium (scale=1.0)", 1.0),
        ("Good (scale=2.0)", 2.0),
        ("Excellent (scale=5.0)", 5.0),
    ]

    for name, sharpness in qualities:
        # 创建 logits，其中正确 token 的 logit 被 sharpness 增强
        base_logits = torch.randn(1, seq_len, vocab_size)
        # 增加正确 token 的 logit 值
        for t in range(seq_len):
            correct_id = target_ids[0, t].item()
            base_logits[0, t, correct_id] += sharpness * 3.0

        ppl = compute_perplexity(base_logits, target_ids)
        print(f"  {name:<25} PPL = {ppl:>8.2f}")


def demo_dataset_perplexity() -> None:
    """演示数据集级别的 perplexity 评估。"""
    print("\n" + "=" * 70)
    print("Dataset-Level Perplexity Evaluation")
    print("=" * 70)

    torch.manual_seed(7)

    vocab_size = 80
    num_batches = 4

    logits_batches = []
    target_batches = []

    # 模拟多个不同长度的 batch
    for i in range(num_batches):
        batch_size = 1
        seq_len = 5 + i * 3  # 变长序列: 5, 8, 11, 14
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits_batches.append(logits)
        target_batches.append(targets)

        # 每个 batch 的 PPL
        batch_ppl = compute_perplexity(logits, targets)
        print(f"\n  Batch {i}: seq_len={seq_len}, PPL={batch_ppl:.2f}")

    # 数据集级别的 PPL（正确加权）
    result = evaluate_perplexity_on_dataset(logits_batches, target_batches)
    print(f"\n  --- Dataset Results ---")
    print(f"  Total tokens evaluated: {result['total_tokens']}")
    print(f"  Average loss:           {result['loss']:.4f}")
    print(f"  Overall PPL:            {result['ppl']:.2f}")

    # 验证：错误的平均方式（对每个 batch 的 PPL 取均值）
    naive_avg = sum(
        compute_perplexity(l, t) for l, t in zip(logits_batches, target_batches)
    ) / len(logits_batches)
    print(f"\n  Naive average PPL:      {naive_avg:.2f}")
    print(f"  (This is WRONG - doesn't weight by token count)")
    print(f"  Proper PPL:             {result['ppl']:.2f}")


def demo_per_token_perplexity() -> None:
    """展示逐 token perplexity 进行分析。"""
    print("\n" + "=" * 70)
    print("Per-Token Perplexity Analysis")
    print("=" * 70)

    torch.manual_seed(1)

    seq_len = 15
    vocab_size = 50
    target_ids = torch.randint(0, vocab_size, (1, seq_len))

    # 让某些 token 比其他 token 更加令人惊讶
    logits = torch.randn(1, seq_len, vocab_size) * 0.5
    # Token 3: 非常令人惊讶（正确 token 的 logit 很低）
    logits[0, 3, target_ids[0, 3]] = -3.0
    # Token 7: 非常容易预测
    logits[0, 7, target_ids[0, 7]] = 10.0
    # Token 11: 非常令人惊讶
    logits[0, 11, target_ids[0, 11]] = -5.0

    per_token_ppl = compute_per_token_perplexity(logits, target_ids)

    print(f"\n  Token-level PPL values:")
    for i, ppl in enumerate(per_token_ppl):
        marker = ""
        if ppl > 20:
            marker = " <-- very surprising!"
        elif ppl < 2:
            marker = " <-- very predictable"
        print(
            f"    pos {i:>2}: target={target_ids[0, i].item():>3}, PPL={ppl:>8.2f}{marker}"
        )

    print(f"\n  Overall PPL: {compute_perplexity(logits, target_ids):.2f}")
    print(f"  Max PPL:     {max(per_token_ppl):.2f}")
    print(f"  Min PPL:     {min(per_token_ppl):.2f}")


def main() -> None:
    demo_basic_perplexity()
    demo_perplexity_comparison()
    demo_dataset_perplexity()
    demo_per_token_perplexity()


if __name__ == "__main__":
    main()
