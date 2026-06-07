"""
从零实现的 DPO（Direct Preference Optimization，直接偏好优化）损失函数。

实现了 DPO 论文（Rafailov et al., 2023）中描述的 DPO 损失，
该损失直接优化策略以满足人类偏好，无需训练单独的 reward model。

    L_DPO = -E[log sigma(beta * (log(pi(yw|x)/ref(yw|x))
                                  - log(pi(yl|x)/ref(yl|x))))]

提供两种变体：
- dpo_loss:          接受完整的 log-probability 张量和 token 索引。
- dpo_loss_logp:     接受预计算的 log-ratio 向量（更轻量级）。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DPO 核心损失函数
# ---------------------------------------------------------------------------


def dpo_loss(
    pi_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    yw_idxs: torch.Tensor,
    yl_idxs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO（直接偏好优化）损失函数（token 索引形式）。

    给定来自 policy 模型和 reference 模型的每个 token 的
    log-probability，此函数提取被选中的 (yw) 和未被选中的 (yl)
    响应所对应的子序列，求和它们的 log-prob，并计算 DPO 目标。

    Args:
        pi_logps:  **policy** 模型的 log-probability。
                   形状 (batch, seq_len)。
        ref_logps: **reference** 模型的 log-probability。
                   形状 (batch, seq_len)。
        yw_idxs:   **被选中（winning）** token 的索引。
                   形状 (batch, n_chosen)。
        yl_idxs:   **未被选中（losing）** token 的索引。
                   形状 (batch, n_rejected)。
        beta:      温度参数，控制 policy 偏离 reference 时受到
                   的惩罚强度。

    Returns:
        对 batch 取平均后的标量 DPO 损失。
    """
    # 对被选中/未被选中的 token 位置求和 log-probability
    pi_yw = pi_logps.gather(1, yw_idxs).sum(dim=1)  # (batch,)
    pi_yl = pi_logps.gather(1, yl_idxs).sum(dim=1)  # (batch,)
    ref_yw = ref_logps.gather(1, yw_idxs).sum(dim=1)  # (batch,)
    ref_yl = ref_logps.gather(1, yl_idxs).sum(dim=1)  # (batch,)

    # 被选中和未被选中的 log(pi / ref)
    pi_log_ratio = pi_yw - pi_yl  # log(pi_w) - log(pi_l)
    ref_log_ratio = ref_yw - ref_yl  # log(ref_w) - log(ref_l)

    # DPO 损失：-log sigma(beta * (pi_log_ratio - ref_log_ratio))
    loss = -F.logsigmoid(beta * (pi_log_ratio - ref_log_ratio)).mean()
    return loss


def dpo_loss_logp(
    pi_log_ratios: torch.Tensor,
    ref_log_ratios: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO 损失变体，接受预计算的 log-ratio。

    Args:
        pi_log_ratios:  每个 batch 项的 log(pi(yw|x)) - log(pi(yl|x))。
                        形状 (batch,)。
        ref_log_ratios: 每个 batch 项的 log(ref(yw|x)) - log(ref(yl|x))。
                        形状 (batch,)。
        beta:           温度 / KL 惩罚系数。

    Returns:
        对 batch 取平均后的标量 DPO 损失。
    """
    loss = -F.logsigmoid(beta * (pi_log_ratios - ref_log_ratios)).mean()
    return loss


# ---------------------------------------------------------------------------
# 工具函数：计算隐含奖励和准确率
# ---------------------------------------------------------------------------


def implied_reward(
    pi_log_ratios: torch.Tensor,
    ref_log_ratios: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO 参数化下的隐含奖励：r(x,y) = beta * log(pi/ref)。"""
    return beta * (pi_log_ratios - ref_log_ratios)


def preference_accuracy(
    pi_log_ratios: torch.Tensor,
    ref_log_ratios: torch.Tensor,
) -> float:
    """policy 对被选中响应赋予比 reference 更高相对概率的 pair 占比。"""
    return (pi_log_ratios > ref_log_ratios).float().mean().item()


# ---------------------------------------------------------------------------
# 合成演示
# ---------------------------------------------------------------------------


def generate_synthetic_dpo_batch(
    batch_size: int,
    seq_len: int,
    n_chosen: int,
    n_rejected: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """为 DPO 创建一个批次的合成 log-probability 张量。

    "被选中"和"未被选中"的 token 位置交错排列，使 policy
    能够学会偏好 winning token。pi_logps 和 ref_logps
    都初始化为随机 log-probability；policy 应在训练过程中
    将概率质量移向被选中的 token。

    Returns:
        (pi_logps, ref_logps, yw_idxs, yl_idxs)
    """
    # 随机 log-probability（负值，因为 log(概率) 为负）
    pi_logps = torch.randn(batch_size, seq_len) * 2.0 - 4.0  # 约 -4
    ref_logps = torch.randn(batch_size, seq_len) * 2.0 - 4.0  # 约 -4
    # 将 ref_logps 分离，使其作为固定的 reference
    ref_logps = ref_logps.detach().clone()

    # 随机选择被选中和未被选中子集的 token 位置
    all_idxs = torch.randperm(seq_len)[: n_chosen + n_rejected]
    yw_idxs = all_idxs[:n_chosen].unsqueeze(0).expand(batch_size, -1)
    yl_idxs = all_idxs[n_chosen:].unsqueeze(0).expand(batch_size, -1)

    return pi_logps, ref_logps, yw_idxs, yl_idxs


def main() -> None:
    """在合成数据上演示 DPO 损失的行为。"""
    torch.manual_seed(42)

    batch_size = 32
    seq_len = 50
    n_chosen = 10
    n_rejected = 10
    beta = 1.0
    lr = 0.1
    num_steps = 100

    # 创建固定的 reference 模型（log-probs 保持冻结）
    _, ref_logps, yw_idxs, yl_idxs = generate_synthetic_dpo_batch(
        batch_size, seq_len, n_chosen, n_rejected
    )

    # 初始化可训练的"policy" log-probability
    pi_logps = torch.randn(batch_size, seq_len) * 2.0 - 4.0
    pi_logps.requires_grad_(True)

    optimizer = torch.optim.Adam([pi_logps], lr=lr)

    print("DPO training on synthetic log-probability tensors")
    print(f"  beta = {beta}, batch = {batch_size}, seq_len = {seq_len}")
    print(
        f"{'step':>6s}  {'loss':>10s}  {'pref_acc':>10s}  "
        f"{'pi_w-pi_l':>12s}  {'ref_w-ref_l':>12s}"
    )
    print("-" * 62)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        loss = dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                # 提取当前的 log-ratio 用于报告
                pi_w = pi_logps.gather(1, yw_idxs).sum(dim=1)
                pi_l = pi_logps.gather(1, yl_idxs).sum(dim=1)
                ref_w = ref_logps.gather(1, yw_idxs).sum(dim=1)
                ref_l = ref_logps.gather(1, yl_idxs).sum(dim=1)

                pi_ratio = pi_w - pi_l
                ref_ratio = ref_w - ref_l
                acc = preference_accuracy(pi_ratio, ref_ratio)
                imp_rew = implied_reward(pi_ratio, ref_ratio, beta)

                print(
                    f"{step:>6d}  {loss.item():>10.4f}  {acc:>10.3f}  "
                    f"{pi_ratio.mean().item():>12.4f}  {ref_ratio.mean().item():>12.4f}"
                )

    # ---- 比较 dpo_loss 与 dpo_loss_logp ----
    print("\n--- dpo_loss_logp 一致性检查 ---")
    with torch.no_grad():
        pi_w = pi_logps.gather(1, yw_idxs).sum(dim=1)
        pi_l = pi_logps.gather(1, yl_idxs).sum(dim=1)
        ref_w = ref_logps.gather(1, yw_idxs).sum(dim=1)
        ref_l = ref_logps.gather(1, yl_idxs).sum(dim=1)

        loss_a = dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta)
        loss_b = dpo_loss_logp(pi_w - pi_l, ref_w - ref_l, beta)
        print(f"  dpo_loss      = {loss_a.item():.6f}")
        print(f"  dpo_loss_logp = {loss_b.item():.6f}")
        print(
            f"  Match?         {'YES' if abs(loss_a.item() - loss_b.item()) < 1e-5 else 'NO'}"
        )

    # ---- 展示 policy 现在偏好被选中的 token ----
    print("\n--- 最终统计 ---")
    with torch.no_grad():
        pi_w = pi_logps.gather(1, yw_idxs).sum(dim=1)
        pi_l = pi_logps.gather(1, yl_idxs).sum(dim=1)
        pi_ratio = pi_w - pi_l
        print(f"  pi(yw) - pi(yl)  mean = {pi_ratio.mean().item():.4f}")
        print(f"  (正值 => policy 偏好被选中的而非未被选中的)")
        print(f"  Final DPO loss          = {loss.item():.6f}")
        print(
            f"  Implied reward mean     = {implied_reward(pi_ratio, ref_ratio, beta).mean().item():.4f}"
        )


if __name__ == "__main__":
    main()
