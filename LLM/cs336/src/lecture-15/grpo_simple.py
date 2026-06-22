"""
简化版 GRPO（Group Relative Policy Optimization，分组相对策略优化）。

GRPO 通过分组相对奖励归一化来消除 critic 网络：

1. 对每个 prompt，采样 G 个候选响应。
2. 为每个响应计算一个标量奖励。
3. 将奖励**在每个组内**归一化为零均值和单位方差。
4. 使用归一化后的值作为 policy-gradient 更新中的 advantage。

参考文献："DeepSeekMath: Pushing the Limits of Mathematical Reasoning
in Open Language Models" (Shao et al., 2024)。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 微型 policy 模型（基于 LSTM 的简单序列生成器）
# ---------------------------------------------------------------------------


class TinyPolicy(nn.Module):
    """一个紧凑的基于 LSTM 的序列生成 policy。

    将 token id 通过 embedding 层映射，传入单层 LSTM，
    并将隐藏状态投影到词汇表大小的 logits 以进行自回归采样。
    """

    def __init__(
        self,
        vocab_size: int = 50,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)  # (B, S, E)
        out, hidden = self.lstm(emb, hidden)  # (B, S, H)
        logits = self.head(out)  # (B, S, V)
        return logits, hidden

    def generate(
        self,
        start_token: torch.Tensor,
        seq_len: int,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """自回归生成序列并返回 log-prob。

        Args:
            start_token: (B, 1) 初始 token id。
            seq_len:     要生成的 token 数量。
            temperature: 采样温度（越高越随机）。

        Returns:
            (tokens, log_probs)
              tokens:    (B, seq_len) 生成的 token id。
              log_probs: (B, seq_len) 每个 token 的 log-probability。
        """
        batch_size = start_token.shape[0]
        tokens: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []

        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None
        current = start_token

        for _ in range(seq_len):
            logits, hidden = self.forward(current, hidden)  # (B, 1, V)
            # 应用 temperature 缩放
            logits_scaled = logits / temperature
            probs = F.softmax(logits_scaled, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()  # (B, 1)
            lp = dist.log_prob(action).squeeze(-1)  # (B, 1)

            tokens.append(action)
            log_probs.append(lp)
            current = action

        tokens_t = torch.cat(tokens, dim=1)  # (B, seq_len)
        log_probs_t = torch.cat(log_probs, dim=1)  # (B, seq_len)
        return tokens_t, log_probs_t


# ---------------------------------------------------------------------------
# 基于简单模式的 reward model
# ---------------------------------------------------------------------------


class PatternRewardModel(nn.Module):
    """简单的 reward model，偏好具有交替高/低 token 模式的序列（如 "ABABAB..."）。

    奖励通过生成的序列与预定义目标模式之间的点积相似度计算，
    并归一化到 [-1, 1] 范围内。
    """

    def __init__(self, pattern_len: int = 4, embed_dim: int = 32) -> None:
        super().__init__()
        # 可学习的模式 embedding
        self.pattern_emb = nn.Parameter(torch.randn(pattern_len, embed_dim))
        self.token_emb = nn.Embedding(100, embed_dim)  # up to vocab size 100

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """为一组序列计算标量奖励。

        Args:
            sequence: (batch, seq_len) token id。

        Returns:
            rewards: (batch,) 标量奖励。
        """
        batch, seq_len = sequence.shape
        # 一次性嵌入目标模式
        target = self.pattern_emb[:seq_len]  # (seq_len, E)
        # 嵌入生成的序列
        seq_emb = self.token_emb(sequence)  # (B, seq_len, E)
        # 计算每个 token 与目标之间的 cosine 相似度
        # 归一化
        target_n = F.normalize(target, dim=-1)  # (seq_len, E)
        seq_n = F.normalize(seq_emb, dim=-1)  # (B, seq_len, E)
        # 每个位置的点积，然后沿序列取平均
        sim = (seq_n * target_n.unsqueeze(0)).sum(dim=-1)  # (B, seq_len)
        rewards = sim.mean(dim=1)  # (B,)
        return rewards


def simple_pattern_reward(
    sequence: torch.Tensor,
    pattern: List[int] | None = None,
) -> torch.Tensor:
    """确定性的模式匹配奖励（无参数）。

    奖励 = token_id % 2 与交替模式预期奇偶性匹配的位置比例。

    Args:
        sequence: (batch, seq_len) token id。
        pattern:  预期的奇偶性序列（默认：交替 0,1,0,1,...）。

    Returns:
        rewards: (batch,) [0, 1] 范围内的标量奖励。
    """
    batch, seq_len = sequence.shape
    if pattern is None:
        pattern = [i % 2 for i in range(seq_len)]
    expected = torch.tensor(pattern, dtype=torch.long)  # (seq_len,)
    actual = sequence % 2  # (batch, seq_len)
    matches = (actual == expected.unsqueeze(0)).float()
    return matches.mean(dim=1)


# ---------------------------------------------------------------------------
# GRPO 训练步骤
# ---------------------------------------------------------------------------


def grpo_step(
    policy: TinyPolicy,
    prompts: torch.Tensor,
    reward_fn: callable,
    seq_len: int,
    group_size: int,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行一次 GRPO 更新。

    对 batch 中的每个 prompt，采样 *group_size* 个响应，计算它们的
    奖励，在每个组内归一化以获得 advantage，然后应用 policy-gradient 更新。

    Args:
        policy:      要更新的 TinyPolicy。
        prompts:     (batch, 1) 初始 token id。
        reward_fn:   可调用对象 (sequence) -> (batch,) 奖励。
        seq_len:     生成的响应长度。
        group_size:  G，每个 prompt 的响应数量。
        temperature: 采样温度。
        eps:         用于数值稳定性的小常数。

    Returns:
        (loss, mean_reward, mean_advantage_abs)
    """
    batch_size = prompts.shape[0]
    all_log_probs: List[torch.Tensor] = []
    all_rewards: List[torch.Tensor] = []

    # ---- 阶段 1：为每个 prompt 采样 G 个响应 ----
    for _ in range(group_size):
        tokens, log_probs = policy.generate(prompts, seq_len, temperature)
        rewards = reward_fn(tokens)  # (batch,)
        all_log_probs.append(log_probs.sum(dim=1))  # (batch,) 求和 log-prob
        all_rewards.append(rewards)

    # 堆叠为 (group_size, batch_size)
    log_probs_mat = torch.stack(all_log_probs, dim=0)  # (G, B)
    rewards_mat = torch.stack(all_rewards, dim=0)  # (G, B)

    # ---- 阶段 2：分组相对 advantage 归一化 ----
    # 对每个 prompt（列），在 G 个响应之间归一化奖励
    mean_r = rewards_mat.mean(dim=0, keepdim=True)  # (1, B)
    std_r = rewards_mat.std(dim=0, keepdim=True) + eps  # (1, B)
    advantages = (rewards_mat - mean_r) / std_r  # (G, B)

    # ---- 阶段 3：Policy gradient 损失 ----
    # L = -mean(advantages * log_prob)  （REINFORCE 风格，但使用 GRPO advantages）
    # 将 advantages 从计算图中分离（它们作为固定权重）
    loss = -(advantages.detach() * log_probs_mat).mean()

    return loss, rewards_mat.mean(), advantages.abs().mean()


# ---------------------------------------------------------------------------
# 训练演示
# ---------------------------------------------------------------------------


def main() -> None:
    """在合成模式匹配任务上使用 GRPO 训练一个微型 policy。"""
    torch.manual_seed(42)

    # 超参数
    vocab_size = 50
    embed_dim = 32
    hidden_dim = 64
    seq_len = 16  # 生成响应的长度
    num_prompts = 8  # 每步的 prompt 批次大小
    group_size = 6  # 每个 prompt 的 G 个响应
    num_steps = 150
    lr = 5e-3
    temperature = 1.0

    policy = TinyPolicy(
        vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    # 合成奖励的目标模式：交替奇偶性
    pattern = [i % 2 for i in range(seq_len)]

    def reward_fn(tokens: torch.Tensor) -> torch.Tensor:
        return simple_pattern_reward(tokens, pattern)

    # 固定的 prompt token 集合（为简单起见，各步之间复用）
    prompt_tokens = torch.randint(0, vocab_size, (num_prompts, 1))

    policy.train()
    print(f"GRPO training on alternating-pattern reward (seq_len={seq_len})")
    print(f"  G = {group_size}, prompts = {num_prompts}, steps = {num_steps}")
    print(
        f"{'step':>6s}  {'loss':>10s}  {'mean_r':>10s}  {'|adv|':>10s}  {'best_r':>10s}"
    )
    print("-" * 56)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss, mean_r, mean_adv_abs = grpo_step(
            policy, prompt_tokens, reward_fn, seq_len, group_size, temperature
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 15 == 0 or step == 1:
            # 同时计算 batch 中的最佳奖励用于报告
            with torch.no_grad():
                best = 0.0
                for _ in range(group_size):
                    tokens, _ = policy.generate(prompt_tokens, seq_len, temperature=1.0)
                    r = reward_fn(tokens)
                    best = max(best, r.max().item())

            print(
                f"{step:>6d}  {loss.item():>10.4f}  {mean_r.item():>10.4f}  "
                f"{mean_adv_abs.item():>10.4f}  {best:>10.4f}"
            )

    # ---- 最终评估 ----
    print("\n--- 最终评估 ---")
    policy.eval()
    with torch.no_grad():
        eval_prompts = torch.randint(0, vocab_size, (4, 1))
        for i in range(4):
            tokens, _ = policy.generate(
                eval_prompts[i : i + 1], seq_len, temperature=1.0
            )
            r = reward_fn(tokens)
            seq_str = " ".join(f"{t.item():2d}" for t in tokens.squeeze())
            parity_str = " ".join(f"{t.item() % 2}" for t in tokens.squeeze())
            print(f"  Prompt {i + 1}: reward = {r.item():.3f}")
            print(f"    tokens: {seq_str}")
            print(f"    parity: {parity_str}")

        # 基线：随机 policy
        random_policy = TinyPolicy(vocab_size=vocab_size, embed_dim=embed_dim)
        print("\n  Random (untrained) baseline:")
        for i in range(4):
            tokens, _ = random_policy.generate(
                eval_prompts[i : i + 1], seq_len, temperature=1.0
            )
            r = reward_fn(tokens)
            parity_str = " ".join(f"{t.item() % 2}" for t in tokens.squeeze())
            print(f"    parity: {parity_str}  (reward = {r.item():.3f})")

        print("\n  Target pattern parity: " + " ".join(str(p) for p in pattern))
        print("  (训练后的 policy 应近似交替奇偶性模式)")


if __name__ == "__main__":
    main()
