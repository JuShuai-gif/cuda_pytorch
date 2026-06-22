"""
简化版 PPO（Proximal Policy Optimization，近端策略优化）训练器。

实现一个 Actor-Critic 架构，使用共享的 LSTM 基础网络，
分别输出 actor（policy logits）和 critic（value）头。

关键组件：
- GAE（Generalized Advantage Estimation，广义优势估计）用于计算 advantage
- PPO clip 损失：L = -min(r_t*A_t, clip(r_t, 1-eps, 1+eps)*A_t)
- Value 损失通过 MSE 计算，外加可选的 entropy bonus
- 在一个合成 next-token 预测任务上进行演示。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Actor-Critic 架构
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """共享基础的 Actor-Critic，使用 LSTM 作为主干网络。

    embedding + LSTM 层是共享的；两个线性头分别产生：
    - actor logits (vocab_size) 用于动作采样
    - critic value  (标量) 用于状态价值估计。
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.actor_head = nn.Linear(hidden_dim, vocab_size)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播。

        Args:
            x:      输入 token id  (batch, seq_len)。
            hidden: 可选的初始 LSTM 隐藏状态。

        Returns:
            (logits, values, hidden_out)
              logits: (batch, seq_len, vocab_size)
              values: (batch, seq_len, 1)
        """
        emb = self.embedding(x)  # (B, S, E)
        lstm_out, hidden_out = self.lstm(emb, hidden)  # (B, S, H)
        logits = self.actor_head(lstm_out)  # (B, S, V)
        values = self.critic_head(lstm_out)  # (B, S, 1)
        return logits, values, hidden_out

    def sample_action(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 categorical 分布中采样并返回 log-prob。

        Args:
            logits: (batch, seq_len, vocab_size)。

        Returns:
            (actions, log_probs)
              actions  : LongTensor   (batch, seq_len)
              log_probs: FloatTensor  (batch, seq_len)
        """
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()  # (B, S)
        log_probs = dist.log_prob(actions)  # (B, S)
        return actions, log_probs

    def get_log_prob(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """*actions* 在 *logits* 给出的分布下的 log-probability。

        Args:
            logits:  (..., vocab_size)
            actions: (...,)  long tensor

        Returns:
            log_probs: (...,)
        """
        log_probs_all = F.log_softmax(logits, dim=-1)
        return log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE（Generalized Advantage Estimation，广义优势估计）
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    terminal_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 GAE advantage 和 return。

    Args:
        rewards:  (seq_len,)  每个时间步的奖励。
        values:   (seq_len+1,)  t=0..T 的 V(s_t)（最后一个为 bootstrap 或 0）。
        gamma:    折扣因子。
        lam:      GAE lambda 参数。
        terminal_value: 在最后一个时间步之后使用的价值。

    Returns:
        (advantages, returns)
          advantages: (seq_len,)
          returns:    (seq_len,)  advantage + value
    """
    seq_len = rewards.shape[0]
    advantages = torch.zeros(seq_len)
    gae = 0.0

    for t in reversed(range(seq_len)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values[:-1]  # V(s_t) + A_t = R_t（近似）
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO clip 损失
# ---------------------------------------------------------------------------


def ppo_clip_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate 目标。

    r_t = pi(a_t|s_t) / pi_old(a_t|s_t)
    L   = -mean(min(r_t*A_t, clip(r_t, 1-eps, 1+eps)*A_t))

    Args:
        logits:     当前 policy 的 logits   (seq_len, vocab_size)。
        old_logits: 旧 policy 的 logits       (seq_len, vocab_size)。
        actions:    采样的动作                  (seq_len,)。
        advantages: GAE advantage              (seq_len,)。
        clip_eps:   裁剪 epsilon。

    Returns:
        标量损失。
    """
    new_log_probs = (
        F.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    )  # (seq_len,)

    old_log_probs = (
        F.log_softmax(old_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    )  # (seq_len,)

    ratio = torch.exp(new_log_probs - old_log_probs)  # (seq_len,)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return policy_loss


def value_loss_fn(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """预测 value 与 GAE return 之间的均方误差。

    Args:
        values:  (seq_len, 1) 或 (seq_len,)
        returns: (seq_len,)

    Returns:
        标量 MSE 损失。
    """
    values = values.squeeze(-1)  # (seq_len,)
    return F.mse_loss(values, returns)


def entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """categorical 分布的平均 entropy，用于鼓励探索。"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# 合成奖励函数
# ---------------------------------------------------------------------------


def synthetic_reward(sequence: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """为生成的序列计算每个 token 的奖励。

    奖励模型：词汇表中上半部分的 token（较大 ID）获得 +1 奖励；
    下半部分的 token 获得 -1。这为 PPO agent 创建了一个清晰的
    学习信号。
    """
    mid = vocab_size // 2
    return torch.where(sequence >= mid, 1.0, -1.0).float()


# ---------------------------------------------------------------------------
# 训练演示
# ---------------------------------------------------------------------------


def main() -> None:
    """在合成 token 偏好任务上使用 PPO 训练一个 actor-critic agent。"""
    torch.manual_seed(42)

    # 超参数
    vocab_size = 100
    embed_dim = 64
    hidden_dim = 128
    seq_len = 20
    num_epochs = 60
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    lr = 1e-3
    value_coef = 0.5
    entropy_coef = 0.01

    model = ActorCritic(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    print(
        f"{'epoch':>6s}  {'total_r':>10s}  {'policy_L':>10s}  "
        f"{'value_L':>10s}  {'entropy':>10s}"
    )
    print("-" * 58)

    # 跟踪 epoch 之间的平均 token id（越高 = 学会了偏好）
    for epoch in range(1, num_epochs + 1):
        # ---- 阶段 1：Rollout（收集一条 trajectory）----
        # 从一个随机的第一个 token 开始
        start_token = torch.randint(0, vocab_size, (1, 1))

        # 自回归生成一个序列
        generated = [start_token]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        old_logits_list: List[torch.Tensor] = []
        actions_list: List[torch.Tensor] = []
        log_probs_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []

        current_input = start_token
        for _ in range(seq_len):
            logits, value, hidden = model(current_input, hidden)
            action, log_prob = model.sample_action(logits)

            old_logits_list.append(logits.squeeze(0))  # (V,)
            actions_list.append(action.squeeze(0))  # ()
            log_probs_list.append(log_prob.squeeze(0))  # ()
            values_list.append(value.squeeze(0).squeeze(-1))  # ()

            current_input = action.unsqueeze(0)  # (1, 1)
            generated.append(current_input)

        # 将列表堆叠为张量
        full_seq = torch.cat(generated, dim=1)  # (1, seq_len+1)
        actions = torch.stack(actions_list)  # (seq_len,)
        old_logits_t = torch.stack(old_logits_list)  # (seq_len, V)
        values_t = torch.stack(values_list)  # (seq_len,)

        # ---- 阶段 2：计算奖励和 advantage ----
        rewards = synthetic_reward(actions, vocab_size)  # (seq_len,)

        # 附加 terminal value（为简单起见使用 0）
        values_ext = torch.cat([values_t, torch.zeros(1)])  # (seq_len+1,)

        advantages, returns = compute_gae(rewards, values_ext, gamma, lam)

        # ---- 阶段 3：PPO 更新（为简单起见，每个 rollout 只更新一次）----
        model.train()
        # 重新前向传播以获取当前 policy 的 logits（需要同时有旧的和新的）
        hidden = None
        current_logits_list: List[torch.Tensor] = []
        current_values_list: List[torch.Tensor] = []
        inp = full_seq[:, :seq_len]  # (1, seq_len)
        for t in range(seq_len):
            token = inp[:, t : t + 1]  # (1, 1)
            logits, value, hidden = model(token, hidden)
            current_logits_list.append(logits.squeeze(0))
            current_values_list.append(value.squeeze(0))

        new_logits = torch.stack(current_logits_list)  # (seq_len, V)
        new_values = torch.stack(current_values_list)  # (seq_len, 1)

        policy_loss = ppo_clip_loss(
            new_logits, old_logits_t, actions, advantages, clip_eps
        )
        v_loss = value_loss_fn(new_values, returns)
        ent = entropy_bonus(new_logits)

        total_loss = policy_loss + value_coef * v_loss + entropy_coef * (-ent)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{epoch:>6d}  {rewards.sum().item():>10.2f}  "
                f"{policy_loss.item():>10.4f}  {v_loss.item():>10.4f}  "
                f"{ent.item():>10.4f}"
            )

    # ---- 最终评估 ----
    print("\n--- 最终评估 ---")
    model.eval()
    with torch.no_grad():
        # 采样 8 个独立序列并显示它们的平均 token ID
        print("采样序列（平均 token id，训练后应增加）：")
        for i in range(8):
            start = torch.randint(0, vocab_size, (1, 1))
            hidden = None
            seq = [start]
            for _ in range(seq_len):
                logits, _, hidden = model(seq[-1], hidden)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs.squeeze(), 1).unsqueeze(0)
                seq.append(action)
            full = torch.cat(seq, dim=1)
            mean_id = full.float().mean().item()
            print(f"  seq {i + 1}: mean_token_id = {mean_id:.2f}")

        # 同时展示随机初始化模型的输出
        random_model = ActorCritic(vocab_size=vocab_size, embed_dim=embed_dim)
        start = torch.randint(0, vocab_size, (1, 1))
        hidden = None
        rseq = [start]
        for _ in range(seq_len):
            logits, _, hidden = random_model(rseq[-1], hidden)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs.squeeze(), 1).unsqueeze(0)
            rseq.append(action)
        rfull = torch.cat(rseq, dim=1)
        print(f"  random baseline: mean_token_id = {rfull.float().mean().item():.2f}")
        print("(训练后的模型应比随机基线偏好更高的 token ID)")


if __name__ == "__main__":
    main()
