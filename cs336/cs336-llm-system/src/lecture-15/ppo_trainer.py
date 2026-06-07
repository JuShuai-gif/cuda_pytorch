"""
Simplified PPO (Proximal Policy Optimization) trainer.

Implements an Actor-Critic architecture with a shared LSTM base network
feeding separate actor (policy logits) and critic (value) heads.

Key components:
- GAE (Generalized Advantage Estimation) for computing advantages
- PPO clip loss:  L = -min(r_t*A_t, clip(r_t, 1-eps, 1+eps)*A_t)
- Value loss via MSE, plus optional entropy bonus
- Demonstrated on a synthetic next-token prediction task.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Actor-Critic architecture
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared-base Actor-Critic with LSTM backbone.

    The embedding + LSTM layers are shared; two linear heads produce
    - actor logits (vocab_size) for action sampling
    - critic value  (scalar)      for state-value estimation.
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
        """Forward pass.

        Args:
            x:      Input token ids  (batch, seq_len).
            hidden: Optional initial LSTM hidden state.

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
        """Sample from categorical distribution and return log-prob.

        Args:
            logits: (batch, seq_len, vocab_size).

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
        """Log-probability of *actions* under the distribution given by *logits*.

        Args:
            logits:  (..., vocab_size)
            actions: (...,)  long tensor

        Returns:
            log_probs: (...,)
        """
        log_probs_all = F.log_softmax(logits, dim=-1)
        return log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    terminal_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards:  (seq_len,)  per-timestep rewards.
        values:   (seq_len+1,)  V(s_t) for t=0..T (last is bootstrap or 0).
        gamma:    Discount factor.
        lam:      GAE lambda parameter.
        terminal_value: Value to use beyond the last timestep.

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

    returns = advantages + values[:-1]  # V(s_t) + A_t = R_t (approx)
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO clip loss
# ---------------------------------------------------------------------------


def ppo_clip_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate objective.

    r_t = pi(a_t|s_t) / pi_old(a_t|s_t)
    L   = -mean(min(r_t*A_t, clip(r_t, 1-eps, 1+eps)*A_t))

    Args:
        logits:     Current-policy logits   (seq_len, vocab_size).
        old_logits: Old-policy logits       (seq_len, vocab_size).
        actions:    Sampled actions          (seq_len,).
        advantages: GAE advantages           (seq_len,).
        clip_eps:   Clipping epsilon.

    Returns:
        Scalar loss.
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
    """Mean-squared-error between predicted values and GAE returns.

    Args:
        values:  (seq_len, 1) or (seq_len,)
        returns: (seq_len,)

    Returns:
        Scalar MSE loss.
    """
    values = values.squeeze(-1)  # (seq_len,)
    return F.mse_loss(values, returns)


def entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """Mean entropy of the categorical distribution to encourage exploration."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Synthetic reward function
# ---------------------------------------------------------------------------


def synthetic_reward(sequence: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Compute a per-token reward for a generated sequence.

    Reward model: tokens from the upper half of the vocabulary (higher
    IDs) receive +1 reward; tokens from the lower half receive -1.
    This creates a clear learning signal for the PPO agent.
    """
    mid = vocab_size // 2
    return torch.where(sequence >= mid, 1.0, -1.0).float()


# ---------------------------------------------------------------------------
# Training demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    """Train an actor-critic agent with PPO on a synthetic token-preference task."""
    torch.manual_seed(42)

    # Hyper-parameters
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

    # Track mean token id over epochs (higher = learned preference)
    for epoch in range(1, num_epochs + 1):
        # ---- Phase 1: Rollout (collect a trajectory) ----
        # Start from a random first token
        start_token = torch.randint(0, vocab_size, (1, 1))

        # Auto-regressively generate a sequence
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

        # Stack lists into tensors
        full_seq = torch.cat(generated, dim=1)  # (1, seq_len+1)
        actions = torch.stack(actions_list)  # (seq_len,)
        old_logits_t = torch.stack(old_logits_list)  # (seq_len, V)
        values_t = torch.stack(values_list)  # (seq_len,)

        # ---- Phase 2: Compute rewards and advantages ----
        rewards = synthetic_reward(actions, vocab_size)  # (seq_len,)

        # Append terminal value (0 for simplicity)
        values_ext = torch.cat([values_t, torch.zeros(1)])  # (seq_len+1,)

        advantages, returns = compute_gae(rewards, values_ext, gamma, lam)

        # ---- Phase 3: PPO update (single epoch per rollout for simplicity) ----
        model.train()
        # Re-forward to get current policy logits (we need both old and new)
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

    # ---- Final evaluation ----
    print("\n--- Final evaluation ---")
    model.eval()
    with torch.no_grad():
        # Sample 8 independent sequences and show their mean token IDs
        print("Sampled sequences (mean token id, should increase with training):")
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

        # Also show what a randomly initialized model would produce
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
        print("(Trained model should prefer higher token IDs than random baseline)")


if __name__ == "__main__":
    main()
