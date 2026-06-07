"""
Simplified GRPO (Group Relative Policy Optimization).

GRPO eliminates the critic network by computing advantages through
group-relative reward normalisation:

1. For each prompt, sample G candidate responses.
2. Compute a scalar reward for each response.
3. Normalise rewards to zero mean and unit variance **within each group**.
4. Use the normalised values as advantages in a policy-gradient update.

Reference: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
in Open Language Models" (Shao et al., 2024).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tiny policy model (simple LSTM-based sequence generator)
# ---------------------------------------------------------------------------


class TinyPolicy(nn.Module):
    """A compact LSTM-based policy for sequence generation.

    Maps token ids through an embedding layer, feeds them into a
    single-layer LSTM, and projects the hidden state to vocabulary-size
    logits for autoregressive sampling.
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
        """Autoregressively generate a sequence and return log-probs.

        Args:
            start_token: (B, 1) initial token ids.
            seq_len:     Number of tokens to generate.
            temperature: Sampling temperature (higher = more random).

        Returns:
            (tokens, log_probs)
              tokens:    (B, seq_len) generated token ids.
              log_probs: (B, seq_len) per-token log-probabilities.
        """
        batch_size = start_token.shape[0]
        tokens: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []

        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None
        current = start_token

        for _ in range(seq_len):
            logits, hidden = self.forward(current, hidden)  # (B, 1, V)
            # Apply temperature scaling
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
# Simple pattern-based reward model
# ---------------------------------------------------------------------------


class PatternRewardModel(nn.Module):
    """Simple reward model that prefers sequences with an alternating
    high/low token pattern (like "ABABAB...").

    Rewards are computed as the dot-product similarity between the
    generated sequence and a pre-defined target pattern, normalised
    to [-1, 1] range.
    """

    def __init__(self, pattern_len: int = 4, embed_dim: int = 32) -> None:
        super().__init__()
        # Learnable pattern embeddings
        self.pattern_emb = nn.Parameter(torch.randn(pattern_len, embed_dim))
        self.token_emb = nn.Embedding(100, embed_dim)  # up to vocab size 100

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute scalar rewards for a batch of sequences.

        Args:
            sequence: (batch, seq_len) token ids.

        Returns:
            rewards: (batch,) scalar rewards.
        """
        batch, seq_len = sequence.shape
        # Embed the target pattern once
        target = self.pattern_emb[:seq_len]  # (seq_len, E)
        # Embed the generated sequence
        seq_emb = self.token_emb(sequence)  # (B, seq_len, E)
        # Compute cosine similarity between each token and target
        # Normalise
        target_n = F.normalize(target, dim=-1)  # (seq_len, E)
        seq_n = F.normalize(seq_emb, dim=-1)  # (B, seq_len, E)
        # Dot product per position, then mean over sequence
        sim = (seq_n * target_n.unsqueeze(0)).sum(dim=-1)  # (B, seq_len)
        rewards = sim.mean(dim=1)  # (B,)
        return rewards


def simple_pattern_reward(
    sequence: torch.Tensor,
    pattern: List[int] | None = None,
) -> torch.Tensor:
    """Deterministic pattern-matching reward (no parameters).

    Reward = fraction of positions where token_id % 2 matches the
    expected parity from an alternating pattern.

    Args:
        sequence: (batch, seq_len) token ids.
        pattern:  Expected parity sequence (default: alternating 0,1,0,1,...).

    Returns:
        rewards: (batch,) scalar rewards in [0, 1].
    """
    batch, seq_len = sequence.shape
    if pattern is None:
        pattern = [i % 2 for i in range(seq_len)]
    expected = torch.tensor(pattern, dtype=torch.long)  # (seq_len,)
    actual = sequence % 2  # (batch, seq_len)
    matches = (actual == expected.unsqueeze(0)).float()
    return matches.mean(dim=1)


# ---------------------------------------------------------------------------
# GRPO training step
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
    """Perform one GRPO update.

    For each prompt in the batch, sample *group_size* responses, compute
    their rewards, normalise within each group to obtain advantages, and
    apply a policy-gradient update.

    Args:
        policy:      The TinyPolicy to update.
        prompts:     (batch, 1) initial token ids.
        reward_fn:   Callable (sequence) -> (batch,) rewards.
        seq_len:     Length of generated response.
        group_size:  G, number of responses per prompt.
        temperature: Sampling temperature.
        eps:         Small constant for numerical stability.

    Returns:
        (loss, mean_reward, mean_advantage_abs)
    """
    batch_size = prompts.shape[0]
    all_log_probs: List[torch.Tensor] = []
    all_rewards: List[torch.Tensor] = []

    # ---- Phase 1: Sample G responses per prompt ----
    for _ in range(group_size):
        tokens, log_probs = policy.generate(prompts, seq_len, temperature)
        rewards = reward_fn(tokens)  # (batch,)
        all_log_probs.append(log_probs.sum(dim=1))  # (batch,) summed log-prob
        all_rewards.append(rewards)

    # Stack into (group_size, batch_size)
    log_probs_mat = torch.stack(all_log_probs, dim=0)  # (G, B)
    rewards_mat = torch.stack(all_rewards, dim=0)  # (G, B)

    # ---- Phase 2: Group-relative advantage normalisation ----
    # For each prompt (column), normalise rewards across the G responses
    mean_r = rewards_mat.mean(dim=0, keepdim=True)  # (1, B)
    std_r = rewards_mat.std(dim=0, keepdim=True) + eps  # (1, B)
    advantages = (rewards_mat - mean_r) / std_r  # (G, B)

    # ---- Phase 3: Policy gradient loss ----
    # L = -mean(advantages * log_prob)  (REINFORCE-style, but with GRPO advantages)
    # Detach advantages from the computation graph (they act as fixed weights)
    loss = -(advantages.detach() * log_probs_mat).mean()

    return loss, rewards_mat.mean(), advantages.abs().mean()


# ---------------------------------------------------------------------------
# Training demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    """Train a tiny policy with GRPO on a synthetic pattern-matching task."""
    torch.manual_seed(42)

    # Hyper-parameters
    vocab_size = 50
    embed_dim = 32
    hidden_dim = 64
    seq_len = 16  # length of generated responses
    num_prompts = 8  # batch of prompts per step
    group_size = 6  # G responses per prompt
    num_steps = 150
    lr = 5e-3
    temperature = 1.0

    policy = TinyPolicy(
        vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    # Target pattern for the synthetic reward: alternating parity
    pattern = [i % 2 for i in range(seq_len)]

    def reward_fn(tokens: torch.Tensor) -> torch.Tensor:
        return simple_pattern_reward(tokens, pattern)

    # Fixed set of prompt tokens (we reuse them across steps for simplicity)
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
            # Also compute the best reward in the batch for reporting
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

    # ---- Final evaluation ----
    print("\n--- Final evaluation ---")
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

        # Baseline: random policy
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
        print("  (Trained policy should approximate the alternating parity pattern)")


if __name__ == "__main__":
    main()
