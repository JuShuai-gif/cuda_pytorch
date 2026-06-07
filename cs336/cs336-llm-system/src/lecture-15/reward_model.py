"""
Reward model with Bradley-Terry pairwise comparison loss.

Implements a tiny transformer-based reward model that maps a token
sequence to a scalar reward, then trains it on synthetic preference
pairs using the Bradley-Terry objective:

    L = -log sigma(r_chosen - r_rejected)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tiny Transformer Reward Model
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding injected into the embedding stream."""

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
        """Add positional encoding to *x* of shape (batch, seq, embed_dim)."""
        return x + self.pe[:, : x.size(1)]


class RewardModel(nn.Module):
    """Tiny transformer that outputs a scalar reward for a token sequence.

    Architecture:  Embedding -> PositionalEncoding -> N transformer
    encoder layers -> mean pool -> linear -> scalar.
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
        """Return scalar reward per sequence.

        Args:
            input_ids: LongTensor of shape (batch, seq_len).

        Returns:
            FloatTensor of shape (batch,) with reward scores.
        """
        # Embed and add positional information
        emb = self.embedding(input_ids)  # (B, S, E)
        emb = self.pos_encoder(emb)
        # Pass through transformer encoder
        enc_out = self.encoder(emb)  # (B, S, E)
        # Mean pool over the sequence dimension
        pooled = enc_out.mean(dim=1)  # (B, E)
        # Project to scalar reward
        reward = self.head(pooled).squeeze(-1)  # (B,)
        return reward


# ---------------------------------------------------------------------------
# Bradley-Terry pairwise loss
# ---------------------------------------------------------------------------


def bradley_terry_loss(
    r_chosen: torch.Tensor,
    r_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise comparison loss.

    L = -log sigma(r_chosen - r_rejected)

    Args:
        r_chosen:  Predicted rewards for the preferred (chosen) sequences.
        r_rejected: Predicted rewards for the dispreferred (rejected) sequences.

    Returns:
        Scalar loss averaged over the batch.
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def pairwise_accuracy(
    r_chosen: torch.Tensor,
    r_rejected: torch.Tensor,
) -> float:
    """Fraction of pairs where r_chosen > r_rejected."""
    return (r_chosen > r_rejected).float().mean().item()


# ---------------------------------------------------------------------------
# Synthetic preference data generation
# ---------------------------------------------------------------------------


def generate_synthetic_pairs(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic preference pairs.

    The "chosen" sequences receive a small positive bias added to a subset
    of their token ids so that a well-trained model can learn to
    distinguish them from "rejected" sequences.

    Returns:
        (chosen_ids, rejected_ids, dummy_rewards) where dummy_rewards
        are ground-truth scalar values used only for sanity checking.
    """
    # Both sequences are randomly sampled, but chosen ones have a
    # systematically higher "quality signal" in their token distribution.
    chosen_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    # Rejected sequences use tokens from the lower half of the vocabulary
    # on average, making them distinguishable.
    rejected_ids = torch.randint(1, max(2, vocab_size // 3), (batch_size, seq_len))

    # Synthesised "true" rewards (not used in training, just for sanity)
    true_r_c = chosen_ids.float().mean(dim=1)  # higher average id -> higher reward
    true_r_r = rejected_ids.float().mean(dim=1)
    return chosen_ids, rejected_ids, true_r_c, true_r_r


# ---------------------------------------------------------------------------
# Training demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    """Train a reward model on synthetic preference pairs and show loss."""
    torch.manual_seed(42)

    # Hyper-parameters (tiny model, fast training)
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

    # Final evaluation
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
        print("(r_chosen should be substantially higher than r_rejected)")


if __name__ == "__main__":
    main()
