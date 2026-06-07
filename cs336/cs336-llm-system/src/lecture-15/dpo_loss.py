"""
DPO (Direct Preference Optimization) loss from scratch.

Implements the DPO loss as described in the DPO paper (Rafailov et al.,
2023), which directly optimises a policy to satisfy human preferences
without training a separate reward model.

    L_DPO = -E[log sigma(beta * (log(pi(yw|x)/ref(yw|x))
                                  - log(pi(yl|x)/ref(yl|x))))]

Two variants are provided:
- dpo_loss:          Takes full log-probability tensors and token indices.
- dpo_loss_logp:     Takes pre-computed log-ratio vectors (lighter weight).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core DPO loss functions
# ---------------------------------------------------------------------------


def dpo_loss(
    pi_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    yw_idxs: torch.Tensor,
    yl_idxs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Direct Preference Optimisation loss (token-index form).

    Given per-token log-probabilities from the policy and reference models,
    this function extracts the sub-sequences corresponding to the chosen
    (yw) and rejected (yl) responses, sums their log-probs, and computes
    the DPO objective.

    Args:
        pi_logps:  Log-probabilities from the **policy** model.
                   Shape (batch, seq_len).
        ref_logps: Log-probabilities from the **reference** model.
                   Shape (batch, seq_len).
        yw_idxs:   Indices of the **chosen** (winning) tokens.
                   Shape (batch, n_chosen).
        yl_idxs:   Indices of the **rejected** (losing) tokens.
                   Shape (batch, n_rejected).
        beta:      Temperature parameter controlling how strongly the
                   policy is penalised for diverging from the reference.

    Returns:
        Scalar DPO loss averaged over the batch.
    """
    # Sum log-probabilities over the chosen / rejected token positions
    pi_yw = pi_logps.gather(1, yw_idxs).sum(dim=1)  # (batch,)
    pi_yl = pi_logps.gather(1, yl_idxs).sum(dim=1)  # (batch,)
    ref_yw = ref_logps.gather(1, yw_idxs).sum(dim=1)  # (batch,)
    ref_yl = ref_logps.gather(1, yl_idxs).sum(dim=1)  # (batch,)

    # log(pi / ref) for chosen and rejected
    pi_log_ratio = pi_yw - pi_yl  # log(pi_w) - log(pi_l)
    ref_log_ratio = ref_yw - ref_yl  # log(ref_w) - log(ref_l)

    # DPO loss: -log sigma(beta * (pi_log_ratio - ref_log_ratio))
    loss = -F.logsigmoid(beta * (pi_log_ratio - ref_log_ratio)).mean()
    return loss


def dpo_loss_logp(
    pi_log_ratios: torch.Tensor,
    ref_log_ratios: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO loss variant that takes pre-computed log-ratios.

    Args:
        pi_log_ratios:  log(pi(yw|x)) - log(pi(yl|x)) for each batch item.
                        Shape (batch,).
        ref_log_ratios: log(ref(yw|x)) - log(ref(yl|x)) for each batch item.
                        Shape (batch,).
        beta:           Temperature / KL-penalty coefficient.

    Returns:
        Scalar DPO loss averaged over the batch.
    """
    loss = -F.logsigmoid(beta * (pi_log_ratios - ref_log_ratios)).mean()
    return loss


# ---------------------------------------------------------------------------
# Utility: compute implied reward and accuracy
# ---------------------------------------------------------------------------


def implied_reward(
    pi_log_ratios: torch.Tensor,
    ref_log_ratios: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Implied reward under the DPO parametrisation: r(x,y) = beta * log(pi/ref)."""
    return beta * (pi_log_ratios - ref_log_ratios)


def preference_accuracy(
    pi_log_ratios: torch.Tensor,
    ref_log_ratios: torch.Tensor,
) -> float:
    """Fraction of pairs where the policy assigns higher relative
    probability to the chosen response than the reference does."""
    return (pi_log_ratios > ref_log_ratios).float().mean().item()


# ---------------------------------------------------------------------------
# Synthetic demonstration
# ---------------------------------------------------------------------------


def generate_synthetic_dpo_batch(
    batch_size: int,
    seq_len: int,
    n_chosen: int,
    n_rejected: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a batch of synthetic log-probability tensors for DPO.

    The "chosen" and "rejected" token positions are interleaved so that
    the policy can learn to favour the winning tokens.  Both pi_logps and
    ref_logps start as random log-probabilities; the policy is expected
    to shift mass toward the chosen tokens during training.

    Returns:
        (pi_logps, ref_logps, yw_idxs, yl_idxs)
    """
    # Random log-probabilities (negative values, as log of probabilities)
    pi_logps = torch.randn(batch_size, seq_len) * 2.0 - 4.0  # around -4
    ref_logps = torch.randn(batch_size, seq_len) * 2.0 - 4.0  # around -4
    # Detach ref_logps so it acts as a fixed reference
    ref_logps = ref_logps.detach().clone()

    # Randomly choose token positions for chosen and rejected subsets
    all_idxs = torch.randperm(seq_len)[: n_chosen + n_rejected]
    yw_idxs = all_idxs[:n_chosen].unsqueeze(0).expand(batch_size, -1)
    yl_idxs = all_idxs[n_chosen:].unsqueeze(0).expand(batch_size, -1)

    return pi_logps, ref_logps, yw_idxs, yl_idxs


def main() -> None:
    """Demonstrate DPO loss behaviour on synthetic data."""
    torch.manual_seed(42)

    batch_size = 32
    seq_len = 50
    n_chosen = 10
    n_rejected = 10
    beta = 1.0
    lr = 0.1
    num_steps = 100

    # Create a fixed reference model (log-probs stay frozen)
    _, ref_logps, yw_idxs, yl_idxs = generate_synthetic_dpo_batch(
        batch_size, seq_len, n_chosen, n_rejected
    )

    # Initialise trainable "policy" log-probabilities
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
                # Extract current log-ratios for reporting
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

    # ---- Compare dpo_loss vs dpo_loss_logp ----
    print("\n--- dpo_loss_logp consistency check ---")
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

    # ---- Show that the policy now prefers chosen tokens ----
    print("\n--- Final statistics ---")
    with torch.no_grad():
        pi_w = pi_logps.gather(1, yw_idxs).sum(dim=1)
        pi_l = pi_logps.gather(1, yl_idxs).sum(dim=1)
        pi_ratio = pi_w - pi_l
        print(f"  pi(yw) - pi(yl)  mean = {pi_ratio.mean().item():.4f}")
        print(f"  (positive value => policy prefers chosen over rejected)")
        print(f"  Final DPO loss          = {loss.item():.6f}")
        print(
            f"  Implied reward mean     = {implied_reward(pi_ratio, ref_ratio, beta).mean().item():.4f}"
        )


if __name__ == "__main__":
    main()
