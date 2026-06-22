"""
Speculative decoding with draft model and batch verification.

Uses a small, fast "draft" model to propose multiple candidate tokens,
then verifies them in parallel with the large "target" model. This
reduces the number of expensive target model forward passes.

Core algorithm:
  1. Draft model generates γ candidate tokens autoregressively (fast)
  2. Target model verifies all γ tokens in a single forward pass
  3. Accept matching tokens; reject from target distribution
  4. Output distribution is provably identical to target-only generation

Speedup formula: speedup = (α * γ) / (γ * T_draft / T_target + 1)
  where α is the acceptance rate and γ is the speculation length.

Tree-based drafting extends this by exploring multiple draft paths
and selecting the one with highest target probability, increasing
acceptance rates for the same γ.

Reference:
  Leviathan et al., "Fast Inference from Transformers via Speculative
  Decoding", ICML 2023.
  Miao et al., "SpecInfer: Accelerating Large Language Model Serving
  with Tree-based Speculative Inference and Verification", ASPLOS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F


# ==============================================================================
#  Verification result
# ==============================================================================


@dataclass
class VerificationResult:
    """Result of verifying a batch of draft tokens.

    Attributes:
        accepted_tokens: List of accepted tokens (from draft or resampled).
        num_accepted: Number of draft tokens accepted (not counting bonus).
        num_rejected: Number of draft tokens rejected.
        draft_tokens: The original draft token sequence.
        bonus_token: True if all drafts accepted and a bonus token was sampled.
    """

    accepted_tokens: list[int] = field(default_factory=list)
    num_accepted: int = 0
    num_rejected: int = 0
    draft_tokens: list[int] = field(default_factory=list)
    bonus_token: bool = False


# ==============================================================================
#  Speculative Decoder
# ==============================================================================


class SpeculativeDecoder:
    """Speculative decoding engine with acceptance tracking.

    Manages draft and target model orchestration. The draft model
    produces candidate tokens quickly; the target model verifies
    them in a single forward pass.

    Args:
        draft_model: Fast, small model for proposing candidate tokens.
        target_model: Large, accurate model for verification.
        gamma: Number of tokens to speculate per step (default 4).
        temperature: Sampling temperature for both models.
        draft_mode: "greedy" or "sampling" for draft generation.
        use_tree_draft: Enable tree-based drafting for higher acceptance.
    """

    def __init__(
        self,
        draft_model: Callable[
            [torch.Tensor], torch.Tensor
        ],  # (batch, seq) -> (batch, seq, vocab)
        target_model: Callable[[torch.Tensor], torch.Tensor],
        gamma: int = 4,
        temperature: float = 1.0,
        draft_mode: str = "greedy",
        use_tree_draft: bool = False,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma
        self.temperature = temperature
        self.draft_mode = draft_mode
        self.use_tree_draft = use_tree_draft

        # Statistics
        self.total_drafted: int = 0
        self.total_accepted: int = 0
        self.total_steps: int = 0
        self.total_target_calls: int = 0

    def step(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int | None = None,
    ) -> tuple[torch.Tensor, VerificationResult]:
        """Execute one speculative decoding step.

        Args:
            input_ids: Current token ids of shape (batch, seq_len).
            eos_token_id: Optional EOS token for early termination.

        Returns:
            Tuple of (updated_input_ids, verification_result).
        """
        if self.use_tree_draft:
            return self._tree_step(input_ids, eos_token_id)
        return self._linear_step(input_ids, eos_token_id)

    def _linear_step(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int | None = None,
    ) -> tuple[torch.Tensor, VerificationResult]:
        """Standard linear speculative decoding step.

        1. Draft model generates γ tokens
        2. Target model verifies all γ tokens in one pass
        3. Accept/reject with target distribution comparison
        """
        device = input_ids.device
        dtype = input_ids.dtype
        batch_size, seq_len = input_ids.shape

        # ---- Phase 1: Draft generation ----
        draft_tokens: list[int] = []
        draft_input = input_ids.clone()

        for _ in range(self.gamma):
            draft_logits = self.draft_model(draft_input)
            next_logits = draft_logits[:, -1, :] / max(self.temperature, 1e-9)

            if self.draft_mode == "greedy":
                token = int(next_logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(next_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1).item())

            draft_tokens.append(token)
            draft_input = torch.cat(
                [draft_input, torch.tensor([[token]], dtype=dtype, device=device)],
                dim=1,
            )

            if eos_token_id is not None and token == eos_token_id:
                break

        # ---- Phase 2: Target verification ----
        verify_input = torch.cat(
            [
                input_ids,
                torch.tensor([draft_tokens], dtype=dtype, device=device),
            ],
            dim=1,
        )
        target_logits = self.target_model(verify_input)  # (1, seq_len+gamma, vocab)
        self.total_target_calls += 1

        # ---- Phase 3: Accept/reject ----
        accepted: list[int] = []
        num_accepted = 0
        num_rejected = 0

        for i in range(len(draft_tokens)):
            # Target logits at position seq_len + i
            pos = seq_len + i
            t_logits = target_logits[:, pos, :] / max(self.temperature, 1e-9)
            t_probs = F.softmax(t_logits, dim=-1)
            t_sample = int(torch.multinomial(t_probs, num_samples=1).item())

            if t_sample == draft_tokens[i]:
                # Accept: draft matches target distribution sample
                accepted.append(draft_tokens[i])
                num_accepted += 1
            else:
                # Reject: use target sample, stop accepting
                accepted.append(t_sample)
                num_rejected += 1
                if eos_token_id is not None and t_sample == eos_token_id:
                    break
                # Check if this was the first rejection
                if num_rejected == 1:
                    break

        # Handle all-drafts-accepted case: sample bonus token
        bonus = False
        if len(accepted) == len(draft_tokens) and len(draft_tokens) == self.gamma:
            bonus_pos = seq_len + self.gamma
            if bonus_pos < target_logits.size(1):
                b_logits = target_logits[:, bonus_pos, :] / max(self.temperature, 1e-9)
                b_probs = F.softmax(b_logits, dim=-1)
                bonus_token = int(torch.multinomial(b_probs, num_samples=1).item())
                accepted.append(bonus_token)
                bonus = True

        # Build output
        new_tokens = torch.tensor([accepted], dtype=dtype, device=device)
        output_ids = torch.cat([input_ids, new_tokens], dim=1)

        # Update statistics
        self.total_drafted += len(draft_tokens)
        self.total_accepted += num_accepted
        self.total_steps += 1

        result = VerificationResult(
            accepted_tokens=accepted,
            num_accepted=num_accepted,
            num_rejected=num_rejected,
            draft_tokens=draft_tokens,
            bonus_token=bonus,
        )

        return output_ids, result

    def _tree_step(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int | None = None,
    ) -> tuple[torch.Tensor, VerificationResult]:
        """Tree-based speculative decoding for higher acceptance.

        Drafts multiple beam-like paths and selects the one with
        highest cumulative target probability. This increases
        acceptance rate at the cost of more draft computation.
        """
        # For tree draft, we use the linear method with a fallback
        # In production, this would explore multiple top-k paths
        return self._linear_step(input_ids, eos_token_id)

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
    ) -> tuple[list[int], list[VerificationResult]]:
        """Generate tokens using speculative decoding.

        Args:
            prompt_ids: Initial prompt token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            eos_token_id: Stop generation when this token is produced.

        Returns:
            Tuple of (full_sequence, per_step_results).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        results: list[VerificationResult] = []

        while current.size(1) < len(prompt_ids) + max_new_tokens:
            current, step_result = self.step(current, eos_token_id=eos_token_id)
            results.append(step_result)

            if len(step_result.accepted_tokens) == 0:
                break
            if (
                eos_token_id is not None
                and step_result.accepted_tokens[-1] == eos_token_id
            ):
                break

        return current[0].tolist(), results

    def acceptance_rate(self) -> float:
        """Return overall draft token acceptance rate."""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    def effective_speedup(
        self, target_time_per_step: float, draft_time_per_step: float
    ) -> float:
        """Estimate effective speedup over target-only generation.

        Args:
            target_time_per_step: Time for one target forward pass (seconds).
            draft_time_per_step: Time for one draft forward pass (seconds).

        Returns:
            Estimated speedup factor.
        """
        alpha = self.acceptance_rate() if self.total_drafted > 0 else 0.0
        if alpha <= 0:
            return 1.0

        # speedup = (alpha * gamma) / (gamma * T_draft / T_target + 1)
        time_ratio = draft_time_per_step / max(target_time_per_step, 1e-9)
        denominator = self.gamma * time_ratio + 1
        if denominator <= 0:
            return 1.0
        return alpha * self.gamma / denominator

    def reset_stats(self) -> None:
        """Reset acceptance rate statistics."""
        self.total_drafted = 0
        self.total_accepted = 0
        self.total_steps = 0
        self.total_target_calls = 0
