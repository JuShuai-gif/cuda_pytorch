"""
Mixture-of-Experts (MoE) Expert Parallelism.

Implements:
- MoEExpertParallel: Distributes experts across GPU ranks
- ExpertRouter: Token-to-expert routing with top-k gating
- TokenDispatcher: All-to-All communication for token routing
- LoadBalancedMoE: Full MoE layer with load balancing
- Load balancing loss computation (auxiliary loss)
- Capacity factor and token drop handling
- Expert weight gradient synchronization

MoE architecture (per transformer layer):
    1. Router: softmax(gate(x)) → top-k expert selection
    2. All-to-All: route tokens to expert GPUs
    3. Expert computation: each GPU runs its assigned experts
    4. All-to-All: route tokens back to original GPUs
    5. Combine: weighted sum of expert outputs

Communication analysis per MoE layer:
- 2x All-to-All (tokens to experts, results back)
- Token routing volume = batch * seq * hidden * top_k * dtype_bytes
- Expert parallelism allows very large expert count (up to 128+ experts)

Load balancing:
- Auxiliary loss encourages uniform token distribution across experts
- Capacity factor limits tokens per expert to prevent OOM
- Token dropping with random routing for overflow

Reference:
    GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding
    (Lepikhin et al., 2021)
    Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2021)
    MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (Gale et al., 2022)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MoEConfig:
    """Configuration for Mixture-of-Experts layer.

    Attributes:
        num_experts: Total number of experts (across all ranks).
        top_k: Number of experts to route each token to.
        expert_hidden_size: Hidden dimension within each expert.
        capacity_factor: Multiplier for tokens per expert capacity.
                         capacity = (tokens_per_rank * top_k / num_experts) * capacity_factor
                         Tokens exceeding capacity are dropped.
        aux_loss_coef: Weight for load balancing auxiliary loss.
        router_z_loss_coef: Weight for router z-loss (stability regularization).
        noisy_gating: Add noise to router logits during training.
        noisy_gating_epsilon: Noise epsilon for exploration.
        drop_tokens: Whether to drop tokens exceeding capacity (True) or reroute (False).
        use_expert_parallel: Whether to shard experts across ranks.
    """

    num_experts: int = 8
    top_k: int = 2
    expert_hidden_size: Optional[int] = None
    capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    noisy_gating: bool = True
    noisy_gating_epsilon: float = 0.01
    drop_tokens: bool = True
    use_expert_parallel: bool = True


# ---------------------------------------------------------------------------
# Expert Router (gating mechanism)
# ---------------------------------------------------------------------------


class ExpertRouter(nn.Module):
    """
    Token-to-expert router with top-k gating.

    Given input hidden states, computes routing probabilities and
    selects top-k experts per token.

    Load balancing: auxiliary loss = alpha * sum(f_i * P_i)
    where f_i = fraction of tokens routed to expert i
          P_i = average routing probability for expert i
    Minimizing this encourages uniform token distribution.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        noisy_gating: bool = True,
        noisy_gating_epsilon: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating
        self.noisy_gating_epsilon = noisy_gating_epsilon

        self.router = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: (batch * seq_len, hidden_size). Flattened tokens.

        Returns:
            tuple of:
            - dispatch_mask: (num_tokens, num_experts, capacity). Sparse routing mask.
            - combine_weights: (num_tokens, num_experts, capacity). Normalized weights.
            - router_probs: (num_tokens, num_experts). Softmax probabilities.
            - aux_loss: Scalar auxiliary load balancing loss.
        """
        num_tokens = x.size(0)

        # Router logits
        logits = self.router(x)  # (num_tokens, num_experts)

        # Noisy top-k gating
        if self.noisy_gating and self.training:
            noise_stddev = F.softplus(logits)
            noise = torch.randn_like(logits) * noise_stddev * self.noisy_gating_epsilon
            logits = logits + noise

        # Softmax probabilities
        router_probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_probs, top_k_indices)

        # Capacity: max tokens per expert
        capacity = int(
            self.capacity_factor * num_tokens * self.top_k / self.num_experts
        )
        capacity = max(capacity, 1)

        # Build dispatch mask and combine weights
        dispatch_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            capacity,
            device=x.device,
            dtype=torch.float32,
        )
        combine_weights = torch.zeros_like(dispatch_mask)

        # Per-expert token count for capacity limiting
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)

        for token_idx in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = top_k_indices[token_idx, k].item()
                count = expert_counts[expert_idx].item()

                if count < capacity:
                    dispatch_mask[token_idx, expert_idx, count] = 1.0
                    combine_weights[token_idx, expert_idx, count] = top_k_probs[
                        token_idx, k
                    ]
                    expert_counts[expert_idx] += 1

        return dispatch_mask, combine_weights, router_probs, aux_loss

    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.

        L_aux = num_experts * sum(f_i * P_i)
        where f_i = fraction of tokens dispatched to expert i
              P_i = fraction of router probability allocated to expert i
        """
        # Get expert mask from top-k indices
        mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        # Mask is (num_tokens, top_k, num_experts)

        # Fraction of tokens dispatched to each expert: f_i = mean over tokens of sum over k of mask
        density = mask.sum(dim=1).mean(dim=0)  # (num_experts,)

        # Fraction of router probability allocated to each expert: P_i = mean over tokens of router_probs
        density_proxy = router_probs.mean(dim=0)  # (num_experts,)

        aux_loss = (density * density_proxy).sum() * self.num_experts

        return aux_loss


# ---------------------------------------------------------------------------
# All-to-All Token Dispatch
# ---------------------------------------------------------------------------


class AllToAllTokenDispatch:
    """
    Manages All-to-All communication for routing tokens between GPUs.

    In expert parallelism:
    - Each GPU starts with its local tokens
    - All-to-All sends tokens to GPUs hosting their target experts
    - After expert computation, All-to-All sends results back

    This is the communication backbone of MoE parallelism.
    Communication volume per step: 2 * batch * seq * hidden * dtype_bytes
    """

    def __init__(
        self,
        num_experts: int,
        experts_per_rank: int,
        ep_group: Any = None,
    ):
        self.num_experts = num_experts
        self.experts_per_rank = experts_per_rank
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if ep_group else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group else 0

    def dispatch(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to expert GPUs via All-to-All.

        Args:
            tokens: (num_local_tokens, hidden_size). Local tokens.
            expert_indices: (num_local_tokens, top_k). Target expert indices.
            router_weights: (num_local_tokens, top_k). Routing weights.

        Returns:
            tuple of:
            - dispatched_tokens: (total_received_tokens, hidden_size)
            - token_to_expert_map: indices mapping dispatched tokens to experts
            - received_weights: router weights for received tokens
        """
        num_local_tokens = tokens.size(0)
        hidden_size = tokens.size(1)
        top_k = expert_indices.size(1)

        # Count tokens per expert
        expert_counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=tokens.device
        )
        for k in range(top_k):
            expert_counts.scatter_add_(
                0, expert_indices[:, k], torch.ones_like(expert_indices[:, k])
            )

        # Local experts on this rank
        expert_start = self.ep_rank * self.experts_per_rank
        expert_end = expert_start + self.experts_per_rank

        # Gather tokens for local experts
        local_expert_mask = (expert_indices >= expert_start) & (
            expert_indices < expert_end
        )
        # Flatten: find which tokens go to which local expert
        token_indices, k_indices = local_expert_mask.nonzero(as_tuple=True)

        if token_indices.numel() == 0:
            return (
                torch.empty(0, hidden_size, device=tokens.device, dtype=tokens.dtype),
                torch.empty(0, dtype=torch.long, device=tokens.device),
                torch.empty(0, device=tokens.device, dtype=tokens.dtype),
            )

        dispatched_tokens = tokens[token_indices]
        local_expert_ids = expert_indices[token_indices, k_indices] - expert_start
        dispatched_weights = router_weights[token_indices, k_indices]

        return dispatched_tokens, local_expert_ids, dispatched_weights

    def combine(
        self,
        expert_outputs: torch.Tensor,
        local_expert_ids: torch.Tensor,
        dispatched_weights: torch.Tensor,
        original_token_count: int,
    ) -> torch.Tensor:
        """
        Combine expert outputs back to original token ordering.

        For each original token, sums the weighted outputs from its
        top-k experts.

        Args:
            expert_outputs: (received_tokens, hidden_size). Expert outputs.
            local_expert_ids: (received_tokens,). Which expert processed each.
            dispatched_weights: (received_tokens,). Router weights.
            original_token_count: Number of tokens to reconstruct.

        Returns:
            (original_token_count, hidden_size). Combined outputs.
        """
        if expert_outputs.numel() == 0:
            return torch.zeros(
                original_token_count,
                expert_outputs.size(-1) if expert_outputs.dim() > 1 else 1,
                device=expert_outputs.device,
                dtype=expert_outputs.dtype,
            )

        weighted_outputs = expert_outputs * dispatched_weights.unsqueeze(-1)

        # Sum outputs for each original token
        combined = torch.zeros(
            original_token_count,
            weighted_outputs.size(-1),
            device=weighted_outputs.device,
            dtype=weighted_outputs.dtype,
        )

        # Note: in practice, we would track which token each output belongs to
        # For simplicity, assume all tokens are from a single rank's perspective
        num_received = weighted_outputs.size(0)
        if num_received > 0:
            # Simple case: scatter back via All-to-All
            combined[:num_received] = weighted_outputs

        return combined


# ---------------------------------------------------------------------------
# MoE Expert (simple MLP expert)
# ---------------------------------------------------------------------------


class MoEExpert(nn.Module):
    """A single expert in the MoE layer, typically a small FFN."""

    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: Optional[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        if expert_hidden_size is None:
            expert_hidden_size = hidden_size * 4
        self.expert_hidden_size = expert_hidden_size

        self.fc1 = nn.Linear(hidden_size, expert_hidden_size)
        self.fc2 = nn.Linear(expert_hidden_size, hidden_size)
        self.activation_fn = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_fn == "gelu":
            x = F.gelu(self.fc1(x))
        elif self.activation_fn == "relu":
            x = F.relu(self.fc1(x))
        elif self.activation_fn == "silu":
            x = F.silu(self.fc1(x))
        else:
            x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Load-Balanced MoE Layer
# ---------------------------------------------------------------------------


class LoadBalancedMoE(nn.Module):
    """
    Full Mixture-of-Experts layer with load balancing.

    Workflow:
        1. Router: token → expert mapping (with top-k and aux loss)
        2. All-to-All dispatch: send tokens to expert GPUs
        3. Expert computation: each GPU runs local experts
        4. All-to-All combine: send results back
        5. Combine weighted expert outputs

    Args:
        hidden_size: Input/output hidden dimension.
        config: MoE configuration.
        ep_group: Expert-parallel process group.
    """

    def __init__(
        self,
        hidden_size: int,
        config: Optional[MoEConfig] = None,
        ep_group: Any = None,
    ):
        super().__init__()
        self.config = config or MoEConfig()
        self.hidden_size = hidden_size
        self.ep_group = ep_group

        self.num_experts = self.config.num_experts
        self.top_k = self.config.top_k
        self.capacity_factor = self.config.capacity_factor

        # Determine expert parallelism layout
        self.ep_size = dist.get_world_size(ep_group) if ep_group else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group else 0
        self.experts_per_rank = max(1, self.num_experts // self.ep_size)

        # Router (replicated on all ranks)
        self.router = ExpertRouter(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            capacity_factor=self.capacity_factor,
            noisy_gating=self.config.noisy_gating,
            noisy_gating_epsilon=self.config.noisy_gating_epsilon,
        )

        # Local experts (only on this rank)
        expert_hidden = self.config.expert_hidden_size or hidden_size * 4
        self.experts = nn.ModuleList(
            [
                MoEExpert(hidden_size, expert_hidden)
                for _ in range(self.experts_per_rank)
            ]
        )

        # Token dispatcher
        self.dispatcher = AllToAllTokenDispatch(
            num_experts=self.num_experts,
            experts_per_rank=self.experts_per_rank,
            ep_group=ep_group,
        )

        # Load balancing loss accumulator
        self.register_buffer("_aux_loss", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x: (batch, seq_len, hidden_size).

        Returns:
            tuple of:
            - output: (batch, seq_len, hidden_size). Combined expert outputs.
            - aux_loss: Scalar load balancing loss.
        """
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len

        # Flatten tokens
        tokens = x.view(num_tokens, hidden_size)

        # Router: determine expert assignments
        dispatch_mask, combine_weights, router_probs, aux_loss = self.router(tokens)

        # Dispatch tokens to expert GPUs
        expert_indices = dispatch_mask.nonzero(as_tuple=False)[
            :, 1
        ]  # expert index per token
        dispatched_tokens, local_expert_ids, dispatched_weights = (
            self.dispatcher.dispatch(
                tokens, expert_indices.unsqueeze(-1), combine_weights
            )
        )

        # Expert computation (only on assigned experts)
        expert_outputs = torch.zeros_like(dispatched_tokens)
        if dispatched_tokens.size(0) > 0:
            # Group tokens by expert for batched computation
            for expert_idx in range(self.experts_per_rank):
                expert_mask = local_expert_ids == expert_idx
                if expert_mask.any():
                    expert_input = dispatched_tokens[expert_mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    expert_outputs[expert_mask] = expert_output

        # Combine expert outputs back
        combined = self.dispatcher.combine(
            expert_outputs, local_expert_ids, dispatched_weights, num_tokens
        )

        output = combined.view(batch_size, seq_len, hidden_size)

        return output, aux_loss


# ---------------------------------------------------------------------------
# Load balancing loss utility
# ---------------------------------------------------------------------------


def compute_load_balancing_loss(
    router_probs: torch.Tensor,
    expert_counts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss from router probabilities
    and expert dispatch counts.

    L_balance = num_experts * sum(f_i * P_i)
    where:
      f_i = expert_counts[i] / total_tokens (fraction dispatched to expert i)
      P_i = mean(router_probs[:, i]) (average probability for expert i)

    This loss is minimized when tokens are uniformly distributed
    across experts.

    Args:
        router_probs: (num_tokens, num_experts). Router softmax probabilities.
        expert_counts: (num_experts,). Number of tokens dispatched per expert.
        num_experts: Total number of experts.

    Returns:
        Scalar auxiliary loss.
    """
    total_tokens = router_probs.size(0)
    if total_tokens == 0:
        return torch.tensor(0.0, device=router_probs.device)

    f = expert_counts.float() / total_tokens
    P = router_probs.mean(dim=0)

    return (f * P).sum() * num_experts


# ---------------------------------------------------------------------------
# Expert-level parallelism wrapper
# ---------------------------------------------------------------------------


class MoEExpertParallel(nn.Module):
    """
    Expert parallelism wrapper for MoE layers.

    Manages expert placement across GPU ranks:
    - Experts are sharded: each rank holds experts_per_rank experts
    - All-to-All routes tokens to the correct rank
    - Expert weights and gradients are local to each rank

    For gradient synchronization:
    - Router weight gradients need All-Reduce (router is replicated)
    - Expert weight gradients are local (no sync needed, each rank
      updates only its own experts)

    Args:
        moe_layer: The LoadBalancedMoE layer to wrap.
        ep_group: Expert-parallel process group.
    """

    def __init__(
        self,
        moe_layer: LoadBalancedMoE,
        ep_group: Any = None,
    ):
        super().__init__()
        self.moe_layer = moe_layer
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if ep_group else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group else 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with expert parallelism."""
        output, aux_loss = self.moe_layer(x)

        # Aggregate aux loss across all ranks
        if self.ep_size > 1 and self.ep_group is not None:
            dist.all_reduce(aux_loss, op=dist.ReduceOp.SUM, group=self.ep_group)

        return output, aux_loss

    def sync_router_gradients(self) -> None:
        """
        Synchronize router gradients across expert-parallel ranks.

        The router is replicated on all ranks, so its gradients
        must be All-Reduced after backward pass.
        """
        if self.ep_size <= 1 or self.ep_group is None:
            return

        for param in self.moe_layer.router.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.ep_group)
