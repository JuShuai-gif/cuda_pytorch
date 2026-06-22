"""
Mixture of Experts (MoE) transformer layer.

Implements MoE with:
- ExpertChoiceRouter: Top-K routing with auxiliary load balancing loss
- ExpertFeedForward: Multiple SwiGLU FFN experts
- Token drop strategy for overloaded experts
- Configurable Top-K (K=1,2,4,8)

Based on:
- Shazeer et al., 2017 "Outrageously Large Neural Networks"
- Fedus et al., 2021 "Switch Transformers"
- Jiang et al., 2024 "Mixtral of Experts"
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFeedForward(nn.Module):
    """A single SwiGLU FFN expert.

    Same architecture as a standard SwiGLU FFN but intended to be used
    as one of many experts in a MoE layer.

    Args:
        hidden_size: Input/output dimension.
        expert_intermediate_size: Internal FFN dimension for this expert.
    """

    def __init__(self, hidden_size: int, expert_intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj: nn.Linear = nn.Linear(
            hidden_size, expert_intermediate_size, bias=False
        )
        self.up_proj: nn.Linear = nn.Linear(
            hidden_size, expert_intermediate_size, bias=False
        )
        self.down_proj: nn.Linear = nn.Linear(
            expert_intermediate_size, hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: SiLU(gate(x)) * up(x), then down-project.

        Args:
            x: [num_tokens, hidden_size]

        Returns:
            Output of shape [num_tokens, hidden_size].
        """
        gate: torch.Tensor = F.silu(self.gate_proj(x))
        up: torch.Tensor = self.up_proj(x)
        return self.down_proj(gate * up)


class ExpertChoiceRouter(nn.Module):
    """Top-K expert routing with auxiliary load balancing loss.

    Routes each token to the Top-K experts based on router logits.
    Supports load balancing to encourage uniform expert utilization.

    Supports two routing strategies:
    - Token choice: each token selects Top-K experts (standard)
    - Expert choice: each expert selects Top-C tokens (Switch Transformer)

    Args:
        hidden_size: Input dimension for the router.
        num_experts: Total number of experts.
        num_experts_per_token: Number of experts to route each token to (Top-K).
        capacity_factor: Capacity factor for token dropping (expert capacity =
            (total_tokens / num_experts) * capacity_factor * num_experts_per_token).
        aux_loss_coef: Weight of auxiliary load balancing loss (0 = disabled).
        routing_strategy: "token_choice" or "expert_choice".
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        routing_strategy: str = "token_choice",
    ) -> None:
        super().__init__()
        self.num_experts: int = num_experts
        self.num_experts_per_token: int = num_experts_per_token
        self.capacity_factor: float = capacity_factor
        self.aux_loss_coef: float = aux_loss_coef
        self.routing_strategy: str = routing_strategy

        self.router: nn.Linear = nn.Linear(hidden_size, num_experts, bias=False)

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary load balancing loss.

        Based on the Switch Transformer formulation:
            loss = num_experts * sum(f_i * P_i)
        where f_i is the fraction of tokens dispatched to expert i,
        and P_i is the average routing probability for expert i.

        Args:
            router_probs: [num_tokens, num_experts] softmax probabilities.
            expert_mask: [num_tokens, num_experts] boolean dispatch mask.

        Returns:
            Scalar load balancing loss.
        """
        # Fraction of tokens dispatched to each expert
        density: torch.Tensor = expert_mask.float().mean(dim=0)  # [num_experts]
        # Average routing probability for each expert
        avg_prob: torch.Tensor = router_probs.mean(dim=0)  # [num_experts]
        # Load balance loss
        loss: torch.Tensor = self.num_experts * (density * avg_prob).sum()
        return loss

    def _token_choice_routing(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token-choice routing: each token picks Top-K experts.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            (dispatch_mask, combine_weights, router_probs, aux_loss) tuple.
            - dispatch_mask: [num_tokens, num_experts] boolean
            - combine_weights: [num_tokens, num_experts] float
            - router_probs: [num_tokens, num_experts] float
            - aux_loss: scalar
        """
        num_tokens: int = hidden_states.size(0)

        # Compute router logits and probabilities
        router_logits: torch.Tensor = self.router(hidden_states)  # [T, E]
        router_probs: torch.Tensor = F.softmax(router_logits.float(), dim=-1).to(
            hidden_states.dtype
        )

        # Select Top-K experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )

        # Normalize weights for selected experts
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Create dispatch mask
        expert_mask: torch.Tensor = torch.zeros(
            num_tokens,
            self.num_experts,
            device=hidden_states.device,
            dtype=torch.bool,
        )
        expert_mask.scatter_(1, top_k_indices, True)

        # Create combine weights
        combine_weights: torch.Tensor = torch.zeros(
            num_tokens,
            self.num_experts,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        combine_weights.scatter_(1, top_k_indices, top_k_probs)

        # Compute auxiliary loss
        aux_loss: torch.Tensor = self._compute_load_balancing_loss(
            router_probs, expert_mask
        )

        return expert_mask, combine_weights, router_probs, aux_loss

    def _expert_choice_routing(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert-choice routing: each expert picks Top-C tokens.

        This ensures perfect load balancing but may drop or duplicate tokens.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            (dispatch_mask, combine_weights, router_probs, aux_loss) tuple.
        """
        num_tokens: int = hidden_states.size(0)

        router_logits: torch.Tensor = self.router(hidden_states)
        router_probs: torch.Tensor = F.softmax(router_logits.float(), dim=-1).to(
            hidden_states.dtype
        )

        # Expert capacity
        expert_capacity: int = max(
            1,
            int(
                self.capacity_factor
                * num_tokens
                / self.num_experts
                * self.num_experts_per_token
            ),
        )

        # Each expert selects Top-C tokens
        _, top_indices = torch.topk(router_probs.t(), expert_capacity, dim=-1)  # [E, C]

        # Build dispatch mask
        expert_mask: torch.Tensor = torch.zeros(
            num_tokens,
            self.num_experts,
            device=hidden_states.device,
            dtype=torch.bool,
        )
        for e in range(self.num_experts):
            expert_mask[top_indices[e], e] = True

        # Combine weights are the router probabilities
        combine_weights: torch.Tensor = router_probs * expert_mask.float()
        # Normalize across experts
        weight_sum: torch.Tensor = combine_weights.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        combine_weights = combine_weights / weight_sum

        aux_loss: torch.Tensor = self._compute_load_balancing_loss(
            router_probs, expert_mask
        )

        return expert_mask, combine_weights, router_probs, aux_loss

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            (dispatch_mask, combine_weights, aux_loss) tuple.
            - dispatch_mask: [num_tokens, num_experts] boolean
            - combine_weights: [num_tokens, num_experts] float
            - aux_loss: scalar
        """
        if self.routing_strategy == "expert_choice":
            expert_mask, combine_weights, router_probs, aux_loss = (
                self._expert_choice_routing(hidden_states)
            )
        else:
            expert_mask, combine_weights, router_probs, aux_loss = (
                self._token_choice_routing(hidden_states)
            )

        return expert_mask, combine_weights, aux_loss


class MoETransformerLayer(nn.Module):
    """Transformer layer with Mixture of Experts FFN.

    Standard attention (with optional GQA) followed by MoE FFN where
    each token is routed to a subset of experts.

    Architecture:
        x -> RMSNorm -> Attention (+ residual)
          -> RMSNorm -> MoE FFN with routing (+ residual)

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA).
        head_dim: Per-head dimension.
        num_experts: Total number of FFN experts.
        num_experts_per_token: Number of experts to route each token to.
        expert_intermediate_size: FFN hidden dimension per expert.
        has_shared_experts: If True, also includes shared experts (DeepSeek style).
        num_shared_experts: Number of shared experts (always active).
        capacity_factor: Expert capacity factor for token dropping.
        aux_loss_coef: Weight of auxiliary load balancing loss.
        norm_eps: Epsilon for RMSNorm.
        dropout: Attention dropout.
        use_rope: Whether to apply RoPE.
        routing_strategy: "token_choice" or "expert_choice".
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_intermediate_size: int = 14336,
        has_shared_experts: bool = False,
        num_shared_experts: int = 1,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        use_rope: bool = True,
        routing_strategy: str = "token_choice",
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.has_shared_experts: bool = has_shared_experts
        self.aux_loss_coef: float = aux_loss_coef

        # Pre-norm layers
        from .normalization import RMSNorm

        self.input_norm: RMSNorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_attn_norm: RMSNorm = RMSNorm(hidden_size, eps=norm_eps)

        # Attention
        from .attention import GroupedQueryAttention

        self.attention: GroupedQueryAttention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_rope=use_rope,
        )

        # Expert router
        self.router: ExpertChoiceRouter = ExpertChoiceRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            capacity_factor=capacity_factor,
            aux_loss_coef=aux_loss_coef,
            routing_strategy=routing_strategy,
        )

        # Expert FFNs
        self.experts: nn.ModuleList = nn.ModuleList(
            [
                ExpertFeedForward(hidden_size, expert_intermediate_size)
                for _ in range(num_experts)
            ]
        )

        # Shared experts (always activated, DeepSeek style)
        self.num_shared_experts: int = num_shared_experts
        if has_shared_experts:
            self.shared_experts: nn.ModuleList = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            hidden_size,
                            expert_intermediate_size * num_shared_experts,
                            bias=False,
                        ),
                        nn.SiLU(),
                        nn.Linear(
                            expert_intermediate_size * num_shared_experts,
                            hidden_size,
                            bias=False,
                        ),
                    )
                ]
            )

    def _moe_ffn(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MoE FFN with expert routing.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            (output [batch, seq_len, hidden_size], aux_loss scalar) tuple.
        """
        batch, seq, hidden = hidden_states.shape
        num_tokens: int = batch * seq

        # Flatten for routing
        flat_hidden: torch.Tensor = hidden_states.view(num_tokens, hidden)

        # Route tokens to experts
        expert_mask, combine_weights, aux_loss = self.router(flat_hidden)
        # expert_mask: [T, E] bool, combine_weights: [T, E] float

        # Process tokens through experts
        output: torch.Tensor = torch.zeros_like(flat_hidden)

        for e in range(self.router.num_experts):
            # Tokens routed to this expert
            token_indices: torch.Tensor = expert_mask[:, e].nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue

            expert_input: torch.Tensor = flat_hidden[token_indices]  # [n, hidden]
            expert_output: torch.Tensor = self.experts[e](expert_input)  # [n, hidden]

            # Weight by routing probability
            weights: torch.Tensor = combine_weights[token_indices, e].unsqueeze(-1)
            weighted_output: torch.Tensor = expert_output * weights

            # Scatter back
            output.index_add_(0, token_indices, weighted_output)

        # Add shared experts output if present
        if self.has_shared_experts:
            output = output + self.shared_experts[0](flat_hidden)

        output = output.view(batch, seq, hidden)
        return output, aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Forward pass through the MoE transformer layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos: RoPE cosine table.
            sin: RoPE sine table.
            attention_mask: Optional attention mask.
            kv_cache: Optional KV cache.

        Returns:
            (output, updated_kv_cache, aux_loss) tuple.
        """
        # Pre-norm attention + residual
        residual: torch.Tensor = hidden_states
        normed: torch.Tensor = self.input_norm(hidden_states)
        attn_out, new_kv_cache = self.attention(
            normed,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        hidden_states = residual + attn_out

        # Pre-norm MoE FFN + residual
        residual = hidden_states
        normed = self.post_attn_norm(hidden_states)
        moe_out, aux_loss = self._moe_ffn(normed)
        hidden_states = residual + moe_out

        return hidden_states, new_kv_cache, aux_loss


# Quick test
if __name__ == "__main__":
    batch, seq, hidden = 2, 16, 256
    num_heads, num_kv_heads = 8, 2
    head_dim = hidden // num_heads

    # Test ExpertFeedForward
    ffn = ExpertFeedForward(hidden, 512)
    x = torch.randn(32, hidden)
    out = ffn(x)
    assert out.shape == x.shape
    print(f"ExpertFeedForward: OK, shape={out.shape}")

    # Test ExpertChoiceRouter
    router = ExpertChoiceRouter(
        hidden_size=hidden,
        num_experts=4,
        num_experts_per_token=2,
        aux_loss_coef=0.01,
    )
    mask, weights, aux_loss = router(x)
    assert mask.shape == (32, 4)
    assert weights.shape == (32, 4)
    assert aux_loss.ndim == 0
    print(f"ExpertChoiceRouter: OK, mask={mask.shape}, loss={aux_loss.item():.4f}")

    # Test MoETransformerLayer
    layer = MoETransformerLayer(
        hidden_size=hidden,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=4,
        num_experts_per_token=2,
        expert_intermediate_size=512,
    )
    x = torch.randn(batch, seq, hidden)
    out, kv_cache, aux_loss = layer(x)
    assert out.shape == x.shape, f"MoE layer shape: {out.shape}"
    print(f"MoETransformerLayer: OK, shape={out.shape}, aux_loss={aux_loss.item():.4f}")

    # Test with shared experts
    layer_deepseek = MoETransformerLayer(
        hidden_size=hidden,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=4,
        num_experts_per_token=2,
        expert_intermediate_size=512,
        has_shared_experts=True,
        num_shared_experts=1,
    )
    out2, cache2, loss2 = layer_deepseek(x)
    assert out2.shape == x.shape
    print(f"MoETransformerLayer (shared experts): OK, shape={out2.shape}")

    # Test incremental decoding
    x_step1 = x[:, :1, :]
    out3, kv_cache, _ = layer(x_step1, kv_cache=None)
    x_step2 = x[:, 1:2, :]
    out4, _, _ = layer(x_step2, kv_cache=kv_cache)
    assert out4.shape == (batch, 1, hidden)
    print(f"MoE KV cache decode: OK")

    print("\nAll MoE tests passed!")
