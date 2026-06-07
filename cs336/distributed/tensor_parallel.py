"""
Megatron-style Tensor Parallelism for Transformer models.

Implements:
- ColumnParallelLinear: Split weight matrix by columns, replicate input
- RowParallelLinear: Split weight matrix by rows, All-Reduce output
- TensorParallelAttention: QKV projection + output projection with TP
- TensorParallelMLP: Column-parallel FC1 + Row-parallel FC2
- TensorParallelTransformerLayer: Combined TP attention + MLP block
- VocabularyParallelEmbedding: Split embedding table across GPUs
- ColumnParallelLMHead: Output projection tied with embedding

Communication analysis per block:
- Forward: 1x All-Reduce (RowParallel fc2 in MLP) + 1x All-Reduce (RowParallel out_proj in attention)
- Backward: 1x All-Reduce (ColumnParallel fc1 gradient) + 1x All-Reduce (ColumnParallel qkv gradient)
- Total: 2 forward + 2 backward All-Reduces per transformer layer

Memory: Parameters split evenly → 1/N per GPU initially.
        But activations grow with replicated inputs.
        TP limited to NVLink domain (< 8 GPUs typically).

Reference:
    Megatron-LM: Training Multi-Billion Parameter Language Models
    Using Model Parallelism (Shoeybi et al., 2019)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from cs336.distributed.collective_ops import create_all_reduce, create_all_gather


# ---------------------------------------------------------------------------
# Device Mesh for multi-dimensional parallelism
# ---------------------------------------------------------------------------


@dataclass
class DeviceMesh:
    """
    Logical device mesh for multi-dimensional parallelism.

    Manages mapping of process ranks to (tp, pp, dp) dimensions.
    Creates process subgroups for each parallelism dimension.

    Example: 8 GPUs with tp_size=2, pp_size=2, dp_size=2:
        Mesh layout: [[gpu0, gpu1],   # TP group 0
                       [gpu2, gpu3],   # TP group 1
                       [gpu4, gpu5],   # PP group 0 (with above)
                       [gpu6, gpu7]]   # PP group 1
    """

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    world_size: int = 1
    tp_group: Any = None
    pp_group: Any = None
    dp_group: Any = None
    _rank: int = 0

    def total_gpus(self) -> int:
        return self.tp_size * self.pp_size * self.dp_size


def create_device_mesh(
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
) -> DeviceMesh:
    """
    Create a 3D device mesh for PTD-P parallelism.

    Maps ranks into (tp, pp, dp) coordinates and creates
    process subgroups for each dimension.

    Rank mapping: rank = dp_idx * tp_size * pp_size + pp_idx * tp_size + tp_idx
    (TP inner-most, PP middle, DP outer-most for best NVLink locality)

    Args:
        tp_size: Tensor parallelism size.
        pp_size: Pipeline parallelism size.
        dp_size: Data parallelism size.

    Returns:
        Configured DeviceMesh.
    """
    mesh = DeviceMesh(
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
    )

    if not dist.is_initialized():
        mesh.world_size = mesh.total_gpus()
        return mesh

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    mesh.world_size = world_size
    mesh._rank = rank

    # Create TP groups: ranks with same dp_idx and pp_idx
    tp_group_ranks: list[list[int]] = []
    pp_group_ranks: list[list[int]] = []
    dp_group_ranks: list[list[int]] = []

    for dp in range(dp_size):
        for pp in range(pp_size):
            tp_ranks = [
                dp * tp_size * pp_size + pp * tp_size + tp for tp in range(tp_size)
            ]
            tp_group_ranks.append(tp_ranks)

    for dp in range(dp_size):
        for tp in range(tp_size):
            pp_ranks = [
                dp * tp_size * pp_size + pp * tp_size + tp for pp in range(pp_size)
            ]
            pp_group_ranks.append(pp_ranks)

    for tp in range(tp_size):
        for pp in range(pp_size):
            dp_ranks = [
                dp * tp_size * pp_size + pp * tp_size + tp for dp in range(dp_size)
            ]
            dp_group_ranks.append(dp_ranks)

    # Create process groups for the rank's cohorts
    for tp_ranks in tp_group_ranks:
        if rank in tp_ranks:
            mesh.tp_group = dist.new_group(tp_ranks)
            break

    for pp_ranks in pp_group_ranks:
        if rank in pp_ranks:
            mesh.pp_group = dist.new_group(pp_ranks)
            break

    for dp_ranks in dp_group_ranks:
        if rank in dp_ranks:
            mesh.dp_group = dist.new_group(dp_ranks)
            break

    return mesh


# ---------------------------------------------------------------------------
# Column Parallel Linear
# ---------------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with weight matrix split by columns across TP ranks.

    Forward: y_i = x @ W_i  (no communication needed for output)
    Backward: gradients need All-Reduce for dL/dx

    Weight shape per rank: (in_features, out_features // tp_size)
    Input: replicated (same on all ranks)
    Output: partitioned along last dim

    Use cases:
    - FC1 in MLP (before activation, no comm needed before activation)
    - QKV projection in attention (before splitting heads, no comm needed)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        gather_output: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output

        # Handle non-divisible output features
        base_chunk = out_features // tp_size
        remainder = out_features % tp_size

        if remainder != 0 and tp_rank == tp_size - 1:
            self.partition_out_features = base_chunk + remainder
        else:
            self.partition_out_features = base_chunk + (1 if tp_rank < remainder else 0)

        self.weight = nn.Parameter(
            torch.empty(
                self.partition_out_features, in_features, device=device, dtype=dtype
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.partition_out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights following Megatron convention."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with column-parallel computation.

        Args:
            x: Input tensor of shape (..., in_features). Replicated on all ranks.

        Returns:
            Output tensor of shape (..., partition_out_features).
        """
        y = F.linear(x, self.weight, self.bias)

        if self.gather_output and self.tp_size > 1:
            # All-Gather along last dim to reconstruct full output
            y = create_all_gather(y, gather_dim=-1)

        return y


# ---------------------------------------------------------------------------
# Row Parallel Linear
# ---------------------------------------------------------------------------


class RowParallelLinear(nn.Module):
    """
    Linear layer with weight matrix split by rows across TP ranks.

    Forward: y_i = x_i @ W_i (partial sum), then y = All-Reduce(y_i)
    Backward: gradients flow independently (input already partitioned)

    Weight shape per rank: (in_features // tp_size, out_features)
    Input: partitioned along last dim
    Output: replicated (same on all ranks after All-Reduce)

    Use cases:
    - FC2 in MLP (after activation, All-Reduce partial sums)
    - Output projection in attention (All-Reduce after attention)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # Handle non-divisible input features
        base_chunk = in_features // tp_size
        remainder = in_features % tp_size

        if remainder != 0 and tp_rank == tp_size - 1:
            self.partition_in_features = base_chunk + remainder
        else:
            self.partition_in_features = base_chunk + (1 if tp_rank < remainder else 0)

        self.weight = nn.Parameter(
            torch.empty(
                out_features, self.partition_in_features, device=device, dtype=dtype
            )
        )
        if bias:
            # Bias is replicated (every rank has full bias)
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with row-parallel computation and All-Reduce.

        Args:
            x: Input tensor of shape (..., partition_in_features).

        Returns:
            Output tensor of shape (..., out_features), replicated across ranks.
        """
        y = F.linear(x, self.weight, self.bias)
        # y is a partial sum; All-Reduce to get full result
        if self.tp_size > 1:
            create_all_reduce(y)
        return y


# ---------------------------------------------------------------------------
# Tensor Parallel Attention
# ---------------------------------------------------------------------------


class TensorParallelAttention(nn.Module):
    """
    Multi-head attention with tensor parallelism.

    QKV projection: ColumnParallel (each rank gets a subset of heads)
    Output projection: RowParallel (partial sums All-Reduced)

    Communication per forward pass:
    - None for QKV (column-parallel)
    - None for attention computation (per-head, independent)
    - 1x All-Reduce for output projection
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dropout = dropout

        # Heads per TP rank (partitioned evenly)
        assert num_heads % tp_size == 0, (
            f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
        )
        assert self.num_kv_heads % tp_size == 0, (
            f"num_kv_heads ({self.num_kv_heads}) must be divisible by tp_size ({tp_size})"
        )
        self.heads_per_rank = num_heads // tp_size
        self.kv_heads_per_rank = self.num_kv_heads // tp_size
        self.head_dim = hidden_size // num_heads

        # Column-parallel QKV projection
        total_qkv_out = (num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=total_qkv_out,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Row-parallel output projection
        self.out_proj = RowParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Split hidden dim into (num_heads, head_dim)."""
        batch_size = x.size(0)
        seq_len = x.size(1)
        return x.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge (num_heads, head_dim) back to hidden_dim."""
        batch_size = x.size(0)
        seq_len = x.size(2)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through tensor-parallel attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size). Replicated input.
            attention_mask: Optional causal mask.
            position_ids: Optional position IDs.
            past_key_value: Optional KV cache tuple.
            use_cache: Whether to return KV cache.

        Returns:
            (output, optional KV cache) tuple.
        """
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)

        qkv = self.qkv_proj(hidden_states)

        # Split into Q, K, V
        q_size = self.heads_per_rank * self.head_dim
        k_size = self.kv_heads_per_rank * self.head_dim
        v_size = self.kv_heads_per_rank * self.head_dim

        q = qkv[:, :, :q_size]
        k = qkv[:, :, q_size : q_size + k_size]
        v = qkv[:, :, q_size + k_size :]

        q = self._split_heads(q, self.heads_per_rank)
        k = self._split_heads(k, self.kv_heads_per_rank)
        v = self._split_heads(v, self.kv_heads_per_rank)

        # KV cache handling
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None and past_key_value is None,
        )

        # GQA group expansion: replicate KV heads to match Q heads if needed
        if self.heads_per_rank > self.kv_heads_per_rank:
            groups = self.heads_per_rank // self.kv_heads_per_rank
            attn_output = attn_output.unsqueeze(2).expand(-1, -1, groups, -1, -1)
            attn_output = attn_output.reshape(
                batch_size, self.heads_per_rank, seq_len, self.head_dim
            )

        attn_output = self._merge_heads(attn_output)

        # Row-parallel output projection (includes All-Reduce)
        output = self.out_proj(attn_output)

        return output, new_kv


# ---------------------------------------------------------------------------
# Tensor Parallel MLP
# ---------------------------------------------------------------------------


class TensorParallelMLP(nn.Module):
    """
    Two-layer MLP with Megatron-style tensor parallelism.

    Architecture: ColumnParallel fc1 -> GELU -> RowParallel fc2

    Communication per forward pass:
    - 1x All-Reduce in fc2
    - 0 communication in fc1
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        activation: str = "gelu",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.activation_fn = activation

        self.fc1 = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.fc2 = RowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TP MLP.

        Args:
            x: (batch, seq_len, hidden_size). Replicated input.

        Returns:
            (batch, seq_len, hidden_size). Replicated output.
        """
        h = self.fc1(x)

        if self.activation_fn == "gelu":
            h = F.gelu(h)
        elif self.activation_fn == "relu":
            h = F.relu(h)
        elif self.activation_fn == "silu":
            h = F.silu(h)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_fn}")

        y = self.fc2(h)
        return y


# ---------------------------------------------------------------------------
# Full Tensor Parallel Transformer Layer
# ---------------------------------------------------------------------------


class TensorParallelTransformerLayer(nn.Module):
    """
    A complete transformer layer with Megatron-style tensor parallelism.

    Architecture (Pre-LN):
        LN -> TP Attention (+ residual) -> LN -> TP MLP (+ residual)

    Communication per layer (forward):
    - Attention: 1x All-Reduce (out_proj)
    - MLP: 1x All-Reduce (fc2)

    Total: 2x All-Reduce per layer per forward pass
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: Optional[int] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # Layer norms are replicated (small, not worth splitting)
        self.input_layernorm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.post_attention_layernorm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps, device=device, dtype=dtype
        )

        self.attention = TensorParallelAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

        self.mlp = TensorParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            activation=activation,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through a TP transformer layer.

        Args:
            hidden_states: (batch, seq_len, hidden_size).
            attention_mask: Optional mask.
            position_ids: Optional position IDs.
            past_key_value: KV cache for this layer.
            use_cache: Return KV cache.

        Returns:
            (output, optional KV cache).
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, new_kv = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        hidden_states = residual + mlp_output

        return hidden_states, new_kv


# ---------------------------------------------------------------------------
# Vocabulary Parallel Embedding
# ---------------------------------------------------------------------------


class VocabularyParallelEmbedding(nn.Module):
    """
    Embedding layer split across TP ranks by vocabulary dimension.

    Each rank stores vocab_size // tp_size rows of the embedding table.
    Input tokens are masked to get the local partition, and output
    is gathered (or left partitioned for subsequent TP layers).

    For the LM head: embedding weight can be tied with the output
    projection (weight tying), requiring an All-Reduce in backward.

    Args:
        vocab_size: Total vocabulary size.
        hidden_size: Embedding dimension.
        padding_idx: Padding token index.
        tp_size: Tensor parallelism size.
        tp_rank: This rank's index within TP group.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # Compute local vocabulary range
        base_size = vocab_size // tp_size
        remainder = vocab_size % tp_size
        self.vocab_start_idx = tp_rank * base_size + min(tp_rank, remainder)
        if tp_rank < remainder:
            self.vocab_end_idx = self.vocab_start_idx + base_size + 1
        else:
            self.vocab_end_idx = self.vocab_start_idx + base_size
        self.local_vocab_size = self.vocab_end_idx - self.vocab_start_idx

        self.weight = nn.Parameter(
            torch.empty(self.local_vocab_size, hidden_size, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings for input token IDs.

        Tokens that fall outside the local vocabulary range are masked
        to the padding index (or 0) and contribute 0 to the embedding.

        For full embedding output with TP, each rank outputs zeros for
        tokens outside its partition; the full result is then the sum
        across ranks (via All-Reduce or as input to ColumnParallel).

        Args:
            input_ids: (batch, seq_len). Token IDs in [0, vocab_size).

        Returns:
            (batch, seq_len, hidden_size). Partitioned embedding output.
        """
        # Map global token IDs to local indices, masking out-of-range tokens
        local_ids = input_ids - self.vocab_start_idx
        mask = (local_ids < 0) | (local_ids >= self.local_vocab_size)
        local_ids = local_ids.clamp(0, self.local_vocab_size - 1)

        embeddings = F.embedding(
            local_ids,
            self.weight,
            padding_idx=self.padding_idx,
        )

        # Zero out embeddings for tokens not in this partition
        embeddings[mask] = 0.0

        return embeddings

    def forward_gather(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup and All-Reduce to get full embedding output on all ranks.

        Since the embeddings are partitioned, each rank computes a partial
        embedding (non-zero for its partition), and All-Reduce combines them.

        Args:
            input_ids: (batch, seq_len). Token IDs.

        Returns:
            (batch, seq_len, hidden_size). Full embedding on all ranks.
        """
        local_emb = self.forward(input_ids)
        if self.tp_size > 1:
            create_all_reduce(local_emb)
        return local_emb


class ColumnParallelLMHead(nn.Module):
    """
    Language model head with column-parallel output and tied embedding.

    The output projection weight is the same as the embedding weight
    (weight tying). For tensor parallelism, this means:
    - Each rank computes logits for its vocabulary partition
    - Cross-entropy loss requires All-Gather or parallel loss computation

    Architecture:
        hidden -> Linear(vocab_size) -> logits
        Weight tied with VocabularyParallelEmbedding.weight

    Args:
        hidden_size: Input hidden dimension.
        vocab_size: Total vocabulary size.
        weight: Shared embedding weight (from VocabularyParallelEmbedding).
        tp_size: Tensor parallelism size.
        tp_rank: This rank's index.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        weight: nn.Parameter,
        tp_size: int = 1,
        tp_rank: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # Share weight with embedding
        self.weight = weight
        # Local vocabulary size (from the shared weight's shape)
        self.local_vocab_size = weight.size(0)
        self.vocab_start_idx = tp_rank * (vocab_size // tp_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for the local vocabulary partition.

        Args:
            hidden_states: (batch, seq_len, hidden_size).

        Returns:
            (batch, seq_len, local_vocab_size). Local logits.
        """
        return F.linear(hidden_states, self.weight)

    def forward_all_gather(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute and All-Gather logits for the full vocabulary.

        Args:
            hidden_states: (batch, seq_len, hidden_size).

        Returns:
            (batch, seq_len, vocab_size). Full logits on all ranks.
        """
        local_logits = self.forward(hidden_states)
        if self.tp_size > 1:
            return create_all_gather(local_logits, gather_dim=-1)
        return local_logits
