"""
Production-grade Llama-style transformer model.

Provides a complete LlamaModel implementation with:
- Pre-norm architecture with RMSNorm
- Rotary Position Embedding (RoPE)
- SwiGLU MLP with 8d/3 intermediate size
- Grouped Query Attention (GQA) with configurable group size
- Weight tying (input/output embeddings)
- KV cache for efficient autoregressive inference
- Gradient checkpointing support
- Mixed precision support (fp16/bf16)

This implementation follows the Llama 3 architecture patterns.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LlamaMLP(nn.Module):
    """SwiGLU MLP used in Llama models.

    Uses the SwiGLU activation: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Llama 3 uses intermediate_size = 8 * hidden_size / 3 (rounded to multiple of 256)
    for parameter parity with the standard 4x FFN.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Internal hidden dimension.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj: nn.Linear = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.up_proj: nn.Linear = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj: nn.Linear = nn.Linear(
            intermediate_size, hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            Output of shape [batch, seq_len, hidden_size].
        """
        gate: torch.Tensor = F.silu(self.gate_proj(x))
        up: torch.Tensor = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaAttention(nn.Module):
    """Llama-style grouped query attention with RoPE.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Per-head dimension.
        dropout: Attention dropout probability.
        use_rope: If True, applies rotary position embeddings.
        use_qk_norm: If True, applies QK normalization (DeepSeek-V3 style).
        norm_eps: Epsilon for QK normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_qk_norm: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.use_rope: bool = use_rope
        self.use_qk_norm: bool = use_qk_norm
        self.n_rep: int = num_heads // num_kv_heads
        self.attn_dropout: float = dropout

        self.q_proj: nn.Linear = nn.Linear(
            hidden_size, num_heads * head_dim, bias=False
        )
        self.k_proj: nn.Linear = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
        self.v_proj: nn.Linear = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
        self.o_proj: nn.Linear = nn.Linear(
            num_heads * head_dim, hidden_size, bias=False
        )

        # Optional QK normalization
        if use_qk_norm:
            from .rope import QKNorm

            self.qk_norm: Optional[QKNorm] = QKNorm(head_dim, eps=norm_eps)
        else:
            self.qk_norm = None

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match Q heads."""
        if self.n_rep == 1:
            return kv
        batch, n_kv, seq, d = kv.shape
        kv = kv[:, :, None, :, :].expand(batch, n_kv, self.n_rep, seq, d)
        return kv.reshape(batch, n_kv * self.n_rep, seq, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with GQA and optional RoPE.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos: RoPE cosine table.
            sin: RoPE sine table.
            attention_mask: Optional mask.
            kv_cache: Optional tuple (cached_k, cached_v).

        Returns:
            (output, updated_kv_cache)
        """
        batch_size: int = hidden_states.size(0)
        seq_len: int = hidden_states.size(1)

        # Project Q, K, V
        query_states: torch.Tensor = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states: torch.Tensor = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states: torch.Tensor = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE
        if self.use_rope and cos is not None and sin is not None:
            from .rope import apply_rotary_pos_emb

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Apply QK normalization
        if self.use_qk_norm and self.qk_norm is not None:
            # Note: QKNorm operates per-head; need to flatten extra dims
            # Q: [B, H, S, D] -> [B*S, H, D] -> norm -> reshape back
            # For simplicity, we apply element-wise normalization across head_dim
            query_states_flat = query_states.view(-1, self.head_dim)
            key_states_flat = key_states.view(-1, self.head_dim)
            query_states_flat, key_states_flat = self.qk_norm(
                query_states_flat, key_states_flat
            )
            query_states = query_states_flat.view_as(query_states)
            key_states = key_states_flat.view_as(key_states)

        # KV cache
        new_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            key_states = torch.cat([cached_k, key_states], dim=2)
            value_states = torch.cat([cached_v, value_states], dim=2)
        new_kv_cache = (key_states, value_states)

        # Expand KV heads for GQA
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        # Attention computation with flash attention
        key_len: int = key_states.size(2)
        query_len: int = query_states.size(2)

        causal_mask: torch.Tensor = torch.tril(
            torch.ones(
                query_len, key_len, device=hidden_states.device, dtype=torch.bool
            )
        ).view(1, 1, query_len, key_len)

        attn_output: torch.Tensor = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        output: torch.Tensor = self.o_proj(attn_output)
        return output, new_kv_cache


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer with pre-norm architecture.

    Architecture:
        x -> RMSNorm -> Attention (+ residual)
          -> RMSNorm -> SwiGLU MLP (+ residual)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_rope: bool = True,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size

        from .normalization import RMSNorm

        self.input_norm: RMSNorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_attn_norm: RMSNorm = RMSNorm(hidden_size, eps=norm_eps)

        self.self_attn: LlamaAttention = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_rope=use_rope,
            use_qk_norm=use_qk_norm,
            norm_eps=norm_eps,
        )

        self.mlp: LlamaMLP = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_gradient_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos: RoPE cosine table.
            sin: RoPE sine table.
            attention_mask: Optional attention mask.
            kv_cache: Optional KV cache.
            use_gradient_checkpointing: If True, uses checkpoint for attention.

        Returns:
            (output, updated_kv_cache) tuple.
        """
        # Attention with pre-norm
        residual: torch.Tensor = hidden_states
        normed: torch.Tensor = self.input_norm(hidden_states)

        if use_gradient_checkpointing and self.training:
            attn_out, new_kv_cache = checkpoint(
                self.self_attn,
                normed,
                cos,
                sin,
                attention_mask,
                kv_cache,
                use_reentrant=False,
            )
        else:
            attn_out, new_kv_cache = self.self_attn(
                normed,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        hidden_states = residual + attn_out

        # FFN with pre-norm
        residual = hidden_states
        normed = self.post_attn_norm(hidden_states)
        ffn_out: torch.Tensor = self.mlp(normed)
        hidden_states = residual + ffn_out

        return hidden_states, new_kv_cache


class LlamaModel(nn.Module):
    """Production-grade Llama transformer model.

    Implements a complete Llama-style language model with configurable
    architecture, supporting features like weight tying, gradient checkpointing,
    mixed precision, and KV caching.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of the hidden states.
        num_layers: Number of transformer decoder layers.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads (for GQA).
        intermediate_size: Internal dimension of the SwiGLU MLP.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        rope_theta: Base frequency for rotary position embedding.
        norm_eps: Epsilon value for RMSNorm.
        attn_dropout: Dropout probability for attention weights.
        resid_dropout: Dropout probability for residual connections.
        tie_word_embeddings: If True, shares weights between embedding and LM head.
        use_qk_norm: If True, applies QK normalization.
    """

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 14336,
        max_seq_len: int = 8192,
        rope_theta: float = 500000.0,
        norm_eps: float = 1e-5,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = hidden_size // num_heads
        self.max_seq_len: int = max_seq_len

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        # Token embedding
        self.embed_tokens: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
        self.embed_dropout: nn.Dropout = (
            nn.Dropout(resid_dropout) if resid_dropout > 0 else nn.Identity()
        )

        # Rotary position embedding
        from .rope import RotaryEmbedding

        self.rotary_emb: RotaryEmbedding = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        # Decoder layers
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=self.head_dim,
                    intermediate_size=intermediate_size,
                    dropout=attn_dropout,
                    norm_eps=norm_eps,
                    use_rope=True,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        from .normalization import RMSNorm

        self.norm: RMSNorm = RMSNorm(hidden_size, eps=norm_eps)

        # LM head
        self.lm_head: nn.Linear = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight  # type: ignore[assignment]

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize weights with normal distribution (std=0.02)."""
        std: float = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _get_rope_embeddings(
        self,
        seq_len: int,
        device: torch.device,
        input_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE cos/sin tables with proper position offsets.

        Args:
            seq_len: Number of tokens in the current input.
            device: Target device.
            input_pos: Optional position tensor for incremental decoding.

        Returns:
            (cos, sin) tuple.
        """
        if input_pos is not None:
            start_pos: int = input_pos[0].item() if input_pos.numel() > 0 else 0
        else:
            start_pos = 0
        return self.rotary_emb.forward(seq_len, device=device, start_pos=start_pos)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        input_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        """Full model forward pass.

        Args:
            input_ids: Token IDs of shape [batch, seq_len].
            kv_caches: Optional per-layer KV caches for incremental decoding.
            input_pos: Optional position tensor for incremental decoding.
            attention_mask: Optional attention mask.
            use_gradient_checkpointing: If True, uses gradient checkpointing.

        Returns:
            (logits [batch, seq_len, vocab_size], updated_kv_caches) tuple.
        """
        batch_size: int = input_ids.size(0)
        seq_len: int = input_ids.size(1)

        # Token embeddings
        hidden_states: torch.Tensor = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)

        # RoPE embeddings
        cos, sin = self._get_rope_embeddings(
            seq_len, device=hidden_states.device, input_pos=input_pos
        )

        # Initialize KV caches if not provided
        if kv_caches is None:
            kv_caches = [None] * self.num_layers

        new_kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []

        # Pass through decoder layers
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if i < len(kv_caches) else None
            hidden_states, new_cache = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            new_kv_caches.append(new_cache)

        # Final normalization and LM head
        hidden_states = self.norm(hidden_states)
        logits: torch.Tensor = self.lm_head(hidden_states)

        return logits, new_kv_caches

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive text generation with optional sampling.

        Args:
            input_ids: Prompt token IDs of shape [batch, seq_len].
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: If > 0, only sample from top-k tokens.
            top_p: If < 1.0, nucleus sampling threshold.
            use_cache: If True, uses KV cache for faster inference.

        Returns:
            Generated token IDs of shape [batch, prompt_len + new_tokens].
        """
        self.eval()
        device: torch.device = input_ids.device
        generated: torch.Tensor = input_ids.clone()

        kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] | None = (
            [None] * self.num_layers if use_cache else None
        )
        cur_pos: int = input_ids.size(1)

        for _ in range(max_new_tokens):
            if use_cache and kv_caches is not None and kv_caches[0] is not None:
                current_input = generated[:, -1:]
                input_pos = torch.tensor([cur_pos], device=device)
            else:
                current_input = generated
                input_pos = None

            with torch.no_grad():
                logits, kv_caches = self.forward(
                    current_input,
                    kv_caches=kv_caches,
                    input_pos=input_pos,
                )

            cur_pos += current_input.size(1)
            next_logits: torch.Tensor = logits[:, -1, :]

            # Temperature scaling
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / max(temperature, 1e-7)

                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(
                        next_logits, min(top_k, next_logits.size(-1))
                    )
                    mask = torch.full_like(next_logits, float("-inf"))
                    mask.scatter_(-1, top_k_indices, top_k_values)
                    next_logits = mask

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits = next_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated

    def get_num_params(self, include_embeddings: bool = True) -> int:
        """Return the number of trainable parameters.

        Args:
            include_embeddings: If False, excludes embedding parameters.

        Returns:
            Total number of trainable parameters.
        """
        total: int = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if not include_embeddings and "embed_tokens" in name:
                continue
            total += param.numel()
        return total

    def to_dtype(self, dtype: torch.dtype) -> LlamaModel:
        """Convert model to a specific dtype for mixed precision.

        Args:
            dtype: Target data type (e.g., torch.float16, torch.bfloat16).

        Returns:
            Self, for method chaining.
        """
        return self.to(dtype=dtype)  # type: ignore[return-value]


# Quick test
if __name__ == "__main__":
    # Small test config
    config = {
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 8,
        "num_kv_heads": 2,
        "intermediate_size": 1024,
        "max_seq_len": 512,
        "rope_theta": 10000.0,
    }

    model = LlamaModel(**config)  # type: ignore[arg-type]
    num_params: int = model.get_num_params()

    batch, seq = 2, 32
    input_ids = torch.randint(0, config["vocab_size"], (batch, seq))

    # Forward pass
    logits, kv_caches = model(input_ids)
    assert logits.shape == (batch, seq, config["vocab_size"]), (
        f"Logits shape: {logits.shape}"
    )
    print(f"LlamaModel forward: OK, shape={logits.shape}, params={num_params:,}")

    # Test with KV cache
    prompt = torch.randint(0, config["vocab_size"], (1, 4))
    generated = model.generate(
        prompt, max_new_tokens=3, temperature=0.0, use_cache=True
    )
    assert generated.shape[1] == 7
    print(f"LlamaModel generate: OK, output_len={generated.shape[1]}")

    # Test weight tying
    model_tied = LlamaModel(**config, tie_word_embeddings=True)  # type: ignore[arg-type]
    assert model_tied.lm_head.weight is model_tied.embed_tokens.weight
    print(f"LlamaModel weight tying: OK")

    # Test gradient checkpointing
    model.train()
    logits_gc, _ = model(input_ids, use_gradient_checkpointing=True)
    assert logits_gc.shape == logits.shape
    print(f"LlamaModel gradient checkpointing: OK")

    # Test mixed precision
    model_half = model.to_dtype(torch.bfloat16)
    input_ids_bf16 = input_ids.to("cpu")
    model_half = model_half.to("cpu")
    with torch.no_grad():
        logits_bf16, _ = model_half(input_ids_bf16)
    assert logits_bf16.dtype == torch.bfloat16
    print(f"LlamaModel bfloat16: OK")

    print(f"\nAll LlamaModel tests passed! Parameters: {num_params:,}")
