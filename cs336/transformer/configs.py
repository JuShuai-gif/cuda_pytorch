"""
Model configurations for major transformer architectures.

Provides dataclass-based configurations for:
- Llama3, Llama2
- DeepSeek-V3 (MLA + MoE)
- Mistral (sliding window)
- Gemma
- Mixtral (MoE)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """Base configuration shared across all model architectures.

    All derived configs inherit these base parameters with appropriate defaults.
    """

    vocab_size: int = 128256
    hidden_size: int = 4096
    num_layers: int = 32
    max_seq_len: int = 8192
    norm_eps: float = 1e-5
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        pass


@dataclass
class Llama3Config(BaseConfig):
    """Llama 3 architecture configuration.

    Key specs:
    - Llama 3 8B:  32 layers, 4096 dim, 32 Q heads, 8 KV heads, G=4
    - Llama 3 70B: 80 layers, 8192 dim, 64 Q heads, 8 KV heads, G=8
    - RoPE theta: 500,000
    - SwiGLU MLP with intermediate_size = 8*hidden_size/3 for param parity
    - Pre-norm with RMSNorm
    """

    # Architecture
    num_heads: int = 32
    num_kv_heads: int = 8

    # RoPE
    rope_theta: float = 500000.0
    use_rope: bool = True

    # SwiGLU MLP (Llama 3 uses 8/3 * hidden_size for param parity)
    intermediate_size: int = field(init=False)

    # Attention
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # Weight tying
    tie_word_embeddings: bool = False

    # Misc
    use_qk_norm: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.head_dim = self.hidden_size // self.num_heads
        # Llama 3 uses 3.5x hidden_size for MLP (vs 8d/3 in Llama 2)
        computed = int(3.5 * self.hidden_size)
        self.intermediate_size = self._round_to_multiple(computed, 256)

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @staticmethod
    def _round_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple


@dataclass
class Llama2Config(BaseConfig):
    """Llama 2 architecture configuration.

    Key specs:
    - Llama 2 7B:  32 layers, 4096 dim, 32 heads, MHA
    - Llama 2 13B: 40 layers, 5120 dim, 40 heads, MHA
    - Llama 2 70B: 80 layers, 8192 dim, 64 heads, GQA with 8 KV heads
    - RoPE theta: 10,000
    - SwiGLU MLP with intermediate_size = 8*hidden_size/3
    """

    # Architecture
    num_heads: int = 32
    num_kv_heads: int = 32  # MHA for 7B/13B; 8 KV for 70B

    # RoPE
    rope_theta: float = 10000.0
    use_rope: bool = True

    # SwiGLU MLP
    intermediate_size: int = field(init=False)

    # Attention
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # Weight tying
    tie_word_embeddings: bool = False

    # Misc
    use_qk_norm: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.head_dim = self.hidden_size // self.num_heads
        computed = int(8 * self.hidden_size / 3)
        self.intermediate_size = self._round_to_multiple(computed, 256)

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @staticmethod
    def _round_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple


@dataclass
class DeepSeekV3Config(BaseConfig):
    """DeepSeek-V3 architecture configuration (MLA + MoE).

    Key specs:
    - Multi-head Latent Attention (MLA) with low-rank KV compression
    - DeepSeekMoE: fine-grained experts with shared experts
    - Decoupled RoPE for MLA
    - QK normalization
    - DeepNorm initialization
    - 671B total params, 37B active per token

    MLA reduces KV cache by ~32x through low-rank compression.
    """

    # Architecture
    num_heads: int = 128
    num_kv_heads: int = 128  # Full heads; MLA compresses them

    # MLA parameters
    kv_lora_rank: int = 512  # Low-rank dimension for KV compression
    qk_rope_head_dim: int = 64  # Per-head dim for decoupled RoPE in MLA
    v_head_dim: int = 128  # Per-head dim for value

    # RoPE
    rope_theta: float = 10000.0
    use_rope: bool = True

    # MoE parameters
    num_experts: int = 256  # Total number of fine-grained experts
    num_shared_experts: int = 1  # Shared experts (always activated)
    num_experts_per_token: int = 8  # Top-K routing per token
    expert_intermediate_size: int = 2048  # FFN dim per expert
    routed_scaling_factor: float = 1.0  # Output scaling for routed experts

    # Dense MLP (for non-MoE layers or shared experts)
    intermediate_size: int = field(init=False)

    # Normalization
    norm_eps: float = 1e-6
    use_qk_norm: bool = True

    # Weight tying
    tie_word_embeddings: bool = False

    # DeepNorm initialization scale
    deepnorm_alpha: float = 1.2

    def __post_init__(self) -> None:
        super().__post_init__()
        computed = int(8 * self.hidden_size / 3)
        self.intermediate_size = self._round_to_multiple(computed, 256)

    @staticmethod
    def _round_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple


@dataclass
class MistralConfig(BaseConfig):
    """Mistral architecture configuration with sliding window attention.

    Key specs:
    - Mistral 7B: 32 layers, 4096 dim, 32 Q heads, 8 KV heads
    - Sliding window attention (W=4096) on most layers
    - Global attention on some layers for long-range dependencies
    - RoPE theta: 1,000,000
    """

    # Architecture
    num_heads: int = 32
    num_kv_heads: int = 8

    # RoPE
    rope_theta: float = 1000000.0
    use_rope: bool = True

    # Sliding window
    sliding_window: int = 4096  # Window size for local attention
    # Indices of layers that use global attention (typically every 6th layer)
    global_attn_layers: set[int] = field(default_factory=set)

    # SwiGLU MLP
    intermediate_size: int = field(init=False)

    # Weight tying
    tie_word_embeddings: bool = False

    # Misc
    use_qk_norm: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.head_dim = self.hidden_size // self.num_heads
        computed = int(8 * self.hidden_size / 3)
        self.intermediate_size = self._round_to_multiple(computed, 256)

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        # Default global attention layers: every 6th layer
        if not self.global_attn_layers:
            self.global_attn_layers = {
                i for i in range(self.num_layers) if (i + 1) % 6 == 0
            }

    @staticmethod
    def _round_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple


@dataclass
class GemmaConfig(BaseConfig):
    """Gemma architecture configuration (Google).

    Key specs:
    - Gemma 2B:  18 layers, 2048 dim, 8 heads, 1 KV head (MQA)
    - Gemma 7B:  28 layers, 3072 dim, 16 heads, 16 KV heads (MHA)
    - Uses GeGLU activation (GELU-gated) instead of SwiGLU
    - RoPE theta: 10,000
    - Pre-norm with RMSNorm
    """

    # Architecture
    num_heads: int = 8
    num_kv_heads: int = 1  # MQA variant

    # RoPE
    rope_theta: float = 10000.0
    use_rope: bool = True

    # Activation: "gelu_pytorch_tanh" or "swiglu"
    activation: str = "gelu_pytorch_tanh"

    # MLP
    intermediate_size: int = field(init=False)

    # Weight tying
    tie_word_embeddings: bool = False

    # Misc
    use_qk_norm: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.head_dim = self.hidden_size // self.num_heads
        computed = int(8 * self.hidden_size / 3)
        self.intermediate_size = self._round_to_multiple(computed, 256)

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @staticmethod
    def _round_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple


@dataclass
class MixtralConfig(BaseConfig):
    """Mixtral MoE architecture configuration.

    Key specs:
    - Mixtral 8x7B: 32 layers, 4096 dim, 32 Q heads, 8 KV heads
    - 8 experts, Top-2 routing
    - RoPE theta: 1,000,000
    - SwiGLU FFN per expert
    - Same architecture as Mistral but with MoE FFN layers
    """

    # Architecture
    num_heads: int = 32
    num_kv_heads: int = 8

    # RoPE
    rope_theta: float = 1000000.0
    use_rope: bool = True

    # MoE
    num_experts: int = 8
    num_experts_per_token: int = 2  # Top-K routing
    expert_intermediate_size: int = field(init=False)
    # Dense FFN intermediate (same as SwiGLU for non-MoE layers)
    intermediate_size: int = field(init=False)

    # Weight tying
    tie_word_embeddings: bool = False

    # Misc
    use_qk_norm: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.head_dim = self.hidden_size // self.num_heads
        computed = int(8 * self.hidden_size / 3)
        self.intermediate_size = self._round_to_multiple(computed, 256)
        self.expert_intermediate_size = self.intermediate_size

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @staticmethod
    def _round_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple


# Pre-defined model configurations for easy instantiation


def llama3_8b_config() -> Llama3Config:
    """Llama 3 8B configuration."""
    return Llama3Config(
        vocab_size=128256,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        max_seq_len=8192,
        rope_theta=500000.0,
        norm_eps=1e-5,
    )


def llama3_70b_config() -> Llama3Config:
    """Llama 3 70B configuration."""
    return Llama3Config(
        vocab_size=128256,
        hidden_size=8192,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        max_seq_len=8192,
        rope_theta=500000.0,
        norm_eps=1e-5,
    )


def llama2_7b_config() -> Llama2Config:
    """Llama 2 7B configuration."""
    return Llama2Config(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        max_seq_len=4096,
    )


def mistral_7b_config() -> MistralConfig:
    """Mistral 7B configuration with sliding window attention."""
    return MistralConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        max_seq_len=32768,
        sliding_window=4096,
        rope_theta=1000000.0,
    )


def mixtral_8x7b_config() -> MixtralConfig:
    """Mixtral 8x7B MoE configuration."""
    return MixtralConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        max_seq_len=32768,
        num_experts=8,
        num_experts_per_token=2,
        rope_theta=1000000.0,
    )


def gemma_2b_config() -> GemmaConfig:
    """Gemma 2B configuration."""
    return GemmaConfig(
        vocab_size=256000,
        hidden_size=2048,
        num_layers=18,
        num_heads=8,
        num_kv_heads=1,
        max_seq_len=8192,
        intermediate_size=16384,
    )


def gemma_7b_config() -> GemmaConfig:
    """Gemma 7B configuration."""
    return GemmaConfig(
        vocab_size=256000,
        hidden_size=3072,
        num_layers=28,
        num_heads=16,
        num_kv_heads=16,
        max_seq_len=8192,
        intermediate_size=24576,
    )
