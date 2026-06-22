"""
Transformer Variants Module

Production-grade implementations of major transformer architectures:

Attention:
- MultiHeadAttention, GroupedQueryAttention, MultiQueryAttention (attention.py)
- MultiHeadLatentAttention - DeepSeek-V3 style (mla.py)
- SlidingWindowAttention - Mistral style (sliding_window.py)

Models:
- LlamaModel - LLaMA 2/3 style with GQA + SwiGLU (llama.py)
- MoETransformerLayer - Mixture of Experts with load balancing (mix_of_experts.py)

Positional Encoding:
- RotaryEmbedding - Standard RoPE (rope.py)
- YaRN - Context length extrapolation (rope.py)
- QKNorm - Query-Key normalization (rope.py)

Configurations:
- Llama3Config, Llama2Config, DeepSeekV3Config, MistralConfig,
  GemmaConfig, MixtralConfig (configs.py)

Utilities:
- Model loading, weight mapping, checkpoint save/load (model_loader.py)
"""

from .attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)

from .mla import MultiHeadLatentAttention

from .sliding_window import (
    MixedAttentionLayer,
    SlidingWindowAttention,
)

from .llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
)

from .mix_of_experts import (
    ExpertChoiceRouter,
    ExpertFeedForward,
    MoETransformerLayer,
)

from .rope import (
    QKNorm,
    RotaryEmbedding,
    YaRN,
    apply_rotary_pos_emb,
)

from .normalization import (
    DeepNorm,
    RMSNorm,
)

from .configs import (
    BaseConfig,
    DeepSeekV3Config,
    GemmaConfig,
    Llama2Config,
    Llama3Config,
    MistralConfig,
    MixtralConfig,
    gemma_2b_config,
    gemma_7b_config,
    llama2_7b_config,
    llama3_70b_config,
    llama3_8b_config,
    mistral_7b_config,
    mixtral_8x7b_config,
)

from .model_loader import (
    WeightMapper,
    convert_config_to_llama3,
    load_checkpoint,
    load_hf_checkpoint,
    save_checkpoint,
)

__all__ = [
    # Attention
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "MultiHeadLatentAttention",
    "SlidingWindowAttention",
    "MixedAttentionLayer",
    # Model
    "LlamaModel",
    "LlamaAttention",
    "LlamaDecoderLayer",
    "LlamaMLP",
    # MoE
    "MoETransformerLayer",
    "ExpertChoiceRouter",
    "ExpertFeedForward",
    # Positional Encoding
    "RotaryEmbedding",
    "YaRN",
    "QKNorm",
    "apply_rotary_pos_emb",
    # Normalization
    "RMSNorm",
    "DeepNorm",
    # Configs
    "BaseConfig",
    "Llama3Config",
    "Llama2Config",
    "DeepSeekV3Config",
    "MistralConfig",
    "GemmaConfig",
    "MixtralConfig",
    "llama3_8b_config",
    "llama3_70b_config",
    "llama2_7b_config",
    "mistral_7b_config",
    "mixtral_8x7b_config",
    "gemma_2b_config",
    "gemma_7b_config",
    # Model Loading
    "WeightMapper",
    "load_hf_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
    "convert_config_to_llama3",
]
