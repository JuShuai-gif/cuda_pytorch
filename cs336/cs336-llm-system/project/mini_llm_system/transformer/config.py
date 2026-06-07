"""
MiniLLM 模型配置。

使用 dataclass 定义小型现代 LLM 系统的所有超参数。
"""

from dataclasses import dataclass, field


@dataclass
class MiniLLMConfig:
    """MiniLLM Transformer 模型的配置类。"""

    # 词汇表
    vocab_size: int = 32000
    pad_token_id: int = 0

    # 模型维度
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4  # 分组查询注意力 (GQA) - 小于 num_heads
    intermediate_size: int = 3072

    # 序列长度
    max_seq_len: int = 2048

    # RoPE（旋转位置编码）
    use_rope: bool = True
    rope_theta: float = 10000.0

    # 归一化
    norm_eps: float = 1e-5

    # 注意力
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.head_dim = self.hidden_size // self.num_heads
        # 验证 GQA 兼容性
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
