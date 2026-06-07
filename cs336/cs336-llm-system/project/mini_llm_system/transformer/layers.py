"""
Transformer 层以及完整的 MiniLLM 模型。

实现内容包括：
- SwiGLU_FFN：使用 SwiGLU 激活的前馈网络（LLaMA 风格）。
- TransformerBlock：RMSNorm + GQA 注意力 + RMSNorm + SwiGLU_FFN，带残差连接。
- MiniLLM：完整模型，包含 embedding、transformer 块、最终归一化层和 LM head。

架构遵循 Pre-norm（LLaMA 风格）模式：在每个子层之前进行归一化，
并使用残差连接包裹。
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

# 允许直接运行此文件或作为包的一部分运行
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.attention import GroupedQueryAttention
from transformer.config import MiniLLMConfig
from transformer.normalization import RMSNorm
from transformer.rotary_embedding import RotaryEmbedding


class SwiGLU_FFN(nn.Module):
    """
    使用 SwiGLU 激活的前馈网络。

    与传统 FFN（ReLU(gate(x)) * up(x)）不同，SwiGLU 使用：
        output = SiLU(gate_proj(x)) * up_proj(x)
    然后再进行下投影。

    Args:
        hidden_size: 输入/输出维度。
        intermediate_size: 内部维度（通常约为 hidden_size 的 4 倍）。
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
        """
        Args:
            x: 形状为 [batch, seq_len, hidden_size] 的张量。

        Returns:
            形状为 [batch, seq_len, hidden_size] 的输出张量。
        """
        # SwiGLU：SiLU(gate(x)) * up(x)，然后进行下投影
        gate: torch.Tensor = F.silu(self.gate_proj(x))
        up: torch.Tensor = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """
    遵循 LLaMA pre-norm 架构的单个 transformer 块。

    架构：
        x -> RMSNorm -> GQA 注意力（+ 残差）
          -> RMSNorm -> SwiGLU FFN（+ 残差）

    注意力和 FFN 均使用 pre-normalization 并带残差连接。
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.num_kv_heads: int = config.num_kv_heads
        self.head_dim: int = config.head_dim

        # Pre-norm 层
        self.input_norm: RMSNorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attn_norm: RMSNorm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # 注意力
        self.attention: GroupedQueryAttention = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dropout=0.0,
            use_rope=config.use_rope,
        )

        # 前馈网络
        self.ffn: SwiGLU_FFN = SwiGLU_FFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        单个 transformer 块的前向传播。

        Args:
            hidden_states: 形状为 [batch, seq_len, hidden_size] 的张量。
            cos: RoPE 余弦表。
            sin: RoPE 正弦表。
            attention_mask: 可选的注意力掩码。
            kv_cache: 可选的 KV cache，用于增量解码。

        Returns:
            (output, updated_kv_cache) 元组。
        """
        # Pre-norm 注意力 + 残差
        residual: torch.Tensor = hidden_states
        normed: torch.Tensor = self.input_norm(hidden_states)
        attn_out, new_kv_cache = self.attention(
            normed, cos=cos, sin=sin, attention_mask=attention_mask, kv_cache=kv_cache
        )
        hidden_states = residual + attn_out

        # Pre-norm FFN + 残差
        residual = hidden_states
        normed = self.post_attn_norm(hidden_states)
        ffn_out: torch.Tensor = self.ffn(normed)
        hidden_states = residual + ffn_out

        return hidden_states, new_kv_cache


class MiniLLM(nn.Module):
    """
    遵循 LLaMA 架构的小型现代 LLM。

    架构：
        Token Embedding
            -> TransformerBlock x num_layers
            -> RMSNorm（最终层）
            -> LM Head（线性投影到 vocab_size）

    LM head 与 token embedding 共享权重（weight tying）。

    Args:
        config: 指定模型超参数的 MiniLLMConfig。
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config: MiniLLMConfig = config
        self.vocab_size: int = config.vocab_size
        self.hidden_size: int = config.hidden_size
        self.num_layers: int = config.num_layers
        self.max_seq_len: int = config.max_seq_len

        # Token embedding 层
        self.embed_tokens: nn.Embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        # 旋转位置编码
        self.rotary_emb: RotaryEmbedding = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # Transformer 块
        self.layers: nn.ModuleList = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # 最终归一化层
        self.norm: RMSNorm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # LM head（输出投影到词汇表）
        self.lm_head: nn.Linear = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Weight tying：共享 embedding 和 LM head 的权重
        self.lm_head.weight = self.embed_tokens.weight  # type: ignore[assignment]

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """使用较小的随机值初始化权重。"""
        std: float = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        整个模型的前向传播。

        Args:
            input_ids: Token ID，形状为 [batch, seq_len]。
            kv_caches: 可选的 KV cache 列表，每层一个。
            input_pos: 可选的位置张量 [seq_len]，用于增量解码时的 RoPE 偏移量。

        Returns:
            (logits [batch, seq_len, vocab_size], 更新后的 KV caches) 元组。
        """
        batch_size: int = input_ids.size(0)
        seq_len: int = input_ids.size(1)

        # Token embedding
        hidden_states: torch.Tensor = self.embed_tokens(input_ids)

        # 获取带有正确位置偏移量的 RoPE cos/sin
        if input_pos is not None:
            # 在增量解码期间，输入是位于 input_pos 位置的 1 个 token
            start_pos: int = input_pos[0].item() if input_pos.numel() > 0 else 0
        else:
            start_pos = 0
        cos, sin = self.rotary_emb.forward(
            seq_len, device=hidden_states.device, start_pos=start_pos
        )

        # 如果未提供 KV cache，则初始化为空列表
        if kv_caches is None:
            kv_caches = [None] * self.num_layers

        new_kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []

        # 依次通过所有 transformer 块
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if i < len(kv_caches) else None
            hidden_states, new_cache = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                kv_cache=layer_cache,
            )
            new_kv_caches.append(new_cache)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # LM head：投影到词汇表
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
        """
        自回归文本生成。

        Args:
            input_ids: 提示词 token ID，形状为 [batch, seq_len]。
            max_new_tokens: 最大生成 token 数。
            temperature: 采样温度（1.0 = 不变，<1.0 = 更尖锐的分布）。
            top_k: 如果 > 0，则仅从 top-k 个 token 中采样。
            top_p: 如果 < 1.0，nucleus 采样阈值。
            use_cache: 是否使用 KV cache 以加快生成速度。

        Returns:
            生成的 token ID，形状为 [batch, prompt_len + new_tokens]。
        """
        self.eval()
        device: torch.device = input_ids.device
        generated: torch.Tensor = input_ids.clone()

        kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] | None = (
            [None] * self.num_layers if use_cache else None
        )
        cur_pos: int = input_ids.size(1)

        for _ in range(max_new_tokens):
            # 确定要处理序列的哪一部分
            if use_cache and kv_caches is not None and kv_caches[0] is not None:
                # 解码步骤：仅处理正确位置上的最后一个 token
                current_input = generated[:, -1:]
                input_pos = torch.tensor([cur_pos], device=device)
            else:
                # Prefill：处理到目前为止的所有 token
                current_input = generated
                input_pos = None

            with torch.no_grad():
                logits, kv_caches = self.forward(
                    current_input, kv_caches=kv_caches, input_pos=input_pos
                )

            cur_pos += current_input.size(1)

            # 获取最后一个位置的 logits
            next_logits: torch.Tensor = logits[:, -1, :]

            # 应用温度
            if temperature > 0:
                next_logits = next_logits / max(temperature, 1e-7)
            else:
                # temperature 为 0 时使用贪心解码
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                continue

            # 应用 top-k 过滤
            if top_k > 0:
                top_k_values: torch.Tensor
                top_k_indices: torch.Tensor
                top_k_values, top_k_indices = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1))
                )
                mask: torch.Tensor = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(-1, top_k_indices, top_k_values)
                next_logits = mask

            # 应用 top-p（nucleus）采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs: torch.Tensor = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # 移除累积概率超过阈值的 token
                sorted_indices_to_remove: torch.Tensor = cumulative_probs > top_p
                # 偏移以始终保留至少第一个 token
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove: torch.Tensor = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            # 从过滤后的分布中采样
            probs: torch.Tensor = F.softmax(next_logits, dim=-1)
            next_token: torch.Tensor = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated

    def get_num_params(self) -> int:
        """返回可训练参数的总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 快速测试
if __name__ == "__main__":
    config = MiniLLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        intermediate_size=1024,
        max_seq_len=512,
    )

    model = MiniLLM(config)
    num_params: int = model.get_num_params()

    batch, seq = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    # 前向传播
    logits, kv_caches = model(input_ids)
    assert logits.shape == (batch, seq, config.vocab_size), (
        f"Logits shape: {logits.shape}"
    )
    print(f"Forward pass: OK, shape={logits.shape}, params={num_params:,}")

    # 测试文本生成
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    generated = model.generate(
        prompt, max_new_tokens=5, temperature=0.8, use_cache=True
    )
    assert generated.shape[1] == 4 + 5, f"Generated shape: {generated.shape}"
    print(f"Generation: OK, output length={generated.shape[1]}")

    # 测试 SwiGLU FFN
    ffn = SwiGLU_FFN(hidden_size=256, intermediate_size=1024)
    x = torch.randn(2, 16, 256)
    out = ffn(x)
    assert out.shape == x.shape, f"FFN shape: {out.shape}"
    print(f"SwiGLU_FFN: OK, shape={out.shape}")

    # 测试 TransformerBlock
    block = TransformerBlock(config)
    x = torch.randn(2, 16, 256)
    out, _ = block(x)
    assert out.shape == x.shape, f"Block shape: {out.shape}"
    print(f"TransformerBlock: OK, shape={out.shape}")

    print(f"\nAll model tests passed! Parameters: {num_params:,}")
