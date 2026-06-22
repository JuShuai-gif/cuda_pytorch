"""
自回归文本生成策略。

实现的生成方法：
- generate_greedy: 始终选择概率最高的 token。
- generate_sampling: 从概率分布中采样。

所有生成方法均支持：
- temperature 缩放（控制随机性）
- top-k 过滤（限制在 k 个最可能的 token 中）
- top-p (nucleus) 过滤（限制在累积概率达到 p 的 token 中）
- KV cache 加速（预分配，避免动态重分配）
- 流式生成（逐个产出 token）
"""

from __future__ import annotations

import os
import sys
from typing import Generator, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def _apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """对 logits 应用 temperature、top-k 和 top-p 过滤。"""
    if temperature > 0:
        logits = logits / temperature
    else:
        return logits

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove: torch.Tensor = (
            logits < torch.topk(logits, top_k)[0][..., -1, None]
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs: torch.Tensor = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_mask: torch.Tensor = cumulative_probs > top_p
        sorted_mask[..., 0] = False
        indices_to_remove = sorted_mask.scatter(
            dim=-1, index=sorted_indices, src=sorted_mask
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


@torch.no_grad()
def generate_greedy(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    eos_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    greedy 生成：始终选择概率最高的 token。

    模型的 forward() 调用方式为 (input_ids, kv_caches, input_pos)。
    KV cache 直接以模型原生的 tuple 格式存储 ——
    无需冗余的 KVCacheManager 复制。
    """
    model.eval()
    generated: torch.Tensor = input_ids.clone()
    num_layers: int = len(model.layers)
    kv_caches: list | None = [None] * num_layers if use_cache else None

    # Prefill 阶段：处理完整 prompt，捕获 KV cache
    if use_cache:
        logits, kv_caches = model.forward(input_ids, kv_caches=None)
        cur_pos: int = input_ids.size(1)
    else:
        cur_pos = input_ids.size(1)

    for _ in range(max_new_tokens):
        if use_cache and kv_caches is not None:
            current_input = generated[:, -1:]
            input_pos = torch.tensor([cur_pos], device=generated.device)
            logits, kv_caches = model.forward(
                current_input, kv_caches=kv_caches, input_pos=input_pos
            )
            cur_pos += 1
        else:
            logits, _ = model.forward(generated, kv_caches=None)

        next_logits: torch.Tensor = logits[:, -1, :]
        next_token: torch.Tensor = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated


@torch.no_grad()
def generate_sampling(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """基于采样的生成，支持 temperature 和过滤参数。"""
    model.eval()
    generated: torch.Tensor = input_ids.clone()
    num_layers: int = len(model.layers)
    kv_caches: list | None = [None] * num_layers if use_cache else None

    if use_cache:
        logits, kv_caches = model.forward(input_ids, kv_caches=None)
        cur_pos: int = input_ids.size(1)
    else:
        cur_pos = input_ids.size(1)

    for _ in range(max_new_tokens):
        if use_cache and kv_caches is not None:
            current_input = generated[:, -1:]
            input_pos = torch.tensor([cur_pos], device=generated.device)
            logits, kv_caches = model.forward(
                current_input, kv_caches=kv_caches, input_pos=input_pos
            )
            cur_pos += 1
        else:
            logits, _ = model.forward(generated, kv_caches=None)

        next_logits = logits[:, -1, :]
        filtered = _apply_sampling_filters(next_logits, temperature, top_k, top_p)
        probs: torch.Tensor = F.softmax(filtered, dim=-1)
        next_token: torch.Tensor = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated


@torch.no_grad()
def generate_streaming(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> Generator[torch.Tensor, None, None]:
    """流式生成：逐个产出 token。"""
    model.eval()
    generated: torch.Tensor = input_ids.clone()
    num_layers: int = len(model.layers)
    kv_caches: list | None = [None] * num_layers if use_cache else None

    if use_cache:
        logits, kv_caches = model.forward(input_ids, kv_caches=None)
        cur_pos: int = input_ids.size(1)
    else:
        cur_pos = input_ids.size(1)

    for _ in range(max_new_tokens):
        if use_cache and kv_caches is not None:
            current_input = generated[:, -1:]
            input_pos = torch.tensor([cur_pos], device=generated.device)
            logits, kv_caches = model.forward(
                current_input, kv_caches=kv_caches, input_pos=input_pos
            )
            cur_pos += 1
        else:
            logits, _ = model.forward(generated, kv_caches=None)

        next_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            filtered = _apply_sampling_filters(next_logits, temperature, top_k, top_p)
            probs = F.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)
        yield next_token

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break


# 快速测试
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformer.config import MiniLLMConfig
    from transformer.layers import MiniLLM

    config = MiniLLMConfig(
        vocab_size=200,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        max_seq_len=128,
    )
    model = MiniLLM(config)

    prompt = torch.randint(0, config.vocab_size, (1, 4))

    # 测试 greedy 生成
    gen = generate_greedy(model, prompt, max_new_tokens=5, use_cache=False)
    assert gen.shape[1] == 9
    print(f"Greedy: {gen.shape}, tokens={gen[0, -5:].tolist()}")

    # 测试 sampling 生成
    gen = generate_sampling(
        model, prompt, max_new_tokens=5, temperature=0.8, use_cache=False
    )
    assert gen.shape[1] == 9
    print(f"Sampling: {gen.shape}")

    # 测试 streaming 生成
    print("Streaming:", end=" ")
    tokens = []
    for token in generate_streaming(model, prompt, max_new_tokens=5, use_cache=False):
        tokens.append(token.item())
        print(token.item(), end=" ")
    print()
    assert len(tokens) == 5
    print("All generation tests passed!")
