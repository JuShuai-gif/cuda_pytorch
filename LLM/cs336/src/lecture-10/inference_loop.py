"""
自回归生成循环。

实现了以下采样策略：
  - 贪心解码（Greedy）：始终选取概率最高的下一个 token
  - 温度采样（Temperature）：按温度系数缩放后采样
  - Top-K 采样：从概率最高的 k 个 token 中采样
  - Top-P（Nucleus）采样：从累积概率 ≥ p 的最小集合中采样
  - 生成过程中的 KV cache 加速
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


# =========================================================================
# 生成配置
# =========================================================================


@dataclass
class GenerationConfig:
    """文本生成的配置参数。"""

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 0  # 0 表示禁用
    top_p: float = 1.0  # 1.0 表示禁用
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] | None = None
    do_sample: bool = True
    pad_token_id: int = 0
    eos_token_id: int | None = None

    def validate(self) -> None:
        """验证配置参数的有效性。"""
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


# =========================================================================
# 采样策略
# =========================================================================


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    贪心解码：选取概率最高的 token。

    参数：
        logits: (batch, vocab_size) 原始 logits

    返回：
        (batch,) 形状的 token id 张量
    """
    return logits.argmax(dim=-1)


def temperature_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    带温度缩放的随机采样。

    温度越高 → 越随机；温度越低 → 越确定。
    temperature=0 等价于贪心解码。

    参数：
        logits: (batch, vocab_size) 原始 logits
        temperature: 缩放因子（越高越随机）

    返回：
        (batch,) 形状的采样 token id 张量
    """
    if temperature < 1e-6:
        return greedy_sample(logits)

    # 用 temperature 缩放 logits
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """
    Top-K 采样：从概率最高的 k 个 token 中采样。

    参数：
        logits: (batch, vocab_size) 原始 logits
        temperature: 温度缩放系数
        top_k: 保留的 top token 数量

    返回：
        (batch,) 形状的采样 token id 张量
    """
    if top_k <= 0:
        return temperature_sample(logits, temperature)

    # 获取 top-k 的 logits，其余设为 -inf
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    min_top_k = top_k_values[:, -1].unsqueeze(-1)
    logits = torch.where(
        logits < min_top_k, torch.full_like(logits, float("-inf")), logits
    )

    return temperature_sample(logits, temperature)


def top_p_sample(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """
    Nucleus（Top-P）采样：从累积概率 ≥ top_p 的最小 token 集合中采样。

    参数：
        logits: (batch, vocab_size) 原始 logits
        temperature: 温度缩放系数
        top_p: 累积概率阈值

    返回：
        (batch,) 形状的采样 token id 张量
    """
    if top_p >= 1.0:
        return temperature_sample(logits, temperature)

    logits = logits / (temperature + 1e-9)
    probs = F.softmax(logits, dim=-1)

    # 按概率降序排列
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 移除累积概率超过 top_p 的 token
    sorted_indices_to_remove = cumulative_probs > top_p
    # 平移一位，确保至少保留一个 token
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # 将掩码散列回原始索引
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return temperature_sample(logits, temperature)


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generated_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    使用配置的策略采样下一个 token。

    参数：
        logits: (batch, vocab_size) 原始 logits
        config: 生成配置
        generated_ids: 之前已生成的 token，用于重复惩罚

    返回：
        (batch,) 形状的下一个 token id 张量
    """
    # 应用重复惩罚
    if config.repetition_penalty != 1.0 and generated_ids is not None:
        for i in range(logits.size(0)):
            for token_id in generated_ids[i].unique():
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= config.repetition_penalty
                else:
                    logits[i, token_id] *= config.repetition_penalty

    if not config.do_sample or config.temperature < 1e-6:
        return greedy_sample(logits)

    if config.top_k > 0:
        return top_k_sample(logits, config.temperature, config.top_k)
    elif config.top_p < 1.0:
        return top_p_sample(logits, config.temperature, config.top_p)
    else:
        return temperature_sample(logits, config.temperature)


# =========================================================================
# 生成循环
# =========================================================================


def generate(
    model: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
    kv_cache: object | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    自回归生成循环。

    参数：
        model: 一个接受 (input_ids) 并返回 (logits) 的函数。
               期望输出形状：(batch, seq_len, vocab_size)
        input_ids: 初始 token id，形状 (batch, prompt_len)
        config: 生成配置
        kv_cache: 可选的 KV cache 对象
        verbose: 是否打印进度

    返回：
        包含 prompt 的完整序列，形状 (batch, prompt_len + new_tokens)
    """
    if config is None:
        config = GenerationConfig()
    config.validate()

    batch_size, prompt_len = input_ids.shape
    generated = list(input_ids.unbind(dim=1))  # 一系列 (batch,) 形状的张量
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    for step in range(config.max_new_tokens):
        # 在 decode 阶段（使用 KV cache）只输入最后一个 token
        if kv_cache is not None and step > 0:
            current_input = generated[-1].unsqueeze(-1)  # (batch, 1)
        else:
            # Prefill 或无缓存：输入完整序列
            current_input = torch.stack(generated, dim=1)

        # 模型前向传播
        logits = model(current_input)  # (batch, seq_len, vocab_size)

        # 获取最后一个位置的 logits
        next_logits = logits[:, -1, :]  # (batch, vocab_size)

        # 采样下一个 token
        next_token = sample_next_token(
            next_logits, config, torch.stack(generated, dim=1)
        )

        # 检查是否遇到 EOS（结束符）
        if config.eos_token_id is not None:
            is_eos = next_token == config.eos_token_id
            finished = finished | is_eos
            # 对已完成序列，将 EOS 替换为 pad
            next_token = torch.where(
                finished, torch.full_like(next_token, config.pad_token_id), next_token
            )

        generated.append(next_token)

        if verbose:
            print(
                f"  Step {step}: token={next_token.item()}, finished={finished.item()}"
            )

        if finished.all():
            break

    return torch.stack(generated, dim=1)


# =========================================================================
# 演示用的 Dummy 模型
# =========================================================================


class DummyLLM(torch.nn.Module):
    """用于演示的简化 transformer 风格模型。"""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            batch_first=True,
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        # 创建因果掩码（causal mask）
        seq_len = input_ids.size(1)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)
        x = self.transformer(x, mask)
        return self.lm_head(x)


# =========================================================================
# 演示
# =========================================================================


def demo_generation_strategies() -> None:
    """演示不同的生成策略。"""
    print("=" * 70)
    print("Generation Strategies Demo")
    print("=" * 70)

    torch.manual_seed(42)

    # 创建一个小的 dummy 模型
    model = DummyLLM(vocab_size=1000, hidden_size=128)
    model.eval()

    # 创建一个简单的 prompt
    prompt = torch.randint(0, 1000, (1, 5))
    print(f"\nPrompt shape: {prompt.shape}")

    strategies = [
        ("Greedy", GenerationConfig(max_new_tokens=10, do_sample=False)),
        ("Temperature=0.8", GenerationConfig(max_new_tokens=10, temperature=0.8)),
        ("Temperature=2.0 (hot)", GenerationConfig(max_new_tokens=10, temperature=2.0)),
        ("Top-K=5", GenerationConfig(max_new_tokens=10, temperature=1.0, top_k=5)),
        ("Top-P=0.9", GenerationConfig(max_new_tokens=10, temperature=1.0, top_p=0.9)),
    ]

    for name, config in strategies:
        with torch.no_grad():
            output = generate(model, prompt, config, verbose=False)
        print(f"\n  {name}:")
        print(f"    Input:  {prompt[0].tolist()}")
        print(f"    Output: {output[0].tolist()} (len={output.size(1)})")


def demo_generation_with_kv_cache() -> None:
    """对比有/无 KV cache 的生成速度。"""
    print("\n" + "=" * 70)
    print("Generation Speed: With vs Without KV Cache")
    print("=" * 70)

    import time

    torch.manual_seed(42)

    # 使用更大的模型来展示差异
    model = DummyLLM(vocab_size=1000, hidden_size=256)
    model.eval()

    prompt = torch.randint(0, 1000, (1, 32))
    config = GenerationConfig(max_new_tokens=20, do_sample=True, temperature=0.7)

    # 不使用 KV cache（朴素方式）
    def naive_model_fn(input_ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids)

    start = time.perf_counter()
    with torch.no_grad():
        output_naive = generate(
            lambda ids: naive_model_fn(ids),
            prompt,
            config,
            kv_cache=None,
            verbose=False,
        )
    naive_time = time.perf_counter() - start

    # 使用模拟 KV cache（仅测量循环开销）
    # 在实际场景中，KV cache 会大幅减少计算量
    print(f"\n  Naive generation (no KV cache): {naive_time:.4f}s")
    print(f"  Output length: {output_naive.size(1)}")

    print(f"\n  注意：在实际模型中，KV cache 避免了在每个 decode 步骤")
    print(f"  重新计算所有历史 token 的 K 和 V。这会将注意力计算")
    print(f"  从每步 O(n²) 降低到每步 O(n)，其中 n 是当前序列长度。")


def demo_temperature_effect() -> None:
    """展示 temperature 对 token 概率的影响。"""
    print("\n" + "=" * 70)
    print("Temperature Effect on Token Probabilities")
    print("=" * 70)

    torch.manual_seed(0)
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.2, 0.1, -1.0, -2.0, -5.0]])

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"\n  Logits: {logits[0].tolist()}")
    print(
        f"\n  {'Temp':<8} {'Token 0':<12} {'Token 1':<12} {'Token 2':<12} {'Token 3':<12} {'Entropy':<10}"
    )
    print("  " + "-" * 65)

    for t in temperatures:
        probs = F.softmax(logits / t, dim=-1)[0]
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        print(
            f"  {t:<8} {probs[0].item():<12.4f} {probs[1].item():<12.4f} {probs[2].item():<12.4f} {probs[3].item():<12.4f} {entropy:<10.4f}"
        )

    print(f"\n  temperature → 0：分布趋向 one-hot（贪心）")
    print(f"  temperature → ∞：分布趋向均匀")


def main() -> None:
    demo_generation_strategies()
    demo_generation_with_kv_cache()
    demo_temperature_effect()


if __name__ == "__main__":
    main()
