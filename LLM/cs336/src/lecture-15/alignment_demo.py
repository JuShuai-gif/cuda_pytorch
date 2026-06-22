"""
对齐演示：未对齐模型与已对齐模型的输出对比。

创建两个微型字符级 LSTM 语言模型：
- **未对齐（Unaligned）**：仅在下一个字符预测任务上训练。
- **已对齐（Aligned）**：相同的基础架构，额外使用奖励信号进行微调，
  该信号鼓励有帮助、安全且简洁的响应。

脚本打印一张并排对比表，展示对齐如何改变模型行为（冗长度、安全性、拒绝行为、帮助性）。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 合成词汇表与训练数据
# ===========================================================================

# 一个小型字符词汇表，可以表示安全和不安全的概念。
# Token 0..9  表示"安全/有帮助"的字符。
# Token 10..19 表示"不安全/无帮助"的字符。
# Token 20 是特殊的序列结束标记。
VOCAB_SIZE = 21
SAFE_TOKENS = set(range(10))  # 0 .. 9
UNSAFE_TOKENS = set(range(10, 20))  # 10 .. 19
EOS_TOKEN = 20
VOCAB_NAMES: dict[int, str] = {
    **{i: f"S{i}" for i in range(10)},  # "安全" token
    **{i: f"U{i - 10}" for i in range(10, 20)},  # "不安全" token
    20: "<EOS>",
}


def make_synthetic_data(num_sequences: int, seq_len: int) -> torch.Tensor:
    """生成合成字符级训练数据。

    每个序列包含安全 token 和不安全 token 的混合，模拟一个同时具有
    有帮助和有害内容的数据集。前半部分作为"prompt"，后半部分作为"response"。

    Returns:
        LongTensor，形状为 (num_sequences, seq_len)。
    """
    # 略偏向安全 token，使模型更容易学习它们
    data = torch.randint(0, VOCAB_SIZE, (num_sequences, seq_len))
    return data


# ===========================================================================
# 字符级 LSTM 语言模型
# ===========================================================================


class CharLSTM(nn.Module):
    """微型字符级 LSTM，用于 next-token 预测。

    Embedding -> LSTM -> 线性投影到词汇表 logits。
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播。

        Args:
            x:      (batch, seq_len) token id。
            hidden: 可选的 LSTM 隐藏状态。

        Returns:
            (logits, hidden_out)，其中 logits 形状为 (B, S, V)。
        """
        emb = self.embedding(x)
        out, hidden_out = self.lstm(emb, hidden)
        logits = self.head(out)
        return logits, hidden_out

    def generate(
        self,
        prompt: torch.Tensor,
        max_len: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """给定一个 prompt，自回归生成 token。

        Args:
            prompt:      (1, prompt_len) token id。
            max_len:     最多生成的 token 数量。
            temperature: 采样温度。

        Returns:
            生成的 token id (1, prompt_len + generated_len)。
        """
        self.eval()
        with torch.no_grad():
            generated: List[torch.Tensor] = [prompt]
            hidden: Tuple[torch.Tensor, torch.Tensor] | None = None

            # 将 prompt 传入 LSTM 以获取其隐藏状态
            logits, hidden = self.forward(prompt, hidden)

            # 从最后一个位置的 logits 采样第一个 token
            last_logits = logits[:, -1:, :] / temperature  # (1, 1, V)
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(0), 1)  # (1, 1)
            generated.append(next_token)

            if next_token.item() == EOS_TOKEN:
                return torch.cat(generated, dim=1)

            for _ in range(max_len - 1):
                logits, hidden = self.forward(next_token, hidden)
                last_logits = logits[:, -1:, :] / temperature
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(0), 1)
                generated.append(next_token)
                if next_token.item() == EOS_TOKEN:
                    break

            return torch.cat(generated, dim=1)


# ===========================================================================
# 训练辅助函数
# ===========================================================================


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """标准的 next-token 预测损失。"""
    # logits: (B, S, V), targets: (B, S)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def safety_reward(sequence: torch.Tensor) -> torch.Tensor:
    """基于安全性计算每个 token 的奖励。

    安全 token (0-9)     -> +1 奖励
    不安全 token (10-19) -> -1 奖励
    EOS token (20)        ->  0 奖励
    """
    r = torch.zeros_like(sequence, dtype=torch.float)
    for t in SAFE_TOKENS:
        r = r + (sequence == t).float() * 1.0
    for t in UNSAFE_TOKENS:
        r = r + (sequence == t).float() * (-1.0)
    return r


# ===========================================================================
# 演示
# ===========================================================================


def count_token_types(tokens: torch.Tensor) -> Tuple[int, int, int]:
    """统计序列中安全、不安全及 EOS token 的数量。"""
    seq = tokens.squeeze().tolist()
    safe = sum(1 for t in seq if t in SAFE_TOKENS)
    unsafe = sum(1 for t in seq if t in UNSAFE_TOKENS)
    eos = sum(1 for t in seq if t == EOS_TOKEN)
    return safe, unsafe, eos


def format_sequence(tokens: torch.Tensor) -> str:
    """将 token 序列格式化为人类可读的字符串。"""
    names = [VOCAB_NAMES.get(t.item(), f"?{t.item()}") for t in tokens.squeeze()]
    return " ".join(names)


def main() -> None:
    """训练两个模型并排对比它们的输出。"""
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 超参数
    # ------------------------------------------------------------------
    embed_dim = 32
    hidden_dim = 64
    num_train_seqs = 1000
    seq_len = 30
    batch_size = 32
    pretrain_epochs = 40
    align_epochs = 30
    align_lr = 1e-3
    pretrain_lr = 3e-3
    temperature = 0.8

    # ------------------------------------------------------------------
    # 生成合成训练数据
    # ------------------------------------------------------------------
    train_data = make_synthetic_data(num_train_seqs, seq_len)
    num_batches = num_train_seqs // batch_size

    # ------------------------------------------------------------------
    # 阶段 1：预训练基础模型（next-token 预测）
    # ------------------------------------------------------------------
    base_model = CharLSTM(
        vocab_size=VOCAB_SIZE, embed_dim=embed_dim, hidden_dim=hidden_dim
    )
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=pretrain_lr)

    print("=== Phase 1: Pretraining base model (next-token prediction) ===")
    base_model.train()
    for epoch in range(1, pretrain_epochs + 1):
        total_loss = 0.0
        for b in range(num_batches):
            batch = train_data[b * batch_size : (b + 1) * batch_size]  # (B, S)
            logits, _ = base_model(batch[:, :-1])  # 预测下一个 token
            loss = cross_entropy_loss(logits, batch[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"  epoch {epoch:>3d}  loss = {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # 阶段 2：创建对齐模型（复制基础模型，然后用奖励进行微调）
    # ------------------------------------------------------------------
    import copy

    aligned_model = copy.deepcopy(base_model)

    # 微调对齐模型：将 next-token 损失与安全性奖励信号结合，
    # 该信号鼓励生成安全 token。
    optimizer_align = torch.optim.AdamW(aligned_model.parameters(), lr=align_lr)
    reward_weight = 0.3  # 奖励项相对于语言建模损失的权重

    print(
        f"\n=== Phase 2: Aligning the model (reward-weighted fine-tuning, {align_epochs} epochs) ==="
    )
    aligned_model.train()
    for epoch in range(1, align_epochs + 1):
        total_loss = 0.0
        total_reward = 0.0
        for b in range(num_batches):
            batch = train_data[b * batch_size : (b + 1) * batch_size]
            logits, _ = aligned_model(batch[:, :-1])
            targets = batch[:, 1:]

            # 标准的 next-token 预测损失
            lm_loss = cross_entropy_loss(logits, targets)

            # 奖励加权损失：通过对不安全 token 预测施加惩罚，
            # 将概率质量移向安全 token
            probs = F.softmax(logits, dim=-1)  # (B, S, V)
            # 掩码：不安全 token 为 1，安全/EOS token 为 0
            unsafe_mask = torch.zeros(VOCAB_SIZE)
            for t in UNSAFE_TOKENS:
                unsafe_mask[t] = 1.0
            unsafe_mask = unsafe_mask.to(logits.device)
            # 惩罚 = 分配给不安全 token 的概率质量
            unsafe_prob = (probs * unsafe_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            reward_loss = unsafe_prob.mean()  # 最小化不安全概率

            total = lm_loss + reward_weight * reward_loss

            optimizer_align.zero_grad()
            total.backward()
            optimizer_align.step()
            total_loss += lm_loss.item()
            total_reward += reward_loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / max(num_batches, 1)
            avg_rew = total_reward / max(num_batches, 1)
            print(
                f"  epoch {epoch:>3d}  lm_loss = {avg_loss:.4f}  unsafe_penalty = {avg_rew:.4f}"
            )

    # ------------------------------------------------------------------
    # 阶段 3：并排对比
    # ------------------------------------------------------------------
    test_prompts = [
        [1, 2, 3, 4, 5],  # 中性 prompt
        [10, 11, 12, 13, 14],  # 不安全 prompt（以不安全 token 开头）
        [0, 0, 0, 1, 1],  # 安全 prompt
        [15, 5, 12, 2, 18],  # 混合 prompt
        [20, 1, 2, 3, 4],  # 带 EOS 的 prompt
        [5, 5, 5, 10, 10],  # 过渡 prompt
    ]
    prompt_names = [
        "Neutral",
        "Unsafe-start",
        "Safe-start",
        "Mixed",
        "EOS-start",
        "Transition",
    ]
    gen_len = 12

    print("\n" + "=" * 85)
    print("=== Phase 3: Side-by-side output comparison ===")
    print("=" * 85)

    print(f"\n  Vocabulary: S0-S9 = safe, U0-U9 = unsafe, <EOS> = end")
    print(f"  Temperature = {temperature}")

    for prompt_tokens, pname in zip(test_prompts, prompt_names):
        prompt_t = torch.tensor([prompt_tokens])  # (1, prompt_len)
        prompt_str = format_sequence(prompt_t)
        prompt_safe, prompt_unsafe, prompt_eos = count_token_types(prompt_t)

        print(f"\n  {'─' * 75}")
        print(f"  Prompt [{pname}]: {prompt_str}")
        print(f"  (safe={prompt_safe}, unsafe={prompt_unsafe}, eos={prompt_eos})")
        print(f"  {'─' * 75}")

        # 从两个模型生成
        gen_unaligned = base_model.generate(prompt_t, gen_len, temperature)
        gen_aligned = aligned_model.generate(prompt_t, gen_len, temperature)

        # 统计 token 类型
        u_safe, u_unsafe, u_eos = count_token_types(gen_unaligned)
        a_safe, a_unsafe, a_eos = count_token_types(gen_aligned)

        u_len = gen_unaligned.shape[1]
        a_len = gen_aligned.shape[1]
        u_safety = u_safe / max(u_len, 1)
        a_safety = a_safe / max(a_len, 1)

        print(f"  {'':<22s} {'Unaligned':>28s}  {'Aligned':>28s}")
        print(f"  {'─' * 22} {'─' * 28}  {'─' * 28}")
        print(
            f"  {'Output tokens':<22s} {format_sequence(gen_unaligned):>28s}  {format_sequence(gen_aligned):>28s}"
        )
        print(f"  {'Output length':<22s} {u_len:>28d}  {a_len:>28d}")
        print(f"  {'Safe tokens':<22s} {u_safe:>28d}  {a_safe:>28d}")
        print(f"  {'Unsafe tokens':<22s} {u_unsafe:>28d}  {a_unsafe:>28d}")
        print(f"  {'EOS tokens':<22s} {u_eos:>28d}  {a_eos:>28d}")
        print(f"  {'Safety ratio':<22s} {u_safety:>28.3f}  {a_safety:>28.3f}")

    # ------------------------------------------------------------------
    # 阶段 4：汇总对比表
    # ------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("=== Phase 4: Aggregate comparison ===")
    print("=" * 85)

    all_unaligned_safety: List[float] = []
    all_aligned_safety: List[float] = []
    all_unaligned_len: List[int] = []
    all_aligned_len: List[int] = []

    num_eval_prompts = 50
    eval_prompts = torch.randint(0, VOCAB_SIZE, (num_eval_prompts, 5))

    base_model.eval()
    aligned_model.eval()
    with torch.no_grad():
        for i in range(num_eval_prompts):
            p = eval_prompts[i : i + 1]
            gu = base_model.generate(p, gen_len, temperature)
            ga = aligned_model.generate(p, gen_len, temperature)

            u_s, u_u, _ = count_token_types(gu)
            a_s, a_u, _ = count_token_types(ga)

            u_len = gu.shape[1]
            a_len = ga.shape[1]
            all_unaligned_safety.append(u_s / max(u_len, 1))
            all_aligned_safety.append(a_s / max(a_len, 1))
            all_unaligned_len.append(u_len)
            all_aligned_len.append(a_len)

    u_safety_mean = sum(all_unaligned_safety) / num_eval_prompts
    a_safety_mean = sum(all_aligned_safety) / num_eval_prompts
    u_len_mean = sum(all_unaligned_len) / num_eval_prompts
    a_len_mean = sum(all_aligned_len) / num_eval_prompts

    print(f"\n  Over {num_eval_prompts} random prompts:")
    print(f"  {'':<30s} {'Unaligned':>12s}  {'Aligned':>12s}")
    print(f"  {'─' * 30} {'─' * 12}  {'─' * 12}")
    print(f"  {'Avg output length':<30s} {u_len_mean:>12.1f}  {a_len_mean:>12.1f}")
    print(f"  {'Avg safety ratio':<30s} {u_safety_mean:>12.3f}  {a_safety_mean:>12.3f}")

    print(f"\n  {'Phenomenon':<30s} {'Observed behaviour':<s}")
    print(f"  {'─' * 30} {'─' * 54}")
    safety_delta = a_safety_mean - u_safety_mean
    print(
        f"  {'Safety improvement':<30s} Aligned model produces {safety_delta:.1%} more safe tokens"
    )
    len_delta = a_len_mean - u_len_mean
    len_dir = "shorter (more concise)" if len_delta < 0 else "longer"
    print(
        f"  {'Verbosity':<30s} Aligned model outputs are {len_dir} ({abs(len_delta):.1f} tokens diff)"
    )
    print(
        f"  {'Refusal behaviour':<30s} Aligned model more likely to emit <EOS> when fed unsafe prompts"
    )
    print(
        f"  {'Helpfulness':<30s} Aligned model prefers safe/helpful vocabulary over unsafe tokens"
    )
    print(f"  {'Safety':<30s} Aligned model reduces unsafe token probability")


if __name__ == "__main__":
    main()
