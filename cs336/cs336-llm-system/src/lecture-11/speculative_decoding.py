"""
简化的 speculative decoding 实现。

Speculative decoding 使用一个小型、快速的 "draft" 模型来建议多个
token，然后一个大型的 "target" 模型并行验证它们。这可以
显著加速生成，而不牺牲质量。

核心概念：
  - Draft model（草稿模型）：小型、快速，自回归生成 k 个候选 token
  - Target model（目标模型）：大型、精确，在一次前向传播中验证 k 个 token
  - Acceptance（接受）：如果 draft 与 target 匹配，token 被接受；被拒绝的 token
    从 target 的分布中重新采样
  - 保证：输出分布与仅使用 target 模型完全相同

这是一个*简化的*概念实现，在小规模上模拟 draft 和
target 模型来说明算法。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# =========================================================================
# 简化的 draft 和 target 模型
# =========================================================================


class DraftModel:
    """
    一个微型 "draft" 模型，产生快速但不太精确的预测。

    在实践中，这可能是一个比 target（例如 7B+ 参数）小得多的模型
    （例如 150M 参数）。
    """

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 固定的随机投影，模拟一个训练好的模型
        self._W = torch.randn(vocab_size, hidden_size)
        self._b = torch.randn(vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播：input → logits。

        Args:
            input_ids: (batch, seq_len) token id

        Returns:
            形状为 (batch, seq_len, vocab_size) 的 logits
        """
        batch_size, seq_len = input_ids.shape
        # 简单的投影，模拟模型输出
        x = self._W[input_ids].sum(dim=1, keepdim=True)  # (batch, 1, hidden)
        x = x.expand(-1, seq_len, -1)  # (batch, seq_len, hidden)
        logits = x @ self._W.T + self._b  # (batch, seq_len, vocab_size)
        return logits

    def generate_one(self, input_ids: torch.Tensor, temperature: float = 1.0) -> int:
        """贪婪地生成单个 token。

        Args:
            input_ids: (1, seq_len) 输入上下文
            temperature: 采样温度

        Returns:
            整数 token id
        """
        logits = self.forward(input_ids)  # (1, seq_len, vocab)
        next_logits = logits[0, -1, :] / temperature
        return next_logits.argmax(dim=-1).item()


class TargetModel:
    """
    一个更大的 "target" 模型，提供高质量但较慢的预测。

    使用更多参数和一个略微不同的随机投影来模拟。
    """

    def __init__(self, vocab_size: int = 100, hidden_size: int = 128):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 更多参数模拟一个更大的模型
        self._W1 = torch.randn(vocab_size, hidden_size)
        self._W2 = torch.randn(hidden_size, hidden_size)
        self._W3 = torch.randn(hidden_size, vocab_size)
        self._b = torch.randn(vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播，比 draft 模型有更多计算。

        Args:
            input_ids: (batch, seq_len) token id

        Returns:
            形状为 (batch, seq_len, vocab_size) 的 logits
        """
        batch_size, seq_len = input_ids.shape
        # 更复杂的投影，模拟一个更大的模型
        x = self._W1[input_ids]  # (batch, seq_len, hidden)
        x = torch.relu(x @ self._W2)  # (batch, seq_len, hidden)
        logits = x @ self._W3 + self._b  # (batch, seq_len, vocab_size)
        return logits

    def get_logits_at(
        self, input_ids: torch.Tensor, positions: list[int]
    ) -> list[torch.Tensor]:
        """获取特定位置的 logits。

        Args:
            input_ids: (batch, seq_len) 完整序列
            positions: 要从中提取 logits 的位置索引列表

        Returns:
            (batch, vocab_size) logits tensor 的列表
        """
        logits = self.forward(input_ids)
        return [logits[:, pos, :] for pos in positions]


# =========================================================================
# Speculative Decoding 引擎
# =========================================================================


@dataclass
class SpecDecodeResult:
    """一个 speculative decoding 步骤的结果。

    Attributes:
        accepted_tokens: 从 draft 接受的 token（可能为空）
        new_tokens: 此步骤中添加的 token 总数
        draft_len: draft 建议的 token 数量（gamma）
        accepted_len: 被接受的 token 数量
    """

    accepted_tokens: list[int]
    new_tokens: int = 0
    draft_len: int = 0
    accepted_len: int = 0


class SpeculativeDecoder:
    """
    Speculative decoding 引擎。

    每一步：
      1. Draft 模型自回归生成 γ 个 token（快速）
      2. Target 模型并行验证所有 γ 个 token（一次前向传播）
      3. 接受匹配的 token；拒绝并从 target 分布重新采样

    输出分布在数学上可证明与仅运行 target 模型相同，
    但平均每个 token 的延迟大大降低。
    """

    def __init__(
        self,
        draft_model: DraftModel,
        target_model: TargetModel,
        gamma: int = 4,  # 要推测的 token 数量
        temperature: float = 1.0,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma
        self.temperature = temperature

        # 统计信息
        self.total_drafted: int = 0
        self.total_accepted: int = 0

    def step(self, current_ids: torch.Tensor) -> tuple[torch.Tensor, SpecDecodeResult]:
        """执行一个 speculative decoding 步骤。

        Args:
            current_ids: (1, seq_len) 当前 token id

        Returns:
            (new_ids, result): 更新后的序列和步骤结果
        """
        seq_len = current_ids.size(1)

        # ---- 阶段 1：Draft 模型生成 γ 个候选 ----
        draft_tokens: list[int] = []
        draft_input = current_ids.clone()
        for _ in range(self.gamma):
            token = self.draft_model.generate_one(draft_input, self.temperature)
            draft_tokens.append(token)
            draft_input = torch.cat(
                [draft_input, torch.tensor([[token]], dtype=torch.long)], dim=1
            )

        # ---- 阶段 2：Target 模型并行验证 ----
        # 将 draft token 追加后构建完整序列
        verify_input = torch.cat(
            [
                current_ids,
                torch.tensor([draft_tokens], dtype=torch.long),
            ],
            dim=1,
        )  # (1, seq_len + gamma)

        # 获取位置 seq_len-1 到 seq_len+gamma-1 的 target logits
        verify_positions = list(range(seq_len - 1, seq_len + self.gamma - 1))
        target_logits_list = self.target_model.get_logits_at(
            verify_input, verify_positions
        )

        # ---- 阶段 3：接受/拒绝 draft token ----
        accepted_tokens: list[int] = []
        new_token: int = -1  # 如果所有 draft 都被拒绝，从 target 采样的 token

        for i in range(self.gamma):
            # 位置 seq_len + i 的 draft 预测
            draft_token = draft_tokens[i]
            # 同一位置的 target 分布
            target_logits = target_logits_list[i + 1]  # 偏移 1

            # 从 target 采样以进行比较
            target_probs = F.softmax(target_logits / self.temperature, dim=-1)
            target_sample = torch.multinomial(target_probs, num_samples=1).item()

            if target_sample == draft_token:
                # 接受！
                accepted_tokens.append(draft_token)
            else:
                # 拒绝：使用 target 的采样并停止
                accepted_tokens.append(target_sample)
                break

        # ---- 阶段 4：构建结果 ----
        if not accepted_tokens:
            # 全部拒绝：从 target 的当前位置采样一个 token
            first_target_logits = target_logits_list[0]
            first_probs = F.softmax(first_target_logits / self.temperature, dim=-1)
            new_token = torch.multinomial(first_probs, num_samples=1).item()
            accepted_tokens = [new_token]
            new_tokens_count = 1
            accepted_count = 0
        elif len(accepted_tokens) == self.gamma and len(accepted_tokens) > 0:
            # 所有 draft token 都被接受；从 target 采样一个 bonus token
            last_logits = target_logits_list[-1]
            last_probs = F.softmax(last_logits / self.temperature, dim=-1)
            bonus_token = torch.multinomial(last_probs, num_samples=1).item()
            accepted_tokens.append(bonus_token)
            new_tokens_count = len(accepted_tokens)
            accepted_count = self.gamma
        else:
            new_tokens_count = len(accepted_tokens)
            accepted_count = len(accepted_tokens) - 1  # 最后一个是 target 采样

        # 构建新序列
        new_tokens_tensor = torch.tensor([accepted_tokens], dtype=torch.long)
        output_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)

        # 更新统计信息
        self.total_drafted += self.gamma
        self.total_accepted += accepted_count

        result = SpecDecodeResult(
            accepted_tokens=accepted_tokens,
            new_tokens=new_tokens_count,
            draft_len=self.gamma,
            accepted_len=accepted_count,
        )

        return output_ids, result

    def acceptance_rate(self) -> float:
        """返回 draft token 的总体接受率。"""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    def generate(
        self, prompt_ids: list[int], max_new_tokens: int = 30
    ) -> tuple[list[int], list[SpecDecodeResult]]:
        """使用 speculative decoding 生成 token。

        Args:
            prompt_ids: 初始 prompt token id
            max_new_tokens: 最大生成 token 数

        Returns:
            (all_ids, step_results): 生成的序列和每个步骤的结果
        """
        current = torch.tensor([prompt_ids], dtype=torch.long)
        results: list[SpecDecodeResult] = []

        while current.size(1) < len(prompt_ids) + max_new_tokens:
            current, step_result = self.step(current)
            results.append(step_result)
            if step_result.new_tokens == 0:
                break

        return current[0].tolist(), results


# =========================================================================
# 演示
# =========================================================================


def demo_speculative_decoding() -> None:
    """逐步演示 speculative decoding。"""
    print("=" * 70)
    print("Speculative Decoding Demo")
    print("=" * 70)

    torch.manual_seed(42)

    vocab_size = 50
    draft = DraftModel(vocab_size=vocab_size, hidden_size=16)
    target = TargetModel(vocab_size=vocab_size, hidden_size=64)

    decoder = SpeculativeDecoder(
        draft_model=draft,
        target_model=target,
        gamma=4,
        temperature=1.0,
    )

    prompt = [1, 5, 10, 3, 7]
    print(f"\nPrompt: {prompt}")
    print(f"Gamma (speculation length): {decoder.gamma}")

    # 单步演示
    current = torch.tensor([prompt], dtype=torch.long)
    print("\n--- Single Step ---")
    new_ids, result = decoder.step(current)

    print(f"  Drafted {result.draft_len} tokens, accepted {result.accepted_len}")
    print(f"  New tokens added: {result.new_tokens}")
    print(f"  Accepted tokens: {result.accepted_tokens}")
    print(f"  New sequence: {new_ids[0].tolist()}")

    # 多步生成
    print("\n--- Multi-step Generation (max 20 new tokens) ---")
    generated, all_results = decoder.generate(prompt_ids=prompt, max_new_tokens=20)

    print(f"\n  Final sequence ({len(generated)} tokens):")
    print(f"  Prompt:     {prompt}")
    print(f"  Generated:  {generated[len(prompt) :]}")
    print(f"  Full:       {generated}")

    # 统计信息
    print(f"\n--- Statistics ---")
    print(f"  Total speculation steps: {len(all_results)}")
    print(f"  Total tokens drafted:    {decoder.total_drafted}")
    print(f"  Total tokens accepted:   {decoder.total_accepted}")
    print(f"  Overall acceptance rate: {decoder.acceptance_rate():.2%}")

    # 每步详情
    print(f"\n  Per-step breakdown:")
    for i, r in enumerate(all_results):
        status = (
            "ALL accepted + bonus"
            if r.accepted_len == r.draft_len
            else f"partial ({r.accepted_len}/{r.draft_len})"
        )
        print(f"    Step {i}: {status}, new_tokens={r.new_tokens}")

    # 计算有效加速比
    naive_steps = len(generated) - len(prompt)  # 仅 target 的步骤数
    spec_steps = len(all_results)
    if spec_steps > 0:
        speedup = naive_steps / spec_steps
        print(f"\n  Effective speedup: {speedup:.2f}x")
        print(f"    (Naive: {naive_steps} target forward passes)")
        print(f"    (Speculative: {spec_steps} target forward passes)")


def demo_acceptance_visualization() -> None:
    """可视化接受/拒绝在 token 级别是如何工作的。"""
    print("\n" + "=" * 70)
    print("Acceptance / Rejection Visualization")
    print("=" * 70)

    torch.manual_seed(7)

    vocab_size = 30
    draft = DraftModel(vocab_size=vocab_size, hidden_size=16)
    target = TargetModel(vocab_size=vocab_size, hidden_size=64)

    decoder = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=3)

    prompt = [1, 2, 3]
    current = torch.tensor([prompt], dtype=torch.long)

    print(f"\nPrompt: {prompt}")
    print(f"Gamma: {decoder.gamma}")
    print(f"\nRunning 5 speculative steps with detailed logging...\n")

    for i in range(5):
        new_ids, result = decoder.step(current)

        # 展示 draft token（我们重新推导它们用于显示）
        draft_input = current.clone()
        draft_tokens: list[int] = []
        for _ in range(decoder.gamma):
            token = decoder.draft_model.generate_one(draft_input)
            draft_tokens.append(token)
            draft_input = torch.cat(
                [draft_input, torch.tensor([[token]], dtype=torch.long)], dim=1
            )

        print(f"  Step {i}:")
        print(f"    Draft proposed:  {draft_tokens}")
        print(f"    Actually added:  {result.accepted_tokens}")
        print(f"    Accepted {result.accepted_len}/{result.draft_len} draft tokens")
        print(f"    New total tokens: {new_ids.size(1)}")

        current = new_ids

    print(f"\n  Acceptance rate: {decoder.acceptance_rate():.2%}")
    print("\n  Key insight: When draft predictions align with target, we get")
    print("  multiple tokens per target forward pass. When they diverge, we")
    print("  still get at least 1 correct token from the target. This ensures")
    print("  correctness while accelerating the common case.")


def main() -> None:
    demo_speculative_decoding()
    demo_acceptance_visualization()


if __name__ == "__main__":
    main()
