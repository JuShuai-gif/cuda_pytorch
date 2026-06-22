"""
简化的 continuous batching 调度器。

Continuous batching 是一种服务优化策略，请求可以在运行中的 batch
动态加入或移除，无需等待所有请求完成。

核心概念：
  - Request queue（请求队列）：等待处理的入站请求
  - Running batch（运行中的 batch）：当前正在一起处理的请求
  - Prefill priority（prefill 优先）：新请求在加入 decode batch 之前
    会获得一个专属的 prefill 步骤
  - Dynamic add/remove（动态添加/移除）：完成的请求被驱逐；
    新请求在内存允许时被接纳

这是一个*简化的*概念实现，在 token ID 层级使用
一个假模型来演示调度逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch


# =========================================================================
# 请求与调度器数据结构
# =========================================================================


class RequestStatus(Enum):
    """系统中生成请求的状态。"""

    QUEUED = auto()  # 在请求队列中等待
    PREFILLING = auto()  # 正在处理 prompt（prefill 阶段）
    DECODING = auto()  # 逐个生成 token
    FINISHED = auto()  # 所有 token 已生成（或已停止）


@dataclass
class GenerationRequest:
    """单个生成请求。

    Attributes:
        request_id: 唯一标识符
        prompt_ids: 输入 token id
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        generated_ids: 目前已生成的 token
        status: 当前状态
        step_count: 已执行的 decode 步骤数
    """

    request_id: int
    prompt_ids: list[int]
    max_new_tokens: int = 100
    temperature: float = 1.0
    generated_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.QUEUED
    step_count: int = 0

    @property
    def is_finished(self) -> bool:
        """检查此请求是否已完成生成。"""
        return self.status == RequestStatus.FINISHED

    @property
    def total_len(self) -> int:
        """总 token 数量（prompt + 已生成的）。"""
        return len(self.prompt_ids) + len(self.generated_ids)


@dataclass
class BatchSlot:
    """运行中 decode batch 的一个槽位。

    将请求映射到其在 batch 维度中的位置。

    Attributes:
        batch_idx: 在 batch tensor 中的位置
        request_id: 占用此槽位的请求
        kv_cache_len: 此槽位当前的 KV cache 长度
    """

    batch_idx: int
    request_id: int
    kv_cache_len: int = 0


# =========================================================================
# 用于演示的假模型
# =========================================================================


class FakeLLMForBatching:
    """
    返回确定但会变化的 logits 的假语言模型。

    在实际系统中，这将是一个完整的 transformer 模型。
    这里我们模拟 prefill 和 decode 阶段，使用可控的输出形状。
    """

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._rng = torch.Generator()
        self._rng.manual_seed(12345)

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """模拟 prefill：处理完整 prompt，返回 logits。

        Args:
            input_ids: (batch, prompt_len) — 可以具有不同长度，
                       无效位置用 -1 填充

        Returns:
            形状为 (batch, prompt_len, vocab_size) 的 logits
        """
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size, generator=self._rng)
        return logits

    def decode(self, input_ids: torch.Tensor, kv_cache_len: int) -> torch.Tensor:
        """模拟一个 decode 步骤。

        Args:
            input_ids: (batch, 1) — 每个序列一个 token
            kv_cache_len: 当前 KV cache 长度，用于位置编码

        Returns:
            形状为 (batch, 1, vocab_size) 的 logits
        """
        batch_size = input_ids.size(0)
        logits = torch.randn(batch_size, 1, self.vocab_size, generator=self._rng)
        # 稍微偏向 token 0，使序列倾向于结束
        logits[:, :, 0] += 2.0
        return logits


# =========================================================================
# Continuous Batching 调度器
# =========================================================================


class ContinuousBatchingScheduler:
    """
    管理请求队列和运行中的 decode batch 的调度器。

    每一步：
      1. 检查已完成的请求 → 从运行中的 batch 移除
      2. 检查队列中的新请求 → 执行 prefill
      3. 将 prefill 完成的请求添加到 decode batch
      4. 对所有活跃请求执行一个 decode 步骤
    """

    def __init__(
        self,
        model: FakeLLMForBatching,
        max_batch_size: int = 4,
        max_total_tokens: int = 512,
        eos_token_id: int = 0,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.eos_token_id = eos_token_id

        self.request_queue: list[GenerationRequest] = []
        self.running_requests: dict[int, GenerationRequest] = {}
        self.batch_slots: list[BatchSlot] = []

        # 统计信息
        self.total_requests_processed: int = 0
        self.step_counter: int = 0

    def submit(self, request: GenerationRequest) -> None:
        """将一个新的生成请求提交到队列。

        Args:
            request: 生成请求
        """
        request.status = RequestStatus.QUEUED
        self.request_queue.append(request)

    def step(self) -> None:
        """执行一个调度步骤。

        核心循环：对新请求进行 prefill，对现有 batch 进行 decode，
        驱逐已完成的请求。
        """
        self.step_counter += 1

        # ---- 阶段 1：驱逐已完成的请求 ----
        self._evict_finished()

        # ---- 阶段 2：接纳新请求（prefill）----
        self._admit_requests()

        # ---- 阶段 3：Decode 步骤 ----
        if self.batch_slots:
            self._decode_step()

    def _evict_finished(self) -> None:
        """从运行中的 batch 移除已完成的请求。"""
        to_remove: list[int] = []

        for slot in self.batch_slots:
            req = self.running_requests.get(slot.request_id)
            if req is None or req.is_finished:
                to_remove.append(slot.request_id)

        for req_id in to_remove:
            req = self.running_requests.pop(req_id, None)
            if req:
                req.status = RequestStatus.FINISHED
                self.total_requests_processed += 1

        # 重建 batch slot（移除已完成的，重新索引）
        self.batch_slots = [
            slot
            for slot in self.batch_slots
            if slot.request_id in self.running_requests
        ]
        # 重新索引 batch 位置
        for i, slot in enumerate(self.batch_slots):
            slot.batch_idx = i

    def _admit_requests(self) -> None:
        """通过 prefill 从队列中接纳新请求。

        从队列中选取请求，最多到 max_batch_size，执行
        prefill，然后在它们合适的情况下将其添加到运行中的 batch。
        """
        available_slots = self.max_batch_size - len(self.batch_slots)
        if available_slots <= 0:
            return

        # 选取要进行 prefill 的请求（最多到可用 slot 数）
        prefill_candidates = self.request_queue[:available_slots]
        if not prefill_candidates:
            return

        # ---- Prefill：在 batch 中处理 prompt ----
        # 打包 prompt（它们可能具有不同长度；填充到最大长度）
        max_prompt_len = max(len(r.prompt_ids) for r in prefill_candidates)
        batch_ids = torch.full(
            (len(prefill_candidates), max_prompt_len), -1, dtype=torch.long
        )
        for i, req in enumerate(prefill_candidates):
            batch_ids[i, : len(req.prompt_ids)] = torch.tensor(req.prompt_ids)

        # Prefill 前向传播
        logits = self.model.prefill(batch_ids)

        # 从每个 prompt 的最后一个位置采样第一个 token
        last_logits = logits[
            torch.arange(len(prefill_candidates)),
            torch.tensor([len(r.prompt_ids) - 1 for r in prefill_candidates]),
        ]
        first_tokens = last_logits.argmax(dim=-1)

        # 从队列移到运行中
        for i, req in enumerate(prefill_candidates):
            self.request_queue.remove(req)
            req.status = RequestStatus.DECODING
            req.generated_ids.append(first_tokens[i].item())
            req.step_count += 1
            self.running_requests[req.request_id] = req

            # 创建 batch slot
            slot = BatchSlot(
                batch_idx=len(self.batch_slots),
                request_id=req.request_id,
                kv_cache_len=req.total_len,
            )
            self.batch_slots.append(slot)

    def _decode_step(self) -> None:
        """对运行中 batch 的所有请求执行一个 decode 步骤。

        将每个请求的最后一个已生成 token 馈送给模型，
        采样下一个 token，并追加它。
        """
        batch_size = len(self.batch_slots)

        # 收集每个请求的最后一个已生成 token
        last_tokens = torch.zeros(batch_size, 1, dtype=torch.long)
        for slot in self.batch_slots:
            req = self.running_requests[slot.request_id]
            last_tokens[slot.batch_idx, 0] = req.generated_ids[-1]

        # Decode 前向传播
        logits = self.model.decode(last_tokens, kv_cache_len=0)
        next_logits = logits[:, -1, :]  # (batch, vocab)
        next_tokens = next_logits.argmax(dim=-1)

        # 追加 token 并检查 EOS / 最大长度
        for slot in self.batch_slots:
            req = self.running_requests[slot.request_id]
            token = next_tokens[slot.batch_idx].item()

            if token == self.eos_token_id:
                req.status = RequestStatus.FINISHED
            elif req.step_count >= req.max_new_tokens:
                req.status = RequestStatus.FINISHED
            else:
                req.generated_ids.append(token)
                req.step_count += 1
                slot.kv_cache_len = req.total_len

    def is_idle(self) -> bool:
        """检查是否没有剩余工作要做。"""
        return len(self.request_queue) == 0 and len(self.running_requests) == 0

    def status_report(self) -> str:
        """返回可读的状态报告。"""
        lines = [
            f"Step {self.step_counter}:",
            f"  Queue: {len(self.request_queue)} requests",
            f"  Running: {len(self.running_requests)}/{self.max_batch_size} slots",
            f"  Total processed: {self.total_requests_processed}",
        ]
        for slot in self.batch_slots:
            req = self.running_requests.get(slot.request_id)
            if req:
                lines.append(
                    f"    Slot {slot.batch_idx}: req={req.request_id}, "
                    f"tokens={req.total_len}, "
                    f"steps={req.step_count}/{req.max_new_tokens}"
                )
        return "\n".join(lines)


# =========================================================================
# 演示
# =========================================================================


def demo_continuous_batching() -> None:
    """演示 continuous batching 调度器的生命周期。"""
    print("=" * 70)
    print("Continuous Batching Scheduler Demo")
    print("=" * 70)

    model = FakeLLMForBatching(vocab_size=50)
    scheduler = ContinuousBatchingScheduler(
        model=model,
        max_batch_size=3,
        max_total_tokens=512,
        eos_token_id=0,
    )

    # 提交若干具有不同 prompt 长度的请求
    requests = [
        GenerationRequest(request_id=1, prompt_ids=[1, 2, 3], max_new_tokens=5),
        GenerationRequest(request_id=2, prompt_ids=[4, 5, 6, 7, 8], max_new_tokens=3),
        GenerationRequest(request_id=3, prompt_ids=[9, 10], max_new_tokens=4),
        GenerationRequest(request_id=4, prompt_ids=[11, 12, 13, 14], max_new_tokens=6),
        GenerationRequest(request_id=5, prompt_ids=[15, 16], max_new_tokens=2),
    ]

    for req in requests:
        scheduler.submit(req)

    print(f"\nInitial queue: {[r.request_id for r in scheduler.request_queue]}")
    print(f"Max batch size: {scheduler.max_batch_size}")

    # 运行调度器直到所有请求处理完毕
    print("\n--- Scheduling Loop ---")
    max_steps = 30
    for _ in range(max_steps):
        if scheduler.is_idle():
            break
        scheduler.step()
        report = scheduler.status_report()
        print(report)
        print()

    # 最终总结
    print("--- Final Summary ---")
    for req in requests:
        print(
            f"  Request {req.request_id}: "
            f"prompt={req.prompt_ids}, "
            f"generated={req.generated_ids}, "
            f"status={req.status.name}"
        )

    print(f"\n  Total steps executed: {scheduler.step_counter}")
    print(f"  Requests processed: {scheduler.total_requests_processed}")
    print("\n  Key insight: Requests with different lengths are batched together.")
    print("  Short requests finish early and free up batch slots for new requests,")
    print("  maximizing GPU utilization compared to static batching.")


def demo_prefill_priority() -> None:
    """演示 prefill 优先：新请求获得专属的 prefill 步骤。"""
    print("\n" + "=" * 70)
    print("Prefill Priority Demo")
    print("=" * 70)

    model = FakeLLMForBatching(vocab_size=50)
    scheduler = ContinuousBatchingScheduler(model=model, max_batch_size=2)

    # 首先提交一个长时间运行的请求
    long_req = GenerationRequest(request_id=10, prompt_ids=[1, 2], max_new_tokens=10)
    scheduler.submit(long_req)

    # 运行几个步骤
    print("\n--- Step 1: Admit first request ---")
    scheduler.step()
    print(scheduler.status_report())

    # 在生成中途提交一个新请求
    print("\n--- Step 2: New request arrives mid-generation ---")
    new_req = GenerationRequest(request_id=20, prompt_ids=[3, 4, 5], max_new_tokens=3)
    scheduler.submit(new_req)
    scheduler.step()
    print(scheduler.status_report())

    # 继续直到空闲
    print("\n--- Remaining steps ---")
    for _ in range(20):
        if scheduler.is_idle():
            break
        scheduler.step()

    print(scheduler.status_report())
    print("\n  Key insight: New requests are prefill-prioritized and join the")
    print("  running batch immediately, without waiting for in-progress requests")
    print("  to complete. This minimizes time-to-first-token (TTFT).")


def main() -> None:
    demo_continuous_batching()
    demo_prefill_priority()


if __name__ == "__main__":
    main()
