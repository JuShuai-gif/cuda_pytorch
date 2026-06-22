"""
工业级异构调度器 — 借鉴 vLLM/DeepSpeed/Megatron-LM 设计模式

核心组件：
1. BlockManager     — PagedAttention 风格的 KV cache 块管理器（借鉴 vLLM）
2. SequenceManager  — 序列生命周期管理（借鉴 vLLM Sequence/SequenceGroup）
3. Scheduler        — 连续批处理（continuous batching）调度器（借鉴 vLLM）
4. MemoryPlanner    — 显存感知的调度策略（借鉴 vLLM + DeepSpeed）
5. PPScheduler      — 流水线并行 1F1B 调度器（借鉴 Megatron-LM）
6. TPScheduler      — 张量并行调度器（借鉴 Megatron-LM）
7. ZEROStageManager — ZeRO 分片策略管理器（借鉴 DeepSpeed ZeRO-1/2/3）
8. WorkloadBalancer — 异构 GPU 负载均衡（借鉴 FlexFlow/Alpa）
9. NCCLCommManager  — NCCL 通信管理（基于 torch.distributed 封装）
10. HybridScheduler — 混合并行（TP+PP+DP）调度器
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence as SequenceType, Set, Tuple

# ==============================================================================
# 常量定义
# ==============================================================================

BLOCK_SIZE = 16  # 每个 KV cache 块容纳的 token 数（借鉴 vLLM）
NUM_LAYERS = 32  # 默认 Transformer 层数
NUM_HEADS = 32  # 默认注意力头数
HEAD_DIM = 128  # 默认注意力头维度
BYTES_PER_ELEMENT = 2  # fp16/bf16 每个元素字节数


# ==============================================================================
# 1. BlockManager — PagedAttention 风格的 KV cache 块管理器
# ==============================================================================


class BlockManager:
    """vLLM 风格的 KV cache 块管理器

    设计原理（借鉴 vLLM vllm/core/block/ 模块）:
    - 将 KV cache 空间划分为固定大小的物理块（block_size 个 token/块）
    - 使用空闲块列表（free block list）管理未分配块，O(1) 分配/回收
    - 每个序列维护逻辑块表（logical block table），映射到物理块
    - 支持 prefix caching：相同 prompt 前缀的序列共享物理块，通过引用计数管理
    - 物理块在物理设备上对应一段连续显存，逻辑上按层索引

    核心数据结构:
    - free_blocks: List[int] — 空闲物理块 ID 栈
    - block_ref_count: Dict[int, int] — 每个物理块的引用计数
    - allocated_blocks: Set[int] — 已分配物理块集合（调试用）
    """

    def __init__(
        self,
        block_size: int = BLOCK_SIZE,
        num_blocks: int = 4096,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 空闲物理块栈（栈顶弹出，O(1) 分配）
        self.free_blocks: List[int] = list(range(num_blocks))

        # 引用计数：物理块被几个序列引用（prefix caching 依赖此机制）
        self.block_ref_count: Dict[int, int] = {i: 0 for i in range(num_blocks)}

        # 已分配块追踪（prefix caching 命中时不增加计数）
        self.allocated_blocks: Set[int] = set()

        # 统计信息
        self.num_alloc_ops = 0
        self.num_free_ops = 0
        self.num_prefix_cache_hits = 0

    def allocate(self) -> Optional[int]:
        """分配一个空闲物理块。

        从 free_blocks 栈顶弹出，O(1) 时间复杂度。
        分配后引用计数设为 1。

        Returns:
            物理块 ID；若无空闲块则返回 None。
        """
        if not self.free_blocks:
            return None
        block_id = self.free_blocks.pop()
        self.block_ref_count[block_id] = 1
        self.allocated_blocks.add(block_id)
        self.num_alloc_ops += 1
        return block_id

    def allocate_with_prefix_cache(self, block_id: Optional[int]) -> Optional[int]:
        """尝试复用已有物理块（prefix caching）。

        如果 block_id 对应的物理块已存在且可共享，仅增加引用计数；
        否则分配新块。

        借鉴 vLLM: 相同前缀的多个序列共享 KV cache 物理块。
        """
        if block_id is not None and block_id in self.allocated_blocks:
            self.block_ref_count[block_id] += 1
            self.num_prefix_cache_hits += 1
            return block_id
        return self.allocate()

    def free(self, block_id: int) -> None:
        """释放一个物理块。

        递减引用计数；仅当引用计数归零时才将块归还到 free_blocks。
        这是 prefix caching 安全性的关键：多个序列共享同一物理块时，
        只有所有引用者都释放后，块才真正变为空闲。

        借鉴 vLLM: block_manager.py 中的 _free_block 逻辑。
        """
        if block_id not in self.block_ref_count:
            return
        ref = self.block_ref_count[block_id]
        if ref <= 0:
            return
        self.block_ref_count[block_id] = ref - 1
        if ref == 1:
            # 最后一个引用者释放，块归还到空闲池
            self.free_blocks.append(block_id)
            self.allocated_blocks.discard(block_id)
        self.num_free_ops += 1

    def get_num_free_blocks(self) -> int:
        """返回当前空闲块数量。"""
        return len(self.free_blocks)

    def can_allocate(self, num_blocks: int) -> bool:
        """检查是否有足够空闲块可供分配。"""
        return self.get_num_free_blocks() >= num_blocks

    def get_usage_ratio(self) -> float:
        """返回块使用率（0.0 ~ 1.0）。"""
        if self.num_blocks == 0:
            return 0.0
        return len(self.allocated_blocks) / self.num_blocks

    def get_prefix_cache_hit_rate(self) -> float:
        """返回 prefix cache 命中率。"""
        total = self.num_alloc_ops + self.num_prefix_cache_hits
        if total == 0:
            return 0.0
        return self.num_prefix_cache_hits / total

    def block_size_bytes(self) -> int:
        """计算每个物理块占用的字节数（K cache + V cache，单层）。

        每个 block 存储单层 K+V cache:
        - K cache: block_size × num_heads × head_dim × BYTES_PER_ELEMENT
        - V cache: 同上
        - 总计: 2 × 上述
        由于每层独立分配物理块，block_size 计算单层开销。
        """
        return 2 * self.block_size * self.num_heads * self.head_dim * BYTES_PER_ELEMENT

    def reset(self) -> None:
        """重置块管理器到初始状态。"""
        self.free_blocks = list(range(self.num_blocks))
        self.block_ref_count = {i: 0 for i in range(self.num_blocks)}
        self.allocated_blocks.clear()
        self.num_alloc_ops = 0
        self.num_free_ops = 0
        self.num_prefix_cache_hits = 0


# ==============================================================================
# 2. 序列管理 — 借鉴 vLLM Sequence / SequenceGroup
# ==============================================================================


class SequenceStatus(Enum):
    """序列状态枚举（借鉴 vLLM SequenceStatus）。"""

    WAITING = 1  # 等待调度，尚未分配 KV cache 块
    RUNNING = 2  # 正在运行中，持有 KV cache 块
    SWAPPED = 3  # 被换出（显存不足时，KV cache 移至 CPU）
    FINISHED = 4  # 已完成生成或达到 max_tokens


@dataclass
class Sequence:
    """单个推理请求序列。

    借鉴 vLLM Sequence 设计:
    - prompt: 输入的 token IDs
    - output_tokens: 已生成的 token IDs
    - block_table: 逻辑块表，映射 (layer_idx, logical_block_idx) -> physical_block_id
    - status: 当前生命周期状态

    计算逻辑块数: num_logical_blocks = ceil(prompt_len + len(output_tokens)) / block_size
    """

    seq_id: int
    prompt: List[int]
    prompt_len: int
    max_output_len: int
    output_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    # 逻辑块表：(layer_idx, logical_block_idx) -> physical_block_id
    block_table: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # 调度时间戳
    arrival_time: float = 0.0
    last_scheduled_time: float = 0.0

    # 每个 token 的生成时间（用于 SLA 监控）
    token_generation_times: List[float] = field(default_factory=list)

    def num_logical_blocks(self, block_size: int = BLOCK_SIZE) -> int:
        """该序列当前需要的逻辑块数（每层）。"""
        total_tokens = self.prompt_len + len(self.output_tokens)
        return max(1, (total_tokens + block_size - 1) // block_size)

    @property
    def num_generated_tokens(self) -> int:
        """已生成的 token 数量。"""
        return len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        """序列是否已完成生成。"""
        return self.num_generated_tokens >= self.max_output_len

    @property
    def total_tokens(self) -> int:
        """当前总 token 数（prompt + 已生成）。"""
        return self.prompt_len + self.num_generated_tokens


@dataclass
class SequenceGroup:
    """一组相关序列（例如 beam search 中的多个 beam）。

    借鉴 vLLM SequenceGroup:
    - request_id: 全局唯一的请求标识
    - seqs: 该组内的所有序列（beam search 时有多个候选）
    - sampling_params: 采样参数（temperature, top_p, top_k 等）
    """

    request_id: str
    seqs: List[Sequence]
    sampling_params: Dict = field(default_factory=dict)

    @property
    def is_finished(self) -> bool:
        """该组内所有序列是否都已完成。"""
        return all(s.is_finished for s in self.seqs)

    @property
    def num_running_seqs(self) -> int:
        """当前正在运行的序列数。"""
        return sum(1 for s in self.seqs if s.status == SequenceStatus.RUNNING)

    def total_logical_blocks(self, block_size: int = BLOCK_SIZE) -> int:
        """该组内所有序列需要的逻辑块总数。"""
        return sum(s.num_logical_blocks(block_size) for s in self.seqs)

    def max_total_tokens(self) -> int:
        """该组内序列的最大 token 数（用于 batch size 限制）。"""
        return max(s.total_tokens for s in self.seqs) if self.seqs else 0


# ==============================================================================
# 3. Scheduler — 连续批处理调度器
# ==============================================================================


class Scheduler:
    """连续批处理（continuous batching）调度器 — 借鉴 vLLM Scheduler

    核心概念（借鉴 vLLM vllm/core/scheduler.py）:
    - waiting queue: 等待调度的新请求
    - running queue: 当前正在运行的请求（每步生成一个 token）
    - swapped queue: 被换出到 CPU 的请求（显存不足时）

    调度策略（借鉴 vLLM 的 _schedule 方法）:
    1. 每步尝试从 waiting 队列取新请求，分配 KV cache 块
    2. 若显存不足，换出 running 中低优先级的请求（preemption）
    3. 若仍不足，拒绝当前新请求
    4. 合并 waiting（已分配）+ running 形成当步执行 batch
    5. 执行推理步骤后，更新每个序列状态
    6. 完成或达到 max_tokens 的序列释放资源

    优先级策略:
    - 短 prompt 优先（减少首 token 延迟，TTFT 优化）
    - 长 running 序列可被换出（避免饥饿 + 提高吞吐）
    """

    def __init__(
        self,
        block_manager: BlockManager,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        max_waiting_queue_size: int = 1024,
    ) -> None:
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_waiting_queue_size = max_waiting_queue_size

        # 三个队列（借鉴 vLLM）
        self.waiting: List[SequenceGroup] = []
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

        # 调度统计
        self.num_preempted = 0  # 被抢占次数
        self.num_swapped_out = 0  # 换出次数
        self.num_swapped_in = 0  # 换入次数
        self.num_completed_requests = 0  # 已完成请求数
        self.num_rejected_requests = 0  # 被拒绝的请求数
        self.total_scheduled_tokens = 0  # 累计调度 token 数

        # Preemption 模式: "recompute" 表示换出时不保存 KV cache（需要重算）
        self.preemption_mode: str = "recompute"

    def add_request(self, seq_group: SequenceGroup) -> bool:
        """将新请求加入 waiting 队列。

        Returns:
            True 如果成功加入；False 如果队列已满。
        """
        if len(self.waiting) >= self.max_waiting_queue_size:
            self.num_rejected_requests += 1
            return False
        self.waiting.append(seq_group)
        return True

    def schedule(self) -> Tuple[List[SequenceGroup], List[SequenceGroup], bool]:
        """执行一步调度 — 借鉴 vLLM Scheduler._schedule()

        Returns:
            scheduled: 本轮要执行的序列组列表
            ignored: 本轮被忽略/换出的序列组列表
            is_prefill: 是否包含 prefill 阶段的新序列
        """
        scheduled: List[SequenceGroup] = []
        is_prefill = False

        # 追踪已调度的序列数和 token 数（跨 waiting + running）
        num_scheduled_seqs = 0
        total_scheduled_tokens = 0

        # ---- 第 1 阶段: 调度 waiting 队列中的新请求 ----
        remaining_waiting: List[SequenceGroup] = []

        for sg in self.waiting:
            sg_num_seqs = len(sg.seqs)
            sg_tokens = sum(s.total_tokens for s in sg.seqs)

            # 检查序列数和 token 限制
            if num_scheduled_seqs + sg_num_seqs > self.max_num_seqs:
                remaining_waiting.append(sg)
                continue
            if total_scheduled_tokens + sg_tokens > self.max_num_batched_tokens:
                remaining_waiting.append(sg)
                continue

            # 计算该序列组需要的总块数
            num_blocks_needed = sg.total_logical_blocks()

            if self.block_manager.can_allocate(num_blocks_needed):
                # 有足够显存，尝试分配 KV cache 块
                success = self._allocate_blocks_for_group(sg)
                if success:
                    for s in sg.seqs:
                        s.status = SequenceStatus.RUNNING
                    scheduled.append(sg)
                    num_scheduled_seqs += sg_num_seqs
                    total_scheduled_tokens += sg_tokens
                    is_prefill = True
                    continue

            # 分配失败：显存不足或分配出错
            if self.running and self.preemption_mode == "recompute":
                # 尝试换出最低优先级的 running 序列以腾出空间
                freed = self._preempt_lowest_priority()
                if freed >= num_blocks_needed and self.block_manager.can_allocate(
                    num_blocks_needed
                ):
                    success = self._allocate_blocks_for_group(sg)
                    if success:
                        for s in sg.seqs:
                            s.status = SequenceStatus.RUNNING
                        scheduled.append(sg)
                        num_scheduled_seqs += sg_num_seqs
                        total_scheduled_tokens += sg_tokens
                        is_prefill = True
                        continue

            # 仍然无法调度，留在 waiting 队列
            remaining_waiting.append(sg)

        self.waiting = remaining_waiting

        # ---- 第 2 阶段: 添加 running 队列中未完成的序列 ----
        # 未调度的 running 序列保留在 running 中（下轮重新尝试）
        unscheduled_running: List[SequenceGroup] = []
        total_tokens = total_scheduled_tokens
        num_seqs = num_scheduled_seqs

        for sg in self.running:
            sg_num_seqs = len(sg.seqs)
            sg_tokens = sum(s.total_tokens for s in sg.seqs)

            if num_seqs + sg_num_seqs <= self.max_num_seqs:
                if total_tokens + sg_tokens <= self.max_num_batched_tokens:
                    scheduled.append(sg)
                    num_seqs += sg_num_seqs
                    total_tokens += sg_tokens
                    continue

            unscheduled_running.append(sg)

        # 未调度的 running 序列保留，swapped 序列归入 ignored
        self.running = unscheduled_running
        ignored = self.swapped

        # 更新统计
        self.total_scheduled_tokens += total_tokens

        return scheduled, ignored, is_prefill

    def update_after_step(self, scheduled: List[SequenceGroup]) -> None:
        """推理步骤后更新状态 — 借鉴 vLLM Scheduler._update_after_step()

        对每个完成生成的序列释放其 KV cache 块。
        未完成的序列合并到 self.running（保留之前 unscheduled 的 running 序列）。
        """
        still_running: List[SequenceGroup] = []

        for sg in scheduled:
            for s in sg.seqs:
                # 模拟 token 生成：追加一个占位 token
                s.output_tokens.append(0)

            if not sg.is_finished:
                still_running.append(sg)
            else:
                self._free_sequence_group(sg)
                self.num_completed_requests += 1

        # 合并：保留之前未被调度的 running 序列，加上本步仍 active 的序列
        self.running = self.running + still_running

    def try_swap_in(self) -> int:
        """尝试将 swapped 队列中的序列换回 running。

        Returns:
            成功换回的序列组数量。
        """
        if not self.swapped:
            return 0

        remaining_swapped: List[SequenceGroup] = []
        swapped_in_count = 0

        for sg in self.swapped:
            num_blocks_needed = sg.total_logical_blocks()
            if self.block_manager.can_allocate(num_blocks_needed):
                success = self._allocate_blocks_for_group(sg)
                if success:
                    for s in sg.seqs:
                        s.status = SequenceStatus.RUNNING
                    self.running.append(sg)
                    swapped_in_count += 1
                    self.num_swapped_in += 1
                    continue
            remaining_swapped.append(sg)

        self.swapped = remaining_swapped
        return swapped_in_count

    def get_queue_sizes(self) -> Dict[str, int]:
        """返回各队列的当前大小。"""
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
        }

    def get_stats(self) -> Dict:
        """返回调度器统计摘要。"""
        return {
            "num_preempted": self.num_preempted,
            "num_swapped_out": self.num_swapped_out,
            "num_swapped_in": self.num_swapped_in,
            "num_completed": self.num_completed_requests,
            "num_rejected": self.num_rejected_requests,
            "total_scheduled_tokens": self.total_scheduled_tokens,
            "block_usage_ratio": self.block_manager.get_usage_ratio(),
            "prefix_cache_hit_rate": self.block_manager.get_prefix_cache_hit_rate(),
        }

    def _allocate_blocks_for_group(self, sg: SequenceGroup) -> bool:
        """为一个 SequenceGroup 中的所有序列分配 KV cache 块。

        Returns:
            True 如果所有块分配成功；False 如果中途失败（已回滚）。
        """
        for s in sg.seqs:
            for layer_idx in range(self.block_manager.num_layers):
                for lb in range(s.num_logical_blocks()):
                    block_id = self.block_manager.allocate()
                    if block_id is None:
                        # 分配失败，回滚整个 group
                        self._rollback_group_alloc(sg)
                        return False
                    s.block_table[(layer_idx, lb)] = block_id
        return True

    def _rollback_group_alloc(self, sg: SequenceGroup) -> None:
        """回滚一个 SequenceGroup 的所有块分配。"""
        for s in sg.seqs:
            self._rollback_seq_alloc(s)

    def _rollback_seq_alloc(self, seq: Sequence) -> None:
        """回滚单个序列的所有块分配。"""
        for key in list(seq.block_table.keys()):
            block_id = seq.block_table.pop(key)
            self.block_manager.free(block_id)

    def _free_sequence_group(self, sg: SequenceGroup) -> None:
        """释放序列组占用的所有 KV cache 块。"""
        for s in sg.seqs:
            for block_id in s.block_table.values():
                self.block_manager.free(block_id)
            s.block_table.clear()
            s.status = SequenceStatus.FINISHED

    def _preempt_lowest_priority(self) -> int:
        """换出最低优先级的 running 序列组。

        借鉴 vLLM: 选择运行最久（可能陷入长文本生成）的序列换出，
        释放其 KV cache 块供新请求使用。如果 preemption_mode 为 "recompute"，
        被换出的序列需要重新计算 KV cache（不保存中间状态）。

        Returns:
            释放的块数量。
        """
        if not self.running:
            return 0

        # 简化策略：换出 prompt_len 最大（通常最占显存）的序列组
        victim_idx = max(
            range(len(self.running)),
            key=lambda i: self.running[i].max_total_tokens(),
        )
        victim = self.running.pop(victim_idx)

        freed_blocks = 0
        for s in victim.seqs:
            freed_blocks += len(s.block_table)
            for block_id in list(s.block_table.values()):
                self.block_manager.free(block_id)
            s.block_table.clear()
            s.status = SequenceStatus.SWAPPED

        self.swapped.append(victim)
        self.num_swapped_out += 1
        self.num_preempted += 1
        return freed_blocks


# ==============================================================================
# 4. MemoryPlanner — 显存感知的调度策略
# ==============================================================================


class MemoryPlanner:
    """显存感知的调度策略 — 借鉴 vLLM + DeepSpeed 的内存管理模式

    策略:
    1. 预留模型权重显存（fixed overhead）— GPU 总显存中减去模型大小
    2. 为 KV cache 分配 block 池 — 剩余显存全部用于 KV cache
    3. 支持 watermark 策略 — 超过阈值时禁止新请求进入
    4. 动态 block 数量和大小调整 — 根据 batch 特征自适应

    借鉴 DeepSpeed: ZeRO-3 将优化器状态、梯度、参数全部 offload，
    释放显存给计算和 KV cache。
    """

    def __init__(
        self,
        total_memory_gb: float,
        model_size_gb: float,
        block_size: int = BLOCK_SIZE,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        watermark: float = 0.9,
        reserved_memory_gb: float = 2.0,
    ) -> None:
        self.total_memory_gb = total_memory_gb
        self.model_size_gb = model_size_gb
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.watermark = watermark
        self.reserved_memory_gb = reserved_memory_gb

        # 可用于 KV cache 的显存
        self.available_kv_cache_gb = max(0.0, total_memory_gb - model_size_gb - reserved_memory_gb)

        # 计算每个 block 的字节数和总块数
        self.bytes_per_block = self._compute_bytes_per_block()
        self.num_total_blocks = int(self.available_kv_cache_gb * (1024**3) / self.bytes_per_block)
        self.used_blocks = 0

        # 允许在运行时动态调整的 watermark
        self._current_watermark = watermark

    def _compute_bytes_per_block(self) -> int:
        """计算每个 KV cache 物理块占用的字节数。

        每个 block 存储 K cache + V cache 各一份，每个 token 存储
        num_heads × head_dim 个元素。

        K cache: num_layers × block_size × num_heads × head_dim × bytes_per_element
        V cache: 同上
        总计: 2 × 上述
        """
        per_token_bytes = self.num_heads * self.head_dim * BYTES_PER_ELEMENT
        return 2 * self.num_layers * self.block_size * per_token_bytes

    def can_accept_new(self, requested_blocks: int) -> bool:
        """检查是否能接受需要 requested_blocks 个块的新请求。

        借鉴 vLLM: 使用 watermark 策略防止 KV cache 耗尽。
        当使用率超过 watermark 时，拒绝新请求（仅允许 running 序列继续）。
        """
        if self.num_total_blocks == 0:
            return False
        projected_usage = (self.used_blocks + requested_blocks) / self.num_total_blocks
        return projected_usage < self._current_watermark

    def reserve_blocks(self, num_blocks: int) -> bool:
        """预留 num_blocks 个块。

        Returns:
            True 如果预留成功（不超过 watermark）。
        """
        if not self.can_accept_new(num_blocks):
            return False
        self.used_blocks += num_blocks
        return True

    def release_blocks(self, num_blocks: int) -> None:
        """释放 num_blocks 个块的预留。"""
        self.used_blocks = max(0, self.used_blocks - num_blocks)

    def get_available_blocks(self) -> int:
        """当前可分配的块数（考虑 watermark 限制）。"""
        max_allowed = int(self.num_total_blocks * self._current_watermark)
        return max(0, max_allowed - self.used_blocks)

    def get_usage_ratio(self) -> float:
        """返回当前 KV cache 使用率。"""
        if self.num_total_blocks == 0:
            return 1.0
        return self.used_blocks / self.num_total_blocks

    def update_watermark(self, new_watermark: float) -> None:
        """动态调整 watermark（例如根据 SLA 要求）。"""
        self._current_watermark = max(0.1, min(0.99, new_watermark))

    def estimate_max_concurrent_sequences(self, avg_prompt_len: int, avg_output_len: int) -> int:
        """估算在给定平均 prompt/output 长度下可并发服务的序列数。"""
        avg_tokens_per_seq = avg_prompt_len + avg_output_len
        blocks_per_seq = max(1, (avg_tokens_per_seq + self.block_size - 1) // self.block_size)
        if blocks_per_seq == 0:
            return 0
        return self.get_available_blocks() // blocks_per_seq

    def get_stats(self) -> Dict:
        """返回显存规划器统计摘要。"""
        return {
            "total_memory_gb": self.total_memory_gb,
            "model_size_gb": self.model_size_gb,
            "reserved_memory_gb": self.reserved_memory_gb,
            "available_kv_cache_gb": round(self.available_kv_cache_gb, 2),
            "bytes_per_block": self.bytes_per_block,
            "num_total_blocks": self.num_total_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.num_total_blocks - self.used_blocks,
            "watermark": self._current_watermark,
            "usage_pct": f"{self.get_usage_ratio() * 100:.1f}%",
        }


# ==============================================================================
# 5. PPScheduler — 流水线并行 1F1B 调度器
# ==============================================================================


class PPScheduler:
    """流水线并行（Pipeline Parallelism）1F1B 调度器 — 借鉴 Megatron-LM

    实现 1F1B (one-forward-one-backward) 调度策略。

    借鉴 Megatron-LM megatron/core/pipeline_parallel/schedules.py:
    - get_pp_rank_microbatches() 计算每 rank 的 warmup/steady/cooldown 阶段
    - num_warmup_microbatches = pp_size - pp_rank - 1
    - Bubble ratio: (pp_size - 1) / (pp_size - 1 + num_microbatches)

    调度阶段:
    1. Warmup: 前 k 个 microbatch 只做 forward（填充 pipeline）
    2. Steady: 每个 step 做 1 forward + 1 backward（1F1B）
    3. Cooldown: 最后 k 个 microbatch 只做 backward（清空 pipeline）

    优化建议: num_microbatches >> pp_size 以减小 bubble 比例。
    """

    def __init__(self, pp_size: int, num_microbatches: int) -> None:
        if pp_size < 1:
            raise ValueError(f"pp_size must be >= 1, got {pp_size}")
        if num_microbatches < 1:
            raise ValueError(f"num_microbatches must be >= 1, got {num_microbatches}")

        self.pp_size = pp_size
        self.num_microbatches = num_microbatches

    def bubble_ratio(self) -> float:
        """计算 pipeline bubble 比例。

        借鉴 Megatron-LM: bubble_time = (pp_size - 1) / (pp_size - 1 + num_microbatches)
        当 num_microbatches >> pp_size 时，bubble 趋近于 0。
        """
        if self.pp_size <= 1:
            return 0.0
        return (self.pp_size - 1) / (self.pp_size - 1 + self.num_microbatches)

    def efficiency(self) -> float:
        """计算 pipeline 效率（1 - bubble_ratio）。"""
        return 1.0 - self.bubble_ratio()

    def num_warmup_microbatches(self, pp_rank: int) -> int:
        """计算指定 rank 的 warmup microbatch 数量。

        借鉴 Megatron-LM get_pp_rank_microbatches():
        num_warmup = pp_size - pp_rank - 1

        例如 PP=4:
        - rank 0: warmup=3（需要 3 个 warmup forward 才能到达 1F1B）
        - rank 1: warmup=2
        - rank 2: warmup=1
        - rank 3: warmup=0（最末 rank，立即进入 1F1B）
        """
        return max(0, self.pp_size - pp_rank - 1)

    def schedule_1f1b(self) -> List[Tuple[int, int, str, int]]:
        """生成完整的 1F1B 调度计划。

        按照 Megatron-LM 的实现逻辑:
        - 每个 rank 维护自己的 vmid (virtual microbatch id)
        - vmid 在 [0, num_warmup + 2 * (M - num_warmup) - 1] 范围内
        - 如果 vmid < num_warmup: 仅 forward
        - 否则: forward(vmid) + backward(vmid - num_warmup)

        Returns:
            List of (global_step, stage_id, "F"|"B", microbatch_id)
            按 global_step 排序，同一 step 内的操作按 stage_id 排序。
        """
        if self.pp_size <= 1:
            # 无 pipeline 并行，所有 microbatch 顺序执行
            return [(mb, 0, "F", mb) for mb in range(self.num_microbatches)]

        schedule: List[Tuple[int, int, str, int]] = []

        for rank in range(self.pp_size):
            num_warmup = self.num_warmup_microbatches(rank)
            num_remaining = self.num_microbatches - num_warmup
            # 该 rank 的总虚拟 step 数
            total_vmids = num_warmup + 2 * num_remaining

            for vmid in range(total_vmids):
                fwd_mb = vmid
                bwd_mb = vmid - num_warmup

                if 0 <= fwd_mb < self.num_microbatches:
                    schedule.append((vmid, rank, "F", fwd_mb))
                if 0 <= bwd_mb < self.num_microbatches:
                    schedule.append((vmid, rank, "B", bwd_mb))

        return schedule

    def get_rank_schedule(self, pp_rank: int) -> List[Tuple[str, int]]:
        """获取指定 rank 的本地调度序列。

        Returns:
            List of ("F"|"B", microbatch_id) 按执行顺序排列。
        """
        full = self.schedule_1f1b()
        return [
            (op, mb)
            for step, rank, op, mb in sorted(full, key=lambda x: (x[0], x[1]))
            if rank == pp_rank
        ]

    def rank_idle_steps(self, pp_rank: int) -> int:
        """计算指定 rank 的 idle（等待）步数。

        Idle 步 = 总时间步 - (该 rank 的有效操作数)
        由于 pipeline 调度中每个 rank 的总虚拟步数等于全局步数，
        idle 步就是没有进行有效计算的时间步。
        """
        rank_schedule = self.get_rank_schedule(pp_rank)
        total_steps = self.total_steps()
        # 每个 rank 有 total_steps 个全局步，其中有效操作为 len(rank_schedule)
        # idle = total_steps - len(rank_schedule) 可能为负说明调度比预期紧凑
        return max(0, total_steps - len(rank_schedule))

    def total_steps(self) -> int:
        """总全局时间步数（包括 bubble）。"""
        if self.pp_size <= 1:
            return self.num_microbatches
        return self.num_microbatches + self.pp_size - 1

    def get_stats(self) -> Dict:
        """返回调度统计。"""
        return {
            "pp_size": self.pp_size,
            "num_microbatches": self.num_microbatches,
            "bubble_ratio": round(self.bubble_ratio(), 4),
            "efficiency": round(self.efficiency(), 4),
            "total_steps": self.total_steps(),
            "useful_forward_steps": self.num_microbatches,
            "useful_backward_steps": self.num_microbatches,
        }


# ==============================================================================
# 6. TPScheduler — 张量并行调度器
# ==============================================================================


class TPScheduler:
    """张量并行（Tensor Parallelism）调度器 — 借鉴 Megatron-LM

    借鉴 Megatron-LM megatron/core/tensor_parallel/ 模块:
    - ColumnParallelLinear: 权重按列切分，输出 all-reduce（forward）
    - RowParallelLinear: 权重按行切分，输入 all-reduce（forward）

    通信模式（借鉴 Megatron-LM TP）:
    - ColumnParallel: Y = X @ [W1 | W2 | ... | Wn]
        每个 rank 计算本地 Y_i = X @ W_i，然后 all-reduce 合并所有 Y_i
    - RowParallel: Y = [X1 | X2 | ... | Xn] @ W
        每个 rank 计算 Y_i = X_i @ W_i，然后 reduce-scatter 得到最终结果

    Megatron-LM TP 的典型配置:
    - Attention QKV: ColumnParallel (3 * hidden / tp_size columns per rank)
    - Attention Output: RowParallel
    - FFN h_to_4h: ColumnParallel
    - FFN 4h_to_h: RowParallel
    """

    def __init__(self, tp_size: int) -> None:
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        self.tp_size = tp_size

    def partition_col_linear_weight(self, weight_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """ColumnParallel: 按输出维度（列）切分权重。

        借鉴 Megatron-LM ColumnParallelLinear:
        - 原始权重: [hidden_out, hidden_in]
        - 切分后每 rank: [hidden_out / tp_size, hidden_in]

        Args:
            weight_shape: (output_dim, input_dim)
        Returns:
            每 rank 的本地权重形状列表。
        """
        out_dim, in_dim = weight_shape
        per_rank_out = out_dim // self.tp_size
        return [(per_rank_out, in_dim) for _ in range(self.tp_size)]

    def partition_row_linear_weight(self, weight_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """RowParallel: 按输入维度（行）切分权重。

        借鉴 Megatron-LM RowParallelLinear:
        - 原始权重: [hidden_out, hidden_in]
        - 切分后每 rank: [hidden_out, hidden_in / tp_size]

        Args:
            weight_shape: (output_dim, input_dim)
        Returns:
            每 rank 的本地权重形状列表。
        """
        out_dim, in_dim = weight_shape
        per_rank_in = in_dim // self.tp_size
        return [(out_dim, per_rank_in) for _ in range(self.tp_size)]

    def col_parallel_communication_cost(
        self, batch_size: int, seq_len: int, hidden_dim: int
    ) -> int:
        """估算 ColumnParallel AllReduce 的单次通信量（字节）。

        AllReduce 发送量 = 2 * (P-1) / P * data_size
        data_size = batch_size * seq_len * hidden_dim * element_size
        """
        data_size = batch_size * seq_len * hidden_dim * BYTES_PER_ELEMENT
        factor = 2 * (self.tp_size - 1) / self.tp_size if self.tp_size > 1 else 0
        return int(data_size * factor)

    def row_parallel_communication_cost(
        self, batch_size: int, seq_len: int, hidden_dim: int
    ) -> int:
        """估算 RowParallel ReduceScatter 的单次通信量（字节）。"""
        return self.col_parallel_communication_cost(batch_size, seq_len, hidden_dim)

    def get_num_allreduces_per_layer(self) -> int:
        """每层 Transformer 需要的 AllReduce 次数。

        借鉴 Megatron-LM: 每层 4 次 AllReduce（forward 2 + backward 2）
        - Forward: QKV projection + Output projection
        - Backward: 对应的梯度 AllReduce
        """
        return 4

    def estimate_total_communication_per_step(
        self, num_layers: int, batch_size: int, seq_len: int, hidden_dim: int
    ) -> int:
        """估算每训练步的总 TP 通信量（字节）。"""
        comm_per_layer = 2 * self.col_parallel_communication_cost(batch_size, seq_len, hidden_dim)
        return num_layers * comm_per_layer * 2  # forward + backward

    def get_stats(self) -> Dict:
        """返回 TP 配置统计。"""
        return {
            "tp_size": self.tp_size,
            "allreduces_per_layer": self.get_num_allreduces_per_layer(),
        }


# ==============================================================================
# 7. ZEROStageManager — ZeRO 分片策略管理器
# ==============================================================================


class ZEROStage(Enum):
    """DeepSpeed ZeRO 优化阶段（借鉴 DeepSpeed ZeRO 论文）。"""

    STAGE_0 = 0  # 无分片（DDP）
    STAGE_1 = 1  # 分片 Optimizer States（4x 内存节省）
    STAGE_2 = 2  # + 分片 Gradients（8x 内存节省）
    STAGE_3 = 3  # + 分片 Parameters（Nx 内存节省，N=GPU 数量）


class ZEROStageManager:
    """DeepSpeed ZeRO 分片策略管理器 — 借鉴 DeepSpeed ZeRO-1/2/3

    各阶段内存占用分析（借鉴 DeepSpeed ZeRO 论文 Table 1）:

    | Component      | Standard | ZeRO-1      | ZeRO-2      | ZeRO-3      |
    |---------------|----------|-------------|-------------|-------------|
    | Parameters    | N × Ψ    | N × Ψ       | N × Ψ       | Ψ           |
    | Gradients     | N × Ψ    | N × Ψ       | Ψ           | Ψ           |
    | Optim States  | N × KΨ   | KΨ          | KΨ          | KΨ          |
    | Total         | N×(2+K)Ψ | (2N+K)Ψ     | (N+K+1)Ψ     | (K+3)Ψ       |

    其中 Ψ = 模型参数量, N = GPU 数量, K = 优化器状态倍数（Adam: K=12）

    通信模式:
    - ZeRO-1: AllGather optimizer states（每 optimizer step 1 次）
    - ZeRO-2: + ReduceScatter gradients（每 backward step 1 次）
    - ZeRO-3: + AllGather parameters（每层 forward/backward 各 1 次）
    """

    # Adam 优化器状态倍数（fp32 param + fp32 momentum + fp32 variance = 12 bytes/param,
    # 除以 fp16 param 的 2 bytes = 6, 但论文中 K=12 是因为用 fp32 参数计算）
    K_FACTOR = 12

    def __init__(
        self, num_gpus: int, model_params: int, stage: ZEROStage = ZEROStage.STAGE_0
    ) -> None:
        self.num_gpus = num_gpus
        self.model_params = model_params  # 模型参数总量（fp16）
        self.stage = stage
        self.bytes_per_param = BYTES_PER_ELEMENT  # fp16

    def parameter_memory_mb(self) -> float:
        """模型参数内存占用（MB）。"""
        return self.model_params * self.bytes_per_param / (1024**2)

    def optimizer_state_memory_mb(self) -> float:
        """优化器状态内存占用（MB）。

        Adam: fp32 param copy (4B) + fp32 momentum (4B) + fp32 variance (4B) = 12B/param
        """
        return self.model_params * self.K_FACTOR / (1024**2)

    def gradient_memory_mb(self) -> float:
        """梯度内存占用（MB，fp16 梯度）。"""
        return self.model_params * self.bytes_per_param / (1024**2)

    def total_memory_per_gpu_mb(self) -> float:
        """每 GPU 的内存占用（MB）。"""
        param_mem = self.parameter_memory_mb()
        grad_mem = self.gradient_memory_mb()
        opt_mem = self.optimizer_state_memory_mb()

        if self.stage == ZEROStage.STAGE_0:
            # 每 GPU 持有完整副本
            return param_mem + grad_mem + opt_mem

        if self.stage == ZEROStage.STAGE_1:
            # Optimizer states 分片到所有 GPU
            return param_mem + grad_mem + opt_mem / self.num_gpus

        if self.stage == ZEROStage.STAGE_2:
            # Optimizer states + Gradients 分片
            return param_mem + grad_mem / self.num_gpus + opt_mem / self.num_gpus

        if self.stage == ZEROStage.STAGE_3:
            # Parameters + Gradients + Optimizer states 全部分片
            return param_mem / self.num_gpus + grad_mem / self.num_gpus + opt_mem / self.num_gpus

        return 0.0

    def memory_savings_ratio(self) -> float:
        """相对于无分片（stage 0）的内存节省比例。"""
        baseline = ZEROStageManager(
            self.num_gpus, self.model_params, ZEROStage.STAGE_0
        ).total_memory_per_gpu_mb()
        if baseline == 0:
            return 0.0
        return 1.0 - self.total_memory_per_gpu_mb() / baseline

    def communication_overhead_per_step(self, hidden_dim: int, num_layers: int) -> Dict[str, float]:
        """估算每个 optimizer step 的额外通信量（MB）。

        借鉴 DeepSpeed ZeRO 通信分析:
        - ZeRO-1: AllGather optimizer states = (K * Ψ / N) * (N-1) / N
        - ZeRO-2: + ReduceScatter gradients = 2 * Ψ * (N-1) / N
        - ZeRO-3: + AllGather params × num_layers × 2（forward + backward）
        """
        results: Dict[str, float] = {}

        # ZeRO-1: AllGather of partitioned optimizer states
        opt_bytes = self.model_params * self.K_FACTOR
        zero1_comm = opt_bytes * (self.num_gpus - 1) / (self.num_gpus**2)
        results["zero1_comm_mb"] = zero1_comm / (1024**2)

        # ZeRO-2: + ReduceScatter of gradients
        grad_bytes = self.model_params * self.bytes_per_param
        zero2_comm = zero1_comm + grad_bytes * (self.num_gpus - 1) / self.num_gpus
        results["zero2_comm_mb"] = zero2_comm / (1024**2)

        # ZeRO-3: + AllGather of parameters per layer (forward & backward)
        param_bytes = self.model_params * self.bytes_per_param
        zero3_comm = zero2_comm + param_bytes * 2 * (self.num_gpus - 1) / self.num_gpus
        results["zero3_comm_mb"] = zero3_comm / (1024**2)

        return results

    def get_stats(self) -> Dict:
        """返回 ZeRO 配置统计。"""
        return {
            "stage": self.stage.name,
            "num_gpus": self.num_gpus,
            "model_params_millions": round(self.model_params / 1e6, 2),
            "param_memory_mb": round(self.parameter_memory_mb(), 2),
            "gradient_memory_mb": round(self.gradient_memory_mb(), 2),
            "optimizer_state_memory_mb": round(self.optimizer_state_memory_mb(), 2),
            "total_memory_per_gpu_mb": round(self.total_memory_per_gpu_mb(), 2),
            "memory_savings_ratio": round(self.memory_savings_ratio(), 4),
        }


# ==============================================================================
# 8. WorkloadBalancer — 异构 GPU 负载均衡器
# ==============================================================================


class WorkloadBalancer:
    """异构 GPU 工作负载均衡器 — 借鉴 FlexFlow / Alpa 的自动并行化策略

    在异构 GPU 环境中（不同型号 GPU、不同显存大小、不同带宽），
    根据各 GPU 的计算能力和显存容量合理地分配工作量。

    分配策略:
    1. capacity: 按显存容量比例分配（显存敏感的 workload）
    2. speed: 按带宽/计算速度加权分配（计算敏感的 workload）
    3. hybrid: 综合考虑容量和速度
    """

    def __init__(
        self,
        device_capacities: Dict[int, float],
        device_bandwidths: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Args:
            device_capacities: {device_id: capacity (任意单位，如 GB 显存)}
            device_bandwidths: {device_id: bandwidth (任意单位，如 GB/s)}，可选
        """
        self.device_capacities = device_capacities
        self.device_bandwidths = device_bandwidths or {}
        self.device_ids = sorted(device_capacities.keys())

    def balance_by_capacity(self, total_work: int) -> Dict[int, int]:
        """按显存容量权重分配工作量。

        借鉴 FlexFlow: 显存较大的 GPU 分配更多的 micro-batch 或更大的序列。
        """
        total_capacity = sum(self.device_capacities.values())
        if total_capacity == 0:
            return {dev: 0 for dev in self.device_ids}

        allocation: Dict[int, int] = {}
        allocated_sum = 0

        for dev in self.device_ids:
            if dev == self.device_ids[-1]:
                # 最后一个设备承担余数（修正 rounding error）
                allocation[dev] = total_work - allocated_sum
            else:
                share = int(total_work * self.device_capacities[dev] / total_capacity)
                allocation[dev] = share
                allocated_sum += share

        return allocation

    def balance_by_speed(self, total_work: int) -> Dict[int, int]:
        """按带宽/计算速度权重分配工作量。

        带宽较高的 GPU 分配更多的计算任务。
        """
        if not self.device_bandwidths:
            return {dev: total_work // len(self.device_ids) for dev in self.device_ids}

        total_speed = sum(self.device_bandwidths.values())
        if total_speed == 0:
            return {dev: 0 for dev in self.device_ids}

        allocation: Dict[int, int] = {}
        allocated_sum = 0

        for dev in self.device_ids:
            if dev == self.device_ids[-1]:
                allocation[dev] = total_work - allocated_sum
            else:
                share = int(total_work * self.device_bandwidths[dev] / total_speed)
                allocation[dev] = share
                allocated_sum += share

        return allocation

    def balance_hybrid(self, total_work: int, capacity_weight: float = 0.5) -> Dict[int, int]:
        """综合考虑容量和速度的混合负载均衡。

        Args:
            capacity_weight: 容量权重（0~1），剩余权重分配给速度。
        """
        cap_alloc = self.balance_by_capacity(int(total_work * capacity_weight))
        speed_alloc = self.balance_by_speed(int(total_work * (1 - capacity_weight)))

        result: Dict[int, int] = {}
        for dev in self.device_ids:
            result[dev] = cap_alloc.get(dev, 0) + speed_alloc.get(dev, 0)

        return result

    def get_imbalance_score(self, allocation: Dict[int, int]) -> float:
        """计算负载不均衡度（0 表示完全均衡）。

        使用变异系数（CV = std/mean）衡量不均衡程度。
        """
        if not allocation:
            return 0.0
        values = list(allocation.values())
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance) / mean

    def get_stats(self) -> Dict:
        """返回负载均衡器统计。"""
        return {
            "num_devices": len(self.device_ids),
            "device_ids": self.device_ids,
            "capacities": self.device_capacities,
            "bandwidths": self.device_bandwidths,
        }


# ==============================================================================
# 9. NCCLCommManager — NCCL 通信管理
# ==============================================================================


class NCCLCommManager:
    """NCCL 通信管理器 — 基于 torch.distributed 的高效通信封装

    借鉴 DeepSpeed comm 模块和 Megatron-LM p2p_communication 模块:
    - AllReduce: 环形算法（大消息）或 Tree 算法（小消息）
    - AllGather: 收集所有 rank 的 tensor
    - ReduceScatter: 归约后分散到各 rank
    - P2P Send/Recv: 流水线并行的层间通信
    - Broadcast: 权重初始化时的广播

    在不直接依赖 NCCL Python 绑定的情况下，通过 torch.distributed
    接口实现所有通信原语。
    """

    def __init__(self, world_size: int = 1, rank: int = 0) -> None:
        self.world_size = world_size
        self.rank = rank

    @staticmethod
    def is_available() -> bool:
        """检查 torch.distributed 是否已初始化。"""
        try:
            import torch.distributed as dist

            return dist.is_available() and dist.is_initialized()
        except (ImportError, RuntimeError):
            return False

    def all_reduce(self, tensor: "torch.Tensor", op: str = "sum") -> "torch.Tensor":
        """AllReduce 操作 — 所有 rank 归约后得到相同结果。

        借鉴 NCCL: 小消息（<256KB）用 Tree，大消息用 Ring 算法。
        """
        import torch.distributed as dist

        if dist.is_initialized():
            reduce_op = getattr(dist.ReduceOp, op.upper(), dist.ReduceOp.SUM)
            dist.all_reduce(tensor, op=reduce_op)
        return tensor

    def all_gather(self, tensor: "torch.Tensor") -> List["torch.Tensor"]:
        """AllGather 操作 — 收集所有 rank 的数据到每个 rank。

        借鉴 DeepSpeed ZeRO-3: 每层 forward/backward 前 AllGather 参数。
        """
        import torch
        import torch.distributed as dist

        if dist.is_initialized():
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered, tensor)
            return gathered
        return [tensor]

    def reduce_scatter(self, tensor: "torch.Tensor", op: str = "sum") -> "torch.Tensor":
        """ReduceScatter 操作 — 归约后每个 rank 得到一部分结果。

        借鉴 DeepSpeed ZeRO-2: backward 后用 ReduceScatter 归约梯度。
        """
        import torch
        import torch.distributed as dist

        if dist.is_initialized():
            input_list = list(tensor.chunk(self.world_size))
            output = torch.empty_like(input_list[0])
            reduce_op = getattr(dist.ReduceOp, op.upper(), dist.ReduceOp.SUM)
            dist.reduce_scatter(output, input_list, op=reduce_op)
            return output
        return tensor.chunk(self.world_size)[0]

    def broadcast(self, tensor: "torch.Tensor", src: int = 0) -> "torch.Tensor":
        """Broadcast 操作 — 从 root rank 广播到所有 rank。"""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor

    def send(self, tensor: "torch.Tensor", dst: int) -> None:
        """P2P 发送 — 流水线并行 forward 输出传递。

        借鉴 Megatron-LM p2p_communication.send_forward()。
        """
        import torch.distributed as dist

        if dist.is_initialized():
            dist.send(tensor, dst=dst)

    def recv(self, tensor: "torch.Tensor", src: int) -> "torch.Tensor":
        """P2P 接收 — 流水线并行 forward 输入接收。"""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.recv(tensor, src=src)
        return tensor

    def barrier(self) -> None:
        """同步屏障 — 所有 rank 在此同步。"""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def estimate_allreduce_time(
        data_size_mb: float, bandwidth_gbs: float = 100.0, world_size: int = 8
    ) -> float:
        """估算 AllReduce 的执行时间。

        使用环形算法模型:
        - 总数据移动量 = 2 * (P-1) / P * data_size
        - 时间 = 总数据移动量 / bandwidth

        Args:
            data_size_mb: 数据大小（MB）
            bandwidth_gbs: 单向链路带宽（GB/s）
            world_size: GPU 数量
        Returns:
            估算的 AllReduce 延迟（毫秒）
        """
        if world_size <= 1:
            return 0.0
        data_transferred_mb = data_size_mb * 2 * (world_size - 1) / world_size
        return data_transferred_mb / bandwidth_gbs * 1000  # ms

    def get_stats(self) -> Dict:
        """返回通信管理器状态。"""
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "is_dist_initialized": self.is_available(),
        }


# ==============================================================================
# 10. HybridScheduler — 混合并行调度器
# ==============================================================================


class HybridScheduler:
    """混合并行（TP + PP + DP）调度器 — 借鉴 Megatron-LM + DeepSpeed

    将 TP、PP、DP 三种并行策略组合成一个完整的调度方案。

    借鉴 Megatron-LM 的 hybrid parallelism:
    1. 节点内使用 TP（NVLink 高带宽利于高频 AllReduce）
    2. 节点间使用 PP（减少跨机通信量，仅 P2P 层间传递）
    3. 整体使用 DP 做 batch 级并行

    典型配置示例（Megatron-LM + DeepSpeed 启发）:
    - 8 卡单机: TP=8
    - 16 卡双机: TP=8, PP=2
    - 32 卡四机: TP=8, PP=4
    """

    def __init__(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        num_microbatches: int = 1,
    ) -> None:
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.num_microbatches = num_microbatches

        self.world_size = tp_size * pp_size * dp_size

        # 子调度器
        self.tp_scheduler = TPScheduler(tp_size)
        self.pp_scheduler = PPScheduler(pp_size, num_microbatches)

    def is_valid(self) -> bool:
        """验证混合并行配置是否合理。"""
        return (
            self.tp_size > 0 and self.pp_size > 0 and self.dp_size > 0 and self.num_microbatches > 0
        )

    def pp_bubble_ratio(self) -> float:
        """PP 的 bubble 比例（全局视图）。"""
        return self.pp_scheduler.bubble_ratio()

    def pp_efficiency(self) -> float:
        """PP 的效率。"""
        return self.pp_scheduler.efficiency()

    def estimate_communication_overhead(
        self,
        num_layers: int,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Dict[str, float]:
        """估算混合并行的总通信开销（MB/step）。

        TP 通信: 每层多次 AllReduce（高带宽需求 → 适合 NVLink 节点内）
        PP 通信: 每 microbatch P2P（低带宽需求 → 适合跨机 InfiniBand）
        DP 通信: 每 optimizer step AllReduce 梯度（中等带宽需求）
        """
        # TP 通信
        tp_comm_mb = self.tp_scheduler.estimate_total_communication_per_step(
            num_layers, batch_size, seq_len, hidden_dim
        ) / (1024**2)

        # PP 通信（P2P 每层边界 activations）
        pp_comm_per_mb = batch_size * seq_len * hidden_dim * BYTES_PER_ELEMENT
        pp_comm_mb = pp_comm_per_mb * self.pp_size * self.num_microbatches / (1024**2)

        # DP 通信（梯度 AllReduce，每 optimizer step 1 次）
        # 简化为模型总参数量的通信
        dp_comm_mb = 0  # 由 ZEROStageManager 单独计算

        return {
            "tp_comm_mb_per_step": round(tp_comm_mb, 2),
            "pp_comm_mb_per_step": round(pp_comm_mb, 2),
            "dp_comm_mb_per_step": round(dp_comm_mb, 2),
        }

    def get_stats(self) -> Dict:
        """返回混合并行调度统计。"""
        return {
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "dp_size": self.dp_size,
            "world_size": self.world_size,
            "num_microbatches": self.num_microbatches,
            "pp_bubble_ratio": round(self.pp_bubble_ratio(), 4),
            "pp_efficiency": round(self.pp_efficiency(), 4),
        }


# ==============================================================================
# 11. 模拟推理引擎 — 端到端演示
# ==============================================================================


class SimulationEngine:
    """模拟推理引擎 — 将 Scheduler + BlockManager + MemoryPlanner 组装起来。

    提供端到端的推理模拟，展示 continuous batching 的完整流程。
    """

    def __init__(
        self,
        block_size: int = BLOCK_SIZE,
        num_blocks: int = 4096,
        num_layers: int = NUM_LAYERS,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 4096,
    ) -> None:
        self.block_manager = BlockManager(
            block_size=block_size, num_blocks=num_blocks, num_layers=num_layers
        )
        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        self.memory_planner = MemoryPlanner(
            total_memory_gb=80.0,  # 典型 A100 80GB
            model_size_gb=60.0,  # llama-70B 大约 60GB（量化后）
            block_size=block_size,
            num_layers=num_layers,
        )

        # 模拟统计
        self.total_requests_processed = 0
        self.total_tokens_generated = 0
        self.step_count = 0

    def add_single_request(self, request_id: str, prompt_len: int, max_output_len: int) -> None:
        """添加一个推理请求（单个序列，非 beam search）。"""
        prompt = list(range(prompt_len))  # 用连续的 token ID 占位
        seq = Sequence(
            seq_id=self.total_requests_processed,
            prompt=prompt,
            prompt_len=prompt_len,
            max_output_len=max_output_len,
        )
        sg = SequenceGroup(request_id=request_id, seqs=[seq])
        self.scheduler.add_request(sg)
        self.total_requests_processed += 1

    def step(self) -> Dict:
        """执行一步调度和推理模拟。

        Returns:
            本步的统计信息。
        """
        scheduled, ignored, is_prefill = self.scheduler.schedule()

        if not scheduled:
            return {
                "step": self.step_count,
                "num_scheduled_seqs": 0,
                "num_scheduled_groups": 0,
                "num_ignored": 0,
                "is_prefill": False,
                "num_completed": 0,
                "queue_sizes": self.scheduler.get_queue_sizes(),
                "block_usage": round(self.block_manager.get_usage_ratio(), 3),
            }

        # 模拟推理：每个 scheduled 序列生成了一个 token
        num_completed_before = self.scheduler.num_completed_requests
        self.scheduler.update_after_step(scheduled)
        num_completed = self.scheduler.num_completed_requests - num_completed_before

        self.step_count += 1

        result = {
            "step": self.step_count,
            "num_scheduled_seqs": sum(len(sg.seqs) for sg in scheduled),
            "num_scheduled_groups": len(scheduled),
            "num_ignored": len(ignored),
            "is_prefill": is_prefill,
            "num_completed": num_completed,
            "queue_sizes": self.scheduler.get_queue_sizes(),
            "block_usage": round(self.block_manager.get_usage_ratio(), 3),
        }

        # 更新 MemoryPlanner
        used_blocks = self.block_manager.num_blocks - self.block_manager.get_num_free_blocks()
        self.memory_planner.used_blocks = used_blocks

        return result

    def run_until_idle(self, max_steps: int = 10000) -> Dict:
        """持续运行调度直到所有请求处理完毕或达到最大步数。

        Returns:
            最终统计信息。
        """
        for _ in range(max_steps):
            queue_sizes = self.scheduler.get_queue_sizes()
            if queue_sizes["waiting"] == 0 and queue_sizes["running"] == 0:
                break
            self.step()

        return self.get_final_stats()

    def get_final_stats(self) -> Dict:
        """获取最终统计摘要。"""
        return {
            "total_steps": self.step_count,
            "total_requests": self.total_requests_processed,
            "completed_requests": self.scheduler.num_completed_requests,
            "preemptions": self.scheduler.num_preempted,
            "swapped_out": self.scheduler.num_swapped_out,
            "swapped_in": self.scheduler.num_swapped_in,
            "prefix_cache_hit_rate": round(self.block_manager.get_prefix_cache_hit_rate(), 4),
            "final_block_usage": round(self.block_manager.get_usage_ratio(), 3),
        }
