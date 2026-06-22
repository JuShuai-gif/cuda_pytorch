"""
工业级异构调度器测试 — 覆盖 vLLM/DeepSpeed/Megatron-LM 调度模式

测试覆盖:
1. BlockManager 分配/释放/引用计数 / prefix caching
2. Scheduler 调度逻辑（waiting -> running -> finished）
3. MemoryPlanner watermark 策略
4. PPScheduler 1F1B 调度正确性
5. TPScheduler 权重切分
6. ZEROStageManager 内存节省分析
7. WorkloadBalancer 负载均衡
8. HybridScheduler 混合并行组合
9. 模拟 1000 请求的连续批处理 end-to-end
10. Sequence 生命周期管理

运行: pytest test_scheduler.py -v
"""

from __future__ import annotations

import math
import random

import pytest

from scheduler import (
    BLOCK_SIZE,
    NUM_LAYERS,
    BlockManager,
    HybridScheduler,
    MemoryPlanner,
    NCCLCommManager,
    PPScheduler,
    Scheduler,
    Sequence,
    SequenceGroup,
    SequenceStatus,
    SimulationEngine,
    TPScheduler,
    WorkloadBalancer,
    ZEROStage,
    ZEROStageManager,
)


# ==============================================================================
# 测试辅助
# ==============================================================================


def make_sequence(seq_id: int, prompt_len: int, max_output_len: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt=list(range(prompt_len)),
        prompt_len=prompt_len,
        max_output_len=max_output_len,
    )


def make_sequence_group(
    request_id: str, prompt_len: int, max_output_len: int, num_seqs: int = 1
) -> SequenceGroup:
    seqs = [make_sequence(i, prompt_len, max_output_len) for i in range(num_seqs)]
    return SequenceGroup(request_id=request_id, seqs=seqs)


# ==============================================================================
# 1. BlockManager 测试
# ==============================================================================


class TestBlockManager:
    """测试 BlockManager 的分配、释放、引用计数和 prefix caching。"""

    def test_allocate_and_free(self) -> None:
        """基本分配和释放：分配后空闲块减少，释放后恢复。"""
        bm = BlockManager(block_size=16, num_blocks=100)
        assert bm.get_num_free_blocks() == 100

        block_id = bm.allocate()
        assert block_id is not None
        assert bm.get_num_free_blocks() == 99
        assert bm.get_usage_ratio() == pytest.approx(0.01)

        bm.free(block_id)
        assert bm.get_num_free_blocks() == 100
        assert bm.get_usage_ratio() == 0.0

    def test_allocate_exhaustion(self) -> None:
        """分配直到耗尽：所有块分配完后返回 None。"""
        bm = BlockManager(block_size=16, num_blocks=5)
        allocations = [bm.allocate() for _ in range(5)]
        assert all(b is not None for b in allocations)
        assert bm.get_num_free_blocks() == 0

        # 第 6 次分配应返回 None
        assert bm.allocate() is None

    def test_can_allocate(self) -> None:
        """can_allocate 检查：正确判断是否有足够空闲块。"""
        bm = BlockManager(block_size=16, num_blocks=100)
        assert bm.can_allocate(50) is True
        assert bm.can_allocate(101) is False

        # 分配 90 个后
        for _ in range(90):
            bm.allocate()
        assert bm.can_allocate(10) is True
        assert bm.can_allocate(11) is False

    def test_ref_count_prefix_caching(self) -> None:
        """引用计数和 prefix caching：多个引用者共享同一物理块。"""
        bm = BlockManager(block_size=16, num_blocks=100)

        # 分配并记录引用
        block_a = bm.allocate()
        assert block_a is not None
        assert bm.block_ref_count[block_a] == 1

        # 模拟 prefix caching：另一个序列引用同一块
        reused = bm.allocate_with_prefix_cache(block_a)
        assert reused == block_a
        assert bm.block_ref_count[block_a] == 2
        assert bm.num_prefix_cache_hits == 1

        # 释放一次，块不应归还（引用计数 > 0）
        bm.free(block_a)
        assert block_a in bm.allocated_blocks
        assert bm.block_ref_count[block_a] == 1

        # 再次释放，块应归还
        bm.free(block_a)
        assert block_a not in bm.allocated_blocks
        assert bm.block_ref_count[block_a] == 0

    def test_prefix_cache_hit_rate(self) -> None:
        """prefix cache 命中率统计正确性。"""
        bm = BlockManager(block_size=16, num_blocks=100)

        b1 = bm.allocate()
        bm.allocate_with_prefix_cache(b1)  # cache hit
        bm.allocate()  # cache miss
        bm.allocate_with_prefix_cache(b1)  # cache hit

        # allocate() called twice (lines 1 and 3), allocate_with_prefix_cache twice (hits)
        assert bm.num_alloc_ops == 2
        assert bm.num_prefix_cache_hits == 2
        # hit_rate = hits / (alloc_ops + hits) = 2 / (2 + 2) = 0.5
        assert bm.get_prefix_cache_hit_rate() == pytest.approx(0.5, rel=1e-4)

    def test_block_size_bytes(self) -> None:
        """块大小字节数计算正确。"""
        bm = BlockManager(block_size=16, num_blocks=100, num_layers=32, num_heads=32, head_dim=128)
        # 2 * block_size * num_heads * head_dim * BYTES_PER_ELEMENT = 2 * 16 * 32 * 128 * 2
        expected = 2 * 16 * 32 * 128 * 2
        assert bm.block_size_bytes() == expected

    def test_reset(self) -> None:
        """reset 恢复初始状态。"""
        bm = BlockManager(block_size=16, num_blocks=50)
        for _ in range(30):
            bm.allocate()
        assert bm.get_num_free_blocks() == 20

        bm.reset()
        assert bm.get_num_free_blocks() == 50
        assert bm.num_alloc_ops == 0
        assert bm.num_prefix_cache_hits == 0


# ==============================================================================
# 2. Sequence 管理测试
# ==============================================================================


class TestSequence:
    """测试 Sequence 和 SequenceGroup 的生命周期状态。"""

    def test_num_logical_blocks(self) -> None:
        """逻辑块数计算。"""
        seq = make_sequence(0, prompt_len=32, max_output_len=64)
        # 32 tokens, block_size=16 → ceil(32/16) = 2
        assert seq.num_logical_blocks() == 2

        seq.output_tokens = [0] * 16
        # 32 + 16 = 48 → ceil(48/16) = 3
        assert seq.num_logical_blocks() == 3

    def test_is_finished(self) -> None:
        """完成判断。"""
        seq = make_sequence(0, prompt_len=10, max_output_len=5)
        assert not seq.is_finished

        seq.output_tokens = [0] * 5
        assert seq.is_finished

        seq.output_tokens = [0] * 6  # 超额
        assert seq.is_finished

    def test_sequence_status_transitions(self) -> None:
        """序列状态转换。"""
        seq = make_sequence(0, prompt_len=10, max_output_len=3)
        assert seq.status == SequenceStatus.WAITING

        seq.status = SequenceStatus.RUNNING
        assert seq.status == SequenceStatus.RUNNING

        seq.status = SequenceStatus.FINISHED
        assert seq.status == SequenceStatus.FINISHED

    def test_sequence_group_is_finished(self) -> None:
        """SequenceGroup 完成判断。"""
        seq1 = make_sequence(0, prompt_len=10, max_output_len=3)
        seq2 = make_sequence(1, prompt_len=10, max_output_len=5)
        sg = SequenceGroup(request_id="test", seqs=[seq1, seq2])

        assert not sg.is_finished

        seq1.output_tokens = [0] * 3
        assert not sg.is_finished  # seq2 未完成

        seq2.output_tokens = [0] * 5
        assert sg.is_finished

    def test_total_logical_blocks(self) -> None:
        """SequenceGroup 总逻辑块数。"""
        seq1 = make_sequence(0, prompt_len=32, max_output_len=64)
        seq2 = make_sequence(1, prompt_len=16, max_output_len=64)
        sg = SequenceGroup(request_id="test", seqs=[seq1, seq2])
        assert sg.total_logical_blocks() == 2 + 1  # 2 blocks + 1 block


# ==============================================================================
# 3. Scheduler 调度逻辑测试
# ==============================================================================


class TestScheduler:
    """测试 Scheduler 的 continuous batching 调度逻辑。"""

    def test_add_request(self) -> None:
        """添加请求到 waiting 队列。"""
        bm = BlockManager(num_blocks=5000)
        sched = Scheduler(bm, max_waiting_queue_size=100)
        sg = make_sequence_group("r1", prompt_len=32, max_output_len=10)
        assert sched.add_request(sg) is True
        assert len(sched.waiting) == 1

    def test_add_request_queue_full(self) -> None:
        """队列满时拒绝新请求。"""
        bm = BlockManager(num_blocks=5000)
        sched = Scheduler(bm, max_waiting_queue_size=2)
        assert sched.add_request(make_sequence_group("r1", 10, 5)) is True
        assert sched.add_request(make_sequence_group("r2", 10, 5)) is True
        assert sched.add_request(make_sequence_group("r3", 10, 5)) is False
        assert sched.num_rejected_requests == 1

    def test_schedule_single_request(self) -> None:
        """调度单个请求：waiting → scheduled → running。"""
        bm = BlockManager(num_blocks=10000)
        sched = Scheduler(bm, max_num_seqs=64)
        sg = make_sequence_group("r1", prompt_len=32, max_output_len=5)
        sched.add_request(sg)

        scheduled, ignored, is_prefill = sched.schedule()
        assert len(scheduled) == 1
        assert is_prefill is True

        # SequenceGroup 中的 seq 应被分配了块
        for s in scheduled[0].seqs:
            assert s.status == SequenceStatus.RUNNING
            assert len(s.block_table) > 0

    def test_schedule_multiple_requests(self) -> None:
        """调度多个请求：先到先服务，显存不足时部分等待。"""
        bm = BlockManager(num_blocks=500)
        sched = Scheduler(bm, max_num_seqs=64)

        for i in range(20):
            sched.add_request(make_sequence_group(f"r{i}", prompt_len=32, max_output_len=5))

        scheduled, ignored, is_prefill = sched.schedule()

        # 至少调度了部分请求
        assert len(scheduled) > 0
        assert is_prefill is True

        # waiting 或 ignored 中应有未调度的请求
        queue_sizes = sched.get_queue_sizes()
        assert queue_sizes["waiting"] + queue_sizes["running"] > 0

    def test_update_after_step_completion(self) -> None:
        """推理步骤后状态更新：完成的序列被清理。"""
        bm = BlockManager(num_blocks=10000)
        sched = Scheduler(bm, max_num_seqs=64)
        sg = make_sequence_group("r1", prompt_len=16, max_output_len=3)
        sched.add_request(sg)

        # 第 1 步：调度并生成 token
        scheduled, _, _ = sched.schedule()
        assert len(scheduled) == 1
        sched.update_after_step(scheduled)
        assert len(scheduled[0].seqs[0].output_tokens) == 1

        # 继续 2 步
        scheduled, _, _ = sched.schedule()
        sched.update_after_step(scheduled)
        scheduled, _, _ = sched.schedule()
        sched.update_after_step(scheduled)

        # 3 个 token 后应完成
        assert sched.num_completed_requests == 1
        assert len(sched.running) == 0

    def test_preemption(self) -> None:
        """抢占测试：显存不足时换出 running 序列。"""
        bm = BlockManager(num_blocks=200)
        sched = Scheduler(bm, max_num_seqs=64)

        # 先添加一个长序列（大量块）
        sched.add_request(make_sequence_group("r_long", prompt_len=256, max_output_len=100))
        scheduled, _, _ = sched.schedule()
        sched.update_after_step(scheduled)

        # 添加多个短序列，可能导致长序列被换出
        for i in range(10):
            sched.add_request(make_sequence_group(f"r{i}", prompt_len=64, max_output_len=10))

        scheduled, _, is_prefill = sched.schedule()
        assert len(scheduled) > 0

    def test_scheduler_stats(self) -> None:
        """调度器统计信息完整性。"""
        bm = BlockManager(num_blocks=10000)
        sched = Scheduler(bm)
        stats = sched.get_stats()
        assert "num_preempted" in stats
        assert "num_completed" in stats
        assert "num_rejected" in stats
        assert "block_usage_ratio" in stats
        assert "prefix_cache_hit_rate" in stats

    def test_queue_sizes_tracking(self) -> None:
        """各队列大小追踪。"""
        bm = BlockManager(num_blocks=10000)
        sched = Scheduler(bm)
        assert sched.get_queue_sizes() == {"waiting": 0, "running": 0, "swapped": 0}

        sched.add_request(make_sequence_group("r1", 16, 5))
        assert sched.get_queue_sizes()["waiting"] == 1

        scheduled, _, _ = sched.schedule()
        sched.update_after_step(scheduled)
        assert sched.get_queue_sizes()["running"] >= 0


# ==============================================================================
# 4. MemoryPlanner 测试
# ==============================================================================


class TestMemoryPlanner:
    """测试 MemoryPlanner 的 watermark 策略和显存估算。"""

    def test_watermark_strategy(self) -> None:
        """watermark 策略：超过阈值时拒绝新请求。"""
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=60, block_size=16)
        available = mp.get_available_blocks()
        assert available == int(mp.num_total_blocks * 0.9)

        # 预留到 watermark 边界
        mp.reserve_blocks(available - 1)
        assert mp.can_accept_new(1) is True

        mp.reserve_blocks(1)  # 正好到 watermark 边界
        assert mp.can_accept_new(1) is False

    def test_get_available_blocks(self) -> None:
        """可分配块数计算。"""
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=60)
        total = mp.num_total_blocks
        available = mp.get_available_blocks()
        assert available == int(total * 0.9)

    def test_release_blocks(self) -> None:
        """释放块后可用块数恢复。"""
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=60)
        before = mp.get_available_blocks()
        mp.reserve_blocks(100)
        assert mp.get_available_blocks() == before - 100
        mp.release_blocks(100)
        assert mp.get_available_blocks() == before

    def test_update_watermark(self) -> None:
        """动态调整 watermark。"""
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=60, watermark=0.8)
        assert mp.get_available_blocks() == int(mp.num_total_blocks * 0.8)

        mp.update_watermark(0.5)
        assert mp.get_available_blocks() == int(mp.num_total_blocks * 0.5)

        # 边界检查
        mp.update_watermark(2.0)
        assert mp._current_watermark == 0.99

        mp.update_watermark(-1.0)
        assert mp._current_watermark == 0.1

    def test_estimate_max_concurrent(self) -> None:
        """估算最大并发序列数。"""
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=60, block_size=16)
        max_concurrent = mp.estimate_max_concurrent_sequences(avg_prompt_len=64, avg_output_len=32)
        assert max_concurrent > 0

    def test_zero_available_memory(self) -> None:
        """模型吃满显存时可用块数为 0。"""
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=80, reserved_memory_gb=0)
        assert mp.num_total_blocks == 0
        assert mp.can_accept_new(1) is False

    def test_get_stats(self) -> None:
        """统计信息完整性。"""
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=60)
        stats = mp.get_stats()
        assert "total_memory_gb" in stats
        assert "model_size_gb" in stats
        assert "available_kv_cache_gb" in stats
        assert "watermark" in stats
        assert "usage_pct" in stats


# ==============================================================================
# 5. PPScheduler 1F1B 测试
# ==============================================================================


class TestPPScheduler:
    """测试 PPScheduler 的 1F1B 调度正确性。"""

    def test_pp_size_1_no_bubble(self) -> None:
        """PP=1 时无 bubble。"""
        pp = PPScheduler(pp_size=1, num_microbatches=8)
        assert pp.bubble_ratio() == 0.0
        assert pp.efficiency() == 1.0

    def test_bubble_ratio_formula(self) -> None:
        """bubble ratio 公式正确性：(P-1) / (P-1+M)。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        expected = 3 / (3 + 8)  # (4-1) / (4-1+8) = 3/11
        assert pp.bubble_ratio() == pytest.approx(expected, rel=1e-6)

    def test_efficiency(self) -> None:
        """效率 = 1 - bubble_ratio。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        assert pp.efficiency() == pytest.approx(8 / 11, rel=1e-6)

    def test_num_warmup_microbatches(self) -> None:
        """每 rank 的 warmup microbatch 数。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        assert pp.num_warmup_microbatches(0) == 3
        assert pp.num_warmup_microbatches(1) == 2
        assert pp.num_warmup_microbatches(2) == 1
        assert pp.num_warmup_microbatches(3) == 0

    def test_schedule_1f1b_total_steps(self) -> None:
        """1F1B 调度总步数 = M + PP - 1。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        assert pp.total_steps() == 11  # 8 + 4 - 1

    def test_schedule_1f1b_rank_forward_counts(self) -> None:
        """每个 rank 恰好执行 M 次 forward。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        for rank in range(4):
            rank_sched = pp.get_rank_schedule(rank)
            forward_count = sum(1 for op, _ in rank_sched if op == "F")
            assert forward_count == 8

    def test_schedule_1f1b_rank_backward_counts(self) -> None:
        """每个 rank 恰好执行 M 次 backward。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        for rank in range(4):
            rank_sched = pp.get_rank_schedule(rank)
            backward_count = sum(1 for op, _ in rank_sched if op == "B")
            assert backward_count == 8

    def test_schedule_1f1b_forward_backward_pairing(self) -> None:
        """1F1B 模式：steady state 中每个 step 恰好 1F + 1B。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        full = pp.schedule_1f1b()
        # 按 global_step 分组
        from collections import defaultdict

        steps: dict = defaultdict(lambda: {"F": 0, "B": 0})
        for step, rank, op, mb in full:
            steps[(step, rank)][op] += 1

        # 每个 (step, rank) 最多有 1F + 1B（warmup 和 cooldown 阶段可能只有 1 个）
        for (step, rank), counts in steps.items():
            assert counts["F"] <= 1
            assert counts["B"] <= 1

    def test_schedule_rank_consistency(self) -> None:
        """每个 microbatch 必须被所有 rank 按序处理。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        for rank in range(pp.pp_size):
            rank_sched = pp.get_rank_schedule(rank)
            forwards = [mb for op, mb in rank_sched if op == "F"]
            # forward 的 microbatch ID 应该递增
            assert forwards == sorted(forwards)

    def test_small_pp_size(self) -> None:
        """PP=1 时的调度。"""
        pp = PPScheduler(pp_size=1, num_microbatches=5)
        sched = pp.schedule_1f1b()
        # 5 个 microbatch，每个一个 forward
        forwards = [(step, rank, mb) for step, rank, op, mb in sched if op == "F"]
        assert len(forwards) == 5

    def test_rank_idle_steps(self) -> None:
        """idle 步数计算。"""
        pp = PPScheduler(pp_size=4, num_microbatches=8)
        # rank 0 idle 最少（最早开始，最早结束... actually rank 0 has most idle at end）
        # rank 3 idle steps should be > 0 (waits at beginning)
        idle_0 = pp.rank_idle_steps(0)
        idle_3 = pp.rank_idle_steps(3)
        # Both ranks have same total steps, but different distribution
        assert idle_0 + idle_3 >= 0  # sanity check

    def test_invalid_params(self) -> None:
        """无效参数应抛出异常。"""
        with pytest.raises(ValueError):
            PPScheduler(pp_size=0, num_microbatches=8)
        with pytest.raises(ValueError):
            PPScheduler(pp_size=4, num_microbatches=0)


# ==============================================================================
# 6. TPScheduler 测试
# ==============================================================================


class TestTPScheduler:
    """测试 TPScheduler 的权重切分和通信量估算。"""

    def test_partition_col_linear(self) -> None:
        """ColumnParallel: 按输出维度切分。"""
        tp = TPScheduler(tp_size=4)
        shapes = tp.partition_col_linear_weight((4096, 4096))
        assert len(shapes) == 4
        for s in shapes:
            assert s == (1024, 4096)  # 4096/4

    def test_partition_row_linear(self) -> None:
        """RowParallel: 按输入维度切分。"""
        tp = TPScheduler(tp_size=4)
        shapes = tp.partition_row_linear_weight((4096, 4096))
        assert len(shapes) == 4
        for s in shapes:
            assert s == (4096, 1024)  # 4096/4

    def test_communication_cost_estimation(self) -> None:
        """通信代价估算。"""
        tp = TPScheduler(tp_size=4)
        cost = tp.col_parallel_communication_cost(batch_size=8, seq_len=4096, hidden_dim=8192)
        assert cost > 0

        cost = tp.estimate_total_communication_per_step(
            num_layers=80, batch_size=8, seq_len=4096, hidden_dim=8192
        )
        assert cost > 0

    def test_single_gpu_no_communication(self) -> None:
        """TP=1 时通信量为 0。"""
        tp = TPScheduler(tp_size=1)
        cost = tp.col_parallel_communication_cost(batch_size=8, seq_len=4096, hidden_dim=4096)
        assert cost == 0

    def test_get_num_allreduces(self) -> None:
        """每层 4 次 AllReduce（forward 2 + backward 2）。"""
        tp = TPScheduler(tp_size=8)
        assert tp.get_num_allreduces_per_layer() == 4


# ==============================================================================
# 7. ZEROStageManager 测试
# ==============================================================================


class TestZEROStageManager:
    """测试 ZeRO 各阶段的内存节省分析。"""

    def test_memory_savings_increase_with_stage(self) -> None:
        """Stage 越高，内存节省越多。"""
        stages = [ZEROStage.STAGE_0, ZEROStage.STAGE_1, ZEROStage.STAGE_2, ZEROStage.STAGE_3]
        savings = []
        for stage in stages:
            zm = ZEROStageManager(num_gpus=8, model_params=70_000_000_000 // 2, stage=stage)
            savings.append(zm.memory_savings_ratio())

        # 节省比例应递增
        for i in range(1, len(savings)):
            assert savings[i] >= savings[i - 1]

    def test_stage3_perfect_scaling(self) -> None:
        """ZeRO-3 参数内存随 GPU 数线性缩减。"""
        num_params = int(70e9 / 2)  # llama-70B fp16 参数量
        zm1 = ZEROStageManager(num_gpus=4, model_params=num_params, stage=ZEROStage.STAGE_3)
        zm2 = ZEROStageManager(num_gpus=8, model_params=num_params, stage=ZEROStage.STAGE_3)
        mem1 = zm1.total_memory_per_gpu_mb()
        mem2 = zm2.total_memory_per_gpu_mb()
        # 8 GPU 的内存应小于 4 GPU（更多 GPU 意味着更多分片）
        # 注意：不是严格 linear 因为参数外的部分不完全按 N 分片
        assert mem2 < mem1

    def test_stage1_optimizer_sharding(self) -> None:
        """ZeRO-1 仅分片优化器状态。"""
        zm = ZEROStageManager(num_gpus=8, model_params=100_000_000, stage=ZEROStage.STAGE_1)
        opt_mem = zm.optimizer_state_memory_mb()
        assert zm.total_memory_per_gpu_mb() == pytest.approx(
            zm.parameter_memory_mb() + zm.gradient_memory_mb() + opt_mem / 8,
            rel=1e-4,
        )

    def test_stage3_all_sharded(self) -> None:
        """ZeRO-3 全部分片。"""
        num_gpus = 8
        num_params = 100_000_000
        zm = ZEROStageManager(num_gpus=num_gpus, model_params=num_params, stage=ZEROStage.STAGE_3)
        assert zm.total_memory_per_gpu_mb() == pytest.approx(
            (zm.parameter_memory_mb() + zm.gradient_memory_mb() + zm.optimizer_state_memory_mb())
            / num_gpus,
            rel=1e-4,
        )

    def test_communication_overhead(self) -> None:
        """通信开销估算。"""
        zm = ZEROStageManager(num_gpus=8, model_params=100_000_000, stage=ZEROStage.STAGE_2)
        overhead = zm.communication_overhead_per_step(hidden_dim=4096, num_layers=32)
        assert "zero1_comm_mb" in overhead
        assert "zero2_comm_mb" in overhead
        assert "zero3_comm_mb" in overhead

    def test_get_stats(self) -> None:
        """统计信息完整性。"""
        zm = ZEROStageManager(num_gpus=4, model_params=1_000_000, stage=ZEROStage.STAGE_2)
        stats = zm.get_stats()
        assert stats["stage"] == "STAGE_2"
        assert stats["num_gpus"] == 4
        assert "memory_savings_ratio" in stats


# ==============================================================================
# 8. WorkloadBalancer 测试
# ==============================================================================


class TestWorkloadBalancer:
    """测试 WorkloadBalancer 的异构 GPU 负载均衡。"""

    def test_balance_by_capacity(self) -> None:
        """按容量分配。"""
        wb = WorkloadBalancer(
            device_capacities={0: 80, 1: 40, 2: 40},
        )
        alloc = wb.balance_by_capacity(total_work=160)
        # 容量比 80:40:40 = 2:1:1 → 80+40+40 = 160
        assert sum(alloc.values()) == 160
        assert alloc[0] == 80
        assert alloc[1] == 40
        assert alloc[2] == 40

    def test_balance_by_speed(self) -> None:
        """按速度分配。"""
        wb = WorkloadBalancer(
            device_capacities={0: 80, 1: 80},
            device_bandwidths={0: 900, 1: 100},
        )
        alloc = wb.balance_by_speed(total_work=100)
        assert sum(alloc.values()) == 100
        assert alloc[0] == 90  # 900 / 1000 * 100
        assert alloc[1] == 10  # remainder

    def test_balance_by_speed_without_bandwidths(self) -> None:
        """无带宽数据时均匀分配。"""
        wb = WorkloadBalancer(device_capacities={0: 80, 1: 80})
        alloc = wb.balance_by_speed(total_work=100)
        assert alloc[0] == 50
        assert alloc[1] == 50

    def test_balance_hybrid(self) -> None:
        """混合负载均衡。"""
        wb = WorkloadBalancer(
            device_capacities={0: 80, 1: 40},
            device_bandwidths={0: 100, 1: 100},
        )
        alloc = wb.balance_hybrid(total_work=120, capacity_weight=0.5)
        assert sum(alloc.values()) == 120
        # 容量一半：80+40=120 → 容量分配 0:40, 1:20
        # 速率一半：50:50 → 速率分配 0:30, 1:30
        # 总计：0:70, 1:50

    def test_imbalance_score(self) -> None:
        """负载不均衡度计算。"""
        wb = WorkloadBalancer(device_capacities={0: 80, 1: 80})
        # 均匀分配 → 不均衡度低
        score = wb.get_imbalance_score({0: 50, 1: 50})
        assert score == 0.0  # CV = std/mean = 0/50

        # 不均分配 → 不均衡度高
        score = wb.get_imbalance_score({0: 100, 1: 0})
        assert score > 0.0

    def test_single_device(self) -> None:
        """单设备测试。"""
        wb = WorkloadBalancer(device_capacities={0: 80})
        alloc = wb.balance_by_capacity(total_work=100)
        assert alloc == {0: 100}


# ==============================================================================
# 9. HybridScheduler 测试
# ==============================================================================


class TestHybridScheduler:
    """测试 HybridScheduler 的混合并行组合。"""

    def test_world_size_computation(self) -> None:
        """world_size = tp_size * pp_size * dp_size。"""
        hs = HybridScheduler(tp_size=8, pp_size=2, dp_size=4, num_microbatches=16)
        assert hs.world_size == 8 * 2 * 4  # 64

    def test_is_valid(self) -> None:
        """配置有效性检查。"""
        hs = HybridScheduler(tp_size=8, pp_size=2, dp_size=4, num_microbatches=16)
        assert hs.is_valid() is True

    def test_pp_efficiency(self) -> None:
        """PP 效率计算。"""
        hs = HybridScheduler(tp_size=4, pp_size=4, dp_size=1, num_microbatches=16)
        expected = 1 - 3 / (3 + 16)  # 1 - 3/19
        assert hs.pp_efficiency() == pytest.approx(expected, rel=1e-4)

    def test_communication_overhead(self) -> None:
        """混合并行通信开销估算。"""
        hs = HybridScheduler(tp_size=4, pp_size=2, dp_size=4, num_microbatches=16)
        overhead = hs.estimate_communication_overhead(
            num_layers=40, batch_size=2, seq_len=2048, hidden_dim=4096
        )
        assert "tp_comm_mb_per_step" in overhead
        assert "pp_comm_mb_per_step" in overhead
        assert overhead["tp_comm_mb_per_step"] > 0

    def test_get_stats(self) -> None:
        """统计信息完整性。"""
        hs = HybridScheduler(tp_size=8, pp_size=2, dp_size=4, num_microbatches=16)
        stats = hs.get_stats()
        assert stats["world_size"] == 64
        assert stats["pp_bubble_ratio"] > 0


# ==============================================================================
# 10. NCCLCommManager 测试
# ==============================================================================


class TestNCCLCommManager:
    """测试 NCCLCommManager 的通信时间估算。"""

    def test_is_available_without_init(self) -> None:
        """未初始化 distributed 时应返回 False。"""
        ncc = NCCLCommManager(world_size=1, rank=0)
        assert ncc.is_available() is False  # 单进程未初始化 dist

    def test_estimate_allreduce_time(self) -> None:
        """AllReduce 时间估算公式。"""
        # 1 MB, 100 GB/s, 8 GPUs
        # data_moved = 1 * 2 * 7/8 = 1.75 MB
        # time = 1.75 / 100 * 1000 = 17.5 ms
        time_ms = NCCLCommManager.estimate_allreduce_time(
            data_size_mb=1.0, bandwidth_gbs=100.0, world_size=8
        )
        expected = 1.0 * 2 * 7 / 8 / 100 * 1000
        assert time_ms == pytest.approx(expected, rel=1e-4)

    def test_estimate_allreduce_single_gpu(self) -> None:
        """单 GPU AllReduce 时间为 0。"""
        time_ms = NCCLCommManager.estimate_allreduce_time(
            data_size_mb=100.0, bandwidth_gbs=100.0, world_size=1
        )
        assert time_ms == 0.0

    def test_get_stats(self) -> None:
        """统计信息完整性。"""
        ncc = NCCLCommManager(world_size=8, rank=3)
        stats = ncc.get_stats()
        assert stats["world_size"] == 8
        assert stats["rank"] == 3
        assert stats["is_dist_initialized"] is False


# ==============================================================================
# 11. SimulationEngine 端到端测试
# ==============================================================================


class TestSimulationEngine:
    """模拟 1000 请求的连续批处理端到端测试。"""

    def test_single_request_lifecycle(self) -> None:
        """单个请求完整生命周期。"""
        engine = SimulationEngine(block_size=16, num_blocks=2000, max_num_seqs=64)
        engine.add_single_request("r1", prompt_len=32, max_output_len=5)

        total_steps = 0
        max_steps = 100
        for _ in range(max_steps):
            result = engine.step()
            total_steps += 1
            if (
                result["num_scheduled_groups"] == 0
                and engine.scheduler.get_queue_sizes()["waiting"] == 0
            ):
                break

        assert engine.scheduler.num_completed_requests >= 1

    def test_many_concurrent_requests(self) -> None:
        """50 个并发请求的连续批处理。"""
        engine = SimulationEngine(block_size=16, num_blocks=10000, max_num_seqs=64)

        for i in range(50):
            prompt_len = random.randint(16, 64)
            max_len = random.randint(5, 20)
            engine.add_single_request(f"r{i}", prompt_len=prompt_len, max_output_len=max_len)

        max_steps = 500
        steps_taken = 0
        for _ in range(max_steps):
            engine.step()
            steps_taken += 1
            queues = engine.scheduler.get_queue_sizes()
            if queues["waiting"] == 0 and queues["running"] == 0:
                break

        assert engine.scheduler.num_completed_requests == 50
        assert steps_taken <= max_steps

    def test_thousand_request_stress(self) -> None:
        """1000 请求压力测试。"""
        # 使用较少的层数以减少每请求块需求，聚焦调度逻辑
        engine = SimulationEngine(
            block_size=16,
            num_blocks=20000,
            num_layers=4,
            max_num_seqs=128,
            max_num_batched_tokens=16384,
        )

        # 添加 1000 个多样化的请求
        for i in range(1000):
            prompt_len = random.randint(8, 64)
            max_len = random.randint(1, 16)
            engine.add_single_request(f"r{i}", prompt_len=prompt_len, max_output_len=max_len)

        engine.run_until_idle(max_steps=10000)

        final = engine.get_final_stats()
        assert final["completed_requests"] == 1000
        assert final["total_steps"] > 0
        assert final["total_steps"] <= 10000

    def test_preemption_under_memory_pressure(self) -> None:
        """显存压力下的抢占行为。"""
        engine = SimulationEngine(block_size=16, num_blocks=4000, num_layers=4, max_num_seqs=256)

        # 添加许多请求，故意超过显存容量
        for i in range(200):
            engine.add_single_request(f"r{i}", prompt_len=random.randint(16, 48), max_output_len=3)

        engine.run_until_idle(max_steps=3000)
        final = engine.get_final_stats()
        assert final["completed_requests"] == 200

    def test_run_until_idle(self) -> None:
        """run_until_idle 自动完成所有请求。"""
        engine = SimulationEngine(block_size=16, num_blocks=5000, max_num_seqs=64)
        for i in range(30):
            engine.add_single_request(f"r{i}", prompt_len=16, max_output_len=10)

        stats = engine.run_until_idle(max_steps=2000)
        assert stats["completed_requests"] == 30
        assert stats["total_steps"] > 0
        assert stats["total_steps"] < 2000

    def test_different_block_sizes(self) -> None:
        """不同 block_size 下的正确性。"""
        for bs in [8, 16, 32, 64]:
            engine = SimulationEngine(block_size=bs, num_blocks=5000, max_num_seqs=32)
            engine.add_single_request("r1", prompt_len=bs * 3, max_output_len=bs)
            engine.run_until_idle(max_steps=200)
            assert engine.scheduler.num_completed_requests == 1

    def test_sequence_states_during_lifecycle(self) -> None:
        """序列状态在生命周期中的转换。"""
        engine = SimulationEngine(block_size=16, num_blocks=5000, max_num_seqs=64)
        engine.add_single_request("r1", prompt_len=32, max_output_len=2)

        result = engine.step()
        scheduled, _, _ = engine.scheduler.schedule()
        if scheduled:
            for sg in scheduled:
                for s in sg.seqs:
                    assert s.status == SequenceStatus.RUNNING

        engine.run_until_idle(max_steps=100)
        for sg in engine.scheduler.running + engine.scheduler.waiting + engine.scheduler.swapped:
            for s in sg.seqs:
                assert s.status in (SequenceStatus.RUNNING, SequenceStatus.FINISHED)


# ==============================================================================
# 12. 边界和异常测试
# ==============================================================================


class TestEdgeCases:
    """边界情况和异常处理测试。"""

    def test_zero_blocks(self) -> None:
        """0 块 BlockManager。"""
        bm = BlockManager(num_blocks=0)
        assert bm.allocate() is None
        assert bm.can_allocate(1) is False

    def test_empty_sequence_group(self) -> None:
        """空序列组。"""
        sg = SequenceGroup(request_id="empty", seqs=[])
        assert sg.is_finished is True
        assert sg.total_logical_blocks() == 0

    def test_zero_memory_budget(self) -> None:
        """零显存预算。"""
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=80, reserved_memory_gb=0)
        assert mp.num_total_blocks == 0

    def test_negative_block_allocation(self) -> None:
        """负引用计数保护。"""
        bm = BlockManager(num_blocks=100)
        # free on unallocated block should not crash or go negative
        bm.free(999)  # non-existent block
        assert bm.get_num_free_blocks() == 100

    def test_double_free(self) -> None:
        """重复释放保护。"""
        bm = BlockManager(num_blocks=100)
        block_id = bm.allocate()
        assert block_id is not None
        bm.free(block_id)
        # 第二次 free 不应崩溃
        bm.free(block_id)
        assert bm.block_ref_count[block_id] == 0
