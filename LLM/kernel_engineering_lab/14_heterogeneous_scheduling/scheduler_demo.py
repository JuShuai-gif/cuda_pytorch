#!/usr/bin/env python3
"""
异构调度演示 — 借鉴 vLLM/DeepSpeed/Megatron-LM 设计模式

演示内容:
1. BlockManager 的 prefix caching 效果
2. Scheduler continuous batching 完整流程
3. Pipeline parallelism 1F1B 调度与 bubble ratio 计算
4. ZeRO 各阶段内存节省对比
5. 不同调度策略的吞吐量对比
6. WorkloadBalancer 异构 GPU 负载分配
7. MemoryPlanner watermark 策略影响

运行方式:
    python scheduler_demo.py --demo all        # 运行所有演示
    python scheduler_demo.py --demo block      # BlockManager 演示
    python scheduler_demo.py --demo batch      # Continuous batching 演示
    python scheduler_demo.py --demo pp         # Pipeline parallelism 演示
    python scheduler_demo.py --demo zero       # ZeRO 内存节省演示
    python scheduler_demo.py --demo balancer   # 负载均衡演示
    python scheduler_demo.py --demo throughput # 吞吐量对比
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Dict, List

from scheduler import (
    BlockManager,
    HybridScheduler,
    MemoryPlanner,
    PPScheduler,
    Scheduler,
    Sequence,
    SequenceGroup,
    SimulationEngine,
    TPScheduler,
    WorkloadBalancer,
    ZEROStage,
    ZEROStageManager,
)


def print_header(title: str) -> None:
    """打印格式化的标题。"""
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subheader(title: str) -> None:
    """打印格式化的副标题。"""
    print(f"\n  --- {title} ---")


# ==============================================================================
# 1. BlockManager prefix caching 效果演示
# ==============================================================================


def demo_block_manager() -> None:
    """展示 BlockManager 的 prefix caching 效果。

    借鉴 vLLM: 当多个请求共享相同的 prompt 前缀时，
    后续请求可以直接复用前面的 KV cache 块，避免重复计算。
    """
    print_header("BlockManager — PagedAttention KV Cache 块管理（借鉴 vLLM）")

    bm = BlockManager(block_size=16, num_blocks=256)

    # 基本信息
    print_subheader("块参数")
    print(f"  block_size:        {bm.block_size} tokens/block")
    print(f"  num_blocks:        {bm.num_blocks}")
    print(f"  num_layers:        {bm.num_layers}")
    print(f"  bytes_per_block:   {bm.block_size_bytes() / 1024:.1f} KB")
    print(f"  total_kv_cache_mb: {bm.block_size_bytes() * bm.num_blocks / (1024**2):.1f} MB")

    # 模拟 prefix caching 场景
    print_subheader("Prefix Caching 效果演示")
    print("  场景: 3 个请求共享相同的前 32-token prompt 前缀")

    # 请求 1: 分配 3 个块（prompt=48 tokens）
    # 请求 2: 共享前 2 个块（prefix=32 tokens，相同前缀）
    # 请求 3: 共享前 2 个块

    prefix_blocks = [bm.allocate() for _ in range(2)]  # 前缀（32 tokens）
    print(f"\n  前缀分配: 2 个物理块（block {prefix_blocks[0]}, {prefix_blocks[1]}）")
    print(f"  空闲块: {bm.get_num_free_blocks()}")

    # 序列 A: 完整使用前 2 个块，额外分配 1 个
    seq_a_suffix = bm.allocate()
    print(f"\n  序列 A: 复用前缀 + 额外分配 block {seq_a_suffix}")
    print(f"  空闲块: {bm.get_num_free_blocks()}")
    assert prefix_blocks[0] is not None and prefix_blocks[1] is not None

    # 序列 B: 使用 prefix caching 复用前缀块
    reused_b0 = bm.allocate_with_prefix_cache(prefix_blocks[0])
    reused_b1 = bm.allocate_with_prefix_cache(prefix_blocks[1])
    seq_b_suffix = bm.allocate()
    print(f"\n  序列 B: prefix cache 复用 block {reused_b0}, {reused_b1}")
    print(f"         额外分配 block {seq_b_suffix}")
    assert reused_b0 == prefix_blocks[0]
    assert reused_b1 == prefix_blocks[1]
    print(f"  空闲块: {bm.get_num_free_blocks()}")

    # 序列 C: 也复用 prefix
    bm.allocate_with_prefix_cache(prefix_blocks[0])
    bm.allocate_with_prefix_cache(prefix_blocks[1])
    seq_c_suffix = bm.allocate()

    print(f"\n  序列 C: prefix cache 复用相同前缀块，额外分配 block {seq_c_suffix}")
    print(f"  空闲块: {bm.get_num_free_blocks()}")

    # 统计
    print_subheader("Prefix Cache 统计")
    print(f"  分配操作次数:      {bm.num_alloc_ops}")
    print(f"  Cache 命中次数:    {bm.num_prefix_cache_hits}")
    print(f"  Cache 命中率:      {bm.get_prefix_cache_hit_rate():.1%}")
    print(f"  最终使用率:        {bm.get_usage_ratio():.1%}")

    # 对比：不使用 prefix caching 的情况
    blocks_without_cache = 3 * 3  # 每个序列 3 个块
    blocks_with_cache = 2 + 1 + 1 + 1  # 前缀 2 + 3 个后缀
    print(f"\n  不使用 prefix cache 需要: {blocks_without_cache} 块")
    print(f"  使用 prefix cache 需要:   {blocks_with_cache} 块")
    print(
        f"  节省:                     {blocks_without_cache - blocks_with_cache} 块 ("
        f"{(blocks_without_cache - blocks_with_cache) / blocks_without_cache:.0%})"
    )


# ==============================================================================
# 2. Continuous Batching 流程演示
# ==============================================================================


def demo_continuous_batching() -> None:
    """展示 continuous batching 的完整流程。

    借鉴 vLLM: 请求到达后进入 waiting 队列，调度器按步取请求、
    分配 KV cache、执行推理、完成释放。
    """
    print_header("Scheduler — Continuous Batching 调度流程（借鉴 vLLM）")

    engine = SimulationEngine(
        block_size=16, num_blocks=2000, max_num_seqs=32, max_num_batched_tokens=4096
    )

    # 添加一组多样化的请求
    random.seed(42)
    requests = [
        ("short_1", 32, 8),
        ("short_2", 32, 6),
        ("medium_1", 64, 16),
        ("medium_2", 64, 20),
        ("long_1", 128, 32),
        ("long_2", 128, 24),
        ("burst_1", 16, 4),
        ("burst_2", 16, 5),
        ("burst_3", 16, 3),
        ("burst_4", 16, 7),
    ]

    for req_id, prompt_len, max_len in requests:
        engine.add_single_request(req_id, prompt_len, max_len)

    print(f"\n  添加了 {len(requests)} 个请求到 waiting 队列")
    print(f"  队列状态: {engine.scheduler.get_queue_sizes()}")

    # 逐步执行调度
    print_subheader("逐步调度过程")
    step_num = 0
    max_steps = 200

    for _ in range(max_steps):
        result = engine.step()
        step_num = result["step"]
        queues = engine.scheduler.get_queue_sizes()

        # 每 5 步打印一次状态
        if step_num % 5 == 0 or result["is_prefill"]:
            status = "[PREFILL]" if result["is_prefill"] else "[DECODE]"
            print(
                f"  Step {step_num:>4d} {status} "
                f"active={result['num_scheduled_groups']:>3d} "
                f"completed={result['num_completed']:>2d} "
                f"waiting={queues['waiting']:>3d} "
                f"running={queues['running']:>3d} "
                f"block_usage={result['block_usage']:.1%}"
            )

        if queues["waiting"] == 0 and queues["running"] == 0:
            break

    # 最终统计
    print_subheader("最终统计")
    final = engine.get_final_stats()
    print(f"  总步数:             {final['total_steps']}")
    print(f"  总请求数:           {final['total_requests']}")
    print(f"  完成请求数:         {final['completed_requests']}")
    print(f"  抢占次数:           {final['preemptions']}")
    print(f"  换出次数:           {final['swapped_out']}")
    print(f"  Prefix Cache 命中率: {final['prefix_cache_hit_rate']:.1%}")
    print(f"  最终块使用率:       {final['final_block_usage']:.1%}")


# ==============================================================================
# 3. Pipeline Parallelism Bubble Ratio 演示
# ==============================================================================


def demo_pipeline_parallelism() -> None:
    """展示 pipeline parallelism 的 1F1B 调度和 bubble ratio。

    借鉴 Megatron-LM: 展示不同 PP size 和 microbatch 数下的
    bubble ratio 变化，帮助选择合适的并行配置。
    """
    print_header("Pipeline Parallelism — 1F1B 调度与 Bubble Ratio（借鉴 Megatron-LM）")

    # Bubble ratio 分析
    print_subheader("Bubble Ratio 分析: (PP - 1) / (PP - 1 + M)")

    pp_sizes = [2, 4, 8, 16]
    microbatch_counts = [4, 8, 16, 32, 64, 128]

    # 表头
    print(f"\n  {'PP\\M':>5}", end="")
    for m in microbatch_counts:
        print(f"  {'M=' + str(m):>8}", end="")
    print(f"  {'':>5}")
    print(f"  {'-' * 5}", end="")
    for _ in microbatch_counts:
        print(f"  {'-' * 8}", end="")
    print()

    for pp in pp_sizes:
        print(f"  {pp:>3}  ", end="")
        for m in microbatch_counts:
            ps = PPScheduler(pp_size=pp, num_microbatches=m)
            eff = ps.efficiency()
            print(f"  {eff:>7.1%}", end="")
        print()

    # 具体示例: PP=4, M=8 的调度
    print_subheader("PP=4, M=8 的 1F1B 调度计划")
    pp = PPScheduler(pp_size=4, num_microbatches=8)

    print(f"\n  Bubble Ratio: {pp.bubble_ratio():.3f}")
    print(f"  Efficiency:   {pp.efficiency():.3f}")
    print(f"  Total Steps:  {pp.total_steps()}")

    for rank in range(4):
        schedule = pp.get_rank_schedule(rank)
        warmup = pp.num_warmup_microbatches(rank)
        idle = pp.rank_idle_steps(rank)

        ops_str = ""
        w_count = 0
        for i, (op, mb) in enumerate(schedule):
            if i == warmup:
                ops_str += "| "
            ops_str += f"{op}{mb} "

        print(f"\n  Rank {rank} (warmup={warmup}, idle={idle}):")
        print(f"    {ops_str}")

    # 大 M 的 bubble ratio
    print_subheader("大 microbatch 数的 Bubble Ratio")
    for pp in [4, 8]:
        for m in [128, 256, 512]:
            ps = PPScheduler(pp_size=pp, num_microbatches=m)
            print(
                f"  PP={pp}, M={m:>4d}:  bubble={ps.bubble_ratio():.4f}  "
                f"efficiency={ps.efficiency():.4f}"
            )


# ==============================================================================
# 4. ZeRO 内存节省演示
# ==============================================================================


def demo_zero_memory() -> None:
    """展示 ZeRO 各阶段的内存节省效果。

    借鉴 DeepSpeed ZeRO 论文: 比较 ZeRO-0/1/2/3 在不同 GPU 数量下
    的每卡内存占用。
    """
    print_header("ZeRO 内存优化 — 各阶段内存节省分析（借鉴 DeepSpeed）")

    # 以 llama-70B 为例（约 140B fp16 参数）
    model_params = int(70e9 / 2)  # ~35B params in fp16

    print_subheader(f"模型参数: {model_params / 1e9:.1f}B (fp16, llama-70B 级别)")

    gpu_counts = [1, 2, 4, 8, 16, 32]
    stages = [
        ZEROStage.STAGE_0,
        ZEROStage.STAGE_1,
        ZEROStage.STAGE_2,
        ZEROStage.STAGE_3,
    ]

    for stage in stages:
        print(f"\n  --- {stage.name} ---")
        print(
            f"  {'GPUs':>5}  {'Param(MB)':>12}  {'Grad(MB)':>12}  "
            f"{'Opt(MB)':>12}  {'Total(MB)':>12}  {'Saving':>8}"
        )
        print(f"  {'-' * 5}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 8}")

        for n in gpu_counts:
            zm = ZEROStageManager(num_gpus=n, model_params=model_params, stage=stage)
            print(
                f"  {n:>5d}  {zm.parameter_memory_mb():>10.0f}  "
                f"{zm.gradient_memory_mb():>10.0f}  "
                f"{zm.optimizer_state_memory_mb():>10.0f}  "
                f"{zm.total_memory_per_gpu_mb():>10.0f}  "
                f"{zm.memory_savings_ratio():>7.1%}"
            )

    # 通信开销对比
    print_subheader("ZeRO 通信开销分析（以 llama-70B 为例）")
    zm = ZEROStageManager(num_gpus=8, model_params=model_params, stage=ZEROStage.STAGE_3)
    overhead = zm.communication_overhead_per_step(hidden_dim=8192, num_layers=80)
    print(f"  ZeRO-1 通信量: {overhead['zero1_comm_mb']:.1f} MB/step")
    print(f"  ZeRO-2 通信量: {overhead['zero2_comm_mb']:.1f} MB/step")
    print(f"  ZeRO-3 通信量: {overhead['zero3_comm_mb']:.1f} MB/step")


# ==============================================================================
# 5. WorkloadBalancer 演示
# ==============================================================================


def demo_workload_balancer() -> None:
    """展示异构 GPU 负载均衡。

    借鉴 FlexFlow/Alpa: 在显存和带宽不等的异构 GPU 环境中，
    按容量和速度合理分配计算任务。
    """
    print_header("WorkloadBalancer — 异构 GPU 负载均衡（借鉴 FlexFlow/Alpa）")

    # 模拟异构环境: A100 80GB + A100 40GB + V100 32GB
    print_subheader("异构 GPU 环境")
    capacities = {0: 80, 1: 40, 2: 40, 3: 32}  # GB 显存
    bandwidths = {0: 900, 1: 600, 2: 600, 3: 300}  # GB/s (NVLink + PCIe 混合)

    wb = WorkloadBalancer(device_capacities=capacities, device_bandwidths=bandwidths)

    for dev in wb.device_ids:
        cap = capacities[dev]
        bw = bandwidths.get(dev, cap * 10)
        print(f"  GPU {dev}: {cap} GB, ~{bw} GB/s")

    # 按容量分配
    print_subheader("按显存容量分配（显存敏感任务）")
    alloc_cap = wb.balance_by_capacity(total_work=1000)
    for dev in sorted(alloc_cap.keys()):
        bar = "█" * (alloc_cap[dev] // 10)
        print(f"  GPU {dev}: {alloc_cap[dev]:>4d} 单位  {bar}")
    print(f"  不均衡度: {wb.get_imbalance_score(alloc_cap):.4f}")

    # 按速度分配
    print_subheader("按带宽速度分配（计算敏感任务）")
    alloc_spd = wb.balance_by_speed(total_work=1000)
    for dev in sorted(alloc_spd.keys()):
        bar = "█" * (alloc_spd[dev] // 10)
        print(f"  GPU {dev}: {alloc_spd[dev]:>4d} 单位  {bar}")
    print(f"  不均衡度: {wb.get_imbalance_score(alloc_spd):.4f}")

    # 混合分配
    print_subheader("混合分配（容量+速度各 50%）")
    alloc_hyb = wb.balance_hybrid(total_work=1000, capacity_weight=0.5)
    for dev in sorted(alloc_hyb.keys()):
        bar = "█" * (alloc_hyb[dev] // 10)
        print(f"  GPU {dev}: {alloc_hyb[dev]:>4d} 单位  {bar}")
    print(f"  不均衡度: {wb.get_imbalance_score(alloc_hyb):.4f}")


# ==============================================================================
# 6. 吞吐量对比演示
# ==============================================================================


def demo_throughput_comparison() -> None:
    """对比不同调度策略的吞吐量。

    通过 SimulationEngine 比较不同 block_size、不同 watermark
    和不同 batch 限制下的吞吐量。
    """
    print_header("吞吐量对比 — 不同调度策略的影响")

    configs: List[Dict] = [
        {
            "name": "Baseline (block=16, watermark=0.9, max_seqs=64)",
            "block_size": 16,
            "num_blocks": 5000,
            "max_seqs": 64,
            "max_tokens": 4096,
        },
        {
            "name": "Larger blocks (block=32, watermark=0.9, max_seqs=64)",
            "block_size": 32,
            "num_blocks": 2500,  # 保持相同的总 KV cache 大小
            "max_seqs": 64,
            "max_tokens": 4096,
        },
        {
            "name": "More concurrency (block=16, watermark=0.9, max_seqs=128)",
            "block_size": 16,
            "num_blocks": 5000,
            "max_seqs": 128,
            "max_tokens": 8192,
        },
        {
            "name": "Conservative watermark (block=16, watermark=0.7, max_seqs=64)",
            "block_size": 16,
            "num_blocks": 5000,
            "max_seqs": 64,
            "max_tokens": 4096,
        },
    ]

    random.seed(42)
    request_counts = [50, 100, 200]

    print(f"\n  {'配置':<50}  {'请求数':>8}  {'步数':>8}  {'吞吐(token/s)':>14}")
    print(f"  {'-' * 50}  {'-' * 8}  {'-' * 8}  {'-' * 14}")

    for cfg in configs:
        for n_requests in request_counts:
            engine = SimulationEngine(
                block_size=cfg["block_size"],
                num_blocks=cfg["num_blocks"],
                max_num_seqs=cfg["max_seqs"],
                max_num_batched_tokens=cfg["max_tokens"],
            )

            for i in range(n_requests):
                prompt_len = random.randint(16, 96)
                max_len = random.randint(4, 24)
                engine.add_single_request(f"r{i}", prompt_len, max_len)

            start = time.perf_counter()
            engine.run_until_idle(max_steps=10000)
            elapsed = time.perf_counter() - start

            final = engine.get_final_stats()
            tokens_generated = sum(random.randint(4, 24) for _ in range(n_requests))  # approximate
            # More accurate: use the scheduled token count
            tokens = engine.scheduler.total_scheduled_tokens or n_requests * 10
            throughput = tokens / max(elapsed, 0.001)

            short_name = cfg["name"][:48]
            print(
                f"  {short_name:<50}  {n_requests:>8d}  "
                f"{final['total_steps']:>8d}  {throughput:>12.0f}"
            )

    # PP throughput efficiency
    print_subheader("Pipeline Parallelism 效率对比")
    for pp in [1, 2, 4, 8]:
        for m in [8, 16, 32, 64]:
            ps = PPScheduler(pp_size=pp, num_microbatches=m)
            print(
                f"  PP={pp}, M={m:>3d}:  efficiency={ps.efficiency():.4f}  "
                f"bubble={ps.bubble_ratio():.4f}"
            )

    # MemoryPlanner watermark 影响
    print_subheader("MemoryPlanner Watermark 对并发数的影响")
    for watermark in [0.7, 0.8, 0.9, 0.95]:
        mp = MemoryPlanner(total_memory_gb=80, model_size_gb=60, watermark=watermark)
        max_seqs = mp.estimate_max_concurrent_sequences(avg_prompt_len=64, avg_output_len=32)
        print(
            f"  watermark={watermark:.2f}:  最多并发 {max_seqs:>4d} 序列,  "
            f"可用块={mp.get_available_blocks()}"
        )


# ==============================================================================
# 7. TP+PP+DP 混合并行演示
# ==============================================================================


def demo_hybrid_parallelism() -> None:
    """展示混合并行（TP+PP+DP）的配置和开销分析。

    借鉴 Megatron-LM: 组合三种并行策略，平衡计算、通信和显存。
    """
    print_header("混合并行 — TP+PP+DP 组合策略（借鉴 Megatron-LM + DeepSpeed）")

    # 典型配置
    configs = [
        ("1 机 8 卡 (DGX A100)", 8, 1, 1),
        ("2 机 16 卡", 8, 2, 1),
        ("4 机 32 卡", 8, 4, 1),
        ("8 机 64 卡", 8, 4, 2),
        ("16 机 128 卡", 8, 4, 4),
    ]

    print(
        f"\n  {'配置':<24}  {'TP':>4}  {'PP':>4}  {'DP':>4}  "
        f"{'Total':>6}  {'PP Eff':>7}  {'M=16':>8}  {'M=32':>8}  {'M=64':>8}"
    )
    print(
        f"  {'-' * 24}  {'-' * 4}  {'-' * 4}  {'-' * 4}  "
        f"{'-' * 6}  {'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 8}"
    )

    for name, tp, pp, dp in configs:
        total = tp * pp * dp
        hs = HybridScheduler(tp_size=tp, pp_size=pp, dp_size=dp, num_microbatches=16)

        # 不同 microbatch 下的 PP 效率
        eff_16 = PPScheduler(pp_size=pp, num_microbatches=16).efficiency()
        eff_32 = PPScheduler(pp_size=pp, num_microbatches=32).efficiency()
        eff_64 = PPScheduler(pp_size=pp, num_microbatches=64).efficiency()

        print(
            f"  {name:<24}  {tp:>4d}  {pp:>4d}  {dp:>4d}  "
            f"{total:>6d}  {hs.pp_efficiency():>6.1%}  "
            f"{eff_16:>7.1%}  {eff_32:>7.1%}  {eff_64:>7.1%}"
        )

    # 通信开销分析
    print_subheader("通信开销分析（llama-70B 级别, hidden=8192, layers=80）")
    for name, tp, pp, dp in configs:
        hs = HybridScheduler(tp_size=tp, pp_size=pp, dp_size=dp, num_microbatches=16)
        overhead = hs.estimate_communication_overhead(
            num_layers=80, batch_size=2, seq_len=4096, hidden_dim=8192
        )
        tp_comm = overhead["tp_comm_mb_per_step"]
        pp_comm = overhead["pp_comm_mb_per_step"]
        print(f"  {name:<24}:  TP通信={tp_comm:>8.1f} MB/step,  PP通信={pp_comm:>8.1f} MB/step")


# ==============================================================================
# 主入口
# ==============================================================================


DEMO_REGISTRY = {
    "block": (demo_block_manager, "BlockManager prefix caching 效果"),
    "batch": (demo_continuous_batching, "Continuous Batching 调度流程"),
    "pp": (demo_pipeline_parallelism, "Pipeline Parallelism 1F1B 与 Bubble Ratio"),
    "zero": (demo_zero_memory, "ZeRO 各阶段内存节省分析"),
    "balancer": (demo_workload_balancer, "异构 GPU 负载均衡"),
    "throughput": (demo_throughput_comparison, "不同调度策略吞吐量对比"),
    "hybrid": (demo_hybrid_parallelism, "混合并行 TP+PP+DP 组合策略"),
    "all": (None, "运行所有演示"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="异构调度演示 — 借鉴 vLLM/DeepSpeed/Megatron-LM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --demo block        BlockManager 演示
  %(prog)s --demo batch        Continuous Batching 演示
  %(prog)s --demo pp           Pipeline Parallelism 演示
  %(prog)s --demo zero         ZeRO 内存节省演示
  %(prog)s --demo balancer     负载均衡演示
  %(prog)s --demo throughput   吞吐量对比
  %(prog)s --demo hybrid       混合并行策略
  %(prog)s --demo all          运行所有演示
  %(prog)s --list              列出所有演示
        """,
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        choices=list(DEMO_REGISTRY.keys()),
        help="要运行的演示名称（默认: all）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的演示",
    )

    args = parser.parse_args()

    if args.list:
        print("可用的演示:")
        for name, (_, desc) in DEMO_REGISTRY.items():
            label = "[全部]" if name == "all" else ""
            print(f"  {name:<15} {label} {desc}")
        return

    print("=" * 72)
    print("  异构调度演示 — 借鉴 vLLM / DeepSpeed / Megatron-LM 设计模式")
    print("=" * 72)

    if args.demo == "all":
        for name, (func, desc) in DEMO_REGISTRY.items():
            if name == "all":
                continue
            print(f"\n  >>> {name}: {desc}")
            try:
                func()
            except Exception as e:
                print(f"\n  [错误] {name} 执行失败: {e}")
                import traceback

                traceback.print_exc()
    else:
        func, desc = DEMO_REGISTRY[args.demo]
        print(f"\n  >>> {args.demo}: {desc}")
        func()

    print("\n演示完成。")


if __name__ == "__main__":
    main()
