"""
第 05 讲 — GPU 架构：Occupancy 计算器。

根据不同 GPU 架构的 register 使用、shared memory 使用以及
block / grid 约束来计算理论 occupancy。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GPU 限制
# ---------------------------------------------------------------------------


@dataclass
class SMLimits:
    """每个 Streaming Multiprocessor 的硬件资源限制。"""

    name: str
    max_threads_per_sm: int
    max_warps_per_sm: int  # max_threads_per_sm / 32
    max_blocks_per_sm: int
    max_registers_per_sm: int
    max_shared_memory_per_sm_kib: int
    max_threads_per_block: int


# 常见 GPU 的架构限制
SM_LIMITS: Dict[str, SMLimits] = {
    "V100": SMLimits(
        name="V100 (Volta)",
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_kib=96,  # 最多 96 KiB（可配置）
        max_threads_per_block=1024,
    ),
    "A100": SMLimits(
        name="A100 (Ampere)",
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_kib=164,  # 可配置，最多 164 KiB
        max_threads_per_block=1024,
    ),
    "H100": SMLimits(
        name="H100 (Hopper)",
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_kib=228,  # 最多 228 KiB
        max_threads_per_block=1024,
    ),
}


# ---------------------------------------------------------------------------
# Occupancy 计算
# ---------------------------------------------------------------------------


@dataclass
class OccupancyResult:
    """Occupancy 计算结果。"""

    thread_blocks_per_sm: int
    warps_per_sm: int
    threads_per_sm: int

    reg_limited_blocks: int
    smem_limited_blocks: int
    thread_limited_blocks: int

    theoretical_occupancy: float  # 0..1  (active warps / max warps)
    active_warps_per_sm: int
    max_warps_per_sm: int

    registers_per_thread: int
    shared_mem_per_block_kib: float
    threads_per_block: int


def calculate_occupancy(
    registers_per_thread: int,
    shared_mem_per_block_bytes: int,
    threads_per_block: int,
    gpu_name: str = "A100",
) -> OccupancyResult:
    """针对给定 kernel 配置计算理论 occupancy。

    Parameters
    ----------
    registers_per_thread : int
        该 kernel 每个线程使用的 register 数量。
    shared_mem_per_block_bytes : int
        每个 block 的静态 + 动态 shared memory（以字节为单位）。
    threads_per_block : int
        每个 block 的线程数（必须是 warp size = 32 的倍数）。
    gpu_name : str
        'V100'、'A100' 或 'H100' 之一。

    Returns
    -------
    带有详细明细的 OccupancyResult。
    """
    limits = SM_LIMITS[gpu_name]

    if threads_per_block > limits.max_threads_per_block:
        raise ValueError(
            f"threads_per_block ({threads_per_block}) exceeds "
            f"max ({limits.max_threads_per_block})"
        )

    warp_size = 32
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size
    regs_per_block = warps_per_block * warp_size * registers_per_thread
    # 向上取整到分配粒度（NVIDIA 为 256）
    regs_per_block = ((regs_per_block + 255) // 256) * 256

    smem_per_block_kib = shared_mem_per_block_bytes / 1024.0

    # --- 受资源限制的 block 数量 ---

    # Register 限制
    if regs_per_block > 0:
        reg_limited_blocks = limits.max_registers_per_sm // regs_per_block
    else:
        reg_limited_blocks = limits.max_blocks_per_sm

    # Shared memory 限制
    if shared_mem_per_block_bytes > 0:
        smem_limited_blocks = int(
            limits.max_shared_memory_per_sm_kib * 1024 // shared_mem_per_block_bytes
        )
    else:
        smem_limited_blocks = limits.max_blocks_per_sm

    # Thread / warp 限制
    thread_limited_blocks = limits.max_threads_per_sm // threads_per_block

    # 硬件 block 限制
    max_blocks_allowed = limits.max_blocks_per_sm

    # 限制因素（取最小值）
    actual_blocks = min(
        reg_limited_blocks,
        smem_limited_blocks,
        thread_limited_blocks,
        max_blocks_allowed,
    )
    actual_blocks = max(actual_blocks, 1)

    active_warps = actual_blocks * warps_per_block
    occupancy = active_warps / limits.max_warps_per_sm

    return OccupancyResult(
        thread_blocks_per_sm=actual_blocks,
        warps_per_sm=active_warps,
        threads_per_sm=actual_blocks * threads_per_block,
        reg_limited_blocks=reg_limited_blocks,
        smem_limited_blocks=smem_limited_blocks,
        thread_limited_blocks=thread_limited_blocks,
        theoretical_occupancy=occupancy,
        active_warps_per_sm=active_warps,
        max_warps_per_sm=limits.max_warps_per_sm,
        registers_per_thread=registers_per_thread,
        shared_mem_per_block_kib=smem_per_block_kib,
        threads_per_block=threads_per_block,
    )


# ---------------------------------------------------------------------------
# 便捷函数：寻找最优 block 大小
# ---------------------------------------------------------------------------


def find_optimal_block_size(
    registers_per_thread: int,
    shared_mem_per_block_bytes: int,
    gpu_name: str = "A100",
    block_sizes: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """在一组 block 大小中寻找对应的 occupancy 值。

    返回按 occupancy 降序排列的 (threads_per_block, occupancy) 列表。
    """
    if block_sizes is None:
        block_sizes = [32, 64, 128, 256, 512, 1024]

    results: List[Tuple[int, float]] = []
    for bs in block_sizes:
        try:
            occ = calculate_occupancy(
                registers_per_thread, shared_mem_per_block_bytes, bs, gpu_name
            )
            results.append((bs, occ.theoretical_occupancy))
        except ValueError:
            pass  # 跳过无效的 block 大小

    results.sort(key=lambda x: (-x[1], x[0]))
    return results


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Occupancy 计算器 ===\n")

    # 示例：matmul kernel，使用 64 个 register，48 KiB shared memory
    regs = 64
    smem = 48 * 1024  # 48 KiB
    threads = 256

    for gpu in ["V100", "A100", "H100"]:
        occ = calculate_occupancy(regs, smem, threads, gpu)
        print(
            f"\n{gpu}: threads_per_block={threads}, regs/thread={regs}, smem/block={smem // 1024} KiB"
        )
        print(f"  Reg-limited blocks:  {occ.reg_limited_blocks}")
        print(f"  Smem-limited blocks: {occ.smem_limited_blocks}")
        print(f"  Thread-limited blocks: {occ.thread_limited_blocks}")
        print(f"  Actual blocks/SM:    {occ.thread_blocks_per_sm}")
        print(
            f"  Active warps/SM:     {occ.active_warps_per_sm} / {occ.max_warps_per_sm}"
        )
        print(f"  Occupancy:           {occ.theoretical_occupancy:.1%}")

    # 寻找最优 block 大小
    print("\n\n=== Optimal Block Size Scan ===\n")
    best = find_optimal_block_size(regs, smem, "A100")
    for bs, occ_val in best:
        print(f"  block_size={bs:4d}  occupancy={occ_val:.1%}")

    print("\nAll checks passed.")
