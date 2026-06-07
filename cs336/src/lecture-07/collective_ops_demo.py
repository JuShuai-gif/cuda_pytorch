"""
演示集合通信操作。
使用 PyTorch 注释来模拟多进程行为，无需真实的多 GPU 环境。
每个操作都通过具体的数值示例来说明。
"""

from __future__ import annotations

import torch


def demo_broadcast() -> None:
    """Broadcast：一个进程将数据发送给所有其他进程。所有进程收到相同的副本。"""
    print("=" * 60)
    print("BROADCAST：一对多通信")
    print("=" * 60)
    # 假设 rank 0 拥有数据 [1, 2, 3, 4]，其他 rank 初始为零
    data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"  Rank 0 发送：       {data}")
    print(f"  所有 rank 接收到：{data}")
    print(f"  （实践中：torch.distributed.broadcast(tensor, src=0)）")
    print()


def demo_scatter() -> None:
    """Scatter：一个进程将数据块分发给所有进程。"""
    print("=" * 60)
    print("SCATTER：一对多，每个进程收到不同的数据块")
    print("=" * 60)
    # Rank 0 拥有 [0, 1, 2, 3, 4, 5, 6, 7]，共 4 个进程
    full_data = torch.arange(8, dtype=torch.float32)
    world_size = 4
    print(f"  Rank 0 输入：{full_data}")
    chunk_size = len(full_data) // world_size
    for rank in range(world_size):
        chunk = full_data[rank * chunk_size : (rank + 1) * chunk_size]
        print(f"  Rank {rank} 接收到：{chunk}")
    print()


def demo_gather() -> None:
    """Gather：所有进程将数据发送给一个进程（scatter 的逆操作）。"""
    print("=" * 60)
    print("GATHER：多对一，拼接数据块")
    print("=" * 60)
    world_size = 4
    chunks = [torch.tensor([r, r + 1], dtype=torch.float32) for r in range(world_size)]
    for rank, chunk in enumerate(chunks):
        print(f"  Rank {rank} 发送：{chunk}")
    gathered = torch.cat(chunks)
    print(f"  Rank 0 接收到（收集后）：{gathered}")
    print()


def demo_reduce() -> None:
    """Reduce：所有进程贡献数据，结果在某个进程上聚合。"""
    print("=" * 60)
    print("REDUCE：多对一，附带操作（求和、最小值、最大值等）")
    print("=" * 60)
    world_size = 4
    data = [torch.tensor([r * 2.0, r * 2.0 + 1.0]) for r in range(world_size)]
    op = "sum"
    print(f"  操作：{op}")
    for rank, d in enumerate(data):
        print(f"  Rank {rank} 贡献：{d}")
    result = data[0].clone()
    for d in data[1:]:
        result += d
    print(f"  根节点结果：{result}")
    print()


def demo_all_gather() -> None:
    """All-gather：所有进程从所有其他进程收集数据。每个进程都得到完整的拼接结果。"""
    print("=" * 60)
    print("ALL-GATHER：所有进程都得到完整的拼接结果")
    print("=" * 60)
    world_size = 4
    chunks = [
        torch.tensor([r * 3.0, r * 3.0 + 1.0, r * 3.0 + 2.0]) for r in range(world_size)
    ]
    for rank, chunk in enumerate(chunks):
        print(f"  Rank {rank} 发送：{chunk}")
    gathered = torch.cat(chunks)
    for rank in range(world_size):
        print(f"  Rank {rank} 接收到：{gathered}")
    print()


def demo_reduce_scatter() -> None:
    """Reduce-scatter：先 reduce 再 scatter。每个进程得到归约结果的一个数据块。"""
    print("=" * 60)
    print("REDUCE-SCATTER：Reduce + Scatter 组合")
    print("=" * 60)
    world_size = 4
    # 每个 rank 拥有一个完整大小的张量
    data = [
        torch.tensor([r * 4 + i for i in range(8)], dtype=torch.float32)
        for r in range(world_size)
    ]
    for rank, d in enumerate(data):
        print(f"  Rank {rank} 输入：{d}")
    # 求和
    summed = data[0].clone()
    for d in data[1:]:
        summed += d
    print(f"  reduce（求和）之后：{summed}")
    # Scatter 结果
    chunk_size = len(summed) // world_size
    for rank in range(world_size):
        chunk = summed[rank * chunk_size : (rank + 1) * chunk_size]
        print(f"  Rank {rank} 接收到：{chunk}")
    print()


def demo_all_reduce() -> None:
    """All-reduce：reduce + broadcast。所有进程得到相同的归约结果。"""
    print("=" * 60)
    print("ALL-REDUCE：Reduce + Broadcast，DDP 中最常用的操作")
    print("=" * 60)
    world_size = 4
    grads = [torch.tensor([r * 0.5, r * 0.5 + 0.25]) for r in range(world_size)]
    print("  操作：sum（DDP 中梯度同步的典型操作）")
    for rank, g in enumerate(grads):
        print(f"  Rank {rank} 梯度：{g}")
    # 求和并平均（DDP 中梯度同步的典型做法）
    summed = grads[0].clone()
    for g in grads[1:]:
        summed += g
    avg = summed / world_size
    for rank in range(world_size):
        print(f"  Rank {rank} 接收到（平均值）：{avg}")
    print()


def demo_all_to_all() -> None:
    """All-to-all：每个进程将数据分发给所有其他进程（类似转置操作）。"""
    print("=" * 60)
    print("ALL-TO-ALL：每个进程向所有其他进程分发数据（转置）")
    print("=" * 60)
    world_size = 3
    # 每个 rank 有一个矩阵。All-to-all 将列分发给所有 rank。
    data = [
        torch.tensor(
            [[r * 10 + c for c in range(3)] for _ in range(2)], dtype=torch.float32
        )
        for r in range(world_size)
    ]
    for rank, d in enumerate(data):
        print(f"  Rank {rank} 输入：\n{d}")
    # 每个 rank 将第 j 列发送给 rank j
    print("  all-to-all 之后：")
    for rank_out in range(world_size):
        result = torch.tensor(
            [[r * 10 + rank_out for r in range(3)] for _ in range(2)],
            dtype=torch.float32,
        )
        print(f"  Rank {rank_out} 接收到：\n{result}")
    print()


def main() -> None:
    demo_broadcast()
    demo_scatter()
    demo_gather()
    demo_reduce()
    demo_all_gather()
    demo_reduce_scatter()
    demo_all_reduce()
    demo_all_to_all()
    print("集合操作演示完成。")


if __name__ == "__main__":
    main()
