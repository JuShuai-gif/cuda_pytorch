import os
import torch
import torch.distributed as dist


def example():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    # 每个 rank 创建完整的张量（被拆分为块）
    input_tensor = torch.cat(
        [
            torch.ones(3, device=rank) * (rank * 2),
            torch.ones(3, device=rank) * (rank * 2 + 1),
        ]
    )

    print(f"Before Rank {rank}: {input_tensor}")

    # 输出缓冲区（只有一个块）
    output_tensor = torch.empty(3, device=rank)

    dist.reduce_scatter(output_tensor, list(input_tensor.chunk(world_size)))

    print(f"After Rank {rank}: {output_tensor}")


if __name__ == "__main__":
    example()
