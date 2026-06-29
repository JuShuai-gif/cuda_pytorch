import os
import torch
import torch.distributed as dist


def example():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    # 仅在源进程 (rank 0) 准备数据
    if rank == 0:
        scatter_list = [torch.ones(3, device=rank) * i for i in range(world_size)]
        print(scatter_list)
    else:
        scatter_list = None

    # 每个 rank 必须有接收缓冲区
    tensor = torch.empty(3, device=rank)

    dist.scatter(tensor, scatter_list=scatter_list, src=0)

    print(f"rank {rank} 接收到: {tensor}")


if __name__ == "__main__":
    example()
