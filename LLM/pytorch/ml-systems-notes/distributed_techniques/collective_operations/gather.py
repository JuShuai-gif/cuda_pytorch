# torchrun --nproc_per_node=2 脚本.py

import os
import torch
import torch.distributed as dist


def example():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    tensor = torch.ones(1, 5, device=rank) * rank
    print(tensor)

    # 只有目标进程或主进程 (dst=0) 需要内存来收集数据
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None  # 其他进程只发送

    dist.gather(tensor, gather_list=gather_list, dst=0)  # 每个张量被收集到 dst 0

    if rank == 0:
        gathered_tensor = torch.cat(
            gather_list, dim=0
        )  # 改变 dim 查看按行 vs 按列的变化
        print("gathered:\n", gathered_tensor)


if __name__ == "__main__":
    example()
