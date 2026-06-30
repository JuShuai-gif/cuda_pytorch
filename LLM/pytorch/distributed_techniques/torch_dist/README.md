## Torch Distributed

`--nnodes` - 你拥有的机器/服务器数量
`--nproc_per_node` - 每台机器内的 GPU 数量

```bash
# 2 台机器，每台 8 个 GPU = 共 16 个进程
torchrun --nnodes=2 --nproc_per_node=8 train.py
```

### Meta 设备

这是一个抽象设备，记录元数据但不记录数据。这意味着你不需要在 CPU/GPU 上加载张量，但可以检查张量的变换、分析等，而无需实际花费时间加载数据，不会 OOM。

```python
import torch
from torch import nn

model = nn.Linear(10, 5).to("meta")
x = torch.randn(3, 10).to("meta")
out = model(x) # 没有内存分配
print(out.shape) # 输出 torch.Size([3, 5])
```

### 进程组（Process Group）

做分布式训练的核心是一种让进程发现并互相通信的方式。你通过进程组来实现这一点。假设我们有 4 个 GPU，我们需要 GPU 1 和 3 互相通信，GPU 2 和 4 互相通信，但不能与其他 GPU 通信。进程组可以帮助你实现这一点。

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")
# 所有进程现在属于默认的世界组

# 只让 rank 0 和 1 互相通信
group_01 = dist.new_group([0, 1])
# 只让 rank 2 和 3 互相通信
group_23 = dist.new_group([2, 3])
```

### 设备网格（Device Mesh）

Device Mesh 本质上是一种结构化的方式来创建和管理多个进程组。随着你扩展更多 GPU，单独使用进程组会变得相当复杂。

```python
from torch.distributed.device_mesh import init_device_mesh

# 创建 2D mesh：2 个节点 × 每个节点 4 个 GPU
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("pp", "tp"))

# 自动创建子组：
pp_group = mesh["pp"]       # 流水线并行的进程组
tp_group = mesh["tp"]       # 张量并行的进程组
```

### DTensor

分布式训练使用的原生张量类型。

可以分片（shard）、复制（replicate）和部分（partial）操作。

```python
from torch.distributed.tensor import DTensor, Shard, Replicate, Partial

mesh = init_device_mesh("cuda", (4,))

# 沿第 0 维跨设备分片
dt = DTensor.from_local(local_tensor, mesh, [Shard(0)])

# 跨所有设备复制
dt = DTensor.from_local(local_tensor, mesh, [Replicate()])
```
