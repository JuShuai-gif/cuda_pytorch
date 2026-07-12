# DDP Reducer: 梯度同步、Bucket 与通信计算重叠

> Python 端: `torch/nn/parallel/distributed.py`
> C++ 核心: `torch/csrc/distributed/c10d/reducer.cpp`、`reducer.hpp`
> NCCL: `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`
> 通信 Hook: `torch/distributed/algorithms/ddp_comm_hooks/`

## 0. 一句话总览

DDP 的 Reducer 是梯度同步的引擎：它注册 autograd hook 监听每个参数的梯度就绪状态，将参数按 bucket 分组，当 bucket 内所有梯度都就绪后触发异步 all-reduce，通过 bucket 机制和通信计算重叠来隐藏通信延迟。

## 1. 最小例子

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

model = torch.nn.Linear(8, 8).cuda()
ddp_model = DDP(model, device_ids=[rank])

x = torch.randn(4, 8, device="cuda")
loss = ddp_model(x).sum()
loss.backward()
```

## 1.5 实战例子

### 1.5.1 调整 bucket 大小优化训练吞吐

在大模型训练中，bucket 大小直接影响通信计算重叠效果：

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

model = torch.nn.Sequential(*[
    torch.nn.Linear(4096, 4096) for _ in range(20)
]).cuda()

# 不同 bucket 大小时对比:
# 默认 25MB: 通用, 但小模型时 overhead 偏高
ddp_default = DDP(model, device_ids=[rank])
# 小 bucket (5MB): all-reduce 更多但更细粒度重叠
ddp_small_bucket = DDP(
    torch.nn.Sequential(*[torch.nn.Linear(4096, 4096) for _ in range(20)]).cuda(),
    device_ids=[rank],
    bucket_cap_mb=5
)
# 大 bucket (200MB): all-reduce 更少但尾部延迟高
ddp_large_bucket = DDP(
    torch.nn.Sequential(*[torch.nn.Linear(4096, 4096) for _ in range(20)]).cuda(),
    device_ids=[rank],
    bucket_cap_mb=200
)

# 用 nsys profile 观察 bucket 完成时间分布
# 或用 torch.cuda.Event 手动计时
```

实际建议：对 transformer 模型，默认 25MB 在大部分场景表现良好。当模型极深（>50层）时，增大 bucket 减少 all-reduce 次数；当模型浅但单层大时，减小 bucket 加速首 bucket 启动。

### 1.5.2 排查 bucket 大小不均衡导致的通信瓶颈

当某些层计算量差异大时（如 embedding 层 vs 单层 attention），bucket 分配不均衡：

```python
# 查看 Reducer 的 bucket 分配
# 通过日志或 C++ 端 DEBUG 输出
import torch._dynamo.logging as logging

# 设置日志级别查看 bucket 信息
torch._logging.set_logs(distributed=True)

# 输出示例:
# [Reducer] Bucket 0: size=25.0MB, params=3
# [Reducer] Bucket 1: size=12.5MB, params=8
# [Reducer] Bucket 2: size=0.5MB, params=2

# 如果发现某个 bucket 明显偏小:
# - 该 bucket 的 all-reduce 在反向早期触发
# - 但小 bucket 的 NCCL launch overhead 占比高
# - 考虑合并小 bucket: 增大 bucket_cap_mb 或手动重排参数注册顺序
```

解决方案：在模型定义中按计算图拓扑顺序注册参数，使同一阶段的参数在同一个 bucket 中。

### 1.5.3 自定义 DDP communication hook 实现梯度压缩

当跨节点带宽成为瓶颈时，用 communication hook 实现梯度量化：

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

model = torch.nn.Linear(1024, 1024).cuda()
ddp_model = DDP(model, device_ids=[rank])

# 自定义 hook: FP16 压缩梯度
def fp16_compress_hook(state, bucket):
    bucket.set_buffer(bucket.buffer().to(torch.float16))
    fut = state.process_group.allreduce(bucket.buffer())
    # 恢复 FP32
    def decompress(fut):
        buf = fut.value().float()
        return buf
    return fut.then(decompress)

# 注册 hook
ddp_model.register_comm_hook(None, fp16_compress_hook)

x = torch.randn(8, 1024, device="cuda")
loss = ddp_model(x).sum()
loss.backward()
# 梯度传输减半, 但精度略微下降
```

其他常用 hook：`gradient_compression_hook`（随机梯度压缩）、`powerSGD_hook`（低秩近似）。

## 2. 从 Python API 到源码的调用链

```
DDP(model)
  -> torch/nn/parallel/distributed.py: DistributedDataParallel.__init__
  -> 创建 Reducer (C++ 扩展)
  -> Reducer 遍历模型参数, 构建 bucket 映射

loss.backward()
  -> autograd 引擎反向传播
  -> 每个参数梯度就绪时:
     autograd hook (由 Reducer 注册) 被触发
     -> Reducer::autograd_hook()
     -> 标记该参数梯度已就绪
     -> 检查所在 bucket 是否全部就绪
  -> bucket 全部就绪:
     -> Reducer::finalize_bucket()
     -> 启动 all-reduce (ProcessGroupNCCL::allreduce)
     -> mark_variable_ready_done()
  -> 所有 all-reduce 完成:
     -> 将 all-reduce 后的梯度写回 param.grad
```

## 3. 核心源码文件

```
torch/nn/parallel/distributed.py                      # DistributedDataParallel Python 层
torch/csrc/distributed/c10d/reducer.cpp                # Reducer C++ 核心实现
torch/csrc/distributed/c10d/reducer.hpp                # Reducer 头文件
torch/csrc/distributed/c10d/comm.hpp                   # 通信相关
torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp       # NCCL all-reduce 实现
torch/distributed/algorithms/ddp_comm_hooks/           # DDP communication hook 扩展
  ├── default_hooks.py                                 # 默认 hook
  └── quantization_hooks.py                            # 量化通信 hook
torch/csrc/distributed/c10d/reducer_timer.h            # 性能分析计时器
```

## 4. 关键机制源码解读

### 4.1 Reducer 如何注册 autograd hook

在 `reducer.cpp:Reducer::initialize_buckets()` 中：

```cpp
for (auto& variable : variables) {
    auto& variable_index = variable_indices_[&variable];
    // 为每个参数注册 autograd hook
    auto hook = [this, variable_index](const Variable& grad) {
        this->autograd_hook(variable_index);
    };
    variable.register_hook(hook);
}
```

这样，当 autograd engine 计算出某个参数的梯度后，立刻触发 hook，通知 Reducer 该参数已就绪。

### 4.2 参数分 bucket 策略

```cpp
// reducer.cpp: Reducer::initialize_buckets()
// 参数按以下规则分组为一个 bucket:
// 1. 按注册顺序遍历参数
// 2. 累积参数大小直到达到 bucket_size
// 3. 同一 bucket 内参数共享一个 all-reduce

// 默认 bucket_size: ~25MB
// 可通过 DDP(bucket_cap_mb=25) 调整
```

Bucket 大小的影响：
- **过小**：all-reduce 次数多，launch overhead 高
- **过大**：通信计算重叠效果差，尾部延迟长
- **理想值**：刚好让一个 all-reduce 可以和一个前向/反向计算重叠

### 4.3 Bucket 就绪与 all-reduce 触发

```cpp
// reducer.cpp
void Reducer::autograd_hook(VariableIndex index) {
    auto& bucket = find_bucket(index);
    bucket.pending--;  // 标记一个参数就绪

    if (bucket.pending == 0) {
        // bucket 内所有参数梯度已就绪
        finalize_bucket(bucket);
        // 启动 all-reduce
        ProcessGroupNCCL::allreduce(bucket.gradients);
    }
}
```

### 4.4 通信计算重叠

all-reduce 是异步的。在 `ProcessGroupNCCL::allreduce` 中：

```
allreduce 启动后立即返回 (non-blocking)
  -> NCCL kernel 在 GPU 上异步执行
  -> CPU 侧的 autograd 继续处理其他参数的 backward
  -> 最终 loss.backward() 完成时:
     Reducer 调用 wait() 等待所有 in-flight all-reduce 完成
```

这就是 bucket 的核心价值：**部分参数还在计算梯度时，已完成梯度的 bucket 已经开始 all-reduce**，计算和通信在 GPU 上时间重叠。

### 4.5 find_unused_parameters=True 的影响

当 `find_unused_parameters=True` 时：

1. Reducer 在前向传播后记录哪些参数被使用
2. 启动一次额外的 **前向 all-reduce** 来同步哪些参数未使用
3. 反向传播时等待所有 bucket 就绪（包括未使用参数的 bucket）
4. 开销：额外一次同步 + 所有 bucket 都需等待，无法利用部分 bucket 提前 all-reduce

### 4.6 DDP communication hook

```python
# torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py

def allreduce_hook(state, bucket):
    # 替代默认 all-reduce
    bucket.set_buffer(bucket.gradients())
    fut = state.process_group.allreduce(bucket.buffer())
    return fut
```

Hook 接管 bucket 通信，可以实现梯度压缩（量化、稀疏化）等高级策略。

## 5. 和已有笔记的连接

```
distributed_techniques/collective_operations/   — All-reduce 等集合通信操作
distributed_techniques/parallelism_strategies/  — DDP 是数据并行的基础
autograd/                                       — Reducer 依赖 autograd hook
torch.compile/                                  — DDP + compile 时的梯度同步区别
```

## 6. 常见坑点

- **Bucket 大小不是越小越好**：过小的 bucket 导致大量小 all-reduce，NCCL 的 launch overhead 占主导。
- **`find_unused_parameters=True` 增加一次额外同步**，不要在不必要时开启。
- **DDP 和 torch.compile 一起使用时**，Reducer 的 hook 注册时机不同：compiled model 的 grad 在编译图执行完成后才产生。
- **NCCL all-reduce 默认是同步的**（从 Reducer 视角），但底层的 CUDA kernel 是异步的，通过 `cudaStreamSynchronize` 同步。
- **`device_ids` 不匹配时**，DDP 可能静默地执行 CPU-GPU 拷贝，严重影响性能。
- **梯度累加 (`accumulate_grads`) 场景**：autograd hook 在每次 backward 都会触发，需要 Reducer 正确跳过非最终累加轮次。

## 7. 阅读源码时建议搜索的关键词

```bash
# Reducer 初始化
rg -n "Reducer::initialize_buckets" torch/csrc/distributed/c10d/reducer.cpp

# autograd hook 注册
rg -n "register_hook" torch/csrc/distributed/c10d/reducer.cpp

# bucket 就绪判断
rg -n "finalize_bucket" torch/csrc/distributed/c10d/reducer.cpp

# find_unused_parameters 逻辑
rg -n "find_unused" torch/csrc/distributed/c10d/reducer.cpp

# DDP communication hook
rg -n "ddp_comm_hooks" torch/nn/parallel/distributed.py

# NCCL all-reduce
rg -n "allreduce" torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
```
