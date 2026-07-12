# RNG、Generator 与 checkpoint 随机数语义

> Python 端: `torch/random.py`
> C++ CPU: `aten/src/ATen/CPUGeneratorImpl.cpp`
> C++ CUDA: `aten/src/ATen/cuda/CUDAGeneratorImpl.cpp`
> Philox: `aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh`
> Checkpoint: `torch/utils/checkpoint.py`

## 0. 一句话总览

PyTorch 的随机数系统分为 CPU 和 CUDA 两个独立的 Generator，各自维护自己的状态。CUDA 使用 Philox 算法，通过 seed+offset 实现细粒度的并行随机数生成。当 activation checkpoint 与 dropout/BatchNorm 等随机 op 共存时，必须在反向时恢复前向的 RNG 状态才能保证语义正确。

## 1. 最小例子

```python
import torch
from torch.utils.checkpoint import checkpoint

dropout = torch.nn.Dropout(p=0.5)
x = torch.ones(8, requires_grad=True)

def f(t):
    return dropout(t).sum()

torch.manual_seed(0)
y1 = f(x)

torch.manual_seed(0)
y2 = checkpoint(f, x, use_reentrant=False)

print(y1, y2)  # 应该相同（种子相同 + checkpoint 正确恢复 RNG）
```

## 1.5 实战例子

### 1.5.1 多卡场景下设置 RNG 确保可复现性

在 DDP 训练中，不同 rank 上的 dropout、数据增强等随机 op 必须可控：

```python
import torch
import torch.distributed as dist

def setup_reproducibility(rank, world_size):
    # 每个 rank 设置不同的种子
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    # 关键: CUDA conv 的确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 验证 RNG 状态
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    # 多 GPU: 每个设备独立状态
    if torch.cuda.device_count() > 1:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                state = torch.cuda.get_rng_state()
                # 确保每个设备的 RNG 状态符合预期

# 加速设置: 用 Philox 的 seed-only 初始化
# 当 CUDA generator 只用 seed 时, offset 自动递增
gen = torch.Generator(device="cuda")
gen.manual_seed(42)
# gen.initial_seed() -> 42
```

### 1.5.2 排查 checkpoint + dropout 导致的随机数不一致

激活 checkpoint 中如果 RNG 恢复不正确，backward 重算时的 dropout mask 与前向不同：

```python
import torch
from torch.utils.checkpoint import checkpoint

dropout = torch.nn.Dropout(0.5)

def faulty_checkpoint_forward():
    # 错误的: 在 checkpoint 区域内修改全局 RNG
    torch.manual_seed(99)  # 这会导致 backward 重算时 seed 不同
    return dropout(torch.ones(8, requires_grad=True))

x = torch.ones(8, requires_grad=True)

# 正确做法: 不要修改 checkpoint 内部的 RNG
def correct_forward(t):
    return dropout(t).sum()

try:
    # use_reentrant=True 时用户负责 RNG
    # use_reentrant=False 时自动保存/恢复
    y = checkpoint(correct_forward, x, use_reentrant=False)
    y.backward()
    print("梯度:", x.grad)  # 正确的梯度
except RuntimeError as e:
    print(f"RNG 错误: {e}")
```

排查时可以通过 `torch.cuda.get_rng_state()` 在 checkpoint 前后对比 RNG 状态是否一致。

### 1.5.3 torch.compile + Dropout 随机数语义验证

验证 compile 后的 dropout 是否每次产生不同 mask：

```python
import torch

dropout = torch.nn.Dropout(0.5)

@torch.compile
def f(x):
    return dropout(x)

x = torch.ones(8)

# 同一输入多次调用, 观察 dropout mask 是否不同
out1 = f(x)
out2 = f(x)
out3 = f(x)

print("out1:", out1)
print("out2:", out2)
print("out3:", out3)
# 应各不相同 (RNG 正常推进)

# 若全部相同, 说明 compile 缓存了随机数
# 检查: torch._inductor.lowering 中对 dropout 的处理
# 在 inductor 中, dropout 被特殊处理以正确推进 RNG
```

如果发现 compile 后 dropout mask 固定不变，检查 `torch._inductor.lowering.py` 中 `fallback_dropout` 的实现。

## 2. 从 Python API 到源码的调用链

```
torch.manual_seed(42)
  -> torch/random.py: manual_seed()
  -> C: at::manual_seed(42)
  -> CPUGeneratorImpl::set_current_seed(42)
  -> CUDAGeneratorImpl::set_current_seed(42)

torch.randn(3)
  -> aten::randn -> Dispatcher -> CPU/CUDA kernel
  -> CPU: std::mt19937 (从 CPUGeneratorImpl 获取状态)
  -> CUDA: curandStatePhilox (从 CUDAGeneratorImpl 获取 seed/offset)

checkpoint(f, x, use_reentrant=False)
  -> torch/utils/checkpoint.py: CheckpointFunction
  -> forward: 保存 RNG 状态 (torch.get_rng_state, torch.cuda.get_rng_state)
  -> forward: 执行 f(x)
  -> backward: 恢复 RNG 状态 (torch.set_rng_state, torch.cuda.set_rng_state)
  -> backward: 重新执行 f(x) 得到梯度
```

## 3. 核心源码文件

```
torch/random.py                                      # random API
aten/src/ATen/Generator.h                            # Generator 基类
aten/src/ATen/CPUGeneratorImpl.cpp                   # CPU RNG 实现 (mt19937)
aten/src/ATen/cuda/CUDAGeneratorImpl.cpp             # CUDA RNG 实现 (Philox)
aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh     # Philox CUDA 状态结构
c10/core/Generator.h                                 # c10 Generator 类型
torch/utils/checkpoint.py                            # Activation checkpoint
torch/csrc/generic/THPRandom.cpp                     # Python RNG binding
```

## 4. 关键机制源码解读

### 4.1 CPU RNG vs CUDA RNG

| 特性 | CPU Generator | CUDA Generator |
|------|--------------|----------------|
| 算法 | mt19937 (Mersenne Twister) | Philox (counter-based) |
| 状态 | ~2.5KB (624 个 word + index) | seed (uint64) + offset (uint64) |
| 状态大小 | 大，不可在 kernel 间传递 | 很小，适合并行 |
| 线程安全 | 需要 mutex | seed+offset 无共享状态 |
| 恢复成本 | 高（拷贝完整状态） | 低（拷贝两个 uint64） |

```cpp
// CPUGeneratorImpl.cpp
class CPUGeneratorImpl : public GeneratorImpl {
    std::mt19937 engine_;           // Mersenne Twister
    std::mutex mutex_;             // 线程安全
};

// CUDAGeneratorImpl.cpp
class CUDAGeneratorImpl : public GeneratorImpl {
    PhiloxCudaState philox_state_;  // seed + offset
    // offset 随每个 randn/rand 调用递增
};
```

### 4.2 Philox seed/offset 机制

Philox 是 counter-based PRNG：给定 `(seed, offset)`，直接计算第 `offset` 个随机数，**无需生成前面的随机数**。

```cpp
// aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh
struct PhiloxCudaState {
    uint64_t seed_;
    uint64_t offset_;  // 当前偏移
};

// CUDA kernel 中:
// Philox(seed, tid + offset)  每个线程直接计算自己的随机数
```

这为什么适合 CUDA:
- 每个 CUDA 线程可以计算 `Philox(seed, thread_id + global_offset)`
- 线程之间无状态依赖
- 退化到 CPU 时需要预先生成一整个 buffer (但 CPU 上不太常用 Philox)

### 4.3 Activation Checkpoint 的 RNG 保存/恢复

```python
# torch/utils/checkpoint.py (use_reentrant=False)
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        # 保存前向时的 RNG 状态
        ctx.rng_state = torch.get_rng_state()          # CPU RNG
        ctx.cuda_rng_state = torch.cuda.get_rng_state() # CUDA RNG
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 恢复 RNG 状态，确保重新执行时随机数相同
        torch.set_rng_state(ctx.rng_state)
        torch.cuda.set_rng_state(ctx.cuda_rng_state)
        # 重新执行 forward 来计算梯度
        ...
```

### 4.4 Dropout + Checkpoint + Compile 的随机数问题

当三者同时出现时：

1. **Dropout** 在前向生成随机 mask（消耗 RNG）
2. **Checkpoint** 保存 RNG 状态，backward 时恢复
3. **torch.compile** 捕获 forward 图并编译

问题在于：`torch.compile` 编译后的图可能会**重复使用同一个编译结果**，如果编译后的图中包含固定的随机 seed/offset，会导致每次调用都产生相同的 dropout mask。

解决方案：`torch.compile` 在编译时对随机 op 插入 `torch.rand` 调用，确保每次执行时 RNG 状态正常推进。参考 `torch/_inductor/lowering.py` 中对 dropout 的处理。

## 5. 和已有笔记的连接

```
checkpoint/     — Activation checkpoint 依赖 RNG 状态恢复
torch.compile/  — Compile 时的随机数语义需要特殊处理
sdpa_attention/ — FlashAttention 内部使用 Philox 随机数
inductor/       — Inductor lowering 中对随机 op 的处理
tensor/         — Generator 作为 Tensor 的可选属性
```

## 6. 常见坑点

- **CPU RNG 和 CUDA RNG 是独立的**：`torch.manual_seed` 同时设置两者，但 `torch.get_rng_state()` 只获取 CPU 状态。
- **Checkpoint 非重入模式 (`use_reentrant=False`) 正确保存 RNG 状态**，而 `use_reentrant=True` 模式下由用户负责。
- **Philox offset 在不同 backend 下递增方式不同**：同一个 seed 在 CPU 和 CUDA 上产生不同的随机序列。
- **多 GPU 场景下每个设备有独立的 CUDAGenerator**，`torch.cuda.manual_seed_all(seed)` 才能设置所有设备。
- **DataLoader 的 worker 进程各自有独立的 RNG**，需要 `worker_init_fn` 来正确设置。
- **编译图在不同随机 op 之间缺少同步点**，可能导致对 RNG 状态的读取顺序不一致。

## 7. 阅读源码时建议搜索的关键词

```bash
# CPUGenerator 状态
rg -n "class CPUGeneratorImpl" aten/src/ATen/CPUGeneratorImpl.cpp

# CUDA Generator Philox 状态
rg -n "class CUDAGeneratorImpl" aten/src/ATen/cuda/CUDAGeneratorImpl.cpp

# checkpoint RNG 保存/恢复
rg -n "rng_state|get_rng_state|set_rng_state" torch/utils/checkpoint.py

# Philox kernel 使用
rg -n "PhiloxCudaState" aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh

# manual_seed 实现
rg -n "manual_seed" torch/random.py

# inductor 中随机 op 的处理
rg -n "dropout|randn" torch/_inductor/lowering.py
```
