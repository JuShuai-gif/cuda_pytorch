# DeviceGuard、StreamGuard 与 `to()` / copy 语义

> CUDA Guard: `c10/cuda/CUDAGuard.h`
> CUDA Stream: `c10/cuda/CUDAStream.h`
> Device Guard: `c10/core/DeviceGuard.h`
> Copy 实现: `aten/src/ATen/native/cuda/Copy.cu`、`aten/src/ATen/native/Copy.cpp`
> `to()` 实现: `aten/src/ATen/native/TensorConversions.cpp`
> Python 绑定: `torch/csrc/cuda/Module.cpp`

## 0. 一句话总览

`tensor.to("cuda")` 不是修改 tensor 的 device 字段，而是通过 ATen copy kernel 创建一个指定 device 的新 tensor 并拷贝数据。DeviceGuard 保护代码块中的默认 device，StreamGuard 保护默认 CUDA stream。`non_blocking=True` 允许（但不保证）异步 H2D/D2H copy，需要 pinned memory 才能真正异步。

## 1. 最小例子

```python
import torch

cpu = torch.randn(1024, 1024)
pinned = torch.randn(1024, 1024, pin_memory=True)

cuda1 = cpu.to("cuda", non_blocking=True)     # 可能异步（因 CPU tensor 非 pinned）
cuda2 = pinned.to("cuda", non_blocking=True)   # 真正异步（pinned memory）

torch.cuda.synchronize()
```

DeviceGuard 使用示例：

```python
import torch

with torch.cuda.device(1):
    # 此代码块中默认 device 是 cuda:1
    x = torch.randn(3, device="cuda")  # 在 cuda:1 上
    with torch.cuda.device(0):
        y = torch.randn(3, device="cuda")  # 在 cuda:0 上
    z = torch.randn(3, device="cuda")  # 回到 cuda:1
```

## 1.5 实战例子

### 1.5.1 用 pinned memory + non_blocking 实现数据加载重叠

在 DataLoader 中用 pinned memory + `non_blocking=True` 使 H2D copy 与 CPU 预处理重叠：

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import time

data = torch.randn(1000, 1024)
labels = torch.randint(0, 10, (1000,))
dataset = TensorDataset(data, labels)

# 开启 pin_memory
loader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=4)

model = torch.nn.Linear(1024, 10).cuda()

for epoch in range(10):
    t0 = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        # non_blocking=True 使 H2D copy 异步执行
        x_gpu = x.cuda(non_blocking=True)
        y_gpu = y.cuda(non_blocking=True)

        # 训练步骤 - 此时 H2D copy 可能还在进行
        # CUDA 会在需要数据时自动同步
        out = model(x_gpu)
        loss = torch.nn.functional.cross_entropy(out, y_gpu)

        # 手动同步点
        if batch_idx % 10 == 0:
            torch.cuda.synchronize()
    epoch_time = time.time() - t0
    print(f"Epoch {epoch}: {epoch_time:.2f}s")
```

对比 `pin_memory=False` 时的耗时，通常 pinned + non_blocking 可减少 20-40% 的 H2D 延迟。

### 1.5.2 排查多 stream 下 copy_ 的数据竞争

当使用自定义 CUDA stream 做异步计算时，`non_blocking` copy 可能引起数据竞争：

```python
import torch

# 默认 stream
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")

# 创建计算 stream
compute_stream = torch.cuda.Stream()

# 问题场景:
with torch.cuda.stream(compute_stream):
    z = x @ y  # 在 compute_stream 上做矩阵乘法
    # copy_ 在默认 stream 上执行, 与 compute_stream 不同步
    w = z.cpu(non_blocking=True)  # D2H copy 可能在计算完成前开始!

# 解决方案: 在 copy 前插入 stream 同步事件
event = torch.cuda.Event()
event.record(stream=compute_stream)
# 等 compute_stream 完成后再 copy
event.synchronize()
w = z.cpu(non_blocking=False)

# 或使用 StreamGuard:
with torch.cuda.stream(compute_stream):
    z = x @ y
# CUDAStreamGuard 自动恢复默认 stream
w = z.cpu()  # 现在在默认 stream 上, 但 compute_stream 已完成
```

### 1.5.3 用 DeviceGuard 隔离多 GPU 推理的 device 状态

在 multi-GPU 推理服务中，DeviceGuard 确保每个请求不会串 device：

```python
import torch

class MultiGPUInference:
    def __init__(self, model_fn, devices=[0, 1, 2, 3]):
        self.devices = devices
        self.models = {d: model_fn().cuda(d) for d in devices}

    def infer(self, x, target_device=None):
        if target_device is None:
            target_device = self.devices[0]

        # DeviceGuard 确保 device 切换安全
        with torch.cuda.device(target_device):
            # 此代码块内所有无 device 的 tensor 都在 target_device
            model = self.models[target_device]
            # 即使 x 在其他 device, to() 会处理跨 device copy
            return model(x.to("cuda"))

# 使用
service = MultiGPUInference(
    lambda: torch.nn.Linear(1024, 10)
)

x = torch.randn(1, 1024)
out1 = service.infer(x, target_device=0)  # 在 GPU 0
out2 = service.infer(x, target_device=1)  # 在 GPU 1, 不会影响 GPU 0 的状态
```

## 2. 从 Python API 到源码的调用链

```
tensor.to("cuda")
  -> TensorConversions.cpp: to() -> to_device()
  -> Copy.cpp: copy_() (通用 copy kernel)
  -> 如果 device 不同:
     -> cuda/Copy.cu: copy_kernel_cuda
        -> cudaMemcpyAsync (non_blocking) 或 cudaMemcpy (blocking)
  -> 返回新 tensor (新 device 上)

torch.cuda.device(1)
  -> CUDAGuard.h: CUDAGuard
  -> 构造时保存当前 device, 设置 target device
  -> 析构时恢复 device

non_blocking True:
  -> copy_() 设置 Async 标志
  -> Copy.cu 中调用 cudaMemcpyAsync
  -> 仅当源 tensor 是 pinned memory 时才真正异步
```

## 3. 核心源码文件

```
c10/cuda/CUDAGuard.h                               # CUDA DeviceGuard
c10/cuda/CUDAStream.h                              # CUDA Stream
c10/cuda/CUDAGuardImpl.h                           # Guard 策略实现
c10/core/DeviceGuard.h                             # 通用 DeviceGuard
aten/src/ATen/native/cuda/Copy.cu                  # CUDA copy kernel
aten/src/ATen/native/Copy.cpp                      # 通用 copy 逻辑
aten/src/ATen/native/TensorConversions.cpp         # to() 实现
torch/csrc/cuda/Module.cpp                         # Python 绑定 (torch.cuda)
aten/src/ATen/native/TypeProperties.cpp            # dtype/device 兼容性
```

## 4. 关键机制源码解读

### 4.1 `to()` 不是修改 device，而是 copy

```cpp
// aten/src/ATen/native/TensorConversions.cpp
Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking) {
    if (self.device() == device && self.dtype() == dtype) {
        return self;  // 同 device/dtype: 直接返回自身
    }
    // 不同 device: 创建新 tensor + copy_
    Tensor out = at::empty(self.sizes(), self.options().device(device).dtype(dtype));
    out.copy_(self, non_blocking);
    return out;
}
```

关键认识：`to()` 在 device 不同时**一定产生新 tensor**，不是原地修改。

### 4.2 DeviceGuard 和 StreamGuard

```cpp
// c10/cuda/CUDAGuard.h
class CUDAGuard {
    // RAII: 构造时设置 device，析构时恢复
    CUDAGuard(Device device) {
        prev_device_ = getCurrentCUDADevice();
        setCurrentCUDADevice(device);
    }
    ~CUDAGuard() {
        setCurrentCUDADevice(prev_device_);
    }
};

// c10/cuda/CUDAStream.h
class CUDAStreamGuard {
    // RAII: 构造时切换 stream，析构时恢复
    CUDAStreamGuard(CUDAStream stream) {
        prev_stream_ = getCurrentCUDAStream();
        setCurrentCUDAStream(stream);
    }
    ~CUDAStreamGuard() {
        setCurrentCUDAStream(prev_stream_);
    }
};
```

DeviceGuard 保护的是隐式 `cudaSetDevice` 调用的线程局部状态。StreamGuard 保护的是当前 CUDA stream。两者都是 RAII 语义。

### 4.3 non_blocking=True: 允许异步，不保证异步

```cpp
// aten/src/ATen/native/cuda/Copy.cu
void copy_kernel_cuda(Tensor& dst, const Tensor& src, bool non_blocking) {
    if (non_blocking && src.is_pinned()) {
        // 真正异步: cudaMemcpyAsync, 不阻塞 CPU
        cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), bytes,
                       cudaMemcpyDefault, stream);
    } else if (non_blocking) {
        // non_blocking 但 src 未 pinned: 实际上走同步路径
        // (PyTorch 实现中可能仍用 cudaMemcpyAsync, 但主机内存可能被回收)
        cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), bytes,
                       cudaMemcpyDefault, stream);
        // 但仍需确保 src 在 copy 完成前不被修改
    } else {
        // 同步 copy
        cudaMemcpy(dst.data_ptr(), src.data_ptr(), bytes, cudaMemcpyDefault);
    }
}
```

`non_blocking=True` 且使用 pinned memory 时，H2D copy 可以和 CPU 计算重叠。pinned memory 通过 `cudaHostRegister` 锁定物理页面，确保 GPU DMA 可以安全访问。

### 4.4 各种 copy 对应的 ATen op

| Python API | ATen op | 说明 |
|-----------|---------|------|
| `tensor.to("cuda")` | `aten::to.device` | 创建新 tensor + copy |
| `tensor.cuda()` | `aten::_copy_from_and_resize` | 等价于 `to("cuda")` |
| `src.copy_(dst)` | `aten::copy_` | 核心 copy op |
| `tensor.pin_memory()` | `aten::pin_memory` | 分配 pinned buffer |

## 5. 和已有笔记的连接

```
cuda_stream/            — Stream 与 copy 异步语义紧密相关
tensor/                 — Tensor storage 模型决定 copy 行为
memory_allocator/       — Pinned memory 分配、CUDA allocator
torch.compile/          — Compile 时 device/copy 的特殊处理
dispatcher/             — copy_ 也通过 Dispatcher 路由到后端实现
```

## 6. 常见坑点

- **`non_blocking=True` 不保证异步**，只有 pinned memory 的 H2D copy 才真正异步。
- **`tensor.to("cuda")` 产生的 tensor 和原始 tensor 不共享 storage**，是独立拷贝。
- **`with torch.cuda.device(i)` 只影响默认 device**，不影响显式指定 device 的 tensor。
- **多 stream 场景下 `non_blocking` copy 可能引起数据竞争**，需要 stream 同步。
- **DeviceGuard 不是线程安全的**，仅在单线程内有效。
- **`pin_memory=True` 的 DataLoader 会为每个 batch 分配 pinned buffer**，注意显存/内存权衡。

## 7. 阅读源码时建议搜索的关键词

```bash
# copy kernel 实现
rg -n "copy_kernel_cuda" aten/src/ATen/native/cuda/Copy.cu

# to() 实现
rg -n "Tensor to\(" aten/src/ATen/native/TensorConversions.cpp

# CUDAGuard 实现
rg -n "class CUDAGuard" c10/cuda/CUDAGuard.h

# CUDAStreamGuard 实现
rg -n "class CUDAStreamGuard" c10/cuda/CUDAStream.h

# non_blocking 处理
rg -n "non_blocking" aten/src/ATen/native/cuda/Copy.cu

# pinned memory 检测
rg -n "is_pinned" aten/src/ATen/native/cuda/Copy.cu
```
