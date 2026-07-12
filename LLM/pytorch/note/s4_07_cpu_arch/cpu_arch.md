# CPU 架构与内存层次：PyTorch 性能的底层基础

> 关联: AVX/SSE 向量化、cache line、NUMA、线程池、false sharing

## 0. 一句话总览

理解 CPU cache hierarchy、SIMD 向量化、NUMA 拓扑和线程调度，是诊断 PyTorch CPU 数据加载瓶颈、pin_memory 开销、以及为什么 "行优先比列优先快" 的前提。

## 1. 最小例子

```python
import torch
import time

# Row-major vs column-major access: cache line effect
N = 4096
x = torch.randn(N, N)

# Row-major (cache-friendly: contiguous stride=1)
t0 = time.perf_counter()
for i in range(N):
    _ = x[i].sum()          # stride-1 access
t_row = time.perf_counter() - t0

# Column-major (cache-unfriendly: stride=N)
x_t = x.t().contiguous()
t1 = time.perf_counter()
for j in range(N):
    _ = x_t[j].sum()        # also stride-1, but data was transposed
t_col = time.perf_counter() - t1

print(f"Row-major: {t_row:.3f}s, Column-major: {t_col:.3f}s")
```

## 2. 实战例子

### 2.1 Cache Line 对齐与 False Sharing

```python
import torch

# False sharing: 不同线程修改同一 cache line 的不同元素
# PyTorch 内部用 cache line 对齐避免
print(f"CPU cache line size (typical): 64 bytes")
print(f"torch.float32 element: 4 bytes -> 16 elements per cache line")

# 观察 PyTorch 内存对齐
x = torch.randn(1024)
print(f"Tensor data_ptr mod 64: {x.data_ptr() % 64}")
# 通常为 0 -> PyTorch 默认 64-byte 对齐
```

### 2.2 SIMD 向量化检测

```python
import torch

# PyTorch 编译时的向量化支持
print(f"CPU features: {torch._C._check_cpu_feature_present()}")

# MKL/OpenBLAS 在背后用 AVX-512 加速 matmul
A = torch.randn(1024, 1024)
B = torch.randn(1024, 1024)

import time
t0 = time.perf_counter()
C = A @ B
t = time.perf_counter() - t0
print(f"matmul(1024x1024): {t*1000:.1f}ms (uses SIMD under the hood)")

# 等价于: cblas_sgemm with AVX-512 kernel
```

### 2.3 NUMA 感知与线程绑定

```python
import torch

# torch.set_num_threads 控制 MKL 线程池
torch.set_num_threads(4)
print(f"Thread count: {torch.get_num_threads()}")

# NUMA: 跨 socket 访问内存慢 2x+
# PyTorch 内部: at::parallel_for 用线程池 + work stealing
# 来源: aten/src/ATen/ParallelNative.cpp
```

## 3. 核心源码文件

```
aten/src/ATen/Parallel.h               # at::parallel_for 线程池
aten/src/ATen/native/cpu/             # CPU kernel (用 SIMD intrinsic)
c10/util/AlignOf.h                     # 对齐工具
aten/src/ATen/CPUFunctions.h          # CPU 函数声明(由 torchgen 生成)
third_party/ideep/                     # Intel oneDNN 集成
```

## 4. 和已有笔记的连接

```
01_tensor/        — tensor stride 影响 cache line 访问
02_device_copy/   — pin_memory 涉及虚拟内存页锁定
05_dataloader/    — DataLoader 多进程 NUMA 感知
29_memory_allocator/ — CPU allocator 与线程池交互
```

## 5. 搜索关键词

```bash
rg -n "parallel_for" aten/src/ATen/Parallel.h
rg -n "AVX|SSE|SIMD" aten/src/ATen/native/cpu/
rg -n "align" c10/util/AlignOf.h
```
