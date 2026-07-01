# cuDNN / cuBLAS Backend：库调度与算子选择

> 源码: `aten/src/ATen/native/cudnn/`, `aten/src/ATen/native/cuda/`, `torch/backends/cudnn/`

## 0. 一句话总览

PyTorch 的卷积、矩阵乘等密集型算子并不总是执行自家 CUDA kernel，而是优先尝试调用 cuDNN/cuBLAS 库的高性能实现。backend 选择由启发式规则决定，受 `torch.backends.cudnn.benchmark` 等 flag 控制。

## 1. 最小例子

```python
import torch

# 控制 cuDNN 行为的关键 flag
torch.backends.cudnn.benchmark = True   # 自动搜索最快算法
torch.backends.cudnn.deterministic = False  # 允许非确定性算法

conv = torch.nn.Conv2d(3, 16, 3).cuda()
x = torch.randn(2, 3, 32, 32, device="cuda")

# CuDNN 内部: 从算法缓存选最优卷积实现
y = conv(x)
print(f"conv2d output: {list(y.shape)}")

# 清空算法缓存
torch.backends.cudnn.benchmark_limit = 10  # benchmark 最多尝试 10 种算法
```

## 2. 实战例子

### 2.1 cuDNN 算法选择与 benchmark

```python
import torch
import time

if torch.cuda.is_available():
    conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
    x = torch.randn(8, 64, 56, 56, device="cuda")

    # benchmark=True: 首次调用的输入 shape 会被 benchmark 所有算法
    # 之后调用复用找到的最优算法
    torch.backends.cudnn.benchmark = True

    for _ in range(10):
        _ = conv(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        _ = conv(x)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0

    print(f"cuDNN conv time: {t*1000/100:.3f}ms")
    print(f"Algorithm selected from cache after first benchmark")
```

### 2.2 cuBLAS 矩阵乘

```python
if torch.cuda.is_available():
    # torch.matmul 后端:
    # 1. cuBLAS (默认, sgemm)
    # 2. cuBLASLt (layout 灵活)
    # 3. Triton (torch.compile 下 Inductor 生成的)

    A = torch.randn(4096, 4096, device="cuda")
    B = torch.randn(4096, 4096, device="cuda")

    # Eager cuBLAS
    t0 = time.perf_counter()
    C = A @ B
    torch.cuda.synchronize()
    t_cublas = time.perf_counter() - t0

    # Inductor (可能用 Triton gemm)
    @torch.compile
    def f(a, b):
        return a @ b

    f(A, B)  # warmup
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(5):
        f(A, B)
    torch.cuda.synchronize()
    t_inductor = (time.perf_counter() - t1) / 5

    print(f"cuBLAS matmul:   {t_cublas*1000:.2f}ms")
    print(f"Inductor matmul:  {t_inductor*1000:.2f}ms")
```

### 2.3 手动禁用 cuDNN

```python
# 某些场景需要回退到 PyTorch 原生实现
# 原因: 精度、确定性、特定 shape 下的性能

torch.backends.cudnn.enabled = False

conv = torch.nn.Conv2d(3, 16, 3).cuda()
x = torch.randn(2, 3, 32, 32, device="cuda")
y = conv(x)  # 回退到 PyTorch 的 CUDA kernel

torch.backends.cudnn.enabled = True  # 恢复
print(f"Native conv output: {list(y.shape)}")
```

## 3. 核心源码文件

```
aten/src/ATen/native/cudnn/Conv.cpp               # cuDNN 卷积调度
aten/src/ATen/native/cudnn/ConvPlaceholders.cpp   # 占位符(链接时解析)
aten/src/ATen/native/cuda/Blas.cpp                # cuBLAS 矩阵乘
aten/src/ATen/cudnn/Handle.h                      # cuDNN handle 管理
torch/backends/cudnn/__init__.py                  # Python flag 接口
```

## 4. 和已有笔记的连接

```
25_inductor/        — Inductor 生成 Triton kernel 替代 cuBLAS
30_sdpa_attention/  — FlashAttention 替代 cuDNN attention
10_amp/             — AMP 下 cuDNN 选择 FP16 算法
42_cuda_arch/       — GPU 架构影响 cuDNN 算法选择
```

## 5. 搜索关键词

```bash
rg -n "cudnnConvolution" aten/src/ATen/native/cudnn/
rg -n "cublasSgemm\|cublasGemm" aten/src/ATen/native/cuda/
rg -n "benchmark" torch/backends/cudnn/__init__.py
rg -n "algorithm" aten/src/ATen/native/cudnn/Conv.cpp
```
