# Triton 用法总结

## 基础工作流

```python
import torch
import triton
import triton.language as tl

# 1. 定义 kernel（@triton.jit）
# 2. 调用 kernel[grid](args)
# 3. 测试验证
```

---

## 1. Kernel 定义

### 装饰器与编译期常量

```python
@triton.jit
def my_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    ...
```

- `@triton.jit` 将 Python 函数编译为 GPU kernel
- `tl.constexpr` 标记编译期常量（编译时确定，常用于 tile 尺寸），必须用关键字参数传入

### 获取当前 block 编号

```python
pid = tl.program_id(0)         # 一维 grid：1D 索引
pid_m = tl.program_id(0)       # 二维 grid：行方向 (M)
pid_n = tl.program_id(1)       # 二维 grid：列方向 (N)
```

- `tl.program_id(axis)` 返回当前 program（block）在 grid 各维度的索引

### 构造偏移量

```python
# 一维
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

# 二维（配合 broadcasting 做地址计算）
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # shape: [BLOCK_M]
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)        # shape: [BLOCK_N]
offs_k = tl.arange(0, BLOCK_K)                          # shape: [BLOCK_K]

# 二维展开：[BLOCK_M, 1] * stride + [1, BLOCK_K] * stride
a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak
```

- `tl.arange(start, end)` 生成 [start, end) 的整数序列
- `[:, None]` 和 `[None, :]` 做 broadcasting 对齐维度

### 边界保护（Mask）

```python
mask = offsets < n                                    # 一维
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)  # 二维
```

---

## 2. 数据搬运

### 从全局内存加载

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

- `mask`：越界位置跳过加载
- `other`：越界位置填充该值
- 特殊值：`float("-inf")` 用于 max reduction 的安全性

### 写入全局内存

```python
tl.store(z_ptr + offsets, value, mask=mask)
```

- `mask`：只写入有效位置，越界位置跳过

---

## 3. 元素级运算

```python
z = x + y
z = x * 2.0
z = tl.maximum(x, 0.0)      # relu（逐元素）
z = tl.exp(x)                # exp
```

---

## 4. 规约运算（Reduction）

```python
x_max = tl.max(x, axis=0)    # 沿 axis=0 取最大值 → 标量
x_sum = tl.sum(x, axis=0)    # 沿 axis=0 求和 → 标量
```

- Reduction 操作的输入通常是一维向量，输出是标量
- 配合 broadcasting：`x - x_max` 自动将标量广播回向量

### 数值稳定的 softmax 模式

```python
x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
x_max = tl.max(x, axis=0)    # 减最大值防溢出
x = tl.exp(x - x_max)
x_sum = tl.sum(x, axis=0)
y = x / x_sum
```

---

## 5. Tiled Matmul 模式

### 矩阵分块与地址计算

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,      # A 的各维度步长
    stride_bk, stride_bn,      # B 的各维度步长
    stride_cm, stride_cn,      # C 的各维度步长
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        # 加载 A 的 tile  [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak
        a = tl.load(a_ptrs, mask=..., other=0.0)

        # 加载 B 的 tile  [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=..., other=0.0)

        # 矩阵乘法累加
        acc += tl.dot(a, b)

    # 写回结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)
```

- `tl.zeros((M, N), dtype)` 创建零矩阵（寄存器中）
- `tl.dot(a, b)` 矩阵乘法，a 的最后一维和 b 的倒数第二维做收缩

---

## 6. Launch

```python
# grid 计算
grid = (triton.cdiv(n, BLOCK_SIZE),)                    # 一维
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # 二维

# 启动
kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE)
```

- `triton.cdiv(a, b)` 向上取整 `ceil(a / b)`
- 编译期常量（`BLOCK_SIZE` 等）必须用关键字参数传入
- 动态参数（tensor、标量）用位置参数或关键字参数均可

### 二维 grid 的约定

- program_id(0) → 列方向（N）
- program_id(1) → 行方向（M）

---

## 7. Benchmark 工具

```python
ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=[0.5, 0.2, 0.8])
```

---

## 8. 工具

```python
BLOCK_SIZE = triton.next_power_of_2(N)   # 不小于 N 的最小 2 的幂
```

用于 softmax 等需要固定 tile 大小的场景。

---

## 9. 完整示例速查

### 一维逐元素（vector add）

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(z_ptr + offsets, x + y, mask=mask)

def add(x, y):
    z = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, z, n, BLOCK_SIZE=1024)
    return z
```

### 一维 softmax

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0) * N
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    x = tl.exp(x - tl.max(x, axis=0))
    y = x / tl.sum(x, axis=0)
    tl.store(y_ptr + offsets, y, mask=mask)

def softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, y, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return y
```

### 二维 tiled matmul

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak,
                    mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```
