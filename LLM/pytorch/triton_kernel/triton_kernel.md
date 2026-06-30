# Custom Triton Kernel 算子生成源码分析

> Triton 文档: https://triton-lang.org — OpenAI 的 GPU 编程语言
> PyTorch 集成: `torch/autograd/function.py` — `Function.apply()` 自定义 autograd 算子
> Inductor 使用 Triton: `torch/_inductor/codegen/triton.py` — 自动生成 Triton kernel

## 0. 一句话总览

Triton = **用 Python 写 GPU kernel**，自动处理 tiling、向量化、shared memory。PyTorch 通过 `torch.autograd.Function` 包裹自定义 Triton kernel → 作为 drop-in replacement 使用，并获得 autograd 支持。

---

## 一、Triton 编程模型

### 1.1 与传统 CUDA 的对比

| | CUDA C++ | Triton |
|---|---|---|
| Thread 管理 | 手动 (block/thread 索引) | 自动 (program_id) |
| Shared Memory | 手动分配/同步 | 自动（编译器推断） |
| Tiling | 手动分块 | 通过注解自动 |
| 语法 | C++ 扩展 | Python decorator |
| 编译 | nvcc | Triton 编译器（LLVM-based） |

### 1.2 Triton kernel 结构

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    x_ptr, y_ptr, output_ptr,      # 数据指针
    n_elements,                      # 标量参数
    BLOCK_SIZE: tl.constexpr,        # 编译期常量
):
    # 每个 program 处理一个 block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load (可选 mask 防止越界)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    y = tl.load(y_ptr + offsets, mask=offsets < n_elements)

    # Compute
    output = x + y

    # Store
    tl.store(output_ptr + offsets, output, mask=offsets < n_elements)
```

### 1.3 调用 kernel

```python
# grid = (num_blocks,) — 每个 program 的 grid
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
my_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
```

---

## 二、PyTorch 集成: `torch.autograd.Function`

### 2.1 自定义 autograd 函数

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        output = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        my_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
        ctx.save_for_backward(x, y)  # 保存用于 backward
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output * y  # 简化的梯度
        grad_y = grad_output * x
        return grad_x, grad_y
```

### 2.2 使用

```python
output = MyOp.apply(x, y)  # 完全 autograd 兼容
loss = output.sum()
loss.backward()
```

---

## 三、Triton 关键优化技巧

### 3.1 Tiling + Shared Memory

```python
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Shared memory tiles
    a_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    b_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(A + rm[:, None] * K + (k + rk)[None, :], mask=...)
        b_tile = tl.load(B + (k + rk)[:, None] * N + rn[None, :], mask=...)
        acc += tl.dot(a_tile, b_tile)

    tl.store(C + rm[:, None] * N + rn[None, :], acc, mask=...)
```

### 3.2 自动调优 (`@triton.autotune`)

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def tuned_kernel(...):
    ...
```

Triton 自动 benchmark 每种配置，为不同的 `n_elements` 选择最优的 `BLOCK_SIZE`。

---

## 四、PyTorch 内部如何使用 Triton

### 4.1 Inductor → Triton 代码生成

Inductor 将融合后的 IR 节点转换为 Triton kernel：

```
FX Graph → Inductor IR → Scheduler.fuse_nodes → Triton codegen → @triton.jit kernel
```

源码: `torch/_inductor/codegen/triton.py:3131` — `TritonKernel` 类，管理:
- `range_trees` — 循环变量
- `cse` — 公共子表达式缓存
- `prologue/body/suffix` — kernel 代码缓冲区
- `codegen_kernel()` — 输出最终 Triton 代码

### 4.2 自定义 Triton kernel 在 torch.compile 中

```python
@torch.compile
def forward(x):
    # torch.compile 自动对支持的 op 生成 Triton kernel
    return x * 2 + 1

# 等价于: Inductor 生成一个融合 Triton kernel 做 mul + add
```

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `torch.autograd.Function` | `torch/autograd/function.py` | — |
| `Function.apply` | `torch/autograd/function.py` | — |
| Inductor Triton codegen | `torch/_inductor/codegen/triton.py` | 3131 |
| TritonKernel 类 | `torch/_inductor/codegen/triton.py` | — |
| Scheduler fusion | `torch/_inductor/scheduler.py` | 5102 |

---

## 六、可借鉴的工程技巧

1. **声明式编程**: Triton 用 `@triton.jit` + program_id 抽象 GPU 并行模型，程序员写 block 级逻辑，编译器处理 thread 级调度。

2. **autotune**: 运行时 benchmark 多种配置选最优 → 不依赖手工调优 → 跨 GPU 代际适配。

3. **PyTorch 集成模式**: `autograd.Function` 作为 Triton kernel 的 Python 包装器 → 获得 backward + device 检查 + dtype 转换全免费。

4. **mask 防越界**: `tl.load(ptr, mask=offsets < n_elements, other=0)` → 自动处理不规则尺寸（如矩阵一行 1025 个元素）。

5. **共享内存透明化**: Triton 编译器自动推断哪些数据需要放 shared memory，程序员只需描述计算逻辑。
