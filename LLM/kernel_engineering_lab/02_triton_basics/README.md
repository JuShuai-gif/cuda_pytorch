# 02_triton_basics - Triton Kernel 编程

## 工业背景

Triton 是使用 Python 编写 GPU kernel 的标准方式。它为以下项目提供支持：

| 项目 | Triton 用途 |
|---------|-------------|
| **vLLM** | PagedAttention kernel、融合 MoE、自定义量化 |
| **xFormers** | 内存高效 attention、FMHA、SwiGLU 融合 |
| **FlashInfer** | 采样、top-k/top-p、不规则 attention kernel |
| **OpenAI** | 训练栈 kernel（GPT-4、o1 训练） |
| **FlagGems** | 通过 Triton 实现完整 PyTorch 算子覆盖 |

## 何时使用 Triton vs CUDA C++

| 标准 | Triton | CUDA C++ |
|-----------|--------|----------|
| **开发速度** | 快（Python） | 慢（C++、构建系统） |
| **可移植性** | 可通过 ROCm Triton 编译到 AMD | 仅限 NVIDIA |
| **Warp 级操作** | 有限 | 完全控制 |
| **动态并行** | 不支持 | 支持 |
| **自动调优** | 内置 `@triton.autotune` | 手动 |
| **Tensor Core** | 通过 `tl.dot` 自动使用 | 显式 MMA 指令 |
| **内存控制** | 高级抽象 | 精确到字节级别的控制 |

**经验法则**：从 Triton 开始。只有当 Triton 无法表达你所需的功能时，才使用 CUDA C++。

## Block 编程模型

Triton 的核心抽象是 **program**（类似于 CUDA 的 block）：

```
Grid = [grid_m, grid_n]                # Grid 中的 program 数量
Program (pid_m, pid_n):                # 每个 program 处理一个 tile
    load  A[pid_m*BM : (pid_m+1)*BM, :]   # 从全局内存加载
    load  B[:, pid_n*BN : (pid_n+1)*BN]
    compute for k in steps(BLOCK_K):
        acc += A_block @ B_block
    store C[pid_m*BM : (pid_m+1)*BM, pid_n*BN : (pid_n+1)*BN]
```

与 CUDA C++ 的关键区别：
- **无线程级索引** - Triton 自动将工作分布到线程
- **无共享内存管理** - Triton 透明地管理它
- **基于掩码的边界处理** - `tl.load`/`tl.store` 上的 `mask` 参数

## 本模块中的 Kernel

### triton_vector_add
逐元素加法。演示最简单的 Triton 模式。

### triton_elementwise
对现代 LLM 至关重要的三种激活函数：
- **SiLU** (x * sigmoid(x))：用于 SwiGLU - LLaMA、Mistral、Gemma
- **GELU**（tanh 近似）：用于 BERT、GPT-2
- **ReLU** (max(0, x))：基础，但仍被广泛使用

### triton_gemm_basic
简单的分块矩阵乘法。刻意未做优化——这是后续模块的热身，
后续模块将添加分块策略、软件流水线和高级优化。

## 运行测试

```bash
pytest 02_triton_basics/test_triton_basics.py -v
```

## 运行基准测试

```bash
python 02_triton_basics/benchmark_triton_basics.py
```

## 常见陷阱

### Grid/Block 配置错误
- **Grid 太小**：GPU 利用不足
- **Grid 太大**：启动过多 program 的开销
- **Block 大小必须是 2 的幂**，适用于大多数 kernel
- 使用 `triton.cdiv(N, BLOCK_SIZE)` 正确计算 grid 大小

### 共享内存大小
Triton 自动管理共享内存，但复杂的 kernel 可能超出限制。
使用 `triton.autotune` 监控以找到最佳配置。

### 掩码边界情况
加载/存储时始终使用掩码。差一错误会导致静默数据损坏：
```python
mask = offsets < n_elements  # 对于非 2 的幂大小至关重要
```

### 自动调优开销
`@triton.autotune` 在运行时评估许多配置。对于生产环境，
要么缓存自动调优结果，要么在性能分析后硬编码最优配置。
