# GPU Kernel 工程学习路线图

一个为期 8 周的结构化课程，用于掌握使用 CUDA 和 Triton 进行 GPU kernel 开发。

## 前置条件

- 精通 Python 和 C++
- 熟悉 PyTorch tensor 操作
- 基本了解 GPU 架构（SM、warp、内存层次结构）

---

## 第 1 周：CUDA 基础

**目标**：理解 CUDA 编程模型并编写你的第一个 kernel。

### 主题
- SIMT 执行模型：threads、warps、blocks、grids
- 内存层次结构：global、shared、local、constant、texture memory
- Launch 配置：block size、grid size 选择
- 错误检查和使用 `cuda-memcheck` 进行调试

### 练习
1. 实现向量加法 kernel（两个数组的逐元素相加）
2. 实现并行归约 kernel（数组求和）
3. 构建封装 CUDA kernel 的 PyTorch C++ 扩展（`torch.utils.cpp_extension`）
4. 与 `torch.add` 和 `torch.sum` 对比性能

### 资源
- [CUDA C++ Programming Guide - Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- PMPP 第 3-5 章

### 关键指标
- Kernel launch 开销
- Occupancy（每个 SM 的活跃 warp）
- 全局内存吞吐量

---

## 第 2 周：Triton 基础

**目标**：学习 Triton 的 block 级编程模型和 auto-tuning。

### 主题
- Triton 编程模型：`@triton.jit`、program ID、block 级操作
- Triton 中的指针运算和 masking
- Triton 的 auto-tuner（`@triton.autotune`）
- 对比：Triton vs 原生 CUDA 的代码量

### 练习
1. 在 Triton 中实现逐元素操作（ReLU、GELU、SiLU）
2. 实现向量加法 kernel 并与 CUDA 版本对比
3. 使用 `triton.testing.do_bench` 进行微基准测试
4. 使用 `@triton.autotune` 添加 block size 扫描的 auto-tuning

### 资源
- [Triton Language Documentation](https://triton-lang.org/main/programming-guide/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

### 关键指标
- Block size vs occupancy
- Auto-tuner 搜索空间大小
- 代码量对比（Triton 行数 vs CUDA 行数）

---

## 第 3 周：内存带宽优化

**目标**：为内存受限的 kernel 实现接近峰值的内存带宽。

### 主题
- Roofline model：内存受限 vs 计算受限模式
- 合并 vs 分步内存访问模式
- 向量化加载/存储（128 位、256 位，通过 `float4` / 带 mask 的 `tl.load`）
- Shared memory 中的 bank conflict 和 padding
- 对齐要求

### 练习
1. 在不同规模下 benchmark 内存拷贝带宽
2. 实现合并 vs 分步访问模式并测量差异
3. 实现向量化加载（4 元素）并测量加速比
4. 实验 shared memory bank conflict 模式（stride=32）

### 资源
- [CUDA Best Practices Guide - Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [Roofline Model Explained](https://crd.lbl.gov/assets/pubs_presos/parlab08-roofline-talk.pdf)

### 关键指标
- 达到的带宽占峰值的百分比（如 A100：1555 GB/s）
- 全局加载/存储效率（NVIDIA Nsight Compute）
- L1/L2 缓存命中率

---

## 第 4 周：Operator Fusion

**目标**：通过 kernel fusion 消除中间内存传输。

### 主题
- 融合经济学：融合何时减少内存流量 vs 损害 occupancy
- 融合逐元素模式（如 `x = gelu(linear(x))` 在一个 kernel 中）
- 融合归一化：LayerNorm、RMSNorm 作为单个 kernel
- 图级融合：operator DAG 重写
- `torch.compile` 和 `torch._inductor.fx_passes` 融合 pass

### 练习
1. 在单个 Triton kernel 中实现融合 GELU + Dropout
2. 在 Triton 中实现融合 RMSNorm（reduce + normalize）
3. 实现一个简单融合 pass，重写 `x = relu(matmul(x, W))` 模式
4. 基准测试：未融合 vs 融合，通过 `torch.cuda.memory_stats` 测量内存流量

### 资源
- [PyTorch Inductor CPU/GPU backend](https://dev-discuss.pytorch.org/t/torchinductor-update-4/1266)
- [Horace He's "Making Deep Learning Go Brrrr"](https://horace.io/brrr_intro.html)

### 关键指标
- 内存流量减少（每次前向传播读/写的字节数）
- Kernel 数量减少
- 墙上时钟加速比

---

## 第 5 周：Matmul 分块

**目标**：使用 blocking 实现高性能矩阵乘法。

### 主题
- 朴素 matmul：内存访问模式和带宽瓶颈
- 全局内存 blocking（分块）：减少冗余加载
- Shared memory blocking：shared memory 中的分块
- 寄存器 blocking 和 warp 级矩阵乘法
- Double buffering 用于重叠计算和加载
- `triton.ops.matmul` 内部机制分析

### 练习
1. 在 Triton 中实现朴素 matmul，测量带宽
2. 添加全局内存分块（K-blocking）
3. 使用 `tl.store` / `tl.load` 添加 shared memory 分块
4. 实现寄存器 blocking（每个线程 4x4 微分块）
5. 扫参：在 A100/H100 上 sweep block size BM × BN × BK

### 资源
- [Triton Matmul Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [CUDA C++ Best Practices Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication)
- [How to Optimize a CUDA Matmul Kernel (Simon Boehm)](https://siboehm.com/articles/22/CUDA-MMM)

### 关键指标
- TFLOPS（占峰值的百分比，如 A100：312 TFLOPS fp16 tensor core）
- Shared memory bank conflict 检测
- Double buffering 加速比

---

## 第 6 周：Flash Attention

**目标**：使用分块和在线 softmax 实现 IO-aware 注意力。

### 主题
- 标准 attention 的内存瓶颈：O(N^2) 中间矩阵
- 在线 softmax：计算 softmax 无需物化完整注意力矩阵
- 分块 attention：逐块 Q*K^T 与在线重新缩放
- 反向传播：节省内存的重新计算策略
- FlashAttention-2：因果 mask 和跨序列长度的并行化
- 与 PyTorch `scaled_dot_product_attention` 集成

### 练习
1. 在 Triton 中实现在线 softmax（数值稳定）
2. 实现分块前向传播（FlashAttention 前向算法）
3. 实现分块反向传播（重新计算 P = softmax(S)）
4. 基准测试 vs `torch.nn.functional.scaled_dot_product_attention`
5. 使用 `torch.cuda.max_memory_allocated` 分析内存使用

### 资源
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Triton FlashAttention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

### 关键指标
- 峰值内存使用（GB）
- 运行时间 vs 序列长度扩展性
- 相对于 PyTorch 标准注意力的加速比

---

## 第 7 周：CUDA 系统编程

**目标**：利用 CUDA streams、异步操作和内存管理。

### 主题
- CUDA streams：并发执行和重叠
- 异步内存拷贝（`cudaMemcpyAsync`）
- CUDA graphs：捕获和重放
- Pinned（page-locked）memory 用于更快的 host-device 传输
- Unified memory 和按需页面迁移
- MPS（Multi-Process Service）用于并发 kernel 执行
- NCCL 基础用于多 GPU 通信

### 练习
1. 使用两个 stream 实现计算-拷贝重叠（double buffering）
2. 将训练迭代捕获到 CUDA graph 中并测量重放加速比
3. Benchmark pinned vs pageable memory 传输吞吐量
4. 为独立操作实现多 stream kernel launch 管线

### 资源
- [CUDA C++ Programming Guide - Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

### 关键指标
- Kernel 并发度（并发 kernel 数量）
- 计算-拷贝重叠比率
- CUDA graph 捕获开销 vs 重放加速比

---

## 第 8 周：图优化与代码生成

**目标**：构建一个简单的图优化器并理解代码生成。

### 主题
- 计算图 IR（Intermediate Representation）设计
  - 节点类型：compute、memory、control flow
  - Operator 表示和元数据
- 图级优化 pass：
  - 逐元素融合（水平和垂直）
  - Dead Code Elimination（DCE）
  - Constant Folding 和 Propagation
  - Common Subexpression Elimination（CSE）
  - 内存规划和 buffer 复用
- 代码生成：将 IR lower 到 Triton/CUDA 源代码
- 与 auto-tuning 集成用于分块大小选择
- `torch.fx` 用于图捕获，`torch._inductor` 用于 lowering

### 练习
1. 设计一个简单的 IR，包含 `Node` 和 `Graph` 类
2. 实现逐元素融合 pass
3. 实现 Dead Code Elimination
4. 实现一个简单代码生成器，将融合 ops lower 到 Triton
5. 端到端构建：图 -> 优化 -> codegen -> benchmark

### 资源
- [PyTorch Dynamo / torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- [torch.fx documentation](https://pytorch.org/docs/stable/fx.html)
- [TVM/Relay IR design](https://tvm.apache.org/docs/arch/relay_intro.html)
- [XLA: Optimizing Compiler for ML](https://www.tensorflow.org/xla/architecture)

### 关键指标
- 融合前后 kernel 数量
- 内存流量减少比率
- 端到端延迟改善
- 优化 pass 数量

---

## 每周检查清单模板

每周记录：

```
第 [N] 周：
  - [ ] 完成所有练习
  - [ ] 运行并保存基准测试结果
  - [ ] 记录性能指标
  - [ ] 写下关键洞察（1-2 句话）
  - [ ] 回顾前一周的指标，检查是否有性能回退
```

## 最终顶点项目

在第 8 周结束时，实现一个完整的融合 attention + MLP block：

1. 将 transformer block 前向传播捕获为图
2. 应用融合（attention + projection + activation + dropout）
3. 生成融合 Triton kernel
4. 与 PyTorch eager 和 `torch.compile` 进行 benchmark 对比
5. 记录实现的加速比和内存减少
