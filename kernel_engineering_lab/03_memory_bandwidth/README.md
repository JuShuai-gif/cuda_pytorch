# 03_memory_bandwidth - 内存带宽优化

## 工业背景：为什么内存带宽在 LLM 推理中至关重要

在 transformer 推理中（尤其是解码/自回归阶段），**大多数操作受内存带宽限制**，
而非计算限制。这意味着它们的性能受限于数据在 GPU DRAM 和计算单元之间的传输速度，
而非计算单元本身的速度。

### 内存墙问题

| 组件 | 速度 |
|-----------|-------|
| GPU 计算（H100 fp16） | ~989 TFLOPS |
| HBM3 带宽（H100） | ~3.35 TB/s |
| GDDR6X 带宽（RTX 4090） | ~1.0 TB/s |

对于逐元素乘法（每 12 字节读/写 1 FLOP），H100 上可达到的最大 TFLOPS 为
3.35 / 12 = **0.28 TFLOPS**——仅为峰值计算的 0.03%！这就是内存墙的核心：
CPU/GPU 计算能力的增长远快于内存带宽。

### Transformer Block 中受内存带宽限制的操作

在典型的 transformer 解码步骤中（batch=1，生成一个 token）：

```
Q = x @ W_q            # 矩阵乘法：计算受限
K = x @ W_k            # 矩阵乘法：计算受限
V = x @ W_v            # 矩阵乘法：计算受限
attn = softmax(Q @ K^T)  # Softmax：内存带宽受限（归约）
attn = attn @ V        # 矩阵乘法：计算受限
out = attn @ W_o       # 矩阵乘法：计算受限
x = x + out            # 残差：内存带宽受限
x = layernorm(x)       # LayerNorm：内存带宽受限
h = x @ W_up           # 矩阵乘法：计算受限
h = silu(h) * gate     # 激活：内存带宽受限
out = h @ W_down       # 矩阵乘法：计算受限
x = x + out            # 残差：内存带宽受限
x = rmsnorm(x)         # RMSNorm：内存带宽受限
```

虽然矩阵乘法受计算限制，但它们之间的每个逐元素操作都**受内存带宽限制**。
在 batch=1 的解码阶段，许多矩阵乘法也变得受内存带宽限制，因为 M 维度仅为 1。

## 算术强度和 Roofline 模型

**算术强度** = FLOPs / 内存传输字节数

Roofline 模型将性能（GFLOPS）与算术强度作图：
- **低算术强度（<~10 FLOP/byte）**：性能上限为内存带宽
- **高算术强度（>~100 FLOP/byte）**：性能上限为峰值计算

| 操作 | ~FLOPs/元素 | ~Bytes/元素 | 算术强度（FLOP/byte） | 限制因素 |
|-----------|---------------|----------------|----------------|-------|
| 逐元素加法 | 1 | 12 | 0.08 | 内存 |
| ReLU | 1 | 12 | 0.08 | 内存 |
| GELU | ~5 | 12 | 0.42 | 内存 |
| LayerNorm | ~10 | 12 | 0.83 | 内存 |
| Softmax | ~5 | 8 | 0.63 | 内存 |
| 矩阵乘法（128x128） | 2*128³ | 3*128²*4 | ~85 | 计算 |
| 矩阵乘法（4096x4096） | 2*4096³ | 3*4096²*4 | ~682 | 计算 |

## 常见陷阱

### 1. 切片导致的非连续张量
```python
x = torch.randn(1024, 1024)
x_sliced = x[:, 256:768]  # 创建了一个分步视图！
# x_sliced.is_contiguous() == False
# 由于非合并内存访问，对 x_sliced 的逐元素操作会更慢
```
**解决方法**：当需要连续布局时调用 `.contiguous()`，或者更好的做法是，
避免在内存带宽受限操作之前进行不必要的切片，从而产生非连续视图。

### 2. 转置创建分步视图
```python
x = torch.randn(1024, 768)
x_t = x.t()  # 分步视图，相同数据
# x_t.is_contiguous() == False
# 如果然后执行 x_t * y_t，你会得到比 x * y 低得多的带宽
```
**解决方法**：仅在真正需要时使用 `.contiguous()`。在矩阵乘法 kernel 中而不是
内存带宽受限操作中进行转置，或交换矩阵乘法操作数的顺序。

### 3. 不当的内存合并
在 CUDA/Triton 中，warp 内的线程应访问连续的内存地址以获得最佳性能。
分步访问模式会导致**每个 warp 多次加载缓存行**，浪费带宽。

### 4. 切片后重塑张量
```python
x = torch.randn(1024, 1024)[:, :512]  # 非连续
x = x.reshape(-1)  # 隐式调用 .contiguous()
# 这会创建一份拷贝——请注意隐藏的内存开销
```

## 关键文件

| 文件 | 描述 |
|------|-------------|
| `analysis.py` | 带宽估算、连续 vs 分步测量、roofline 模型 |
| `triton_copy.py` | Triton 拷贝 kernel：简单、向量化、分步 |
| `test_memory_bandwidth.py` | 拷贝正确性的 pytest 测试 |
| `benchmark_memory_bandwidth.py` | 全面的带宽基准测试 |

## 运行测试

```bash
pytest 03_memory_bandwidth/test_memory_bandwidth.py -v
```

## 运行基准测试

```bash
python 03_memory_bandwidth/benchmark_memory_bandwidth.py
```

## 运行分析

```bash
python 03_memory_bandwidth/analysis.py
```

## 参考文献

- **Roofline 模型**：Williams, Waterman, Patterson (2009) - "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures"
- **GPU 内存层次结构**：NVIDIA CUDA Programming Guide - Chapter 5, "Performance Guidelines"
- **Transformer 算术强度**：Dao et al. (2022) - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
