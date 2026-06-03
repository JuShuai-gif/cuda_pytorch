# 06_attention_flash_like - 分块注意力与在线 Softmax

## 工业背景：FlashAttention 对 LLM 的影响

注意力是 transformer 架构中的 O(N^2) 瓶颈。在 FlashAttention 出现之前，标准方法将完整的 N*N 注意力矩阵物化在 HBM 中，使得训练或服务长上下文（如 32K+ tokens）的模型变得不可能。

FlashAttention（Dao et al., 2022）通过使注意力 IO-aware 而彻底革新了这一领域：使用在线 softmax 算法分块计算注意力，而不是将完整注意力矩阵写入 HBM。这将内存复杂度从 O(N^2) 降低到 O(N)，并实现了 2-4 倍加速。

### 生产影响

每个主流的 LLM 服务框架都使用 FlashAttention 或其衍生版本：

| 框架 | 注意力后端 | 关键优化 |
|-----------|------------------|------------------|
| vLLM | FlashAttention + PagedAttention | KV cache 分页 |
| TensorRT-LLM | FlashAttention + FlashInfer | Multi-query、GQA 支持 |
| HuggingFace TGI | FlashAttention-2 | FP8 量化 |
| llama.cpp | 自定义 CUDA kernel | 量化注意力 |
| MLX (Apple) | 自定义 Metal kernel | 统一内存 |

### 在线 Softmax 算法

关键洞察：softmax 可以增量计算而无需存储所有 scores。

标准 softmax：
```
m = max(scores)
P = exp(scores - m) / sum(exp(scores - m))
```

在线 softmax（逐块）：
```
对每个分块：
  m_new = max(m_old, max(s_tile))
  l_new = l_old * exp(m_old - m_new) + sum(exp(s_tile - m_new))
  acc = acc * exp(m_old - m_new) + P_tile @ V_tile
  m_old = m_new, l_old = l_new
最终：
  O = acc / l
```

这种重新缩放方式在保持数值稳定性的同时，永远不需要在内存中保存完整的注意力矩阵。

### 内存复杂度分析

| 方法 | 内存 | 何时 OOM？ |
|--------|--------|-----------|
| 朴素（物化 S） | O(B*H*N^2) | 4096 tokens × 32 heads × 4 bytes = 2 GB |
| 分块（在线 softmax） | O(B*H*N*D) | KV cache 占主导 |
| FlashAttention | O(B*H*N*D) | 接近最优 |

以 LLaMA-7B 在 seq_len=4096 为例：
- 朴素：每层约 2 GB，仅注意力矩阵
- 分块：每层约 1 MB（分块）+ 64 MB（KV cache）
- 在 seq_len=32K：朴素约需 128 GB，分块仍约 512 MB

### Prefill vs Decode 优化策略

**Prefill（Q_len == KV_len）：**
- 计算密集型：O(Q_len * KV_len * D) 次操作
- 通过在 Q 和 KV 两个维度分块进行优化
- 使用大分块以提高计算/内存比
- 可以跨 Q 分块并行化（多个 thread block）

**Decode（Q_len=1，KV_len >> Q_len）：**
- 内存密集型：O(KV_len * D) 次读取，O(D) 次写入
- Q 很小（1 token），因此不需要 Q 维度的分块
- 优化内存带宽：K/V 分块的合并读取
- 瓶颈是每步读取整个 KV cache
- 每个生成的 token 需要 O(N*D) 次内存读取

### 常见陷阱

1. **Mask 处理**：因果 mask 必须考虑分块边界。
   对于分块注意力，每个 Q 分块只看到 KV 位置 <= 其位置。

2. **数值稳定性**：在线 softmax 的重新缩放可能累积小误差。
   即使输入是 float16/bf16，内部始终使用 float32 计算。

3. **分块大小选择**：
   - BLOCK_M 太小：过多的 program 启动，GPU 利用率低
   - BLOCK_M 太大：寄存器压力过大，并发线程更少
   - BLOCK_N：通常 32-128；在计算强度和 shared memory 之间平衡
   - BLOCK_D：必须是 head_dim 的除数；常用 32/64/128

4. **缩放因子**：始终使用 1/sqrt(d)。对于大 d（如 256），点积可能很大，没有适当缩放会导致 softmax 饱和。

5. **Shared memory 限制**：分块大小决定 shared memory 使用量：
   BLOCK_M * BLOCK_D + BLOCK_N * BLOCK_D 个元素。在大多数 GPU 上，每个 SM 的 48-164 KB shared memory 限制了分块组合。

## 模块结构

- `naive_attention.py`：PyTorch 和 Triton 实现，物化完整注意力矩阵
- `tiled_attention.py`：使用在线 softmax 的分块注意力，FlashAttention 的核心思想
- `flash_attention_kv_cache.py`：专门的 prefill 和 decode kernel，带 KV cache 模拟
- `test_attention.py`：全面的正确性测试
- `benchmark_attention.py`：跨 LLM 配置的性能基准测试

## 参考文献

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024
