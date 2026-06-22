# 08_memory_management - GPU 内存管理

## 工业背景：KV Cache 内存压力

GPU 内存是 LLM 服务中的主要容量约束。仅 KV cache 就可以消耗数十 GB（取决于上下文长度）。理解内存管理是决定服务 4 个并发用户还是 40 个的关键。

### 内存问题

以 LLaMA-70B 模型为例（fp16, batch_size=32, seq_len=4096）：
- **模型权重**：约 140 GB（2 bytes × 700 亿参数）
- **KV cache**：约 70 GB（2 × 80 layers × 8 KV heads × 32 batch × 4096 seq × 128 dim × 2 bytes）
- **Activations**：约 2-5 GB（前向传播期间的中间 tensor）
- **总计**：约 220 GB（远超 H100 的 80GB，需要 tensor parallelism）

KV cache 随以下因素线性增长：
- 序列长度（上下文窗口）
- Batch size（并发请求）
- 层数

Grouped Query Attention（GQA）通过跨 Q heads 共享 K/V heads 来减少这一开销。

### PyTorch Caching Allocator

PyTorch 使用一个缓存分配器，以块为单位管理 GPU 内存：

```
应用程序请求
    |
    v
[PyTorch CUDA Caching Allocator]
    |
    |-- 有已缓存的空闲块？ --> 是 --> 返回已缓存块
    |                                          |
    |                                        否
    |                                          |
    v                                          v
[cudaMalloc]                              [cudaFree]
    |                                          |
    v                                          v
[CUDA Driver]                            返回至池中
    |
    v
[GPU Physical Memory]
```

**为什么缓存很重要：**
- `cudaMalloc`/`cudaFree` 很昂贵（每个约数百 us）
- 它们会同步整个 device
- 没有缓存时，PyTorch 会在每次创建/删除 tensor 时调用它们
- 缓存分配器通过保留已释放的内存以供重用，分摊了这一开销

**关键指标：**
| 指标 | 含义 |
|--------|---------|
| `allocated_bytes()` | 活跃 tensor 占用的内存 |
| `reserved_bytes()` | 分配器持有的内存（已分配 + 已缓存的空闲块） |
| `max_allocated_bytes()` | 峰值已分配内存 |
| `empty_cache()` | 将所有已缓存块释放回 CUDA driver |

### vLLM PagedAttention

vLLM 的 PagedAttention 受操作系统虚拟内存分页的启发：

**传统方法：**
每个序列获得一个连续的 KV cache，大小为 `max_seq_len * num_heads * head_dim * 2`。
问题：对于 max_seq_len=4096 但实际序列长度=100，约 96% 的缓存被浪费。

**PagedAttention 方法：**
- 将 KV cache 划分为固定大小块（如每个 block 16 个 tokens）
- 每个序列有一个**block table**，将逻辑位置映射到物理 block
- Block 从共享池按需分配
- 无预分配浪费，无碎片

这使得**内存共享**成为可能：多个序列可以共享公共前缀 block（system prompt）。在 beam search 期间，子序列共享父级 block。

### 常见陷阱

1. **碎片 OOM**：`reserved` 有足够的空闲空间但 `allocated` 无法容纳一块连续的分配。解决方案：使用 `empty_cache()`，减少 tensor 大小，或整理碎片（分配一个大 tensor 然后释放它）。

2. **中间 tensor OOM**：如 `x + y` 这样的操作会分配临时输出。如果 x 和 y 很大，该时刻的总内存是 `sizeof(x) + sizeof(y) + sizeof(z)`。解决方案：使用 in-place 操作、融合 kernel 或 gradient checkpointing。

3. **忘记 `torch.cuda.empty_cache()`**：删除大 tensor 后，缓存的内存仍然被 reserved。其他进程或框架会将其视为"已使用"。在关键内存操作前始终调用 `empty_cache()`。

4. **In-place op 形状兼容性**：In-place 操作要求匹配的形状和 strides。如果 x 和 y 形状不兼容，`x.add_(y)` 会失败。In-place broadcasting 受限。

5. **引用循环导致内存泄漏**：涉及 CUDA tensor 的 Python 引用循环会延迟垃圾回收。使用 `gc.collect()` + `torch.cuda.empty_cache()` 进行清理。

6. **训练期间的 activation 内存**：前向传播的 activations 必须存储以用于反向传播。单个 transformer layer 可以产生输入大小 4-10 倍的 activations。Gradient checkpointing 通过在反向传播时重新计算 activations 而非存储它们，以计算换取内存。

## 模块结构

- `allocator_observe.py`：观察 PyTorch caching allocator 行为
- `memory_reuse.py`：内存复用模式（in-place、buffer 池化、checkpointing）
- `kv_cache_memory.py`：KV cache 大小计算和 PagedAttention 概念
- `test_memory_management.py`：正确性测试
- `benchmark_memory_management.py`：性能基准测试

## 参考文献

- PyTorch CUDA Semantics 文档
- vLLM：Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
- PyTorch Blog：Understanding GPU Memory
- NVIDIA：CUDA C++ Best Practices Guide - Memory Optimizations
