# Lecture 11: 推理系统 Inference — Part 2: 高级技术

## 本讲核心问题

上一讲我们理解了 KV Cache 和推理的两个阶段。但现实中推理系统远比"一个请求 → prefill → decode → 完成"复杂：数千个用户同时提交不同长度的请求，有的只问"你好"（3 个 token），有的要求"帮我分析这篇 100 页报告"（10 万 token）。如何高效调度这些请求？如何在不牺牲质量的前提下加速生成？本讲深入回答：(1) PagedAttention 如何管理碎片化的 KV Cache？(2) Continuous Batching 如何动态调度？(3) Speculative Decoding 如何用小模型"猜"、大模型"验"？

---

## 通俗解释

**PagedAttention**：想象一个图书馆的书架系统。

- **传统 KV Cache**：为每个读者预留一整排书架（max_seq_len），不管他实际借了多少本书。大多数书架空空如也，浪费大量空间。
- **PagedAttention**：不预留整排书架。每个读者需要时才分配一"页"书架（比如 16 本书一页）。书可以不连续存放——通过一个"索书号本"（block table）记录每页在哪儿。这样：
  - 空间利用率接近 100%（没有内部碎片）
  - 可以随时增加新页（序列变长时）
  - 不同读者可以共享同一页（prefix caching）

这直接来自操作系统的**虚拟内存分页机制**——进程看到连续的地址空间，但物理内存在页框里不连续分布，通过页表做映射。

**Continuous Batching**：想象一个老师同时批改 10 份作业。

- **Static Batching**（老方法）：必须等 10 份作业**全部**批完才能换下一批。如果其中一个学生写了 100 页（长回答），其他 9 个学生写 1 页就完了，也得干等着。
- **Continuous Batching**（新方法）：每批改完一页，马上加入新作业——不等 10 份全完成。老师手上始终有 10 份作业在流转。这就是"不等 batch 全走完就插新请求"。

**Speculative Decoding**：一个大作家（大模型）写文章很慢，但质量高。他雇了一个小助手（draft model，小模型），助手先快速地写出几个词，然后作家快速看一遍——对的保留，错的自己重写。因为验证比从零写要快，总体速度提升了。这就是投机解码的核心：**用快的小模型 draft，用慢的大模型 verify**。

**Prefix Caching**：10 个用户对同一个文档提问，前 5000 个 token 是相同的 prompt。传统做法：10 次 prefill 计算相同的 5000 个 token——重复劳动。Prefix Caching：第一次算完把 KV Cache 存起来，后 9 次直接从缓存取——省了 9× 的计算量。

---

## 数学公式 + 工程意义

### 1. PagedAttention 的虚拟内存映射

PagedAttention 将 KV Cache 划分为固定大小的 **block**（通常 B=16 tokens），每个 block 包含：

```
KV_Block[i] = {
    K: B × num_kv_heads × head_dim × dtype_size,
    V: B × num_kv_heads × head_dim × dtype_size
}
```

每个请求维护一个 **block table**，记录该请求的 sequence 在哪些物理 block 中：

```
block_table[req_id] = [physical_block_0, physical_block_1, ..., physical_block_k]
```

逻辑位置到物理位置的映射：
```
KV[req_id, logical_pos] → {
    block_idx = logical_pos / B,
    offset    = logical_pos % B,
    physical_block = block_table[req_id][block_idx],
    actual_addr = physical_block.base_addr + offset × stride
}
```

**工程意义**：
- **零内部碎片**：每个请求只占用 ⌈seq_len / B⌉ 个 block
- **零外部碎片**：通过统一的内存池管理，物理 block 可以任意分配给任一请求
- **动态扩展**：序列增长时只需追加 block，不需要 pre-allocate

### 2. 内存浪费对比

设平均请求长度为 avg_len，最大请求长度为 max_len，batch size 为 b。

**传统预分配**（contiguous allocation）：
```
wasted_memory = b × (max_len - avg_len) × KV_per_token
```

例如 max_len=32K, avg_len=4K, b=32：
```
wasted_memory = 32 × (32768 - 4096) × (2 × 80 × 8 × 128 × 2) / 8
              = 32 × 28672 × 327680 bytes
              ≈ 300 GB 浪费
```

**PagedAttention**：
```
wasted_memory = (b × avg_len) % B × KV_per_token  // 仅每个请求的最后一个 block 可能有浪费
              ≈ b × B/2 × KV_per_token  (平均浪费半个 block)
              = 32 × 8 × 327680 bytes ≈ 84 MB
```

**结论**：PagedAttention 将内存浪费从 ~300GB 降到 ~84MB，利用率接近 100%。这意味着可以在同样的 80GB HBM 上同时服务 **2-4 倍**的请求。

### 3. Continuous Batching 的调度策略

Continuous Batching 的核心操作：在每个 decode step 结束时，检查是否有请求完成（生成 EOS token）或达到最大长度，如果有，立即从 batch 中移除；同时，如果有等待队列中的新请求，将其 prefill 的结果加入 batch。

```
每步 decode 的调度逻辑：
1. 检查 batch 中每个请求的完成状态
2. 对完成的请求，释放其 KV Cache blocks（放回 free pool）
3. 从 waiting queue 取新请求，执行 prefill
4. 将新请求的 KV Cache 加入 batch
5. 执行当前 batch 的 decode step
```

**挑战**：prefill 和 decode 的混合——prefill 是 compute-bound（大矩阵运算），decode 是 memory-bound。如果在一个 batch 中混合 prefill 和 decode，步调不一致（compute-bound 的 prefill 会拖慢 memory-bound 的 decode）。

**SGLang 的解决方案**：将 prefill 和 decode 分配到不同的 GPU 资源上（chunked prefill），prefill 被拆分成小块，与 decode 交错执行。

### 4. Speculative Decoding 的速度提升分析

设目标大模型（Target Model）单 token 生成时间为 T_target，小 draft 模型为 T_draft（T_draft << T_target）。

**Standard Decoding**：生成 K 个 token 的时间 = K × T_target。

**Speculative Decoding**（每批 draft γ 个 token）：
1. Draft：γ 个 token，用时 γ × T_draft
2. Verify：目标模型一次性处理 prompt + γ 个 draft token，用时 T_target（batch verification 和单 token 时间相近）
3. 接受 α 个 token（接受率，统计期望，通常 0.7-0.9）

有效加速比：
```
speedup ≈ αγ / (γ × T_draft/T_target + 1)
```

当 T_draft/T_target 很小且 α 高时：
```
speedup → αγ
```

例如：T_target = 20ms, T_draft = 2ms, γ = 5, α = 0.8：
```
speedup = 0.8 × 5 / (5 × 2/20 + 1) = 4 / 1.5 ≈ 2.67x
```

**关键洞察**：加速比的来源是验证的并行性——大模型一次性处理 γ 个 token，不需要串行 decode γ 次。

---

## 工业界真实实现

### vLLM 的 PagedAttention 实现细节

vLLM（来自 UC Berkeley）是 PagedAttention 的开创者。核心数据结构：

```python
class BlockTable:
    """
    Maps logical block indices to physical block indices.
    Inspired by OS virtual memory page tables.
    """
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.physical_blocks: list[int] = []  # physical block ids

    def append_block(self, physical_block_id: int):
        """Allocate a new physical block for a growing sequence."""
        self.physical_blocks.append(physical_block_id)

    def get_kv_block(self, logical_pos: int):
        """Compute physical address from logical position."""
        block_idx = logical_pos // self.block_size
        offset = logical_pos % self.block_size
        return self.physical_blocks[block_idx], offset
```

vLLM 的 KV Cache 管理使用 **gpu block allocator**：
- Free block queue：空闲物理 block 的队列
- 每个请求按需分配物理 block（通过 `malloc` 语义的 allocator）
- 序列完成时，所有物理 block 回收到 free queue
- **Copy-on-write**：当多个请求共享 prefix block 时，只有被修改时才复制（写时复制）

### SGLang 的 RadixAttention

SGLang 引入 **Radix Tree**（基数树）来组织和共享 KV Cache：

```
RadixAttention 的数据结构：
- Radix Tree 以 token sequence 为 key
- 每个节点存储该 prefix 对应的 KV Cache blocks
- 新请求到达时，沿着 tree 匹配最长 prefix，复用匹配节点的 KV Cache
- 只对剩余部分做 prefill

示例：
请求序列 A："The quick brown fox jumps over"
请求序列 B："The quick brown fox runs away"

Radix Tree:
  root
   └── "The quick brown fox "
        ├── "jumps over"  → KV blocks [5,7,2]
        └── "runs away"   → KV blocks [8,1]

请求 B 只需 prefill "runs away" 的 2 个 token，
复用 root + "The quick brown fox " 的 KV Cache
```

**效果**：在多轮对话和 RAG 场景中（大量共享 prefix），RadixAttention 可减少 50-70% 的 prefill 计算。

### TensorRT-LLM 的 In-flight Batching

NVIDIA 的实现使用 **GPU-triggered scheduling**，消除了 CPU-GPU 同步开销：
1. GPU 在执行当前 batch 的 decode 时，同时在 device 上判断哪些请求已完成
2. 完成信号通过 CUDA stream 的 event 机制通知 CPU
3. CPU 异步准备下一个 batch 的请求，与 GPU 当前 batch 的执行重叠

这使得请求可以"在飞行中"（in-flight）加入和离开，延迟极低。

---

## CUDA/GPU 视角

### PagedAttention 的 Kernel 实现挑战

PagedAttention 引入了间接寻址——读取 K/V 时需要通过 block table 查找物理 block：

```cuda
// PagedAttention kernel (simplified)
__global__ void paged_attention_kernel(
    float* out,               // [num_heads, head_dim]
    const float* query,       // [num_heads, head_dim]
    const float* kv_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* block_table,   // [max_num_blocks_per_seq]
    int seq_len,
    int block_size
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[HEAD_DIM];

    // Iterate over logical blocks
    for (int pos = 0; pos < seq_len; pos += block_size) {
        int logical_block = pos / block_size;
        int physical_block = block_table[logical_block];  // INDIRECTION!
        int offset = pos % block_size;

        // Load K, V from physical block
        const float* k_block = kv_cache + physical_block * BLOCK_STRIDE;
        // ... compute attention for this block ...
    }
}
```

**关键优化**：block table lookup 引入一次额外的内存访问（通常是 L2 cache miss），但可以通过预取（prefetch）和将 block table 放在 shared memory 来缓解。vLLM 的优化将 block table 的 lookup 开销控制在 1-3% 的总 decode 时间。

### Continuous Batching 的 GPU 利用

传统静态 batching 下，如果 batch 中有一个长请求，其他短请求完成后 GPU 就闲置等它。Continuous Batching 通过动态补充请求，使 GPU 的 SM 利用率一直保持在较高水平：

```
静态 batching: GPU 利用率 = avg_len / max_len（最坏情况）
连续 batching: GPU 利用率 ≈ 1 / (1 + scheduling_overhead)
```

---

## 本讲与整个 LLM 系统的关系

高级推理技术是 LLM 从"能跑"到"能赚钱"的关键：

- **成本经济性**：PagedAttention 将 GPU 利用率从 30% 提升到 95%+，直接决定 API 定价——OpenAI 的定价很大程度上取决于推理效率
- **用户体验**：Continuous Batching 降低了排队延迟（在请求突发时不会让新请求等太久）
- **模型架构**：Speculative Decoding 对 draft model 和 target model 的组合提出了新需求（如 Medusa 的多头预测）
- **协同设计**：PagedAttention 的反向影响了模型设计——如果未来 LLM 原生支持分页式的 KV Cache，效果会更好

---

## 面试问题

1. **PagedAttention 和传统 KV Cache 管理的本质区别是什么？** 从虚拟内存分页的角度回答。

2. **Continuous Batching 如何实现？prefill 和 decode 混合调度有什么挑战？**

3. **Speculative Decoding 的加速比如何计算？** 推导公式，分析 draft 模型速度和接受率的影响。

4. **RadixAttention 和 Prefix Caching 的区别？** 从数据结构（Radix Tree vs Hash Map）和适用场景分析。

5. **PagedAttention 中 block size 如何选择？** 分析 block size 对内存利用率、间接寻址开销的影响。

6. **如果 draft model 和 target model 的 tokenizer 不同怎么办？** 讨论 token-level vs byte-level speculative decoding。

7. **为什么不能简单地把 prefill 和 decode 放在同一个 batch？** 分析 compute-bound vs memory-bound 的资源竞争。

8. **Copy-on-write 在 prefix caching 中如何实现？** 描述多请求共享 KV Cache 时的写保护机制。
