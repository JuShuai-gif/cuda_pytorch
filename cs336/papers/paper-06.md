# Paper-06: vLLM — PagedAttention 与 Continuous Batching

> Kwon et al., 2023. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.

---

## 1. 解决什么问题

LLM 推理服务有一个隐蔽但致命的效率问题：**KV cache 的显存管理极度低效**。

在传统的 LLM 推理系统中（如 Hugging Face Transformers 的 `generate`），每服务一个请求，系统会预分配一块显存来存储该请求的 KV cache。这块显存的大小是固定的——按最大可能生成长度分配。例如，如果 max_new_tokens=2048，系统就预分配 2048 个 token 位置的 KV cache。

这种"静态预分配"带来了三个严重问题：

**问题 1：内部碎片（internal fragmentation）**：大部分请求实际生成的长度远小于 max_new_tokens。一个只生成了 30 个 token 的回复占据了 2048 个 token 的显存空间——**浪费了 98.5% 的显存**。

**问题 2：外部碎片（external fragmentation）**：多个请求的 KV cache 分配在显存的不同位置，释放后留下大小不一的空间。这些碎片无法被大分配利用，即使总空闲显存足够，也无法分配新的请求。类似于操作系统的内存碎片问题。

**问题 3：无法 batch 不同长度的请求**：传统系统中所有 batch 内的请求必须有相同的生成进度——因为 KV cache 是连续分配的，无法动态扩展。这意味着：
- 一个请求完成后，它的 slot 会空闲，直到 batch 中所有请求都完成
- 新请求必须等整个 batch 完成才能加入
- GPU 利用率周期性下降到极低

这个问题的严重程度：传统系统（如 FasterTransformer）的显存利用率常常只有 20-40%。vLLM 要做的，就是借鉴操作系统中的虚拟内存和分页技术，将显存利用率提升到接近 100%。

---

## 2. 核心创新

### 2.1 PagedAttention

PagedAttention 的核心思想：**将 KV cache 分割成固定大小的 page（block），每个 page 可以存储固定数量的 token（如 16 个 token）的 KV 向量**。

类比操作系统的虚拟内存：
- **传统方式**：每个请求分配连续的显存块 = 早期 OS 的连续分配
- **PagedAttention**：每个请求维护一个 page table（页表），映射逻辑位置到物理 page = 现代 OS 的分页管理

具体来说：
- 每个 page 存储 `block_size` 个 token 的 Key 和 Value（如 16 个）
- 请求的 KV cache 是一系列 page 的集合，通过页表 `(block_number → physical_page_index)` 来访问
- 新生成的 token 分配在物理上不连续的新 page 中

Attention 计算需要适配这种分页结构。标准 attention 是：

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

PagedAttention 将 K 和 V 按 page 切分，对每个 Q 的 query，遍历所有 page 中的 K 和 V 块来计算 local attention，然后用 online softmax 聚合。这与 FlashAttention 的 tiling 在数学上完美兼容——FlashAttention 本来就支持分块计算。

### 2.2 Continuous Batching (也叫 Iteration-Level Scheduling)

传统 static batching 是 **request-level** 的——等所有请求完成后，再开始新的一批。Continuous batching 是 **iteration-level** 的——每个 forward pass 后，立即将完成的请求移除，加入等待队列中的新请求。

调度算法的简化流程：
1. 当有空闲 slot 时，从等待队列拉新请求
2. 每个 iteration 执行一次 forward（所有活跃请求一起做）
3. 任何请求完成（生成 EOS），立即释放其 KV cache pages
4. 下一个 iteration 开始前，可能又有新请求加入

这本质上将 batch 从"提前确定"变成了"动态流动"。任何时候 batch 中都是"恰好正在生成中的请求"，最大化了 GPU 利用率。

### 2.3 显存共享

PagedAttention 还支持 KV cache pages 在不同请求间的共享：

**Beam search 共享**：beam search 中每个 candidate 共享相同的 prompt KV cache。不同 candidate 只需要 fork 一份 page table——copy-on-write 策略。

**Parallel sampling 共享**：多个采样结果共享 prompt KV cache。

**Prefix caching**：如果多个请求有相同的前缀（如 system prompt 相同），前缀的 KV cache page 被所有请求共享，避免重复计算。

---

## 3. 为什么有效

从直觉上，PagedAttention 解决了 LLM 推理中最根本的矛盾：**KV cache 的大小在请求开始时是未知的**。

在传统系统中，由于必须预分配，要么高估（浪费显存）要么低估（请求失败）。PagedAttention 的动态 page 分配保证了只占用实际需要的显存，且碎片极小（仅最后一个 page 可能有内部碎片，碎片比 = 1/block_size，如 block_size=16 时约为 6%）。

更深刻的好处是它彻底改变了调度层面的可能性。Continuous batching 能实现的原因是：**由于 KV cache 是分页的，新请求的 KV cache 不需要"插入"到已有的连续内存块中**。page table 的间接寻址使得动态增删变得 trivial——这完全类似于操作系统引入虚拟内存后，进程可以动态增长或收缩而无需在乎物理内存的连续性。

显存共享的价值也不容低估。在对话系统中，system prompt 是固定且较长的（可能数千 token）。如果同时有几百个并发用户，传统系统需要在每个请求中重复计算和存储 system prompt 的 KV cache。PagedAttention 的 prefix caching 将这几百份存储合并为 1 份。

---

## 4. GPU/硬件角度解释

PagedAttention 的设计实际上与被 GPU 硬件深度绑定：

**GPU 显存分配器（CUDA allocator）的特性**：`cudaMalloc` 和 `torch.cuda.empty_cache` 是非常慢的操作——每次调用可能需要毫秒级的时间，对于迭代级调度（每 iteration 可能 10-50ms）来说 unacceptable。vLLM 实现了自己的显存池（类似 slab allocator），预分配大块显存，然后自己管理 page 的分配和回收——绕过了 CUDA allocator 的开销。

**Page size 对性能的影响**：更大的 page（如 32 或 64 token）减少页表大小和 page 管理开销，但增加内部碎片。更小的 page 减少碎片，但增加页表寻址开销。16 token/page 是在吞吐和延迟之间的 sweet spot。

**Attention kernel 中的 page lookup**：PagedAttention 的 kernel 需要在做 attention 时通过 page table 查找 K、V page 的实际物理地址。如果处理不当，这会在每个 token 的 attention 计算中引入 random access 到 page table——对 GPU cache 不友好。vLLM 的优化是：**在 CUDA kernel 中将 page table 加载到 shared memory**，内存访问变为规律性的（按 block 遍历物理 pages）。

**NVLink 和分布式推理**：在多 GPU 推理中，paged KV cache 的引入使得 tensor parallelism 中的 all-reduce 通信模式变为"按 page 的 batch"——同一 page 下的 K、V 分散在不同 GPU 上，通信模式因此更规整。

---

## 5. 工业意义

vLLM 和 PagedAttention 不仅仅是一个学术创新，而是改变了整个 LLM 推理服务的基础设施：

1. **推理吞吐提升 2-4x**：与 HuggingFace Transformers 相比，vLLM 在相同硬件上实现 2-4 倍的吞吐提升（token/s）。这意味着同样服务能力所需 GPU 减少到原来的 1/2 到 1/4。

2. **成为标准推理引擎**：vLLM 已成为开源社区（HuggingFace TGI 除外）最流行的 LLM 推理框架。支持几乎所有主流开源模型（Llama、Mistral、Qwen、DeepSeek 等）和量化方法（GPTQ、AWQ、FP8 等）。

3. **Prefix caching 引爆了 prompt 优化**：由于 system prompt 的重复计算成本降为零，开发者可以写更详细的 system prompt（常达到数千 token）而不必担心增加的延迟。

4. **影响后续推理引擎设计**：SGLang 的 RadixAttention、TensorRT-LLM 的 paged KV cache 都是受到 PagedAttention 启发的变体或增强版本。

5. **SOSP 最佳论文**：这是首个获得操作系统顶级会议最佳论文奖的 LLM 系统论文——它证明了大模型时代的系统问题需要系统级别的思维来解决。

---

## 6. 如何复现

关键实现细节：

1. **Block Manager 的数据结构**：
   ```python
   class BlockManager:
       # 维护 free_blocks（空闲 page 列表）和 used_blocks（已分配 page 列表）
       # 每个请求的 BlockTable 是一个 list of physical_block_ids
       def allocate(self, num_blocks):
           # 从 free_blocks 弹出 num_blocks 个
       def free(self, blocks):
           # 将 blocks 放回 free_blocks
   ```

2. **PagedAttention kernel 结构**：
   ```cuda
   // 伪代码
   for each token in query batch:
       load query q
       for each logical block in block_table[token]:
           physical_idx = block_table[token][logical_idx]
           K_block = load_kv_cache[physical_idx]
           // 计算 q @ K_block^T
           accumulate attention
       // online softmax aggregation
       write output
   ```

3. **Scheduler 的 preemption（抢占）机制**：当显存不足时，将等待中的请求 swap 到 CPU 内存（包括所有的 KV cache pages）。当显存宽裕时再 swap 回来。这类似于 OS 的 page swapping，使得 vLLM 可以在负载高峰不丢弃请求。

4. **Block size 的 tuning**：16 token 是常用值，但对于长上下文模型（128k）可能需要增大到 32。Block size = 1 会退化为逐 token 分配，页表过大（每 token 一个 entry）。Block size = max_seq_len 退化回静态分配。

---

## 7. 面试要点

**必问题**：

1. **PagedAttention 如何解决 KV cache 的内存碎片问题？**
   答：通过将 KV cache 分割为固定大小的 page，每个请求的 KV cache 是物理不连续 page 的链表，通过 page table 逻辑寻址。类似于 OS 虚拟内存分页：物理内存可以不连续，通过页表映射提供逻辑连续性。

2. **Continuous batching 比 static batching 优势在哪里？**
   答：Static batching 是 request-level（一批请求一起结束），完成的请求必须等最慢的。Continuous batching 是 iteration-level（每个 forward 后动态调整 batch），完成的请求立即移除，新请求立即加入——GPU 利用率更平稳，不会周期性空闲。

3. **PagedAttention 和 FlashAttention 如何结合？**
   答：FlashAttention 的分块计算和 PagedAttention 的 page 结构天然兼容——PagedAttention 的 page 直接对应 FlashAttention 的 tiles，都通过 online softmax 聚合不同块的结果。

4. **Prefix caching 的工作原理？节省了什么？**
   答：多个请求的公共前缀（如 system prompt）共享同一批 KV cache pages。节省了两方面：① 重复的 prefill 计算（compute savings）；② 重复 KV cache 的显存占用（memory savings）。

5. **为什么 vLLM 不直接使用 PyTorch 的显存分配器？**
   答：PyTorch 的 `cudaMalloc` 开销太大（每 allocation 毫秒级），对于每 iteration 都需要分配新 page 的场景太慢。vLLM 维护自己的 page pool（预分配所有 page），分配和释放是简单的 O(1) 操作。

6. **PagedAttention 相比传统的 contiguous KV cache 有什么额外开销？**
   答：页表查找开销（约 1-3% 的额外 latency）和非连续内存访问导致的 GPU L1 cache 命中率下降。但由于显存利用率的大幅提升，整体吞吐增益远大于这点 overhead。
