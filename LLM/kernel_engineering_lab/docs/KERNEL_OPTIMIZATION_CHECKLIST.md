# Kernel 优化检查清单

GPU kernel 性能优化的系统化检查清单。在声明一个 kernel "完成"之前，逐项检查每个部分。

---

## 1. 内存访问模式

### 合并访问

- [ ] **全局内存加载是合并的**：warp 内的线程访问连续的内存地址。在现代 GPU（A100/H100）上，一个 warp（32 线程）访问连续的 128 字节对齐段只需一次事务。
- [ ] **对齐检查**：每次加载/存储的基地址对齐到 16、32 或 128 字节（取决于访问宽度）。未对齐访问会浪费带宽。
- [ ] **Stride 分析**：warp 内的 stride-1 访问是最优的。Stride-N（N > 1）导致部分事务利用率。
- [ ] **数组结构（SoA）布局**：在 GPU 上优先使用 SoA 而非结构数组（AoS）。具有 4 个字段的 struct 产生 4 倍内存事务，而 SoA 布局只需 1 倍。

### 向量化

- [ ] **向量化加载/存储**：尽可能使用 `float2`、`float4` 或 128 位/256 位加载。一次 `float4` 加载在一次事务中读取 16 字节，而 4 次 4 字节事务需要 4 倍开销。
- [ ] **Triton 向量化**：使用带适当 `num_elements` 的 `tl.load(ptr, mask=..., other=0.0)` 或通过 `triton.Config` 自动向量化。
- [ ] **CUDA 向量化**：当连续线程访问连续内存时使用 `float4`、`int4`、`double2`。

### Bank Conflict（Shared Memory）

- [ ] **Shared memory stride 检查**：warp 内的线程不应以不同地址访问同一 bank。大多数架构上有 32 个 bank，每个 bank 4 字节。
- [ ] **用于避免 bank conflict 的填充**：当 stride 模式导致冲突时，向 shared memory 数组添加填充列。示例：使用 `__shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1]` 替代 `[BLOCK_SIZE][BLOCK_SIZE]`。
- [ ] **Swizzle 模式**：重映射地址以分散 bank 访问，适应特定问题维度。
- [ ] **Nsight Compute 验证**：检查 "Shared Memory Bank Conflicts" 指标。对良好优化的 kernel 应为 0。

---

## 2. Occupancy 优化

### 寄存器使用

- [ ] **寄存器数量在大多数 GPU 上应 <= 64**：每个线程使用有限的寄存器文件（A100 上每个 SM 64K 寄存器）。更少的寄存器意味着更多 warp，更好的延迟隐藏。
- [ ] **在 CUDA 中使用 `__launch_bounds__`**：告诉编译器每个 block 的最大线程数以优化寄存器分配。
- [ ] **减少活跃变量**：重构代码以限制同时活跃的变量数量。
- [ ] **避免寄存器溢出**：Nsight Compute 的 "Local Memory" 流量应为 0。溢出到全局内存（L1 缓存的 local memory）极其缓慢。

### Shared Memory

- [ ] **每个 block 的 shared memory < SM shared memory 限制**：A100 每个 SM 有 164 KB 可配置 shared memory。如果一个 block 使用 64 KB，每个 SM 只能容纳 2 个 block（最多 32 warp）。
- [ ] **动态 shared memory**：在 CUDA 中使用 `extern __shared__` 或在 Triton 中使用 `tl.alloc` 进行灵活大小调整。
- [ ] **Persistent kernel 模式**：当 shared memory 使用量高时，考虑使用 persistent thread 在多轮迭代间将数据保留在 shared memory 中。

### Block Size

- [ ] **Block size 是 32（warp size）的倍数**：非倍数会导致最后一个 warp 中有浪费的线程。
- [ ] **对于现代架构，block size 是 64 或 128 的倍数**（A100 每个 SM 有 4 个 warp scheduler）。
- [ ] **Grid size 提供足够的 block**：`num_blocks > num_SMs * target_blocks_per_SM`。目标每个 SM 至少 2-4 个 block 以实现良好的延迟隐藏。
- [ ] **Auto-tune block size**：取决于寄存器和 shared memory 使用量。按 2 的幂从 64 扫描到 1024。

---

## 3. 计算优化

### 指令组合

- [ ] **优先使用 FMA（fused multiply-add）**：`a*b + c` 在一条指令中完成。寻找乘法和加法模式的收敛。
- [ ] **在热路径中避免整数除法和取模**：`%`、`/` 很昂贵。预计算或对 2 的幂模使用按位 `&`。
- [ ] **在可接受时使用 fast math**：`__sinf()`、`__cosf()`、`__expf()`（内置函数）比 `sin()`、`cos()`、`exp()` 更快但精度较低。
- [ ] **优先使用 `float` 而非 `double`**：双精度带宽是 2 倍，在消费级 GPU 上的吞吐量是单精度的 1/32，在 A100 tensor core 上慢 2 倍。

### 延迟隐藏

- [ ] **足够的并行度**：线程数 >> SM 数量 × 每个 SM 的 core 数。A100 有 108 SM × 64 CUDA core = 需要 6912 个活跃线程，但需要远多于 6912 以保证隐藏。
- [ ] **Barrier 之间的独立工作**：`__syncthreads()` 之间的操作应相互独立，以允许 warp scheduler 切换。
- [ ] **指令级并行（ILP）**：结构化代码使得独立操作交错排列。编译器可以重排序，但显式结构有帮助。

### Warp Divergence

- [ ] **最小化 warp 内的分支发散**：warp 中的线程以锁步方式执行。`if/then/else` 中部分线程走不同路径会串行化执行。
- [ ] **Warp-uniform 条件**：使分支条件对 warp 中所有线程相同。示例：`if (threadIdx.x < CONSTANT)` 在 CONSTANT >= 32 时是 warp-uniform 的。
- [ ] **小分支使用 Predication**：简短的发散路径（< 8 条指令）可能更适合所有线程通过 predication 执行。
- [ ] **为分支友好性排序输入**：如果工作负载不均衡，对线程排序以将相似工作分组。

---

## 4. 数据复用

### Shared Memory

- [ ] **将数据加载到 shared memory 一次，多次复用**：基于分块的算法（matmul、convolution、attention）将数据 block 到 shared memory 以减少全局内存流量。
- [ ] **将 shared memory 作为软件管理的缓存使用**：使用无 swizzle 模式的显式加载。
- [ ] **Double buffering**：在当前分块上计算的同时加载下一个分块。使用异步 `cp.async`（A100+，CUDA 11.0+）。

### 寄存器

- [ ] **将频繁访问的标量保持在寄存器中**：依赖编译器来完成这一点，但结构化代码使哪些值被复用更加明显。
- [ ] **展开小循环**：CUDA 中的 `#pragma unroll` 或对行程计数较小的循环（< 8）手动展开。减少索引计算开销。

### 缓存

- [ ] **L1 缓存感知的分块**：A100 上的 L1 缓存为 192 KB（与 shared memory 组合，可配置）。缓存行为 128 字节。
- [ ] **缓存行填充**：访问缓存行中的所有字节以最大化利用率。
- [ ] **避免缓存颠簸**：超过缓存容量的重复访问模式会驱逐有用数据。

---

## 5. 异步操作

### CUDA Streams

- [ ] **使用多个 streams 进行重叠**：在 stream 1 上执行 kernel 的同时在 stream 2 上进行内存拷贝。
- [ ] **异步内存拷贝**：使用带 pinned memory 的 `cudaMemcpyAsync`，而非 `cudaMemcpy`。
- [ ] **基于 Event 的同步**：使用 `cudaEventRecord` 和 `cudaStreamWaitEvent` 替代 `cudaStreamSynchronize` 或 `cudaDeviceSynchronize`，以获得细粒度控制。

### CUDA Graphs

- [ ] **对重复的 launch 模式进行图捕获**：捕获一次 kernel launch 序列，多次重放。将 launch 开销减少 2-10 倍。
- [ ] **图更新**：使用 `cudaGraphExecUpdate` 更新 kernel 参数而无需重新捕获。

### 预取

- [ ] **预取数据到 shared memory**：对于顺序分块访问，在处理当前分块时发起对分块 N+1 的加载（double buffering）。
- [ ] **CPU 预取**：使用 `cudaMemPrefetchAsync` 进行统一内存的手动页面迁移触发。

---

## 6. Triton 特定优化

- [ ] **使用 `triton.autotune` 进行 block size 选择**：自动 sweep `BLOCK_SIZE`、`num_warps`、`num_stages`。
- [ ] **利用 `tl.make_block_ptr`**：用于带 tensor core 友好布局的分步 2D 访问模式。
- [ ] **使用 `tl.dot` 进行 tensor core matmul**：当形状对齐到 16x16（fp16/bf16）时自动使用 MMA 指令。
- [ ] **使用 `num_stages` 进行软件流水线**：2-4 stages 用于重叠全局加载与计算。
- [ ] **尽可能避免 `tl.atomic_add`**：使用归约 + scatter 替代以获得更好性能。
- [ ] **对编译时已知值设置 `tl.constexpr`**：使 Triton 编译器能够生成专门的代码。

---

## 7. 优化后验证

- [ ] **正确性**：将输出与参考 PyTorch 实现比较（allclose，rtol=1e-3，atol=1e-5）。
- [ ] **数值稳定性**：在边缘情况输入（零、非常大的值、负值）上检查 NaN/Inf 输出。
- [ ] **性能回退检查**：重新运行 baseline benchmark 以确认改进。
- [ ] **内存检查**：`torch.cuda.synchronize()` + `torch.cuda.reset_peak_memory_stats()` 验证内存使用没有恶化。
- [ ] **规模检查**：跨多个问题规模（小、中、大、超大型）进行 benchmark。某些优化仅在超过某个阈值后才有效。
