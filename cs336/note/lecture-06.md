# Lecture 06: GPU 编程 (Kernels, Triton)

## 本讲核心问题

1. Triton 比 CUDA 好在哪里？为什么不直接用 CUDA C++ 写 kernel？
2. Fused kernel 的"融合"到底节省了什么？为什么 elementwise 操作是融合的主要目标？
3. Matmul tiling 如何通过 shared memory 优化？block 大小如何选择？
4. Roofline model 如何指导我们选择优化策略？
5. 为什么说"LLM 系统优化的本质是减少显存访问"？

## 通俗解释

### Triton ≈ GPU 编程的 Python

手写 CUDA C++ 就像手写汇编——你能精确控制一切，但要花大量时间处理琐碎细节：thread index 计算、shared memory 边界检查、bank conflict 规避。写一个高效的 matmul kernel 可能需要 100+ 行 CUDA 代码。

Triton 让你用 Python 描述"这个操作在 block 级别上怎么做"，然后编译器自动生成高效的 CUDA 代码。一个 matmul 在 Triton 里只需要约 30 行：

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    # Each program processes a BLOCK_M x BLOCK_N tile
    pid = tl.program_id(0)
    # ...tiling logic...
```

Triton 编译器会自动处理 memory coalescing、bank conflict 规避和 warp 调度。

### Fused Kernel ≈ 去超市一次买完所有东西

假设你要做几件事：去买菜、去银行、去邮局。三个选择：

1. **每件事单独跑一趟**（标准 PyTorch）：`y1 = matmul(x, W)` -> `y2 = y1 + bias` -> `y3 = relu(y2)` ——每次操作都从 HBM 读、写回 HBM
2. **Fused kernel**（一次处理完）：读完 x 和 W，做完 matmul + bias + relu，只写最终结果

对于 elementwise 操作（bias add, relu, layer norm），算术强度（AI）极低——通常在 1-10 FLOP/byte 范围。H100 的 ridge point 是 295 FLOP/byte。这意味着这些操作是 **severely memory bound**——98% 的时间在等数据，只有 2% 的时间在计算。Fuse 之后，中间结果不写 HBM，消除了中间的 memory round-trip。

在标准 PyTorch 中，`Linear + LayerNorm + ReLU` 会产生：
- 4 次 HBM 写入（output of each op）
- 4 次 HBM 读取（input of each op）
- Fuse 后只有 1 次 HBM 写入（final output）

**这就是节省的核心——不是少算了，而是少搬了。**

### Matmul Tiling ≈ 搬家时分批搬

要把一个大矩阵乘法全部在 GPU 上算完，数据量远超 shared memory 容量。Tiling 的思路：
1. 把大矩阵切成小块（tiles）
2. 每次把一个 tile 搬到 shared memory（很快）
3. 在 shared memory 里算完这个 tile 的结果（极快）
4. 累加到最终结果（存在 register 里）

这样每个数据从 HBM 只读一次，但在 shared memory 被重复使用多次——实现了 data reuse。没有 tiling 的话，每个数据要反复从 HBM 读。

## 数学公式 + 工程意义

### Matmul Tiling 的 Arithmetic Intensity

对于 `C = A @ B`，A 是 M x K，B 是 K x N：

```
FLOPs = 2 * M * N * K         （每个 C[i,j] 做 K 次乘法 + K 次加法）
HBM reads = M*K + K*N + ?     （如果无 tiling，? 包含反复读取）
HBM writes = M * N

AI(without tiling) = 2*M*N*K / (M*K + K*N + M*N)
                   = 2 / (1/N + 1/M + 1/K)  <-- 当 M,N 很大时 ≈ 2
                   = 2 FLOP/byte  (extremely memory bound!)

AI(with tiling, block size = BM x BK x BN) = 2*BM*BN*BK / (BM*BK + BK*BN + BM*BN)
     for BM=BN=BK=128: AI = 2*128^3 / (3*128^2) = 2*128/3 ≈ 85 FLOP/byte
```

Tiling 将 AI 从 ~2 提升到 ~85——提升了 40x。但 85 仍远低于 H100 的 ridge point（295），所以 matmul 依然是接近 memory bound 的。

这就是为什么 NVIDIA 引入了 Tensor Core——它的 MMA (Matrix Multiply-Accumulate) 指令在一个 warp 内处理 16x16x16 的 tile，比手工 tiling 更高效。

### Softmax Online (Tiled Softmax) 算法

标准 softmax 需要两次 pass：

```
# Pass 1: find max for numerical stability
m = max(x)

# Pass 2: compute exp and sum
s = sum(exp(x - m))
y = exp(x - m) / s
```

这对于 attention 中的 softmax 是个问题——softmax 是逐行（row-wise）的，如果一整行放不进 SRAM，就要做 online/tiled softmax：

```
# Tiled softmax (online algorithm)
For each tile:
    m_new = max(m, max(tile))
    s_new = s * exp(m - m_new) + sum(exp(tile - m_new))
    m = m_new
    s = s_new

# Final: output = values * exp(scores - m) / s
```

这消除了写入完整 attention score 矩阵到 HBM 的需要——正是 FlashAttention 的核心。

### GeLU Kernel 实现 (Elementwise, Fused)

```python
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # GeLU: 0.5 * x * (1 + erf(x/sqrt(2)))
    # Approximate version (faster):
    y = 0.5 * x * (1.0 + tl.math.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x)))

    tl.store(y_ptr + offsets, y, mask=mask)
```

这个 kernel 的 AI ≈ 8 FLOPs / 6 bytes ≈ 1.3 FLOP/byte——severe memory bound。优化方向：(1) fuse 到前面的 matmul 或 bias add 中；(2) 如果后面还有 LayerNorm，也一起 fuse。

AI 如此低，意味着在 H100 上跑单独的 GeLU kernel，GPU 利用率只有 1.3/295 ≈ 0.4%——99.6% 的时间在等数据。

### Softmax Kernel 实现 (Reduction)

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets, mask=offsets < row_start + n_cols, other=-float('inf'))

    # Online softmax
    m = tl.max(x, axis=0)       # Find max (reduction)
    e = tl.exp(x - m)           # Elementwise
    s = tl.sum(e, axis=0)       # Sum (reduction)
    y = e / s                   # Elementwise

    tl.store(y_ptr + offsets, y, mask=offsets < row_start + n_cols)
```

Reduction 操作（max, sum）需要跨线程通信——`tl.max` 和 `tl.sum` 内部使用 warp shuffle 或 shared memory 来汇总结果。AI 略高于 elementwise（因为 reduction 有一些 FLOPs），但仍然是 memory bound。

### Roofline Model 应用

```
# For H100:
ridge_point = 989e12 FLOP/s / 3.35e12 bytes/s = 295 FLOP/byte

# Elementwise kernel (e.g., GeLU):    AI = ~2   -> memory bound -> utilization ~2/295 = 0.7%
# Matmul (small tile):                 AI = ~85  -> memory bound -> utilization ~85/295 = 29%
# Matmul (large M,N):                  AI = ~200 -> near ridge   -> utilization ~68%
# Large matmul w/ Tensor Core:         AI = ~400 -> compute bound -> utilization ~100%
```

优化策略取决于位置：
- **Memory bound kernel**（AI < 295）：减少 HBM 访问（fuse, quantization）
- **Compute bound kernel**（AI > 295）：提高 Tensor Core 利用率（larger tiles, better occupancy）

## 工业界真实实现

### FlashAttention 的 Triton 实现

FlashAttention 实现了 attention 的 online softmax + tiling：

```python
# Simplified FlashAttention forward (conceptual)
for i in range(num_q_blocks):
    # Load Q block into SRAM
    q_block = load_q_block(i)

    # Initialize online softmax state
    m_i, l_i, o_i = init_state()

    for j in range(num_kv_blocks):
        # Load K, V block into SRAM
        k_block = load_kv_block(j)
        v_block = load_kv_block(j)

        # Compute partial attention scores (in SRAM!)
        s = q_block @ k_block.T

        # Online softmax update (in SRAM!)
        m_new = max(m_i, row_max(s))
        l_new = exp(m_i - m_new) * l_i + row_sum(exp(s - m_new))
        o_i = diag(exp(m_i - m_new)) * o_i + exp(s - m_new) @ v_block

        m_i, l_i = m_new, l_new

    # Write final output (only this touches HBM!)
    write_output(i, o_i / l_i)
```

关键数据：中间 attention matrix (NxN) **从未写入 HBM**，只在 SRAM 里计算。对于 N=16K，这意味着每次 attention 节省了 16K^2 * 2 bytes = 512 MB 的 HBM 写入。

FlashAttention-2 进一步优化：将 Q 放在外循环（而非 FlashAttention-1 的 K,V 在外循环），减少了 non-matmul FLOPs，在 H100 上达到 740 TFLOPS（75% of peak）。

### Torch Profiler 实践

```python
from torch.profiler import profile, record_function, ProfilerActivity
import torch

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             with_stack=True,    # Capture Python stack traces
             record_shapes=True, # Record tensor shapes
             profile_memory=True # Track GPU memory allocation
            ) as prof:
    with record_function("model_forward"):
        output = model(input)

# Print table of top CUDA kernels by time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export to Chrome Trace for visualization
prof.export_chrome_trace("trace.json")
```

典型分析流程：
1. 跑 profiler，看哪个 kernel 耗时最长
2. 检查该 kernel 的 memory bandwidth utilization（trace 里会显示）
3. 如果 memory bandwidth 很低但耗时很长 → memory access pattern 有问题（可能是 strided）
4. 如果 memory bandwidth 很高但 still slow → kernel is memory bound, 需要 fuse

### Nsight Systems

NVIDIA Nsight Systems 是比 Torch profiler 更底层的工具：

```bash
# Profile a training script
nsys profile --trace=cuda,nvtx,osrt,cublas \
     --stats=true \
     --output=profile.qdrep \
     python train.py

# Or launch GUI to view
nsys-ui profile.qdrep
```

Nsight Systems 可以看到：
- GPU kernel launch overhead（CPU -> GPU 的 kernel dispatch 延迟）
- Memory copy between CPU and GPU（`cudaMemcpy`）
- NCCL 通信和 kernel 执行的 overlap 情况
- CUDA graph 的 capture 和 replay

对于训练性能优化，常看的是 **NCCL all-reduce 和 forward/backward kernel 的 overlap**——如果通信时间完全暴露在 compute 之后，说明没有充分 overlap，需要调整 pipeline parallelism 的 micro-batch 调度策略。

### vLLM 的 Fused Kernels

vLLM 将 PagedAttention 实现为 fused kernel——把 attention score 计算、softmax、对 V 的加权求和全部融合在一个 Triton kernel 中。同时，因为 KV cache 是分页存储的（不连续），fused kernel 需要额外处理"跨 page 边界"的逻辑。

vLLM 的 fused attention kernel 在 H100 上可达 ~500 TFLOPS（相比峰值 989 TFLOPS），但因为解码时 attention 只占总计算量的一小部分，整体吞吐的瓶颈仍在 MLP 层的 matmul。

## CUDA/GPU 视角

### Warp Divergence

```cuda
// Divergent: threads take different paths
if (threadIdx.x % 2 == 0) {
    x = expensive_func1(x);  // Half the warp compute, other half idle
} else {
    x = expensive_func2(x);  // Half the warp compute, other half idle
}
// Total: both func1 and func2 executed sequentially
```

在 LLM kernel 中，warp divergence 常出现在：
- **Attention mask** 处理：causal mask 的三角区域计算 vs 跳过
- **Mixture of Experts** 的 routing：不同专家处理不同 token
- **Sequence padding**：batch 中不同长度的序列

Triton 编译器会自动识别 `tl.where` 等条件，但在循环内有条件分支时仍需手动优化——比如把 mask 逻辑提前到 block 级别判定。

### Memory Coalescing 在 Matmul 中的实现

对于 `C = A @ B` 的 tiled matmul：

```python
# Load A tile: shape [BLOCK_M, BLOCK_K]
# Each program loads a BLOCK_M x BLOCK_K block of A
offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_ak = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
a_block = tl.load(a_ptrs)  # Coalesced if stride_ak=1 (row-major)
```

关键：`stride_ak=1` 确保相邻线程访问连续内存（coalesced）。如果 A 是 column-major (`stride_am=1`)，访问模式变成 strided，性能可能下降 10-50x。

### Bank Conflicts 在 Shared Memory 中的体现

```python
# Loading into shared memory - potential bank conflict
# If BLOCK_K is a multiple of 32, each column has same bank
a_shared = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
a_shared[offs_m, offs_k] = a_block  # Writing to shared memory

# Reading from shared memory during matmul
acc += tl.dot(a_shared[offs_m, :], b_shared[:, offs_n])  # tl.dot handles bank conflict internally
```

Triton 的 `tl.dot` 自动使用 Tensor Core 的 `mma.sync` 指令，内部有硬件级的 bank conflict 规避（通过 warp-level matrix layout）。这也是为什么推荐用 Triton 而非手写 CUDA——bank conflict 这种细节被编译器优化了。

### 为什么 LLM 系统优化的本质是"减少显存访问"

回到 H100 的数据：

```
HBM bandwidth:  3.35 TB/s
Peak BF16:      989 TFLOPS
Roofline ridge: 295 FLOP/byte
```

每从 HBM 读 1 byte，需要做 295 次 FLOP 才能算力饱和。但实际 LLM 推理中，每 byte 平均只有 20-50 次 FLOP（取决于模型大小和 batch size）。**这意味着所有不 fuse 的 kernel 都在浪费 GPU 算力**。

CUDA 的 async copy（`cp.async`）和 H100 的 TMA 提供了一定缓解（overlap data movement and compute），但它们不会自动 fused kernels——这是算法/系统设计层面必须手动处理的问题。

一个简单的经验法则：

```
优化前：runtime = HBM_read_time + HBM_write_time + compute_time
优化后：runtime = max(optimized_HBM_time, compute_time)

# Optimization is about reducing HBM_time until it ≤ compute_time
# If HBM_time >> compute_time: fuse, quantize, or restructure data layout
# If compute_time >> HBM_time: increase block size, improve occupancy, use Tensor Core
```

## 本讲与整个 LLM 系统的关系

```
Tokenizer -> Embedding -> Attention -> MLP -> Loss -> Optimizer -> Distributed -> Inference
              [_________________________________________________________]
              每一个阶段的 GPU kernel 优化都直接体现在本讲的工具和技术中
```

GPU 编程是整个 LLM 系统优化的"最后一公里"。无论你是做训练加速还是推理部署，最终都体现在写高效 GPU kernel 的能力上。Triton 让这件事从"需要 5 年 CUDA 经验"变成了"Python 工程师也能做"——这也是为什么现在很多 LLM 系统工程师需要掌握 Triton。

## 面试问题

**Q1: 实现一个 fused GELU + bias kernel 在 Triton 中，描述你如何保证 memory coalescing。**

A: 设计思路：(1) bias add 和 GELU 都是 elementwise，融合后只需一次 HBM 读和一次 HBM 写；(2) block size 选 1024，让每个 block 处理连续的 1024 个元素——确保 warp 内的 32 个线程访问连续 32 个 float，完美的 memory coalescing；(3) grid size = ceil(n_elements / BLOCK_SIZE)，用 `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` 计算每个线程的全局索引；(4) `tl.load` 和 `tl.store` 会自动合并连续地址的访问。

```python
@triton.jit
def fused_gelu_bias_kernel(x_ptr, bias_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)         # Coalesced read
    b = tl.load(bias_ptr + offs, mask=mask)       # Coalesced read
    z = x + b
    # GeLU approximation
    y = 0.5 * z * (1 + tl.math.tanh(0.79788456 * z * (1 + 0.044715 * z * z)))
    tl.store(y_ptr + offs, y, mask=mask)          # Coalesced write
```

**Q2: 如何用 roofline model 分析一个 attention kernel 是 memory bound 还是 compute bound？给出计算步骤。**

A: 步骤：(1) 计算 total FLOPs——attention 的核心计算是 `QK^T` (SxS matmul): 2*B*H*S*S*d_head FLOPs 和 `attn_weight*V`: 2*B*H*S*d_head^2 FLOPs；(2) 计算 total bytes accessed——假设无 tiling: B*H*(S*d_head + S*d_head + S*S) * bytes_per_element (Q+K+attention scores)；(3) AI = FLOPs / bytes；(4) 与 H100 ridge (295 FLOP/byte) 比较。对于 S=2048, d_head=128: AI ≈ 2*2048*128 / (128+128+1)*2 ≈ 85 FLOP/byte → memory bound。对于 S=16: AI ≈ 16 → more severely memory bound。

**Q3: Matmul tiling 中，BLOCK_M, BLOCK_N, BLOCK_K 如何选择？考虑什么约束？**

A: 约束：(1) shared memory 大小——`BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N` 的 float 总数不能超过 SM shared memory / sizeof(float) ≈ 48K；(2) register 限制——每个线程需要的 register 数 = accumulator registers + input registers，不能超过 255；(3) occupancy——block 总数不能超过 SM 的 max blocks (32)；(4) Tensor Core 指令要求——BLOCK_K 必须是 16 的倍数（bf16 mma 指令操作 16x16x16 tiles）；(5) memory coalescing——A tile 的加载沿 K 维（如果 A 是 row-major）应该是 warp 对齐的。实践中的典型选择：128x128x32 或 64x128x64，在这个空间做 grid search 找最优的。

**Q4: Triton 相比手写 CUDA 有什么性能损失？什么情况下你会坚持用 CUDA？**

A: Triton 通常能达到手工优化 CUDA 的 85-95% 性能。以下场景可能需要 CUDA：(1) 需要精确控制 warp-level 原语（如 `__shfl_xor_sync` 做 warp 内 reduction）；(2) 使用 H100/B200 特有的 TMA 指令做异步数据搬运；(3) 需要 persistent kernel（kernel 不退出，持续从 global work queue 取任务——适用于小 batch 推理）；(4) 需要 inter-block communication（CUDA cooperative groups 的 `grid_group` 和 `grid.sync()` 让整个 grid 同步——Triton 目前不支持跨 block 同步）；(5) 精度要求极高（如使用 `__float2half_rn` 控制 rounding mode）。但在 90% 的 LLM 系统优化任务中，Triton 足够快且开发效率高得多。
