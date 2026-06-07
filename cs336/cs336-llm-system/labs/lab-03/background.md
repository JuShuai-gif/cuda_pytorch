# 背景知识：Triton Kernel、Fused Operations 与 DDP

## 1. GPU Memory Hierarchy (回忆)

```
┌─────────────────────────────────────────────┐
│ HBM (High Bandwidth Memory)                 │  慢但大 (40-80 GB)
│   ┌───────────────────────────────────────┐ │
│   │ L2 Cache (~40-80 MB)                  │ │
│   │   ┌─────────────────────────────────┐ │ │
│   │   │ SRAM / Shared Memory (~100-200KB)│ │ │  快但小
│   │   └─────────────────────────────────┘ │ │
│   └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

- **SRAM (Shared Memory)**: ~19 TB/s bandwidth, ~200KB/SM on A100
- **HBM**: ~2 TB/s bandwidth, 80GB on A100

核心优化策略：尽量在 SRAM 中完成操作，减少 HBM 读写。

---

## 2. 为什么需要 Kernel Fusion？

### 2.1 PyTorch 的 Eager Execution 问题

```python
# Naive RMSNorm (3 separate kernel launches)
x_sq = x * x                    # Kernel 1: read x, write x_sq
mean_sq = x_sq.mean(dim=-1)     # Kernel 2: read x_sq, write mean_sq
rms = torch.rsqrt(mean_sq + eps) # Kernel 3: read mean_sq, write rms
y = x * rms                     # Kernel 4: read x & rms, write y
# + maybe weight multiply
```

每次 kernel launch 都有：
1. Launch overhead (CPU→GPU communication)
2. HBM read/write between kernels

### 2.2 Fused Kernel 的优势

```
Fused RMSNorm (1 kernel launch):
1. Load tile of x from HBM → SRAM
2. Compute x^2, reduce for mean in SRAM
3. Compute rsqrt
4. normalize and apply weight (all in SRAM)
5. Write result tile back to HBM
```

✅ 减少 kernel launch overhead
✅ 减少 HBM traffic（中间结果不出 SRAM）
✅ 提高 arithmetic intensity

---

## 3. Triton 编程模型

### 3.1 核心抽象

Triton 提供了比 CUDA 更高级的抽象：

| 概念         | Triton                              | CUDA                |
| ------------ | ----------------------------------- | ------------------- |
| 计算单元     | Block (program)                     | Thread Block        |
| 内存管理     | Automatic (tensor of pointers)      | Manual (`__shared__`) |
| 索引         | `tl.program_id(axis)`               | `blockIdx`, `threadIdx` |
| 同步         | Implicit (block-level)              | `__syncthreads()`   |
| 加载/存储    | `tl.load` / `tl.store`（支持 mask） | Manual              |

### 3.2 基本语法

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    x_ptr,          # pointer to input tensor
    y_ptr,          # pointer to output tensor
    N: tl.constexpr, # compile-time constant
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)

    # Compute offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create mask for boundary handling
    mask = offsets < N

    # Load from HBM
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute
    y = x * 2.0

    # Store back to HBM
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 3.3 启动 Kernel

```python
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
my_kernel[grid](x, y, N=N, BLOCK_SIZE=1024)
```

### 3.4 Fused RMSNorm 的设计思路

```
Algorithm: Fused RMSNorm forward
─────────────────────────────────
Input:  x ∈ R^(B, L, D)   (batch, seq_len, hidden_dim)
        w ∈ R^D            (weight)
         eps (small constant)
Output: y ∈ R^(B, L, D)

For each row (b, l):
  // Step 1: compute mean(x^2)
  Load tiles of x[b,l,:] into SRAM
  For each tile:
    tile_sq = tile * tile
    acc += sum(tile_sq)

  // Step 2: rsqrt(mean + eps)
  rms = sqrt(acc / D + eps)
  inv_rms = 1.0 / rms

  // Step 3: normalize and scale
  For each tile:
    tile_out = tile * inv_rms * w_tile
    Store tile_out to y

Key optimization: Do Step 1 (read-only reduction)
then Step 3 (write-only) — all in one kernel,
with no intermediate HBM writes.
```

### 3.5 行级并行 vs 块级并行

对于 RMSNorm，每行独立，所以：

- **并行策略 1**：每个 program 处理一行（small D → 浪费 SM）
- **并行策略 2**：每个 program 处理部分列，多 program 处理一行（需要跨 program 同步）

方案 2 更好，但需要 `tl.atomic_add` 或 two-pass 方法。实践中，对于 hidden_dim ≤ 8192：
- 使用 one-pass，单 program 处理整行即可

---

## 4. Distributed Data Parallel (DDP)

### 4.1 DDP 原理

```python
# Pseudocode for DDP training
for batch in dataloader:
    # Forward: each GPU processes its own micro-batch
    loss = model(batch_local)

    # Backward: compute gradients locally
    loss.backward()

    # Gradient synchronization: AllReduce across GPUs
    # DDP hooks are registered on each parameter
    # The AllReduce is triggered by the backward hook
    # (happens during backward, not after!)
    allreduce_gradients()  # implicit via DDP hooks

    # Optimizer step: each GPU has identical gradients
    optimizer.step()
    optimizer.zero_grad()
```

### 4.2 Gradient Bucketing

DDP 将 gradients 按 bucket 分组（默认 bucket_size=25MB）：

```
Instead of: AllReduce(p1.grad) → AllReduce(p2.grad) → ...
Do:          wait until bucket is full → AllReduce(bucket)
```

好处：
- 减少通信次数（更大的 message size → 更好的带宽利用率）
- 与 backward 计算 overlap（通信可以在计算下一个 bucket 时进行）

### 4.3 DDP 的 Overlap 机制

```
Timeline:
─────────────────────────────────────────────────
GPU 0: | backward layer N | AllReduce bucket 0 | backward layer N-1 | ...
GPU 1: | backward layer N | AllReduce bucket 0 | backward layer N-1 | ...
─────────────────────────────────────────────────

因为模型参数从最后一层到第一层依次计算梯度,
DDP 在反向传播的早期层计算梯度时,
已经触发较早层的 AllReduce。
```

### 4.4 AllReduce 通信量

Ring AllReduce 算法（DDP 默认）：

$$T_{\text{comm}} \approx 2 \cdot (p-1)/p \cdot M/B$$

其中 $p$ = GPU 数, $M$ = 梯度总字节数, $B$ = 网络带宽。

关键：通信量与 GPU 数几乎无关（$p \to \infty$ 时 $T \to 2M/B$）。

### 4.5 DDP vs FSDP vs TP vs PP

| 策略     | 切分什么       | 通信量                          | 适用场景       |
| -------- | -------------- | ------------------------------- | -------------- |
| DDP      | 数据           | AllReduce gradients (per step)  | 模型能放单卡   |
| FSDP     | 参数+优化器    | AllGather params + ReduceScatter | 模型超出单卡   |
| TP       | 层内参数 (列/行) | AllReduce per layer output      | 超大矩阵乘法   |
| PP       | 层             | P2P send/recv activations       | 深度优先       |

---

## 5. 核心公式速查

| 概念                  | 公式/说明                                       |
| --------------------- | ----------------------------------------------- |
| RMSNorm               | $y = \frac{x}{\sqrt{\text{mean}(x^2)+\varepsilon}} \cdot w$ |
| RMSNorm FLOPs         | $O(B \cdot L \cdot D)$，约 4 FLOPs/element      |
| Fused kernel 加速比   | 取决于 bandwidth reduction, 通常 1.5x-3x         |
| Ring AllReduce time   | $T \approx 2\frac{p-1}{p}\frac{M}{B}$           |
