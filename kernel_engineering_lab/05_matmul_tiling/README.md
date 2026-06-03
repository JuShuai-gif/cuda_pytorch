# 05_matmul_tiling - 矩阵乘法分块

## 工业背景：LLM 中的 Matmul

在基于 transformer 的 LLM 中，矩阵乘法占据了 **约 99% 的总 FLOPs**。
每一层执行：

| 操作 | 形状 | 备注 |
|-----------|-------|-------|
| QKV 投影 | (hidden, 3*hidden) | 单个融合投影或 3 个独立投影 |
| Attention Score | (heads, seq, seq) | O(N^2)；由 FlashAttention 优化 |
| Output 投影 | (hidden, hidden) | 聚合多头注意力 |
| FFN up-projection | (hidden, 4*hidden) | 通常 4 倍扩展 |
| FFN down-projection | (4*hidden, hidden) | 压缩回 hidden 维度 |
| Output 投影 | (hidden, vocab) | 最终 logits 层 |

以 LLaMA-7B 为例：hidden=4096, FFN=11008, 32 heads, 32 layers。

**单次前向传播的总 matmul FLOPs**：约 6.6 × 10^12（6.6 TFLOPS）。在 A100（峰值 312 TFLOPS）上，仅 matmul 就需要约 21ms。使用分块可以达到峰值的 70-80%。不使用分块可能只达到 5-10%。

## 分块基础

### 为什么要分块？

分块是 GPU 上矩阵乘法最重要的优化。核心思想：**从 shared memory 复用数据，而不是从 global memory 重复读取**。

不使用分块（朴素）：
```
for each output element (i, j):
    sum = 0
    for k in 0..K:
        sum += A[i, k] * B[k, j]          # 每次内循环 2 次全局内存读取
    C[i, j] = sum
```
A 和 B 的每个元素从全局内存加载 K 次。全局读取总次数：2*M*N*K。

使用分块（BLOCK_M × BLOCK_K 用于 A，BLOCK_K × BLOCK_N 用于 B）：
```
for each output tile (block_m, block_n):
    accumulator = 0
    for k_tile in 0..K step BLOCK_K:
        load A_tile into shared memory    # BLOCK_M * BLOCK_K 次读取
        load B_tile into shared memory    # BLOCK_K * BLOCK_N 次读取
        for each (m, n) in tile:
            for each k in tile:
                accumulator[m,n] += A_tile[m,k] * B_tile[k,n]
    store accumulator to C
```
全局内存中每个元素读取 K/BLOCK_K 次。对于 BLOCK_K=32 和 K=4096，这是 **128 倍的减少**。

### 算术强度

| Kernel 类型 | 全局读取/元素 | 全局写入/元素 | 算术强度 |
|------------|---------------------|----------------------|---------------------|
| 朴素 | 2K | 1 | 2K / (2K+1) ~ 1 |
| 分块（BLOCK_K） | 2K/BLOCK_K | 1 | 2K*BLOCK_K / (2K+BLOCK_K) |

更高的算术强度意味着更好的 GPU 利用率。对于现代 GPU，在大多数实际规模下 matmul 是计算受限（而非内存受限）的。

## 块大小选择策略

### 如何选择 BLOCK_M、BLOCK_N、BLOCK_K

1. **BLOCK_K（内维度）**：控制全局内存复用。
   - 更大的 BLOCK_K = 每个元素更少的全局读取
   - 受 shared memory 限制：2 * BLOCK_K * (BLOCK_M + BLOCK_N) * sizeof(float)
   - 常用值：16, 32, 64

2. **BLOCK_M、BLOCK_N（输出维度）**：控制并行度和寄存器使用量。
   - 更大的分块 = 更少的 thread block，更好的块内复用
   - 太大 = 寄存器溢出，降低 occupancy
   - 常用值：64, 128, 256

3. **Shared memory 限制**：大多数 GPU 上每个 SM 48-164 KB。
   ```
   shared_mem = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float)
   ```
   对于 fp32，shared_mem 必须 <= 48KB（默认）或 164KB（在 H100/A100 上选用）。

4. **Grid 大小**：必须有足够的 block 填满所有 SM。
   ```
   num_blocks = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
   ```
   如果 num_blocks < num_sms * occupancy_factor，GPU 未充分利用。

### L2 缓存局部性（GROUP_M）

优化版 kernel 添加了 GROUP_M 使处理相邻行的 block 保持在一起：
- 无 GROUP_M：block 在 N 位置间循环时切换 M 位置
- 使用 GROUP_M=8：block 处理 8 个相邻行，从 L2 缓存复用 A 分块

## 本模块的 Kernel

### naive_matmul.py
每次迭代直接从 global memory 加载。无 shared memory 复用。
展示了性能问题：**A 和 B 的每个元素从全局内存加载 K 次**。对于大 K，比分块慢 10-20 倍。

### tiled_matmul.py
使用 shared memory 的分块 matmul。每个 program：
1. 加载 A 的一个 BLOCK_M × BLOCK_K 分块和 B 的一个 BLOCK_K × BLOCK_N 分块
2. 使用 `tl.dot` 累加（在可用时映射到 tensor core）
3. 将最终结果写入 global memory

净效果：全局内存流量减少 K/BLOCK_K 倍。

### triton_matmul_optimized.py
扩展的分块 matmul，包含：
- 可配置的 num_warps（2, 4, 8），影响 occupancy
- GROUP_M 用于 L2 缓存局部性
- 预设配置（small/medium/large）
- 支持字分块以实现寄存器级复用

### batched_matmul.py
3D 批量 matmul：C[b] = A[b] @ B[b]。每个 program 处理一个 batch 元素的分块。与 torch.bmm 对比。展示 batch 与分块如何交互。

## 常见陷阱

### 1. Shared Memory 中的 Bank Conflict

当 warp 中多个线程访问同一内存 bank 时，访问会串行化。
影响：对于对齐不好的分块大小，速度降低 2-4 倍。

缓解措施：
- 填充分块维度以避免与 bank 数量（32 bank）的 2 的幂对齐
- 对复杂模式使用 swizzling（置换地址位）
- Triton 通过其内存布局自动处理大多数 bank conflict

### 2. Shared Memory 溢出

分块太大超出每个 SM 的 shared memory：
```
required_smem = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * element_size
```
如果超过 SM 限制（通常默认 48KB，选用最多 164KB），kernel 将无法启动或溢出到 L1/寄存器。

对于 fp16：BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 = 128*32*2 + 32*128*2 = 16KB。可行。
对于 fp32：相同配置 = 32KB。仍然可行。
对于 fp32 且 BLOCK_M=256, BLOCK_N=256, BLOCK_K=64 = 64KB。超出默认 48KB。

### 3. 寄存器溢出

每个线程有有限的寄存器文件（通常 255 个 32 位寄存器）。
如果 kernel 使用的寄存器超过可用数量，值会溢出到 local memory（L1 缓存或 DRAM），导致速度严重下降。

每线程寄存器数 = total_registers / (threads_per_block)
- 65536 total / 256 threads = 256 寄存器/线程（健康）
- 65536 total / 1024 threads = 64 寄存器/线程（大量累加器可能溢出）

一个 BLOCK_M × BLOCK_N 的累加器使用 BLOCK_M * BLOCK_N * num_elements_per_thread 个寄存器。例如 BLOCK_M=128, BLOCK_N=128，每线程 1 元素 = 整个 block 16384 个寄存器。如果有 1024 线程：仅累加器就 16 寄存器/线程。

### 4. Tensor Core 未充分利用

Tensor core 需要特定的分块大小才能激活：
- fp16/bf16：M、N 必须是 16 的倍数，K 必须是 8 的倍数
- int8：M、N 是 32 的倍数，K 是 16 的倍数
- tf32：M、N 是 16 的倍数，K 是 8 的倍数

如果 BLOCK_M、BLOCK_N、BLOCK_K 不满足这些对齐要求，`tl.dot` 会回退到 SIMT FMA，损失 4-8 倍吞吐量。

### 5. Launch 开销

对于非常小的 matmul（M, N < 64），kernel launch 开销占主导。
考虑：
- 将小 matmul 合并为批量操作
- 使用融合的 matmul+activation kernel
- 让 torch.compile 自动处理融合

## 运行测试

```bash
pytest 05_matmul_tiling/test_matmul_tiling.py -v
```

## 运行基准测试

```bash
python 05_matmul_tiling/benchmark_matmul_tiling.py
```

## 参考文献

- **GEMM 和 cuBLAS**：NVIDIA cuBLAS 文档，https://docs.nvidia.com/cuda/cublas/
- **Triton Matmul 教程**：https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- **Roofline 模型**：Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 2009
- **GPU 架构**：NVIDIA CUDA C++ Programming Guide，第 5 章（Performance Guidelines）
