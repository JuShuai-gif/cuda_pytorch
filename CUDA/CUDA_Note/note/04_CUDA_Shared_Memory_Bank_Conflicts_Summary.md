# CUDA Shared Memory bank conflict 与 LDS.128/float4 访存总结

本文总结自 PDF《搞懂 CUDA Shared Memory 上的 bank conflicts 和向量化指令（LDS.128 / float4）的访存特点》。

## 1. 先记住 shared memory 的基本模型

CUDA shared memory 可以按 32 个 bank 来理解：

- shared memory 以 4 bytes / 32 bits 作为一个 word。
- 第 `i` 个 word 放在第 `i mod 32` 个 bank。
- 每个 bank 每个 cycle 可提供 32 bits。
- 所以 32 个 bank 合计每个 cycle 可提供 `32 * 4 = 128 bytes`。
- 一次 shared memory memory transaction 最多覆盖 128 bytes。

这条 `128 bytes / transaction` 是理解 `LDS.64`、`LDS.128` 的关键。

## 2. bank conflict 和 broadcast

分析 shared memory 时，通常只看单个 warp 内的访存。

当一个 warp 发起一次 shared memory 请求：

- 多个线程访问同一个 bank 的同一个 word：触发 broadcast，不算 bank conflict。
- 多个线程访问同一个 bank 的不同 word：产生 bank conflict。
- bank conflict 会让一次请求被拆成多次 memory transaction 串行发出。

例子：

- 2-way bank conflict：通常需要拆成 2 次 transaction。
- 没有 conflict 且总访问量不超过 128 bytes：1 次 transaction 就够。

注意：bank conflict 是针对单次 memory transaction 而言的。即使整个 warp 访问了很多数据，只要硬件拆出的每一次 transaction 内没有多个不同 word 落到同一个 bank，就没有 conflict。

## 3. 每个线程访问 4 bytes：普通情况最简单

如果 32 个线程各访问 4 bytes：

- 整个 warp 总需求是 `32 * 4 = 128 bytes`。
- 没有 bank conflict 时，1 次 transaction 足够。
- 访问连续 word 时，thread 0 访问 word 0，thread 1 访问 word 1，依此类推，刚好覆盖 32 个 bank。

这是官方文档最常讲的情况。

## 4. LDS.64 / uint2 / float2：每线程访问 8 bytes

如果使用 `LDS.64`，或者 C/CUDA 代码里用 `uint2`、`float2` 读 shared memory：

- 每个线程请求 8 bytes，也就是 2 个 word。
- 16 个线程就需要 `16 * 8 = 128 bytes`。
- 所以一个 warp 默认会拆成两个 half warp。
- 每个 half warp 默认产生 1 次 memory transaction。
- 没有合并时，一个 warp 共有 2 次 transaction。

但在特殊模式下，两个 half warp 的请求可以合并为 1 次 transaction。PDF 中给出的合并条件是满足以下任意一种：

- 对所有活跃线程 `i`，线程 `i xor 1` 不活跃，或者和线程 `i` 访问同一地址。
- 对所有活跃线程 `i`，线程 `i xor 2` 不活跃，或者和线程 `i` 访问同一地址。

直觉理解：如果相邻配对或隔 2 配对的线程拿的是同一份数据，硬件可以用 broadcast 复用数据，使每个 half warp 的有效独占数据量下降，从而把两个 half warp 合并进一次 128-byte transaction。

## 5. LDS.128 / uint4 / float4：每线程访问 16 bytes

如果使用 `LDS.128`，或者代码里用 `uint4`、`float4`：

- 每个线程请求 16 bytes，也就是 4 个 word。
- 8 个线程就需要 `8 * 16 = 128 bytes`。
- 所以一个 half warp 会再拆成两个 quarter warp。
- 一个 warp 有 4 个 quarter warp。
- 默认情况下，没有 bank conflict 时，一个 warp 会产生 4 次 transaction。

类似 `LDS.64`，满足上面的 `i xor 1` 或 `i xor 2` 条件时，同一个 half warp 内的两个 quarter warp 可以合并。

关键限制：

- 合并只发生在同一个 half warp 内。
- 两个 half warp 不会再进一步合并。
- 所以 `LDS.128` 最理想也通常是每个 half warp 1 次 transaction，整个 warp 共 2 次 transaction。

## 6. PDF 中几个重要 Case 的含义

### 6.1 连续读取 `uint2`

```cpp
reinterpret_cast<uint2 *>(a)[tid] =
    reinterpret_cast<const uint2 *>(smem)[tid];
```

含义：

- thread `tid` 读取第 `tid` 个 `uint2`。
- 每个 half warp 访问连续 32 个 word。
- 每个 half warp 1 次 transaction。
- 全 warp 共 2 次 transaction。
- 没有 bank conflict。

### 6.2 多个线程读取相同 `uint2`

```cpp
reinterpret_cast<uint2 *>(a)[tid] =
    reinterpret_cast<const uint2 *>(smem)[tid / 2];
```

含义：

- thread 0 和 1 读同一个 `uint2`，thread 2 和 3 读同一个 `uint2`。
- 满足 `i xor 1` 合并条件。
- 两个 half warp 可以合并。
- 全 warp 可降到 1 次 transaction。
- 没有 bank conflict。

### 6.3 `uint4` / `LDS.128` 的默认行为

```cpp
reinterpret_cast<uint4 *>(a)[tid] =
    reinterpret_cast<const uint4 *>(smem)[tid];
```

含义：

- 每个线程读取 16 bytes。
- 每 8 个线程就达到 128 bytes。
- 全 warp 默认 4 个 quarter warp，也就是 4 次 transaction。
- 连续访问时没有 bank conflict。

### 6.4 `uint4` 的合并访问

如果 warp 内线程按照某种地址模式，让每个 half warp 内的两个 quarter warp 满足 `i xor 1` 或 `i xor 2` 的广播/合并条件，那么：

- 前两个 quarter warp 合并成 1 次 transaction。
- 后两个 quarter warp 合并成 1 次 transaction。
- 全 warp 从 4 次 transaction 降到 2 次 transaction。

这也是 GEMM 优化里常见线程布局的动机之一。

### 6.5 `uint4` 的 bank conflict 例子

PDF 中有一个模式会让 half warp 内产生 2-way bank conflict：

```cpp
uint32_t addr = (tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8;
```

这个模式虽然满足合并条件，但合并后的 transaction 内，有多个不同 word 落到同一个 bank，于是每个 half warp 出现 2-way bank conflict。

结果：

- 每个 half warp 原本可合并成 1 次 transaction。
- 但由于 2-way conflict，会拆成 2 次。
- 两个 half warp 合计 4 次 transaction。
- bank conflict 计数为 2。

## 7. 为什么 GEMM 里会关心 4x8 / 8x4 和 Z-order

GEMM 优化中，warp 从 shared memory 读取 tile 到 register 时，常会让一个 warp 内的线程按 `4 x 8` 或 `8 x 4` 的形状组织，并采用类似 Z-order 的排列。

原因是：

- `float4` / `LDS.128` 每线程一次读 16 bytes。
- 如果线程布局让同一个 half warp 内的两个 quarter warp 满足合并条件，就能把 transaction 数从 4 降到 2。
- 更少的 shared memory transaction 可以降低访存压力。
- 在简单 kernel 中，2 次和 4 次 transaction 的差距可能不大。
- 在 GEMM 这类计算密集、流水线较满的场景里，这类减少 transaction 的布局可能带来可观收益。

PDF 提到的实测/论文参考中，合适的 warp 内线程排列在 GEMM 场景可能带来约 13% 的提升。

## 8. 实践判断口诀

看 shared memory 向量化访存时，可以按这个顺序分析：

1. 每线程访问几 bytes？
   - 4 bytes：看整个 warp。
   - 8 bytes：默认按 half warp 看。
   - 16 bytes：默认按 quarter warp 看。

2. 每个 transaction 内是否超过 128 bytes？
   - 超过就一定会拆。
   - 不超过才继续看 conflict。

3. 单次 transaction 内是否有同一个 bank 的不同 word？
   - 有：bank conflict。
   - 没有：无 conflict。

4. 是否满足 `i xor 1` 或 `i xor 2` 的同地址/不活跃条件？
   - `LDS.64`：两个 half warp 可能合并为 1 次 transaction。
   - `LDS.128`：同一 half warp 内两个 quarter warp 可能合并，但两个 half warp 不会再合并。

## 9. 本仓库示例

示例代码放在 [src](./src)：

- [src/shared_memory_bank_examples.cu](./src/shared_memory_bank_examples.cu)：CUDA kernel，对应 PDF 的 `uint2`、`uint4` 访问模式。
- [src/bank_transaction_sim.py](./src/bank_transaction_sim.py)：无 CUDA 环境也可运行的 bank/地址映射模拟器。
- [src/README.md](./src/README.md)：编译、运行和 Nsight Compute 观察方式。

