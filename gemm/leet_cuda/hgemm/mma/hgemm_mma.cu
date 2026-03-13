#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <torch/types.h>
#include <torch/extension.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

/*
在CUDA kernel中，传统的拷贝方式：
Register = Global;
Shared = Register;
这会消耗线程的发射能力

使用 cp.async 的正确姿势是：
流水线：你可以连续提交两组拷贝（Group A, Group B）。在处理 Group A 的数据时，
通过 CP_ASYNC_WAIT_GROUP(1) 只等待 A 完成，让 B 在后台继续跑。这就是 n 参数的用处。

3. 与内存模型的关系
这段代码是 CUDA 弱内存模型（Weak Memory Model） 的典型体现，涉及以下三个核心概念：

A. 异步性（Asynchrony）
在 CUDA 内存模型中，cp.async 建立了一种“发射后不管”的机制。它解耦了指令发射和数据到达。如果没有 COMMIT 和 WAIT，线程会直接读取到旧的、甚至是垃圾数据。

B. 内存一致性（Memory Consistency）
cp.async 指令跳过了 L1 缓存（直接写入 Shared Memory），这改变了数据在存储层级间的可见性路径。

- COMMIT_GROUP 相当于在异步任务队列中插入一个标记（Marker）。

- WAIT_GROUP 充当了内存屏障（Memory Barrier）。它保证了在该指令之后的内存访问（Read/Write）能够看到异步拷贝所写入的最新数据。

C. 线程级协作
虽然 cp.async 是每个线程独立发起的，但它们通常是为了填充整个 Block 共享的内存。因此，内存模型要求在 WAIT_GROUP 之后必须接一个 __syncthreads()（同步原语），以确保 Block 内的其他线程也感知到异步操作的完成。


cp.async 是如何执行指令的？
---

### 1. 发射维度：线程级指令

在代码中，`cp.async` 看起来像是一个普通的线程级指令（每个线程执行自己的 `asm` 代码）。

* **每个线程定义自己的分工**：每个线程会计算出自己负责的 Global Memory 地址（源）和 Shared Memory 地址（目的）。
* **独立提交**：每个线程都会执行 `CP_ASYNC_COMMIT_GROUP()`。

### 2. 执行维度：Warp 级与硬件加速

虽然指令是线程发出的，但在硬件底层（SM 内部）：

* **合并访问（Coalescing）**：就像普通的 `LDU` (Load Unit) 一样，当一个 Warp 的 32 个线程同时发射 `cp.async` 时，硬件的 **Data Management Unit (DMU)** 会尝试将这些请求合并，以最高效的方式从显存拉取数据。
* **非阻塞执行**：传统的拷贝会占用线程的 `LSU` (Load/Store Unit) 直到数据返回。而 `cp.async` 将任务交给一个**专用的异步拷贝引擎**后，Warp 就可以立即去执行下一条指令（比如算术运算）。这就是所谓的 **Warp 调度解耦**。

### 3. 分工模式：每个线程拷贝一部分

这是最常见的用法（Collective Copy）。为了填满一块 Shared Memory 区域，我们会让 Warp 中的线程协同工作。

**举个例子：**
假设我们要把 $128$ 个 `float` 数据搬到 Shared Memory。

* **Warp 分工**：Warp 里的 32 个线程，每个线程负责搬运 $4$ 个 `float`（通过 `cp.async.cg.shared.global.16` 指令，每次搬 16 字节）。
* **地址映射**：
* 线程 0 搬运 `src[0:3]` 到 `dst[0:3]`
* 线程 1 搬运 `src[4:7]` 到 `dst[4:7]`
* ... 依此类推。

### 4. 关键：谁在“等待” (Wait)？

`CP_ASYNC_WAIT_GROUP(n)` 的行为是**线程级的阻塞，但会影响 Warp 的进度**。

1. **线程计数器**：每个物理线程内部都有一个计数器来跟踪它发出的 Commit。
2. **Warp 调度**：当 Warp 执行到 `WAIT_GROUP` 时，如果该 Warp 中某个线程的计数器仍大于 `n`，那么**整个 Warp** 都会被调度器切走，进入等待状态（Stall）。
3. **同步必要性**：
* **重点**：`WAIT_GROUP` 只保证“我这个线程”搬的数据到家了。
* **现实**：通常你的计算逻辑需要使用整个 Warp 或整个 Block 搬来的数据。
* **后果**：如果你不加 `__syncthreads()` 或 `warp_sync`，线程 0 可能会在线程 31 的数据还没到齐时就开始读取 Shared Memory，从而导致结果错误。

---

### 总结：协作模型

| 维度 | 行为 |
| --- | --- |
| **谁负责拷贝？** | **每个线程**负责计算地址并搬运一小块（通常是 4、8 或 16 字节）。 |
| **谁负责调度？** | **Warp 调度器**。它发现数据没到（`WAIT_GROUP` 条件未满足）就会把 Warp 挂起。 |
| **谁负责执行？** | **专用异步拷贝硬件**。它独立于计算单元，在后台静默搬运。 |


*/
/*
作用：提交一个“异步拷贝组”。

逻辑：当你发出一个或多个 cp.async 拷贝请求后，必须调用此指令来“闭合”这个组。这告诉硬件：“刚才那些拷贝任务被归为一组了”。
*/
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

/*
作用：强制等待所有异步操作结束。通常等同于 wait_group 0，但在某些编译器语境下语义更直接。
*/
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

/*
作用：等待异步拷贝任务完成。

逻辑：它并不一定是等待“所有”任务，而是允许保留 n 个最近提交的组仍在运行，而阻塞当前线程直到更早的任务完成。

CP_ASYNC_WAIT_GROUP(0) 表示等待所有已提交的任务完成。

3. 硬件实现原理
在硬件底层，有一个 Async Copy Stage 计数器。

每执行一次 COMMIT_GROUP，计数器 加 1。

每当一个拷贝组在物理上搬运完成，计数器 减 1。

WAIT_GROUP(n) 会检查：while (counter > n) { 挂起线程; }。

4. 关键注意事项
线程安全性：cp.async 是线程级别的指令。虽然整个 Warp 可能会一起调用，但 WAIT_GROUP 必须由每个发起拷贝的线程亲自调用。

配合屏障：WAIT_GROUP(n) 只能保证当前线程看到了数据。如果你的计算逻辑需要读取其他线程异步拷贝过来的数据（这在 Collective Copy 中很常见），必须在 WAIT_GROUP 之后加一个 __syncthreads()。

指令顺序：如果你连续 commit 了 5 次，然后调用 WAIT_GROUP(2)，它保证前 3 次 commit 的任务一定完成了。
*/
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
// 异步拷贝指令，从全局内存拷贝数据到共享内存，使用L1+L2缓存策略
// 举例：CP_ASYNC_CA(shared_ptr, global_ptr, 16) 从global_ptr拷贝16字节到shared_ptr
/*

1. cp.async.ca.shared.global.L2::128B [%0],[%1],%2;
- cp.async:异步拷贝操作
- .ca (Cache All):缓存策略。表示数据在读取时会尝试缓存在 L2 Cache 中。这对于那些在当前阶段会被多次读取的数据非常有用。
- .shared.global: 数据源位于 Global Memory（显存），目的地位于 Shared Memory。
- .L2::128B: 这是一个 Eviction Priority（驱逐优先级） 提示。
它告诉 L2 缓存控制器：“这批数据非常重要，请尝试在 L2 中保留 128 字节的缓存行，不要轻易换出。”这能有效减少之后读取时的延迟。
- [%0], [%1], %2:
  - %0 (dst): Shared Memory 的偏移地址。
  - %1 (src): Global Memory 的 64 位指针。
  - %2 (bytes): 拷贝的字节数（必须是常数，通常为 4, 8 或 16）。

2. 执行时：线程（Thread）发生了什么？
在线程层面，这条指令是**“发射后不管”（Fire and Forget）**：

  1. 非阻塞提交：线程执行到这一行时，只是将“从 A 搬运 N 字节到 B”这个任务提交给了 SM 内部的异步数据管理单元（DMU）。

  2. 寄存器零占用：传统的 LDG (Load Global) 指令需要线程提供寄存器来中转数据（Global -> Register -> Shared）。而 cp.async 是 绕过寄存器 的。

  3. 立即返回：提交完任务后，线程的程序计数器（PC）立即指向下一条指令。此时，数据并没有到达 Shared Memory，搬运才刚刚在后台开始。

3. 执行时：Warp 发生了什么？
在 Warp 层面，会发生高度协作的硬件级优化：

1. 合并访问（Coalescing）：
当一个 Warp 的 32 个线程集体调用这个宏时，硬件会自动检查各线程的 src 地址。如果地址是连续的，硬件会将 32 个小的拷贝请求合并成几个巨大的内存事务（Memory Transactions），极大地提升了总线带宽利用率。
合并事务：如果 Warp 中 32 个线程的 src 地址是连续的，且每个线程搬运 16 字节，那么 32 个线程总共请求 $32 \times 16 = 512$ 字节。

2. 流水线解耦：
通常，Warp 调度器会等待数据返回（产生 Stall）。但由于这是 async 指令，调度器知道该 Warp 并不急于使用这个数据。因此，Warp 可以立即切换到算术计算指令。
  - 结果：你实现了“一边做矩阵乘法（数学运算），一边搬运下一块数据（内存访问）”。这就是隐藏内存延迟的终极手段。

4. 关键限制与细节
- 字节数对齐：为了性能最大化，bytes 最好是 16（即 128 位）。在汇编层面，这对应一条指令搬运 128 位数据。如果每个线程搬 16 字节，整个 Warp 一次就能搬 512 字节。

L2::128B 是“缓存政策”而非“拷贝长度”：
这里的 128B 指的是 L2 Cache 的行大小（Cache Line Size）。它是在告诉硬件：“在搬运时，请以 128 字节为单位优化 L2 缓存的驻留策略（Eviction Priority）”。它并不改变单个线程搬运的数据量。


你可以这样理解你的“128B”结论：
1. 每个线程的 %2 应该设为 16（这是 cp.async 的硬件最优值）。
2. 整个 Warp 的目标是 $16 \times 32 = 512$ 字节。
3. 最终效果：通过一次同步的 Warp 操作，你完美填满了 4 个 128 字节的缓存行。
*/
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// 异步拷贝指令，从全局内存拷贝数据到共享内存，仅使用L2缓存策略
// 举例：CP_ASYNC_CG(shared_ptr, global_ptr, 16) 从global_ptr拷贝16字节到shared_ptr
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))


// 从共享内存加载1个8x8矩阵到寄存器，矩阵元素为16位浮点数
// 举例：LDMATRIX_X1(regA, shared_addr) 从shared_addr加载8x8矩阵到regA
/*
ldmatrix: 指令名称，意为“Load Matrix”。它不是普通的内存读取，而是专门为矩阵形状设计的加载。

- .sync: 同步标志。在执行该指令前，Warp 内的所有线程都会处于同步状态（这是 Tensor Core 要求的硬性前提）。

- .aligned: 对齐要求。要求内存地址 %1 必须是 16 字节对齐的。

- .x1: 拷贝数量。表示从 Shared Memory 加载 1 个 8x8 的矩阵片段（Matrix Fragment）。

- .m8n8: 矩阵形状。表示加载的是一个 8 行 8 列的矩阵块。

- .shared: 源地址位于共享内存。

- .b16: 数据位宽。每个元素是 16 位（即 half 或 bfloat16）。

- {%0}, [%1]: %0 是目标寄存器组（Tensor Core 使用的格式），[%1] 是共享内存的基地址。

ldmatrix 指令由 warp 中 32 个线程协作执行。

warp 被划分为 8 组，每组 4 个线程。
每组线程负责加载一行（或一列）8 个 FP16 元素。

由于每个线程只返回一个 32bit 寄存器，
因此每个线程保存 2 个 FP16。

ldmatrix.x1 表示每线程返回 1 个寄存器，
warp 总共加载一个 8×8 tile。

ldmatrix.x2 / x4 表示每线程返回多个寄存器，
warp 会加载多个 8×8 tile。
*/
#define LDMATRIX_X1(R, addr) asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))


// 从共享内存加载2个连续的8x8矩阵到寄存器
// 举例：LDMATRIX_X2(regA0, regA1, shared_addr) 加载2个8x8矩阵
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))


// 从共享内存加载4个连续的8x8矩阵到寄存器
// 举例：LDMATRIX_X4(regA0, regA1, regA2, regA3, shared_addr) 加载4个8x8矩阵
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))


// 从共享内存加载并转置1个8x8矩阵到寄存器
// 举例：LDMATRIX_X1_T(regA, shared_addr) 加载并转置8x8矩阵
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))


// 从共享内存加载并转置2个连续的8x8矩阵到寄存器
// 举例：LDMATRIX_X2_T(regA0, regA1, shared_addr) 加载并转置2个8x8矩阵
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))


// 从共享内存加载并转置4个连续的8x8矩阵到寄存器
// 举例：LDMATRIX_X4_T(regA0, regA1, regA2, regA3, shared_addr) 加载并转置4个8x8矩阵
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))


// 执行16x8x16的矩阵乘累加操作，使用半精度浮点数
// 举例：HMMA16816(result0, result1, A0, A1, A2, A3, B0, B1, C0, C1) 计算A(16x16) × B(16x8) + C(16x8)
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

HOST_DEVICE_INLINE 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.
// ============================================================================
// Kernel: hgemm_mma_m16n8k16_naive_kernel
// 功能: 基础的半精度矩阵乘法kernel，使用MMA指令
// 特点: 
//   - 每个block只包含1个warp（32个线程）
//   - 每个warp计算一个16x8的输出块
//   - 使用16x8x16的MMA指令
//   - 所有矩阵都是行主序
// 用法:
//   - 输入: A(M×K), B(K×N), C(M×N) 都是半精度浮点数
//   - 输出: C = A × B
//   - 每个block计算一个16x8的输出块
//   - grid维度: (ceil(N/8), ceil(M/16))
//   - block维度: (32, 1, 1)
// 示例调用:
//   dim3 block(32);
//   dim3 grid(div_ceil(N, 8), div_ceil(M, 16));
//   hgemm_mma_m16n8k16_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
// ============================================================================
template<const int MMA_M=16, const int MMA_N=8, const int MMA_K=16>
__global__ void hgemm_mma_m16n8k16_naive_kernel(half* A, half* B, half* C, 
                                                int M, int N, int K) {
  // 块的序号
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  // 一个 K 中有多少个 MMA_K
  const int NUM_K_TILES = div_ceil(K, MMA_K);

  // 
  constexpr int BM = MMA_M; // 16
  constexpr int BN = MMA_N; // 8
  constexpr int BK = MMA_K; // 16

  __shared__ half s_a[MMA_M][MMA_K]; // 16x16
  __shared__ half s_b[MMA_K][MMA_N]; // 16x8


  __shared__ half s_c[MMA_M][MMA_N]; // 16x8

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int lane_id = tid % WARP_SIZE; // 0~31

  // s_a[16][16], 每行16，每线程load 8，需要2线程，共16行，需2x16=32线程
  const int load_smem_a_m = tid / 2; // row 0~15
  const int load_smem_a_k = (tid % 2) * 8; // col 0,8
  // s_b[16][8], 每行8，每线程load 8，需要1线程，共16行，需16线程，只需一半线程加载

  // 只使用前 15 个
  const int load_smem_b_k = tid; // row 0~31, but only use 0~15
  const int load_smem_b_n = 0; // col 0
  const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
  const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n
  if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;

  uint32_t RC[2] = {0, 0};

  #pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    // gmem_a -> smem_a
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;

    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (
      LDST128BITS(A[load_gmem_a_addr]));

    // gmem_b -> smem_b
    if (lane_id < MMA_K) {
      int load_gmem_b_k = k * MMA_K + load_smem_b_k; // global row of b
      int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
      LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (
        LDST128BITS(B[load_gmem_b_addr]));
    }
    __syncthreads(); 

    uint32_t RA[4];  // 寄存器数组，存储从s_a加载的4个8x8矩阵片段（共16x16矩阵）
    uint32_t RB[2];  // 寄存器数组，存储从s_b加载的2个8x8转置矩阵片段（共16x8矩阵）
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // s_a: (0,1)*8 -> 0,8 -> [(0~15),(0,8)]
    // 说明：对s_a使用普通加载，对s_b使用转置加载
    // s_a访问模式：每个线程根据lane_id确定加载位置，32个线程协作加载整个16x16矩阵

    // 计算s_a在共享内存中的加载地址
    // __cvta_generic_to_shared：将通用指针转换为共享内存地址
    // lane_id % 16：确定行索引（0-15），因为s_a是16x16矩阵
    // lane_id / 16：确定列块索引（0或1），乘以8得到实际列偏移（0或8）
    /*
    16 * 16的矩阵
     0   |   1
     2   |   3
    
    */
    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[lane_id % 16][(lane_id / 16) * 8]); 
    
    /*
    地址映射：每个线程根据自己的 lane_id 计算读取起始点。
    这里 A 矩阵是按行存储的。
    X4 含义：每个线程从 SMEM 读取数据，最终整个 Warp 共同加载一个 $16 \times 16$ 的分块。
    结果：数据被填充到 4 个 32 位的寄存器 RA[0-3] 中。
    因为每个寄存器装 2 个 half (16位)，所以每个线程实际持有了 8 个 half 类型的数据
    */
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[lane_id % 16][0]);
    
    /*
    _T (Transpose)：这是关键。mma.m16n8k16 指令要求 B 矩阵在寄存器中必须是列优先布局。
    _T 后缀告诉硬件在加载的同时进行转置。
    X2 含义：加载一个 $16 \times 8$ 的分块，存入 2 个寄存器 RB[0-1] 中（每个线程持有 4 个 half）
    */
    LDMATRIX_X2_T(RB[0], RB[1], load_smem_b_ptr);

    HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

    // 同步所有线程，确保所有线程都完成了当前K tile的计算
    // __syncthreads()：块内线程同步，在加载下一个K tile的数据之前需要这个同步
    __syncthreads();
  }
  
  // s_c[16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
  LDST32BITS(s_c[lane_id / 4    ][(lane_id % 4) * 2]) = LDST32BITS(RC[0]); 
  LDST32BITS(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);

  __syncthreads();

  // store s_c[16][8]
  if (lane_id < MMA_M) {
    // store 128 bits per memory issue.
    int store_gmem_c_m = by * BM + lane_id;
    int store_gmem_c_n = bx * BN;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    LDST128BITS(C[store_gmem_c_addr]) = (LDST128BITS(s_c[lane_id][0]));
  }
}

// 128x128, mma2x4, warp4x4(64,32,16)
// ============================================================================
// Kernel: hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel
// 功能: 优化的半精度矩阵乘法kernel，使用多级tiling策略
// 特点:
//   - 每个block包含8个warp（256个线程）
//   - 每个block计算一个128x128的输出块
//   - 每个warp计算一个64x32的输出块
//   - 每个warp使用2x4个MMA指令（每个MMA计算16x8）
//   - 支持共享内存bank冲突避免（通过padding）
// 参数说明:
//   - MMA_M=16, MMA_N=8, MMA_K=16: 单个MMA指令的维度
//   - MMA_TILE_M=2, MMA_TILE_N=4: 每个warp在M和N方向上的MMA数量
//   - WARP_TILE_M=4, WARP_TILE_N=4: 每个block在M和N方向上的warp数量
//   - A_PAD=0, B_PAD=0: 共享内存padding大小，用于避免bank冲突
// 用法:
//   - 输入: A(M×K), B(K×N), C(M×N) 都是半精度浮点数
//   - 输出: C = A × B
//   - 每个block计算一个128x128的输出块
//   - grid维度: (ceil(N/128), ceil(M/128))
//   - block维度: (256, 1, 1)
// 示例调用:
//   dim3 block(256);
//   dim3 grid(div_ceil(N, 128), div_ceil(M, 128));
//   hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<<<grid, block>>>(A, B, C, M, N, K);
// ============================================================================
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int A_PAD=0, 
         const int B_PAD=0>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16

  __shared__ half s_a[BM][BK+A_PAD]; // 128*16*2=4KB
  __shared__ half s_b[BK][BN+B_PAD]; // 16*128*2=4KB, 16*(128+16)*2=4.5KB

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }
  
  #pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    // gmem -> smem
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (
      LDST128BITS(B[load_gmem_b_addr]));
    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (
      LDST128BITS(A[load_gmem_a_addr]));
    __syncthreads(); 

    // ldmatrix for s_a, ldmatrix.trans for s_b.
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];

    // smem -> reg
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[lane_smem_a_m][lane_smem_a_k]);
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[lane_smem_b_k][lane_smem_b_n]);
      LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }
    
    // MMA compute
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], 
                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                  RB[j][0], RB[j][1], 
                  RC[i][j][0], RC[i][j][1]);
      }
    }
    __syncthreads(); 
  }

  // reg -> gmem, MMA_MxMMA_N=16x8
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      // mapping lane smem index -> global index.
      // [16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
      // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
      // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
      int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
      int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
      // TODO: how to use LDST128BITS here ? reverse the loop order ?
      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]); 
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]); 
    }
  }
}


// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)           \
if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
  throw std::runtime_error("Tensor size mismatch!");  \
}

// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.
// ============================================================================
// 函数: hgemm_mma_m16n8k16_naive
// 功能: PyTorch绑定的基础半精度矩阵乘法
// 用法:
//   - 输入: a(M×K), b(K×N), c(M×N) 都是torch.half类型
//   - 输出: c = a × b (原地更新)
//   - 调用方式: hgemm_mma_m16n8k16_naive(a, b, c)
// 注意事项:
//   - a, b, c必须是torch.half类型
//   - a的列数必须等于b的行数
//   - c的行数必须等于a的行数，列数必须等于b的列数
//   - 使用基础的naive kernel，每个block只包含1个warp
// 示例:
//   torch::Tensor a = torch::randn({M, K}, torch::kHalf).cuda();
//   torch::Tensor b = torch::randn({K, N}, torch::kHalf).cuda();
//   torch::Tensor c = torch::zeros({M, N}, torch::kHalf).cuda();
//   hgemm_mma_m16n8k16_naive(a, b, c);
// ============================================================================
void hgemm_mma_m16n8k16_naive(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16; 

  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
 
  hgemm_mma_m16n8k16_naive_kernel<
    MMA_M, MMA_N, MMA_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// 128x128, mma2x4, warp4x4(64,32,16)
// ============================================================================
// 函数: hgemm_mma_m16n8k16_mma2x4_warp4x4
// 功能: PyTorch绑定的优化半精度矩阵乘法
// 用法:
//   - 输入: a(M×K), b(K×N), c(M×N) 都是torch.half类型
//   - 输出: c = a × b (原地更新)
//   - 调用方式: hgemm_mma_m16n8k16_mma2x4_warp4x4(a, b, c)
// 特点:
//   - 使用优化的kernel，每个block包含8个warp（256线程）
//   - 每个block计算128x128的输出块
//   - 使用共享内存padding避免bank冲突
//   - 性能比naive版本更高
// 注意事项:
//   - a, b, c必须是torch.half类型
//   - a的列数必须等于b的行数
//   - c的行数必须等于a的行数，列数必须等于b的列数
// 示例:
//   torch::Tensor a = torch::randn({M, K}, torch::kHalf).cuda();
//   torch::Tensor b = torch::randn({K, N}, torch::kHalf).cuda();
//   torch::Tensor c = torch::zeros({M, N}, torch::kHalf).cuda();
//   hgemm_mma_m16n8k16_mma2x4_warp4x4(a, b, c);
// ============================================================================
void hgemm_mma_m16n8k16_mma2x4_warp4x4(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16; 
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int A_PAD = 0;
  constexpr int B_PAD = 16;
  constexpr int NUM_THREADS= (
    MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, MMA_N * MMA_TILE_N * WARP_TILE_N), 
            div_ceil(M, MMA_M * MMA_TILE_M * WARP_TILE_M));

  hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}
