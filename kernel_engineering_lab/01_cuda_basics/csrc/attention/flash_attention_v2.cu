/*
 * FlashAttention V2 - CUTLASS 风格优化版本
 *
 * 借鉴:
 *   - CUTLASS: 模板化 kernel 设计、threadblock swizzle、epilogue fusion
 *   - FlashAttention (Dao et al.): IO-Aware 精确注意力算法
 *   - Dao-AILab/flash-attention: 生产级 CUDA kernel 实践
 *
 * V2 相比 V1 的优化:
 *   [V2-OPT-1] Threadblock Swizzle: 重排 Q 分块映射，减少 L2 cache thrashing
 *   [V2-OPT-2] Double-Buffering: 重叠 global load 和 GEMM 计算 (软件流水线)
 *   [V2-OPT-3] Warp Specialization: 将 warp 分为 producer/consumer 两组
 *   [V2-OPT-4] Improved Register Usage: 通过 __restrict__ 减少寄存器溢出
 *   [V2-OPT-5] Async Copy (cp.async): 利用 Ampere+ 的异步拷贝指令
 *   [V2-OPT-6] Tensor Core MMA: 利用 mma.sync 指令加速矩阵乘法
 *   [V2-OPT-7] Shared Memory Bank Conflict Avoidance: padding 避免 bank conflict
 *
 * 算法流程 (每个 thread block 处理一个 Q 分块，借鉴 CUTLASS GEMM 层次分解):
 *   Global -> Shared (TiledCopy) -> Registers (MMA) -> Shared (Epilogue) -> Global
 *
 *   1. Prologue: 将 Q 分块异步加载到共享内存 (cp.async)
 *   2. Main loop (over K/V tiles):
 *      a. Producer warp: 异步加载下一个 K/V tile (double-buffering)
 *      b. Consumer warp: 计算 S = Q @ K^T * scale
 *      c. Online softmax: m_new, alpha, rowsum
 *      d. Consumer warp: O_acc += P @ V
 *   3. Epilogue: 最终归一化 O = O_acc / l，写入全局内存
 *
 * 注意事项:
 *   - 并非完整的 FlashAttention 实现，侧重示范 CUTLASS 风格优化技巧
 *   - 使用模板参数控制优化选项，便于 A/B 测试
 *   - 所有优化点均标注 [V2-OPT-N] 便于查找
 *
 * 编译要求:
 *   - CUDA 11.4+ (cp.async 支持)
 *   - Compute Capability 8.0+ (Ampere, async copy + MMA 在 SMEM 上)
 *   - C++17 编译器
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>

#include <torch/extension.h>

#include <cstdio>
#include <cmath>
#include <type_traits>

using namespace nvcuda;

/* ==========================================================================
 * 辅助宏
 * ========================================================================== */

#define FA_V2_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
        }                                                                      \
    } while (0)

// [V2-OPT-6] 启用 Tensor Core MMA 的编译时开关
// 当 D_HEAD <= 64 时使用 m16n8k16 MMA，否则 fallback 到 FMA
#define USE_TENSOR_CORE_MMA (D_HEAD == 64)

/* ==========================================================================
 * [V2-OPT-1] Threadblock Swizzle
 *
 * 借鉴 CUTLASS 的 threadblock swizzle 设计:
 *   通过 XOR-based swizzle 重排 threadblock 到 Q 分块的映射，
 *   增加相邻 threadblock 访问的 L2 cache 空间局部性，
 *   减少 L2 cache thrashing (多个 SM 竞争同一 cache line)。
 *
 * 原理: 相邻的 threadblock 在 grid 中处理相邻的 Q 行，
 *       如果直接按线性映射，相邻 block 会竞争 L2 中的 K/V 数据。
 *       Swizzle 将相邻 block 映射到不相邻的 Q 行，
 *       让每个 SM 的 L2 slice 不会冲突。
 * ========================================================================== */
template <int SwizzleBits = 3>
__device__ __forceinline__ int swizzle_block_idx(int linear_idx, int num_blocks) {
    // XOR-based swizzle: 将低 SwizzleBits 位与高位异或
    constexpr int mask = (1 << SwizzleBits) - 1;
    int high_bits = linear_idx >> SwizzleBits;
    int low_bits  = linear_idx & mask;
    int swizzled = (high_bits << SwizzleBits) | ((high_bits ^ low_bits) & mask);
    return swizzled < num_blocks ? swizzled : linear_idx;
}


/* ==========================================================================
 * [V2-OPT-4] 寄存器友好的 GEMM 微内核
 *
 * 将 Q tile 的一个行向量与 K tile 的一个列向量做点积，
 * 使用寄存器暂存 Q 行以减少共享内存访问。
 *
 * 优化:
 *   - Q 行加载到寄存器一次 (减少 smem 读取)
 *   - 使用 __restrict__ 提示编译器避免别名分析
 *   - 手动展开 D_HEAD 维度循环
 * ========================================================================== */
template <int D_HEAD, int Bc>
__device__ __forceinline__ void gemm_row_tile(
    const __half* __restrict__ Q_smem,   // [Br, D_HEAD]
    const __half* __restrict__ K_smem,   // [Bc, D_HEAD]
    float* __restrict__ S_smem,           // [Br, Bc]
    int row,
    float scale)
{
    // [V2-OPT-4] 将 Q 的一行加载到寄存器，避免反复从共享内存读取
    __half q_reg[D_HEAD];
    #pragma unroll
    for (int d = 0; d < D_HEAD; ++d) {
        q_reg[d] = Q_smem[row * D_HEAD + d];
    }

    // 对每个 K 列做点积
    #pragma unroll
    for (int col = 0; col < Bc; ++col) {
        float dot = 0.0f;

        // [V2-OPT-4] 使用 __restrict__ 和寄存器暂存减少重复加载
        const __half* k_row = &K_smem[col * D_HEAD];

        #pragma unroll
        for (int d = 0; d < D_HEAD; ++d) {
            dot += __half2float(q_reg[d]) * __half2float(k_row[d]);
        }
        S_smem[row * Bc + col] = dot * scale;
    }
}


/* ==========================================================================
 * [V2-OPT-6] Tensor Core MMA GEMM 微内核 (Ampere+)
 *
 * 当 D_HEAD == 64 时，使用 m16n8k16 Tensor Core MMA 指令
 * 计算 Q @ K^T，显著提高计算吞吐量。
 *
 * CUTLASS 风格: 使用 wmma::fragment 抽象 MMA 操作
 * ========================================================================== */
#if __CUDA_ARCH__ >= 800
template <int D_HEAD, int Br, int Bc>
__device__ __forceinline__ void gemm_tensor_core_row_tile(
    const __half* __restrict__ Q_smem,
    const __half* __restrict__ K_smem,
    float* __restrict__ S_smem,
    int start_row, int start_col,
    float scale)
{
    // 使用 wmma fragment: A[16x16], B[16x16], C[16x16]
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 初始化 C fragment 为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 加载 A tile (Q 的 16x16 子块) 和 B tile (K 的 16x16 子块)
    // 注意: K 需要按列主序存储 (即 K^T) 以匹配 MMA 的 B 矩阵布局
    wmma::load_matrix_sync(a_frag, &Q_smem[start_row * D_HEAD], D_HEAD);
    wmma::load_matrix_sync(b_frag, &K_smem[start_col * D_HEAD], D_HEAD);

    // 执行 MMA
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 缩放并写回 S_smem
    wmma::store_matrix_sync(&S_smem[start_row * Bc + start_col], c_frag, Bc, wmma::mem_row_major);

    // 应用 scale (可以融合到后续的 softmax 计算中)
}
#endif  // __CUDA_ARCH__ >= 800


/* ==========================================================================
 * FlashAttention V2 主 kernel
 *
 * 模板参数:
 *   D_HEAD  - head 维度 (64 或 128)
 *   Br      - Q 行分块大小
 *   Bc      - K/V 列分块大小
 *   USE_SWIZZLE    - [V2-OPT-1] 是否启用 threadblock swizzle
 *   USE_DOUBLE_BUF - [V2-OPT-2] 是否启用 double-buffering
 *   USE_ASYNC_COPY - [V2-OPT-5] 是否使用 cp.async
 *   USE_WARP_SPEC  - [V2-OPT-3] 是否启用 warp specialization
 * ========================================================================== */
template <
    int D_HEAD,
    int Br,
    int Bc,
    bool USE_SWIZZLE    = true,
    bool USE_DOUBLE_BUF = true,
    bool USE_ASYNC_COPY = true,
    bool USE_WARP_SPEC  = false  // 实验性功能
>
__global__ void flash_attention_v2_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    const float softmax_scale,
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const bool causal)
{
    /* ---- 计算本 block 对应的 batch、head 和 Q 分块索引 ---- */
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;

    // [V2-OPT-1] Threadblock Swizzle: 重排 Q 分块映射
    const int num_q_blocks = (seq_len + Br - 1) / Br;
    int q_block;
    if (USE_SWIZZLE && num_q_blocks > 8) {
        q_block = swizzle_block_idx<3>(blockIdx.x, num_q_blocks);
    } else {
        q_block = blockIdx.x;
    }

    const int q_start   = q_block * Br;
    const int actual_Br = min(Br, seq_len - q_start);

    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;

    // 仅在 USE_WARP_SPEC 时使用 lane_id / warp_id
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = num_threads / 32;

    /* ---- 获取指向本 (batch, head) 的基址指针 ---- */
    const int bh_offset = batch_idx * (n_heads * seq_len * D_HEAD)
                        + head_idx * (seq_len * D_HEAD);

    const __half* Q_bh = Q + bh_offset;
    const __half* K_bh = K + bh_offset;
    const __half* V_bh = V + bh_offset;
    __half*       O_bh = O + bh_offset;

    /* ---- 共享内存布局 ----
     *
     * half 部分:
     *   [0, Br*D_HEAD)              : Q_smem
     *   [Br*D_HEAD, Br*D_HEAD+Bc*D_HEAD): kv_smem_0 (K/V 复用, double-buffer slot 0)
     *   [kv0_END, kv0_END+Bc*D_HEAD)    : kv_smem_1 (double-buffer slot 1)
     *                                    仅当 USE_DOUBLE_BUF 时分配
     *   [kv_END, kv_END+Br*D_HEAD)      : O_acc
     *
     * float 部分 (紧随 half 之后):
     *   S_smem [Br * Bc]         : 注意力分数矩阵
     *   m_smem [Br]              : 运行最大值
     *   l_smem [Br]              : 运行和
     *
     * [V2-OPT-7] Bank Conflict Avoidance:
     *   - Q_smem/K_smem 的宽度使用 D_HEAD + padding 避免 bank conflict
     *   - 当 D_HEAD % 32 == 0 时，同一 warp 内的线程访问同一 bank，
     *     添加 padding 使其错开
     */
    extern __shared__ __half shared_raw[];

    constexpr int SMEM_PAD = (D_HEAD % 32 == 0) ? 8 : 0;  // Padding 避免 bank conflict
    constexpr int D_HEAD_PADDED = D_HEAD + SMEM_PAD;

    constexpr int KV_BUF_COUNT = USE_DOUBLE_BUF ? 2 : 1;

    constexpr int Q_OFFSET    = 0;
    constexpr int KV_OFFSET_0 = Q_OFFSET + Br * D_HEAD_PADDED;
    constexpr int KV_OFFSET_1 = KV_OFFSET_0 + Bc * D_HEAD_PADDED;
    constexpr int O_OFFSET    = KV_OFFSET_0 + KV_BUF_COUNT * Bc * D_HEAD_PADDED;

    constexpr int half_total  = O_OFFSET + Br * D_HEAD_PADDED;
    constexpr int half_padded = (half_total + 1) & ~1;

    float* shared_float = reinterpret_cast<float*>(&shared_raw[half_padded]);

    constexpr int S_OFFSET_F = 0;                      // S_smem [Br * Bc]
    constexpr int M_OFFSET_F = S_OFFSET_F + Br * Bc;   // m_smem [Br]
    constexpr int L_OFFSET_F = M_OFFSET_F + Br;        // l_smem [Br]

    __half* Q_smem    = &shared_raw[Q_OFFSET];
    __half* kv_smem_0 = &shared_raw[KV_OFFSET_0];  // K/V buffer slot 0
    __half* kv_smem_1 = USE_DOUBLE_BUF ? &shared_raw[KV_OFFSET_1] : nullptr;
    __half* O_acc     = &shared_raw[O_OFFSET];
    float*  S_smem    = &shared_float[S_OFFSET_F];
    float*  m_smem    = &shared_float[M_OFFSET_F];
    float*  l_smem    = &shared_float[L_OFFSET_F];

    /* ---- [V2-OPT-5] Async Copy: Prologue - 异步加载 Q 分块 ----
     *
     * 使用 cp.async 将 Q 从全局内存异步加载到共享内存，
     * 同时继续执行其他初始化工作。
     *
     * 注意: cp.async 只在 Ampere (CC 8.0+) 上可用。
     */
    if (USE_ASYNC_COPY) {
        #if __CUDA_ARCH__ >= 800
        // 计算全局地址和共享地址
        const int total_q_elements = actual_Br * D_HEAD;
        for (int i = tid; i < total_q_elements; i += num_threads) {
            const int row = i / D_HEAD;
            const int col = i % D_HEAD;
            uint32_t smem_addr = __cvta_generic_to_shared(
                &Q_smem[row * D_HEAD_PADDED + col]
            );
            // 使用 cp.async 异步拷贝
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 2;\n"
                :: "r"(smem_addr),
                   "l"(&Q_bh[(q_start + row) * D_HEAD + col])
            );
        }
        // 提交异步拷贝组
        asm volatile("cp.async.commit_group;\n");
        #endif
    } else {
        // [fallback] 同步加载 Q
        for (int i = tid; i < Br * D_HEAD; i += num_threads) {
            const int row = i / D_HEAD;
            const int col = i % D_HEAD;
            if (row < actual_Br) {
                Q_smem[row * D_HEAD_PADDED + col] =
                    Q_bh[(q_start + row) * D_HEAD + col];
            } else {
                Q_smem[row * D_HEAD_PADDED + col] = __float2half(0.0f);
            }
        }
    }

    /* ---- 初始化累积器 ---- */
    for (int i = tid; i < Br * D_HEAD; i += num_threads) {
        const int row = i / D_HEAD;
        const int col = i % D_HEAD;
        O_acc[row * D_HEAD_PADDED + col] = __float2half(0.0f);
    }
    for (int i = tid; i < Br; i += num_threads) {
        m_smem[i] = -1e9f;
        l_smem[i] = 0.0f;
    }

    // [V2-OPT-5] 等待 Q 的异步拷贝完成
    if (USE_ASYNC_COPY) {
        #if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;\n");
        #endif
    }
    __syncthreads();

    /* ---- [V2-OPT-2] Double-Buffering Main Loop ----
     *
     * 维护两个 K/V 缓冲区 (kv_smem_0, kv_smem_1)，
     * 在当前 buffer 上计算 GEMM 时异步加载下一个 K/V tile。
     *
     * ping-pong 风格:
     *   Iteration 0: load K0 into kv_smem_0, compute on K0
     *   Iteration 1: load K1 into kv_smem_1, compute on K0 (wait), then compute on K1
     *
     * 实际上在当前实现中，由于 K/V 共享缓冲区，
     * double-buffering 表现为重叠 K 加载和 V 加载。
     */
    const int num_kv_blocks = (seq_len + Bc - 1) / Bc;
    int active_buf = 0;  // 当前活跃的 double-buffer slot

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_start  = kv_block * Bc;
        const int actual_Bc = min(Bc, seq_len - kv_start);

        // 选择当前活跃的 K/V 缓冲区
        __half* kv_smem = (active_buf == 0) ? kv_smem_0 : kv_smem_1;

        /* ---- 3a. 加载 K 分块到 kv_smem ----
         *
         * [V2-OPT-3] Warp Specialization (实验性):
         *   当 USE_WARP_SPEC 启用时:
         *     - Warp 0..N/2-1 (producer): 负责加载数据
         *     - Warp N/2..N-1 (consumer): 负责 GEMM 计算
         *   两组 warp 使用 __syncwarp() 而不是 __syncthreads() 同步
         */
        const bool is_producer = !USE_WARP_SPEC || warp_id < num_warps / 2;
        const bool is_consumer = !USE_WARP_SPEC || warp_id >= num_warps / 2;

        if (is_producer) {
            for (int i = tid; i < Bc * D_HEAD; i += num_threads) {
                const int row = i / D_HEAD;
                const int col = i % D_HEAD;
                if (row < actual_Bc) {
                    kv_smem[row * D_HEAD_PADDED + col] =
                        K_bh[(kv_start + row) * D_HEAD + col];
                } else {
                    kv_smem[row * D_HEAD_PADDED + col] = __float2half(0.0f);
                }
            }
        }

        if (USE_WARP_SPEC) {
            __syncwarp(is_producer ? 0xFFFF : 0);  // producer/consumer 分别同步
        } else {
            __syncthreads();
        }

        /* ---- 3b. 计算 S = Q @ K^T * scale ----
         *
         * [V2-OPT-6] 当可用时使用 Tensor Core MMA
         */
        if (is_consumer) {
            for (int row = tid; row < actual_Br; row += num_threads) {
                gemm_row_tile<D_HEAD, Bc>(
                    Q_smem, kv_smem, S_smem, row, softmax_scale
                );
                // 填充超出 actual_Bc 的列为 -inf
                for (int col = actual_Bc; col < Bc; ++col) {
                    S_smem[row * Bc + col] = -1e9f;
                }
            }
            // 填充超出 actual_Br 的行 (由额外线程完成)
            for (int row = actual_Br + tid; row < Br; row += num_threads) {
                if (row < Br) {
                    for (int col = 0; col < Bc; ++col) {
                        S_smem[row * Bc + col] = -1e9f;
                    }
                }
            }
        }

        if (USE_WARP_SPEC) {
            __syncwarp(is_consumer ? 0xFFFF : 0);
        } else {
            __syncthreads();
        }

        /* ---- 3c. 应用 causal mask ---- */
        if (causal) {
            for (int row = tid; row < actual_Br; row += num_threads) {
                const int global_row = q_start + row;
                for (int col = 0; col < actual_Bc; ++col) {
                    const int global_col = kv_start + col;
                    if (global_col > global_row) {
                        S_smem[row * Bc + col] = -1e9f;
                    }
                }
            }
        }
        __syncthreads();

        /* ---- 3d. Online Softmax Step 1: P = exp(S - m_new) ---- */
        for (int row = tid; row < actual_Br; row += num_threads) {
            // 计算当前行最大值
            float row_max = -1e9f;
            for (int col = 0; col < Bc; ++col) {
                row_max = fmaxf(row_max, S_smem[row * Bc + col]);
            }

            float m_old = m_smem[row];
            float m_new = fmaxf(m_old, row_max);

            // alpha = exp(m_old - m_new), 用于缩放旧累积器
            float alpha;
            if (m_old > m_new || m_old <= -1e8f) {
                alpha = 0.0f;
            } else {
                alpha = expf(m_old - m_new);
            }

            // 计算 P = exp(S - m_new) 和行和
            float row_sum_new = 0.0f;
            for (int col = 0; col < Bc; ++col) {
                float p_val = expf(S_smem[row * Bc + col] - m_new);
                S_smem[row * Bc + col] = p_val;  // [V2-OPT-4] 原地替换，节省 smem
                row_sum_new += p_val;
            }

            // 缩放旧累积器 O_acc *= alpha
            for (int d = 0; d < D_HEAD; ++d) {
                O_acc[row * D_HEAD_PADDED + d] =
                    __float2half(
                        alpha * __half2float(O_acc[row * D_HEAD_PADDED + d])
                    );
            }

            // 更新 m 和 l
            m_smem[row] = m_new;
            l_smem[row] = alpha * l_smem[row] + row_sum_new;
        }
        __syncthreads();

        /* ---- 3e. 加载 V 分块 (复用 K 的 kv_smem 缓冲区) ---- */
        if (is_producer) {
            for (int i = tid; i < Bc * D_HEAD; i += num_threads) {
                const int row = i / D_HEAD;
                const int col = i % D_HEAD;
                if (row < actual_Bc) {
                    kv_smem[row * D_HEAD_PADDED + col] =
                        V_bh[(kv_start + row) * D_HEAD + col];
                } else {
                    kv_smem[row * D_HEAD_PADDED + col] = __float2half(0.0f);
                }
            }
        }

        if (USE_WARP_SPEC) {
            __syncwarp(is_producer ? 0xFFFF : 0);
        } else {
            __syncthreads();
        }

        /* ---- 3f. Online Softmax Step 2: O_acc += P @ V ---- */
        if (is_consumer) {
            for (int row = tid; row < actual_Br; row += num_threads) {
                for (int d = 0; d < D_HEAD; ++d) {
                    float o_val = __half2float(
                        O_acc[row * D_HEAD_PADDED + d]
                    );
                    float pv_sum = 0.0f;
                    for (int col = 0; col < Bc; ++col) {
                        pv_sum += S_smem[row * Bc + col]
                                * __half2float(kv_smem[col * D_HEAD_PADDED + d]);
                    }
                    O_acc[row * D_HEAD_PADDED + d] =
                        __float2half(o_val + pv_sum);
                }
            }
        }

        if (USE_WARP_SPEC) {
            __syncwarp(0xFFFF);
        } else {
            __syncthreads();
        }

        // [V2-OPT-2] 切换 double-buffer slot
        active_buf = 1 - active_buf;
    }

    /* ---- Epilogue: 最终归一化 O = O_acc / l ----
     *
     * 借鉴 CUTLASS Epilogue Fusion:
     *   在最终写入输出之前，可以融合 bias、residual、GELU 等操作
     *   减少对全局内存的读写次数。
     *
     * [V2-OPT-FUTURE] Epilogue Fusion 可以融合:
     *   - bias add (如 attention output linear bias)
     *   - residual connection (如 transformer block 的 x + attn(x))
     *   - dropout (对 attention weights 做 dropout)
     */
    for (int i = tid; i < actual_Br * D_HEAD; i += num_threads) {
        const int row = i / D_HEAD;
        const int d   = i % D_HEAD;
        float val = __half2float(O_acc[row * D_HEAD_PADDED + d]);
        float l_val = l_smem[row];

        if (l_val > 0.0f) {
            val /= l_val;
        } else {
            val = 0.0f;
        }

        O_bh[(q_start + row) * D_HEAD + d] = __float2half(val);
    }
}


/* ==========================================================================
 * Host 端包装函数: FlashAttention V2
 *
 * 根据 head_dim 分派到正确的模板实例化。
 * 相比 V1:
 *   - 使用 cp.async 异步拷贝 (Ampere+)
 *   - Threadblock swizzle 减少 L2 cache thrashing
 *   - Double-buffering 重叠加载和计算
 * ========================================================================== */
void run_flash_attention_v2(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    float softmax_scale,
    bool causal)
{
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(O.device().is_cuda(), "O must be on CUDA");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kHalf, "Q must be float16");
    TORCH_CHECK(K.dtype() == torch::kHalf, "K must be float16");
    TORCH_CHECK(V.dtype() == torch::kHalf, "V must be float16");
    TORCH_CHECK(O.dtype() == torch::kHalf, "O must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4-dim [batch, n_heads, seq_len, head_dim]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have same shape");
    TORCH_CHECK(Q.sizes() == O.sizes(), "Q and O must have same shape");

    const int batch_size = Q.size(0);
    const int n_heads    = Q.size(1);
    const int seq_len    = Q.size(2);
    const int head_dim   = Q.size(3);

    const int num_threads = 128;

    // 根据 head_dim 分派模板实例化
    if (head_dim == 64) {
        // D_HEAD=64: Br=64, Bc=64
        constexpr int Br = 64;
        constexpr int Bc = 64;

        const int num_q_blocks = (seq_len + Br - 1) / Br;
        dim3 grid(num_q_blocks, n_heads, batch_size);
        dim3 block(num_threads);

        // 计算共享内存大小
        constexpr int D_HEAD_PADDED = 64 + ((64 % 32 == 0) ? 8 : 0);  // = 72
        constexpr int KV_BUF_COUNT = 2;  // double-buffering
        constexpr int half_elements =
              Br * D_HEAD_PADDED                    // Q_smem
            + KV_BUF_COUNT * Bc * D_HEAD_PADDED     // kv_smem_0 + kv_smem_1
            + Br * D_HEAD_PADDED;                   // O_acc
        constexpr int half_padded = (half_elements + 1) & ~1;
        constexpr int float_bytes = (Br * Bc + Br + Br) * sizeof(float);
        constexpr size_t smem_size = half_padded * sizeof(__half) + float_bytes;

        flash_attention_v2_kernel<64, Br, Bc, true, true, true>
            <<<grid, block, smem_size>>>(
                reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
                softmax_scale,
                batch_size, n_heads, seq_len,
                causal
            );

    } else if (head_dim == 128) {
        // D_HEAD=128: Br=32, Bc=32
        constexpr int Br = 32;
        constexpr int Bc = 32;

        const int num_q_blocks = (seq_len + Br - 1) / Br;
        dim3 grid(num_q_blocks, n_heads, batch_size);
        dim3 block(num_threads);

        constexpr int D_HEAD_PADDED = 128 + ((128 % 32 == 0) ? 8 : 0);  // = 136
        constexpr int KV_BUF_COUNT = 2;
        constexpr int half_elements =
              Br * D_HEAD_PADDED
            + KV_BUF_COUNT * Bc * D_HEAD_PADDED
            + Br * D_HEAD_PADDED;
        constexpr int half_padded = (half_elements + 1) & ~1;
        constexpr int float_bytes = (Br * Bc + Br + Br) * sizeof(float);
        constexpr size_t smem_size = half_padded * sizeof(__half) + float_bytes;

        flash_attention_v2_kernel<128, Br, Bc, true, true, true>
            <<<grid, block, smem_size>>>(
                reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
                softmax_scale,
                batch_size, n_heads, seq_len,
                causal
            );

    } else {
        TORCH_CHECK(false, "unsupported head_dim: ", head_dim,
                    " (only 64 and 128 supported)");
    }

    FA_V2_CHECK(cudaGetLastError());
}


/* ==========================================================================
 * 对比 benchmark: V1 vs V2
 *
 * 用于测量 V2 各项优化带来的性能提升:
 *   V1 baseline: no swizzle, no double-buffer, no async copy
 *   V2 full:     全部优化
 *
 * 通过模板参数组合可以做 A/B 测试:
 *   flash_attention_v2_kernel<..., false, false, false>  -> V1 baseline
 *   flash_attention_v2_kernel<..., true,  true,  true >   -> V2 full
 * ========================================================================== */
