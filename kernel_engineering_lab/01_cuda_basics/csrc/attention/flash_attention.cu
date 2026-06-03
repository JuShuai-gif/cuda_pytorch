/*
 * FlashAttention 前向传播 CUDA kernel
 *
 * 实现 Dao et al. (2022) 提出的 IO-Aware 精确注意力算法。
 * 核心思想：将 Q 分块为 Br 行，将 K/V 分块为 Bc 行，
 * 在共享内存中对每个分块进行计算，避免物化完整的 N×N 注意力矩阵。
 *
 * 算法流程（每个 thread block 处理一个 Q 分块）：
 *   1. 将 Q 分块加载到共享内存
 *   2. 初始化累积器：O_acc = 0, m = -inf, l = 0
 *   3. 遍历所有 K/V 分块：
 *      a. 加载 K 分块到共享内存
 *      b. 计算 S = Q @ K^T * scale
 *      c. 应用 causal mask（如果启用）
 *      d. 在线 softmax 第一步：计算 m_new, P, rowsum
 *      e. 加载 V 分块到共享内存（复用 K 的缓冲区）
 *      f. 在线 softmax 第二步：更新 O_acc, m, l
 *   4. O = O_acc / l，写入全局内存
 *
 * 模板参数：
 *   D_HEAD - head 维度（典型值 64, 128）
 *   Br     - Q 行分块大小（32 或 64）
 *   Bc     - K/V 列分块大小（32 或 64）
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdio>
#include <cmath>

/* --------------------------------------------------------------------------
 * 辅助宏：CUDA 错误检查
 * -------------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
        }                                                                      \
    } while (0)

/* --------------------------------------------------------------------------
 * FlashAttention 前向传播 kernel
 *
 * 每个 thread block 处理一个 (batch, head, Q 分块) 三元组。
 *
 * 网格维度：
 *   gridDim.x = (seq_len + Br - 1) / Br   (Q 分块数量)
 *   gridDim.y = n_heads                   (head 数量)
 *   gridDim.z = batch_size                (batch 数量)
 *
 * 共享内存布局（按偏移量，单位：__half 即 2 字节）：
 *   [0,                  Br*D_HEAD)          : Q_smem     (__half)
 *   [Br*D_HEAD,          Br*D_HEAD+Bc*D_HEAD): kv_smem    (__half, K/V 复用)
 *   [kv_END,             kv_END+Br*D_HEAD)   : O_acc      (__half, 累积器)
 *
 *   对齐后是 float 部分：
 *   S_smem     [Br * Bc]    float  (临时注意力 / P 矩阵)
 *   m_smem     [Br]         float  (运行最大值)
 *   l_smem     [Br]         float  (运行和)
 * -------------------------------------------------------------------------- */
template <int D_HEAD, int Br, int Bc>
__global__ void flash_attention_fwd_kernel(
    const __half* __restrict__ Q,       // [batch, n_heads, seq_len, head_dim]
    const __half* __restrict__ K,       // [batch, n_heads, seq_len, head_dim]
    const __half* __restrict__ V,       // [batch, n_heads, seq_len, head_dim]
    __half* __restrict__ O,             // [batch, n_heads, seq_len, head_dim]
    const float softmax_scale,
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const bool causal)
{
    /* ---- 计算本 block 对应的 batch、head 和 Q 分块索引 ---- */
    const int batch_idx = blockIdx.z;                   // 0 .. batch_size-1
    const int head_idx  = blockIdx.y;                   // 0 .. n_heads-1
    const int q_block    = blockIdx.x;                   // 第几个 Q 分块
    const int q_start    = q_block * Br;                // Q 分块起始行
    const int actual_Br  = min(Br, seq_len - q_start); // 实际有效行数

    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;

    /* ---- 获取指向本 (batch, head) 的基址指针 ---- */
    const int bh_offset = batch_idx * (n_heads * seq_len * D_HEAD)
                         + head_idx * (seq_len * D_HEAD);

    const __half* Q_bh = Q + bh_offset;
    const __half* K_bh = K + bh_offset;
    const __half* V_bh = V + bh_offset;
    __half*       O_bh = O + bh_offset;

    /* ---- 声明共享内存指针 ---- */
    extern __shared__ __half shared_raw[];

    // half 缓冲区偏移（以 __half 元素为单位）
    const int Q_OFFSET  = 0;
    const int KV_OFFSET = Q_OFFSET + Br * D_HEAD;
    const int O_OFFSET  = KV_OFFSET + Bc * D_HEAD;  // 注意：K 和 V 复用同一缓冲区

    // 计算 half 部分总大小，对齐到偶数（保证后续 float* 对齐）
    const int half_total  = O_OFFSET + Br * D_HEAD;
    const int half_padded = (half_total + 1) & ~1;

    // float 部分紧随 half 之后
    float* shared_float = reinterpret_cast<float*>(&shared_raw[half_padded]);

    const int S_OFFSET_F = 0;                        // S_smem  [Br * Bc]
    const int M_OFFSET_F = S_OFFSET_F + Br * Bc;     // m_smem  [Br]
    const int L_OFFSET_F = M_OFFSET_F + Br;          // l_smem  [Br]

    __half* Q_smem  = &shared_raw[Q_OFFSET];
    __half* kv_smem = &shared_raw[KV_OFFSET];   // K/V 复用缓冲区
    __half* O_acc   = &shared_raw[O_OFFSET];
    float*  S_smem  = &shared_float[S_OFFSET_F];
    float*  m_smem  = &shared_float[M_OFFSET_F];
    float*  l_smem  = &shared_float[L_OFFSET_F];

    /* ---- 第一步：将 Q 分块从全局内存加载到共享内存 ---- */
    for (int i = tid; i < Br * D_HEAD; i += num_threads) {
        const int row = i / D_HEAD;
        const int col = i % D_HEAD;
        if (row < actual_Br) {
            Q_smem[i] = Q_bh[(q_start + row) * D_HEAD + col];
        } else {
            Q_smem[i] = __float2half(0.0f);
        }
    }

    /* ---- 第二步：初始化累积器 ---- */
    for (int i = tid; i < Br * D_HEAD; i += num_threads) {
        O_acc[i] = __float2half(0.0f);
    }
    for (int i = tid; i < Br; i += num_threads) {
        m_smem[i] = -1e9f;
        l_smem[i] = 0.0f;
    }
    __syncthreads();

    /* ---- 第三步：遍历所有 K/V 分块 ---- */
    const int num_kv_blocks = (seq_len + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_start  = kv_block * Bc;
        const int actual_Bc = min(Bc, seq_len - kv_start);

        /* ---- 3a. 加载 K 分块到 kv_smem ---- */
        for (int i = tid; i < Bc * D_HEAD; i += num_threads) {
            const int row = i / D_HEAD;
            const int col = i % D_HEAD;
            if (row < actual_Bc) {
                kv_smem[i] = K_bh[(kv_start + row) * D_HEAD + col];
            } else {
                kv_smem[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        /* ---- 3b. 计算 S[Br][Bc] = Q @ K^T * scale ---- */
        for (int row = tid; row < actual_Br; row += num_threads) {
            for (int col = 0; col < actual_Bc; ++col) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D_HEAD; ++d) {
                    dot += __half2float(Q_smem[row * D_HEAD + d])
                         * __half2float(kv_smem[col * D_HEAD + d]);
                }
                S_smem[row * Bc + col] = dot * softmax_scale;
            }
            // 填充超出 actual_Bc 的列为 -inf
            for (int col = actual_Bc; col < Bc; ++col) {
                S_smem[row * Bc + col] = -1e9f;
            }
        }
        // 填充超出 actual_Br 的行
        for (int row = actual_Br + tid; row < Br; row += num_threads) {
            if (row < Br) {
                for (int col = 0; col < Bc; ++col) {
                    S_smem[row * Bc + col] = -1e9f;
                }
            }
        }
        __syncthreads();

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

        /* ---- 3d. 在线 softmax 第一步：计算 P = exp(S - m_new) ---- */
        for (int row = tid; row < actual_Br; row += num_threads) {
            // 计算当前行最大值
            float row_max = -1e9f;
            for (int col = 0; col < Bc; ++col) {
                row_max = fmaxf(row_max, S_smem[row * Bc + col]);
            }

            float m_old = m_smem[row];
            float m_new = fmaxf(m_old, row_max);
            float alpha = (m_old > m_new || m_old <= -1e8f)
                          ? 0.0f : expf(m_old - m_new);

            // 计算 P = exp(S - m_new) 和行和
            float row_sum_new = 0.0f;
            for (int col = 0; col < Bc; ++col) {
                float p_val = expf(S_smem[row * Bc + col] - m_new);
                S_smem[row * Bc + col] = p_val;  // 原地替换为 P
                row_sum_new += p_val;
            }

            // 缩放旧累积器
            for (int d = 0; d < D_HEAD; ++d) {
                O_acc[row * D_HEAD + d] =
                    __float2half(alpha * __half2float(O_acc[row * D_HEAD + d]));
            }

            // 更新 m 和 l（暂存回共享内存）
            m_smem[row] = m_new;
            l_smem[row] = alpha * l_smem[row] + row_sum_new;
        }
        __syncthreads();

        /* ---- 现在可以安全地复用 kv_smem 加载 V ---- */
        /* ---- 3e. 加载 V 分块到 kv_smem（复用缓冲区） ---- */
        for (int i = tid; i < Bc * D_HEAD; i += num_threads) {
            const int row = i / D_HEAD;
            const int col = i % D_HEAD;
            if (row < actual_Bc) {
                kv_smem[i] = V_bh[(kv_start + row) * D_HEAD + col];
            } else {
                kv_smem[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        /* ---- 3f. 线上 softmax 第二步：O_acc += P @ V ---- */
        for (int row = tid; row < actual_Br; row += num_threads) {
            for (int d = 0; d < D_HEAD; ++d) {
                float o_val = __half2float(O_acc[row * D_HEAD + d]);
                float pv_sum = 0.0f;
                for (int col = 0; col < Bc; ++col) {
                    pv_sum += S_smem[row * Bc + col]
                            * __half2float(kv_smem[col * D_HEAD + d]);
                }
                O_acc[row * D_HEAD + d] = __float2half(o_val + pv_sum);
            }
        }
        __syncthreads();
    }

    /* ---- 第四步：最终归一化 O = O_acc / l ---- */
    for (int i = tid; i < actual_Br * D_HEAD; i += num_threads) {
        const int row = i / D_HEAD;
        const int d   = i % D_HEAD;
        float val = __half2float(O_acc[row * D_HEAD + d]);
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
 * Host 端包装函数
 * ========================================================================== */

/*
 * 根据 head_dim 分派到正确的模板实例化。
 *
 * 支持的 head_dim：64, 128
 * 分块大小根据 head_dim 自动选择以适配共享内存容量
 * （典型 48KB 限制，Ada/Ampere 架构）。
 *
 * Q/K/V/O 均为 [batch, n_heads, seq_len, head_dim]，dtype=float16。
 */
void run_flash_attention_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    float softmax_scale,
    bool causal)
{
    TORCH_CHECK(Q.device().is_cuda(),  "Q 必须在 CUDA 上");
    TORCH_CHECK(K.device().is_cuda(),  "K 必须在 CUDA 上");
    TORCH_CHECK(V.device().is_cuda(),  "V 必须在 CUDA 上");
    TORCH_CHECK(O.device().is_cuda(),  "O 必须在 CUDA 上");
    TORCH_CHECK(Q.is_contiguous(),     "Q 必须是 contiguous");
    TORCH_CHECK(K.is_contiguous(),     "K 必须是 contiguous");
    TORCH_CHECK(V.is_contiguous(),     "V 必须是 contiguous");
    TORCH_CHECK(O.is_contiguous(),     "O 必须是 contiguous");
    TORCH_CHECK(Q.dtype() == torch::kHalf, "Q 必须是 float16");
    TORCH_CHECK(K.dtype() == torch::kHalf, "K 必须是 float16");
    TORCH_CHECK(V.dtype() == torch::kHalf, "V 必须是 float16");
    TORCH_CHECK(O.dtype() == torch::kHalf, "O 必须是 float16");
    TORCH_CHECK(Q.dim() == 4, "Q 必须是 4 维 [batch, n_heads, seq_len, head_dim]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q 和 K 维度必须相同");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q 和 V 维度必须相同");
    TORCH_CHECK(Q.sizes() == O.sizes(), "Q 和 O 维度必须相同");

    const int batch_size = Q.size(0);
    const int n_heads    = Q.size(1);
    const int seq_len    = Q.size(2);
    const int head_dim   = Q.size(3);

    const int num_threads = 128;

    // 根据 head_dim 分派模板实例化
    // 注意：K 和 V 复用同一共享内存缓冲区，实际 half 元素为
    //   Q_smem + kv_smem + O_acc（kv_smem 在 K 和 V 之间复用）
    if (head_dim == 64) {
        // D_HEAD=64, Br=64, Bc=64：共享内存 ~40.5KB，适配 48KB 限制
        constexpr int Br = 64;
        constexpr int Bc = 64;

        const int num_q_blocks = (seq_len + Br - 1) / Br;

        dim3 grid(num_q_blocks, n_heads, batch_size);
        dim3 block(num_threads);

        const int half_elements = Br * head_dim       // Q_smem
                                + Bc * head_dim       // kv_smem (K/V 复用)
                                + Br * head_dim;      // O_acc
        const int half_padded = (half_elements + 1) & ~1;
        const int float_bytes  = (Br * Bc + Br + Br) * sizeof(float);
        const size_t smem_size = half_padded * sizeof(__half) + float_bytes;

        flash_attention_fwd_kernel<64, Br, Bc>
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
        // D_HEAD=128, Br=32, Bc=32：共享内存 ~28KB
        constexpr int Br = 32;
        constexpr int Bc = 32;

        const int num_q_blocks = (seq_len + Br - 1) / Br;

        dim3 grid(num_q_blocks, n_heads, batch_size);
        dim3 block(num_threads);

        const int half_elements = Br * head_dim       // Q_smem
                                + Bc * head_dim       // kv_smem (K/V 复用)
                                + Br * head_dim;      // O_acc
        const int half_padded = (half_elements + 1) & ~1;
        const int float_bytes  = (Br * Bc + Br + Br) * sizeof(float);
        const size_t smem_size = half_padded * sizeof(__half) + float_bytes;

        flash_attention_fwd_kernel<128, Br, Bc>
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
        TORCH_CHECK(false, "暂不支持的 head_dim: ", head_dim,
                    "（当前仅支持 64 和 128）");
    }

    CUDA_CHECK(cudaGetLastError());
}
