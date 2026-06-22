/*
 * PagedAttention (vLLM 风格) CUDA kernel
 *
 * 实现 vLLM 中使用的分页注意力机制，用于自回归解码阶段。
 * 此时 Q 只有 1 个 token（seq_len=1），K/V 按固定大小的块存储。
 *
 * 核心概念：
 *   - K/V cache 存储在固定大小的物理块中（block_size 个 token/块）
 *   - block_table 将逻辑块索引映射到物理块索引
 *   - 避免了内存碎片，支持跨请求的 KV cache 共享（prefix caching）
 *   - 每个 thread block 处理一个 (batch, head) 对
 *
 * 算法流程（每个 thread block）：
 *   1. 将当前 head 的 Q 向量加载到共享内存
 *   2. 初始化：O_acc = 0, m = -inf, l = 0
 *   3. 遍历该序列的所有逻辑块：
 *      a. 通过 block_table 获取物理块索引
 *      b. 将 K 块和 V 块从全局内存加载到共享内存
 *      c. 计算 S = Q @ K^T * scale（得到 block_size 个分数）
 *      d. 对最后一个不完整块应用 mask
 *      e. 在线 softmax：
 *         - m_new = max(m_old, max(S))
 *         - alpha = exp(m_old - m_new)
 *         - P = exp(S - m_new)
 *         - l_new = alpha * l_old + sum(P)
 *         - O_acc = alpha * O_acc + P @ V
 *      f. 更新 m = m_new, l = l_new
 *   4. O = O_acc / l，写入全局内存
 *
 * 模板参数：
 *   HEAD_DIM   - head 维度（典型值 64, 128, 256）
 *   BLOCK_SIZE - 每个物理块的 token 数（典型值 16）
 *   NUM_THREADS - 每个 thread block 的线程数（128 或 256）
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
 * 辅助函数：warp 内求和 reduction
 * -------------------------------------------------------------------------- */
__inline__ __device__ float warp_reduce_sum_pa(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/* --------------------------------------------------------------------------
 * 辅助函数：warp 内求最大值 reduction
 * -------------------------------------------------------------------------- */
__inline__ __device__ float warp_reduce_max_pa(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/* --------------------------------------------------------------------------
 * block 级求和 reduction：将所有 warp 的部分和汇总
 *
 * 使用共享内存暂存各 warp 的部分和，然后由 warp 0 完成最终求和。
 * 调用前需要确保 shared_sum 数组有足够的空间（num_warps 个 float）。
 * -------------------------------------------------------------------------- */
__inline__ __device__ float block_reduce_sum(float val, float* shared_sum, int num_warps) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // warp 内 reduction
    val = warp_reduce_sum_pa(val);

    // 每个 warp 的 lane 0 写入共享内存
    if (lane_id == 0) {
        shared_sum[warp_id] = val;
    }
    __syncthreads();

    // warp 0 读取各部分和并完成最终 reduction
    val = (warp_id == 0 && lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum_pa(val);
    }

    // 将结果广播给所有线程（仅 warp 0 的 lane 0 有结果）
    // 这里我们只让调用者（通常是 lane 0 或专用线程）使用结果
    return val;  // 只有 warp 0 lane 0 的值是有效的
}

/* --------------------------------------------------------------------------
 * block 级求最大值 reduction
 * -------------------------------------------------------------------------- */
__inline__ __device__ float block_reduce_max(float val, float* shared_max, int num_warps) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    val = warp_reduce_max_pa(val);

    if (lane_id == 0) {
        shared_max[warp_id] = val;
    }
    __syncthreads();

    val = (warp_id == 0 && lane_id < num_warps) ? shared_max[lane_id] : -1e9f;
    if (warp_id == 0) {
        val = warp_reduce_max_pa(val);
    }
    return val;
}

/* --------------------------------------------------------------------------
 * PagedAttention 核心 kernel
 *
 * 每个 thread block 处理一个 (batch, head) 对。
 *
 * 网格维度：
 *   gridDim.x = num_heads   （每个 head 独立的 thread block）
 *   gridDim.y = batch_size  （每个 batch 元素独立的 thread block）
 *
 * 共享内存布局（所有元素以 __half 为单位偏移）：
 *   [0,               HEAD_DIM)                      : Q_smem     (__half)
 *   [HEAD_DIM,        HEAD_DIM+BLOCK_SIZE*HEAD_DIM)  : K_smem     (__half)
 *   [K_END,           K_END+BLOCK_SIZE*HEAD_DIM)     : V_smem     (__half)
 *
 *   然后对齐后是 float 部分：
 *   S_vals   [BLOCK_SIZE]   float  (临时注意力分数)
 *   O_acc    [HEAD_DIM]     float  (输出累积器)
 *   shared_tmp [num_warps]  float  (reduction 临时空间)
 * -------------------------------------------------------------------------- */
template <int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_kernel(
    const __half* __restrict__ Q,              // [num_heads, head_dim]
    const __half* __restrict__ K_cache,        // [num_blocks, block_size, num_heads, head_dim]
    const __half* __restrict__ V_cache,        // [num_blocks, block_size, num_heads, head_dim]
    const int* __restrict__ block_tables,      // [batch_size, max_blocks_per_seq]
    const int* __restrict__ context_lens,      // [batch_size]
    __half* __restrict__ O,                    // [num_heads, head_dim]
    const float softmax_scale,
    const int num_heads,
    const int max_blocks_per_seq)
{
    /* ---- 计算本 block 对应的 batch 和 head 索引 ---- */
    const int head_idx  = blockIdx.x;   // 0 .. num_heads-1
    const int batch_idx = blockIdx.y;   // 0 .. batch_size-1

    const int tid = threadIdx.x;
    const int num_warps = NUM_THREADS / 32;

    /* ---- 获取该序列的上下文长度 ---- */
    const int context_len = context_lens[batch_idx];
    if (context_len <= 0) {
        // 没有 KV cache，直接写入零
        if (tid < HEAD_DIM) {
            O[head_idx * HEAD_DIM + tid] = __float2half(0.0f);
        }
        return;
    }

    /* ---- 计算需要遍历的逻辑块数量 ---- */
    const int num_logical_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* ---- 声明共享内存 ---- */
    extern __shared__ __half shared_raw[];

    // half 缓冲区偏移（以 __half 即 2 字节为单位）
    const int Q_OFFSET = 0;
    const int K_OFFSET = Q_OFFSET + HEAD_DIM;
    const int V_OFFSET = K_OFFSET + BLOCK_SIZE * HEAD_DIM;

    const int half_total  = V_OFFSET + BLOCK_SIZE * HEAD_DIM;
    const int half_padded = (half_total + 1) & ~1;  // 对齐到偶数

    // float 缓冲区紧随其后
    float* shared_float = reinterpret_cast<float*>(&shared_raw[half_padded]);

    float* S_vals     = shared_float;                        // [BLOCK_SIZE]
    float* O_acc_f    = shared_float + BLOCK_SIZE;           // [HEAD_DIM]
    float* shared_tmp = shared_float + BLOCK_SIZE + HEAD_DIM; // [num_warps]

    __half* Q_smem = &shared_raw[Q_OFFSET];
    __half* K_smem = &shared_raw[K_OFFSET];
    __half* V_smem = &shared_raw[V_OFFSET];

    /* ---- 第一步：将 Q[head] 加载到共享内存 ---- */
    for (int i = tid; i < HEAD_DIM; i += NUM_THREADS) {
        Q_smem[i] = Q[head_idx * HEAD_DIM + i];
    }

    /* ---- 第二步：初始化累积器 ---- */
    for (int i = tid; i < HEAD_DIM; i += NUM_THREADS) {
        O_acc_f[i] = 0.0f;
    }

    float m_global = -1e9f;  // 全局运行最大值
    float l_global = 0.0f;   // 全局运行和

    __syncthreads();

    /* ---- 第三步：遍历所有逻辑块 ---- */
    for (int logical_block = 0; logical_block < num_logical_blocks; ++logical_block) {
        /* 3a. 获取物理块索引 */
        const int physical_block = block_tables[batch_idx * max_blocks_per_seq + logical_block];

        /* 计算该块中实际有效的 token 数（最后一个块可能不完整） */
        const int block_start = logical_block * BLOCK_SIZE;
        const int valid_tokens = min(BLOCK_SIZE, context_len - block_start);

        /* 3b. 将 K 块和 V 块从全局内存加载到共享内存 ---- */
        // K_cache 和 V_cache 的布局：[num_blocks, block_size, num_heads, head_dim]
        // 每个元素在全局内存中的偏移计算：
        //   offset = block_idx * (block_size * num_heads * head_dim)
        //          + tok_idx  * (num_heads * head_dim)
        //          + head_idx * head_dim
        //          + d
        const int block_stride = BLOCK_SIZE * num_heads * HEAD_DIM;
        const int tok_stride   = num_heads * HEAD_DIM;
        const int head_base    = head_idx * HEAD_DIM;

        for (int i = tid; i < BLOCK_SIZE * HEAD_DIM; i += NUM_THREADS) {
            const int tok_in_block = i / HEAD_DIM;
            const int d            = i % HEAD_DIM;

            if (tok_in_block < valid_tokens) {
                const int global_offset = physical_block * block_stride
                                        + tok_in_block * tok_stride
                                        + head_base + d;
                K_smem[i] = __ldg(K_cache + global_offset);
                V_smem[i] = __ldg(V_cache + global_offset);
            } else {
                K_smem[i] = __float2half(0.0f);
                V_smem[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        /* 3c. 计算 S = Q @ K^T * scale ---- */
        // S[tok] = sum_d Q[d] * K[tok][d] * scale
        // 每个线程计算一部分 token 的分数
        for (int tok = tid; tok < valid_tokens; tok += NUM_THREADS) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += __half2float(Q_smem[d])
                     * __half2float(K_smem[tok * HEAD_DIM + d]);
            }
            S_vals[tok] = dot * softmax_scale;
        }
        // 无效 token 设置为 -inf
        for (int tok = tid + valid_tokens; tok < BLOCK_SIZE; tok += NUM_THREADS) {
            if (tok < valid_tokens) continue;  // 只处理无效的
            // 实际上上面的循环已经跳过了，这里只是为了安全
            S_vals[tok] = -1e9f;
        }
        // 显式设置超出 valid_tokens 的部分
        if (tid >= valid_tokens && tid < BLOCK_SIZE) {
            S_vals[tid] = -1e9f;
        }
        __syncthreads();

        /* 3e. 在线 softmax ---- */
        // 步骤 1：计算当前块内 S 的最大值（需要 block 级 reduction）
        float local_max = -1e9f;
        for (int tok = tid; tok < BLOCK_SIZE; tok += NUM_THREADS) {
            // 只考虑有效 token（无效的是 -1e9f，不影响 max）
            local_max = fmaxf(local_max, S_vals[tok]);
        }

        float block_max = block_reduce_max(local_max, shared_tmp, num_warps);
        __syncthreads();

        /* m_new = max(m_global, block_max) */
        // 只有 warp 0 lane 0 持有有效的 block_max
        // 我们用共享内存把它广播出去
        if (threadIdx.x == 0) {
            shared_tmp[0] = fmaxf(m_global, block_max);
        }
        __syncthreads();
        float m_new = shared_tmp[0];

        /* alpha = exp(m_global - m_new) */
        float alpha = expf(m_global - m_new);

        /* 步骤 2：计算 P = exp(S - m_new) 以及行和 */
        float local_sum = 0.0f;
        for (int tok = tid; tok < BLOCK_SIZE; tok += NUM_THREADS) {
            float p_val = expf(S_vals[tok] - m_new);
            S_vals[tok] = p_val;  // 原地替换，供后续 V 加权使用
            local_sum += p_val;
        }

        float block_sum = block_reduce_sum(local_sum, shared_tmp, num_warps);
        __syncthreads();

        /* l_new = alpha * l_global + block_sum */
        float l_new;
        if (threadIdx.x == 0) {
            l_new = alpha * l_global + block_sum;
            shared_tmp[0] = l_new;
        }
        __syncthreads();
        l_new = shared_tmp[0];

        /* 步骤 3：O_acc = alpha * O_acc + P @ V ---- */
        // 对每个 head_dim 维度 d，计算 O_acc[d] += sum_tok(P[tok] * V[tok][d])
        // 先将 O_acc 缩放 alpha
        for (int d = tid; d < HEAD_DIM; d += NUM_THREADS) {
            O_acc_f[d] *= alpha;
        }
        __syncthreads();

        // 然后累加 P @ V
        for (int d = tid; d < HEAD_DIM; d += NUM_THREADS) {
            float pv = 0.0f;
            for (int tok = 0; tok < BLOCK_SIZE; ++tok) {
                pv += S_vals[tok] * __half2float(V_smem[tok * HEAD_DIM + d]);
            }
            O_acc_f[d] += pv;
        }

        /* 更新全局状态 */
        m_global = m_new;
        l_global = l_new;

        __syncthreads();
    }

    /* ---- 第四步：最终归一化并写入全局内存 ---- */
    // O[head, d] = O_acc[d] / l_global
    for (int d = tid; d < HEAD_DIM; d += NUM_THREADS) {
        float val = O_acc_f[d];
        if (l_global > 0.0f) {
            val /= l_global;
        } else {
            val = 0.0f;
        }
        O[head_idx * HEAD_DIM + d] = __float2half(val);
    }
}

/* ==========================================================================
 * Host 端包装函数
 * ========================================================================== */

/*
 * 运行 PagedAttention 的入口函数。
 *
 * 根据 head_dim 自动分派到正确的模板实例化。
 *
 * 参数：
 *   Q              - [num_heads, head_dim] float16，单个 token 的 query
 *   K_cache        - [num_blocks, block_size, num_heads, head_dim] float16
 *   V_cache        - [num_blocks, block_size, num_heads, head_dim] float16
 *   block_tables   - [batch_size, max_blocks_per_seq] int32
 *   context_lens   - [batch_size] int32
 *   O              - [num_heads, head_dim] float16，输出
 *   softmax_scale  - 缩放因子
 */
void run_paged_attention(
    torch::Tensor Q,              // [num_heads, head_dim]
    torch::Tensor K_cache,        // [num_blocks, block_size, num_heads, head_dim]
    torch::Tensor V_cache,        // [num_blocks, block_size, num_heads, head_dim]
    torch::Tensor block_tables,   // [batch_size, max_blocks_per_seq]
    torch::Tensor context_lens,   // [batch_size]
    torch::Tensor O,              // [num_heads, head_dim]
    float softmax_scale)
{
    // 输入校验
    TORCH_CHECK(Q.device().is_cuda(),           "Q 必须在 CUDA 上");
    TORCH_CHECK(K_cache.device().is_cuda(),     "K_cache 必须在 CUDA 上");
    TORCH_CHECK(V_cache.device().is_cuda(),     "V_cache 必须在 CUDA 上");
    TORCH_CHECK(block_tables.device().is_cuda(),"block_tables 必须在 CUDA 上");
    TORCH_CHECK(context_lens.device().is_cuda(),"context_lens 必须在 CUDA 上");
    TORCH_CHECK(O.device().is_cuda(),           "O 必须在 CUDA 上");

    TORCH_CHECK(Q.dtype() == torch::kHalf,           "Q 必须是 float16");
    TORCH_CHECK(K_cache.dtype() == torch::kHalf,     "K_cache 必须是 float16");
    TORCH_CHECK(V_cache.dtype() == torch::kHalf,     "V_cache 必须是 float16");
    TORCH_CHECK(O.dtype() == torch::kHalf,           "O 必须是 float16");
    TORCH_CHECK(block_tables.dtype() == torch::kInt32,"block_tables 必须是 int32");
    TORCH_CHECK(context_lens.dtype() == torch::kInt32,"context_lens 必须是 int32");

    TORCH_CHECK(Q.dim() == 2,            "Q 必须是 2 维 [num_heads, head_dim]");
    TORCH_CHECK(K_cache.dim() == 4,      "K_cache 必须是 4 维 [num_blocks, block_size, num_heads, head_dim]");
    TORCH_CHECK(V_cache.dim() == 4,      "V_cache 必须是 4 维");
    TORCH_CHECK(block_tables.dim() == 2, "block_tables 必须是 2 维 [batch_size, max_blocks_per_seq]");
    TORCH_CHECK(context_lens.dim() == 1, "context_lens 必须是 1 维 [batch_size]");
    TORCH_CHECK(O.dim() == 2,            "O 必须是 2 维 [num_heads, head_dim]");

    TORCH_CHECK(Q.is_contiguous(),           "Q 必须是 contiguous");
    TORCH_CHECK(K_cache.is_contiguous(),     "K_cache 必须是 contiguous");
    TORCH_CHECK(V_cache.is_contiguous(),     "V_cache 必须是 contiguous");
    TORCH_CHECK(block_tables.is_contiguous(),"block_tables 必须是 contiguous");
    TORCH_CHECK(context_lens.is_contiguous(),"context_lens 必须是 contiguous");
    TORCH_CHECK(O.is_contiguous(),           "O 必须是 contiguous");

    const int num_heads   = Q.size(0);
    const int head_dim    = Q.size(1);
    const int batch_size  = block_tables.size(0);
    const int max_blocks  = block_tables.size(1);

    TORCH_CHECK(K_cache.size(2) == num_heads, "K_cache head 数量不匹配");
    TORCH_CHECK(K_cache.size(3) == head_dim,  "K_cache head_dim 不匹配");
    TORCH_CHECK(O.size(0)    == num_heads, "O head 数量不匹配");
    TORCH_CHECK(O.size(1)    == head_dim,  "O head_dim 不匹配");

    constexpr int BLOCK_SIZE = 16;   // 每个物理块 16 个 token
    constexpr int NUM_THREADS = 128;

    // 网格：每个 (batch, head) 一个 thread block
    dim3 grid(num_heads, batch_size);
    dim3 block(NUM_THREADS);

    // 计算共享内存大小
    const int half_elements = head_dim                     // Q_smem
                            + BLOCK_SIZE * head_dim        // K_smem
                            + BLOCK_SIZE * head_dim;       // V_smem
    const int half_padded = (half_elements + 1) & ~1;
    const int num_warps = NUM_THREADS / 32;
    const int float_elements = BLOCK_SIZE + head_dim + num_warps;
    const size_t smem_size = half_padded * sizeof(__half)
                           + float_elements * sizeof(float);

    // 根据 head_dim 分派模板实例化
    if (head_dim == 64) {
        paged_attention_kernel<64, BLOCK_SIZE, NUM_THREADS>
            <<<grid, block, smem_size>>>(
                reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(K_cache.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(V_cache.data_ptr<at::Half>()),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
                softmax_scale,
                num_heads, max_blocks
            );
    } else if (head_dim == 128) {
        paged_attention_kernel<128, BLOCK_SIZE, NUM_THREADS>
            <<<grid, block, smem_size>>>(
                reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(K_cache.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(V_cache.data_ptr<at::Half>()),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
                softmax_scale,
                num_heads, max_blocks
            );
    } else if (head_dim == 256) {
        paged_attention_kernel<256, BLOCK_SIZE, NUM_THREADS>
            <<<grid, block, smem_size>>>(
                reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(K_cache.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(V_cache.data_ptr<at::Half>()),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
                softmax_scale,
                num_heads, max_blocks
            );
    } else {
        TORCH_CHECK(false, "暂不支持的 head_dim: ", head_dim,
                    "（当前仅支持 64、128 和 256）");
    }

    CUDA_CHECK(cudaGetLastError());
}

/* ==========================================================================
 * KV Cache 内存分配辅助函数
 * ========================================================================== */

/*
 * 分配 PagedAttention 所需的 KV cache。
 *
 * 返回一个长度为 2 的 tuple：(K_cache, V_cache)
 * 两个 tensor 的形状均为 [num_blocks, block_size, num_heads, head_dim]。
 *
 * 内存为零初始化，dtype=float16，device=cuda。
 *
 * 参数：
 *   num_blocks - 物理块的总数
 *   block_size - 每个块的 token 数
 *   num_heads  - 注意力头数
 *   head_dim   - 每个头的维度
 */
std::vector<torch::Tensor> allocate_kv_cache(
    int num_blocks, int block_size, int num_heads, int head_dim)
{
    TORCH_CHECK(num_blocks > 0, "num_blocks 必须 > 0");
    TORCH_CHECK(block_size > 0, "block_size 必须 > 0");
    TORCH_CHECK(num_heads  > 0, "num_heads 必须 > 0");
    TORCH_CHECK(head_dim   > 0, "head_dim 必须 > 0");

    auto options = torch::TensorOptions()
        .dtype(torch::kHalf)
        .device(torch::kCUDA)
        .requires_grad(false);

    auto K_cache = torch::zeros({num_blocks, block_size, num_heads, head_dim}, options);
    auto V_cache = torch::zeros({num_blocks, block_size, num_heads, head_dim}, options);

    return {K_cache, V_cache};
}
