#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>

// ============================================================================
// Online Safe Softmax（在线安全 softmax）
//
// 工业背景：Online softmax 是 FlashAttention (Dao et al. 2022) 的核心组件，
// 也是对注意力分数做归一化的数值稳定标准实现。
//
// 相比传统三遍 softmax（找 max → exp → normalize），online softmax 仅需两遍：
//   第一遍：在线维护 running max m 和 running sum l
//   第二遍：计算 exp(x_i - m) / l
//
// 避免了：
//   - 显存中存储整行 exp(x - max) 的中间结果
//   - 多次遍历数据（传统方法需要 3 次遍历）
//
// Online 归约算法（单次遍历）：
//   初始化：m = -inf, l = 0
//   对每个新元素 x_new：
//     m_new = max(m, x_new)
//     l = exp(m - m_new) * l + exp(x_new - m_new)
//     m = m_new
//   最终：softmax(x_i) = exp(x_i - m) / l
//
// 每个 thread block 处理一行数据，使用 warp shuffle 实现高效的归约
// ============================================================================

#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t err_ = (err);                                            \
        if (err_ != cudaSuccess) {                                           \
            throw std::runtime_error(                                        \
                std::string("CUDA error at ") + __FILE__ + ":" +            \
                std::to_string(__LINE__) + " - " +                          \
                cudaGetErrorString(err_));                                   \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// 辅助结构体：保存在线 softmax 的部分归约状态 (running max, running sum)
// ---------------------------------------------------------------------------
struct SoftmaxState {
    float m; // 当前运行最大值
    float l; // 当前运行和 sum(exp(x_i - m))
};

// ---------------------------------------------------------------------------
// 合并两个部分归约状态
//   必须处理 m 为极度负值的情况：
//   当两个状态均未初始化（m = -1e10）时，m - m_new 可能产生 NaN。
//   使用足够小的有限值 -1e10 而非 -inf 来避免此问题。
__device__ __forceinline__ SoftmaxState combine_states(
    const SoftmaxState& a,
    const SoftmaxState& b)
{
    float m_new = fmaxf(a.m, b.m);
    // 如果 a 或 b 状态为哨兵值（m <= -1e9 且 l == 0），
    // 则对应的修正因子为 0（因为 exp(x - m) 中的所有 x 都很小）
    float corr_a = (a.m > -1.0e9f) ? expf(a.m - m_new) : 0.0f;
    float corr_b = (b.m > -1.0e9f) ? expf(b.m - m_new) : 0.0f;
    float l_new = corr_a * a.l + corr_b * b.l;
    return {m_new, l_new};
}

// ---------------------------------------------------------------------------
// 在线更新：将一个元素加入 SoftmaxState
//   这是 combine 的特化版本——将单元素状态 (x, 1.0) 与当前状态合并
//   因为单元素 x 的 softmax sum = exp(x - x) = 1
// ---------------------------------------------------------------------------
__device__ __forceinline__ void update_state(SoftmaxState& state, float val) {
    // 将 val 修正为有限值，避免后续指数运算中 -inf 相关的 NaN
    // -inf - (-inf) = NaN，需要特殊处理
    if (val < -1.0e10f) {
        val = -1.0e10f; // exp(-1e10) ≈ 0，不影响 softmax 结果
    }

    float m_new = fmaxf(state.m, val);
    state.l = expf(state.m - m_new) * state.l
            + expf(val - m_new);
    state.m = m_new;
}

// ============================================================================
// Kernel 1: Online Softmax 前向传播（无 mask）
//
//   网格配置：每行一个 thread block
//     gridDim.x = rows
//     blockDim.x = BLOCK_SIZE（256）
//
//   算法步骤：
//     Phase 1: 每个线程在线归约其负责的元素 → 局部 SoftmaxState
//     Phase 2: warp 内归约 → warp 级别 SoftmaxState
//     Phase 3: 跨 warp 归约（通过 shared memory）→ 全局 SoftmaxState
//     Phase 4: 广播全局 m、l，计算并写入每个元素的 softmax 值
// ============================================================================
template <int BLOCK_SIZE>
__global__ void online_softmax_kernel(
    const __half* __restrict__ x,   // [rows, cols]
    __half* __restrict__ out,       // [rows, cols]
    int rows,
    int cols)
{
    // 每个 block 处理一行数据
    const int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;   // 线程在 warp 内的索引
    const int warp_id = tid >> 5;   // warp 在 block 内的索引
    const int num_warps = blockDim.x >> 5;

    const __half* x_row = x + row_idx * cols;
    __half* out_row = out + row_idx * cols;

    // -----------------------------------------------------------------------
    // Phase 1: 线程级在线归约
    //   每个线程用 online 算法遍历自己负责的元素，维护局部 (m, l)
    //   使用 float 累加器避免 fp16 精度丢失
    // -----------------------------------------------------------------------
    SoftmaxState local;
    local.m = -1.0e10f;  // 使用足够小的有限值，避免 -inf 的 NaN 问题
    local.l = 0.0f;

    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = __half2float(x_row[i]);
        update_state(local, val);
    }

    // 如果本线程处理了至少一个有效元素，local.m 将大于 -1e10
    // 如果 initial m 保持不变（线程未处理任何元素），将其重置以确保归约正确
    if (local.m <= -1.0e9f && local.l == 0.0f) {
        // 该线程没有处理到元素（cols 小于线程数时的边界情况），
        // 使用哨兵状态，对其他线程的归约结果无影响
        local.m = -1.0e10f;
        local.l = 0.0f;
    }

    // -----------------------------------------------------------------------
    // Phase 2: warp 内归约
    //   使用 __shfl_xor_sync butterfly 模式将各线程的 (m, l) 归约到 warp 内统一值
    // -----------------------------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        SoftmaxState peer;
        peer.m = __shfl_xor_sync(0xffffffff, local.m, offset);
        peer.l = __shfl_xor_sync(0xffffffff, local.l, offset);
        local = combine_states(local, peer);
    }

    // -----------------------------------------------------------------------
    // Phase 3: 跨 warp 归约
    //   每个 warp 的 lane 0 将 warp 归约结果写入 shared memory
    //   然后第一个 warp 的线程将这些 warp 结果进一步归约到全局 (m, l)
    // -----------------------------------------------------------------------
    __shared__ float s_m[32]; // 最多支持 32 个 warp（BLOCK_SIZE=1024）
    __shared__ float s_l[32];

    if (lane_id == 0) {
        s_m[warp_id] = local.m;
        s_l[warp_id] = local.l;
    }
    __syncthreads();

    // 只有 warp 0 的所有 32 个线程参与最终 reduction
    SoftmaxState warp_state;
    warp_state.m = -1.0e10f;  // 使用有限哨兵值，避免 combine_states 中产生 NaN
    warp_state.l = 0.0f;
    if (warp_id == 0) {
        warp_state.m = (lane_id < num_warps) ? s_m[lane_id] : -1.0e10f;
        warp_state.l = (lane_id < num_warps) ? s_l[lane_id] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            SoftmaxState peer;
            peer.m = __shfl_xor_sync(0xffffffff, warp_state.m, offset);
            peer.l = __shfl_xor_sync(0xffffffff, warp_state.l, offset);
            warp_state = combine_states(warp_state, peer);
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        s_m[0] = warp_state.m;
        s_l[0] = warp_state.l;
    }
    __syncthreads();

    // 广播全局 (m, l) 到 block 内所有线程
    float global_m = s_m[0];
    float global_l = s_l[0];

    // 处理边界情况：归一化因子为 0（整行均为极小的无效值）
    // 此时所有输出置为 0，避免除零错误
    if (global_l < 1.0e-30f) {
        global_l = 1.0f; // 最终结果 = exp(x - m) / 1，对于极小 x 接近 0
    }

    // -----------------------------------------------------------------------
    // Phase 4: 计算输出
    //   每个线程对自己负责的元素计算：
    //     out[i] = exp(x[i] - global_m) / global_l
    // -----------------------------------------------------------------------
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = __half2float(x_row[i]);
        float prob = expf(val - global_m) / global_l;
        out_row[i] = __float2half_rn(prob);
    }
}

// ============================================================================
// Kernel 2: 带 mask 的 Online Softmax
//
//   支持 attention mask（causal mask、padding mask 等）
//   mask 为非空指针时，mask 中的值会加到输入上：
//     val = x[i] + mask[i]
//   典型 mask 约定：
//     - 有效位置：mask[i] = 0
//     - 无效位置：mask[i] = -inf（或一个非常大的负数）
//   mask 位置参与 max 计算，但因为 exp(mask_value) → 0，不参与 sum
//
//   对于 mask = nullptr 的情况，回退到普通的 online softmax
// ============================================================================
template <int BLOCK_SIZE>
__global__ void masked_online_softmax_kernel(
    const __half* __restrict__ x,       // [rows, cols]
    const __half* __restrict__ mask,    // [rows, cols] 或 nullptr
    __half* __restrict__ out,           // [rows, cols]
    int rows,
    int cols,
    float mask_value)                   // mask 中无效位置的值（通常 -inf）
{
    const int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    const __half* x_row = x + row_idx * cols;
    __half* out_row = out + row_idx * cols;

    // -----------------------------------------------------------------------
    // Phase 1: 线程级在线归约（带 mask）
    //   如果 mask 为空，回退到无 mask 模式
    // -----------------------------------------------------------------------
    SoftmaxState local;
    local.m = -1.0e10f;
    local.l = 0.0f;

    if (mask != nullptr) {
        const __half* mask_row = mask + row_idx * cols;
        for (int i = tid; i < cols; i += BLOCK_SIZE) {
            float val = __half2float(x_row[i]);
            float m = __half2float(mask_row[i]);

            // additive mask：mask[i] = 0（有效）或 -inf（无效）
            // mask_value 用于文档化 mask 无效时的值
            val += m;

            update_state(local, val);
        }
    } else {
        // 无 mask 模式，与 online_softmax_kernel 相同
        for (int i = tid; i < cols; i += BLOCK_SIZE) {
            float val = __half2float(x_row[i]);
            update_state(local, val);
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: warp 内归约
    // -----------------------------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        SoftmaxState peer;
        peer.m = __shfl_xor_sync(0xffffffff, local.m, offset);
        peer.l = __shfl_xor_sync(0xffffffff, local.l, offset);
        local = combine_states(local, peer);
    }

    // -----------------------------------------------------------------------
    // Phase 3: 跨 warp 归约
    // -----------------------------------------------------------------------
    __shared__ float s_m[32];
    __shared__ float s_l[32];

    if (lane_id == 0) {
        s_m[warp_id] = local.m;
        s_l[warp_id] = local.l;
    }
    __syncthreads();

    // 只有 warp 0 的所有 32 个线程参与最终 reduction
    SoftmaxState warp_state;
    warp_state.m = -1.0e10f;  // 使用有限哨兵值，避免 combine_states 中产生 NaN
    warp_state.l = 0.0f;
    if (warp_id == 0) {
        warp_state.m = (lane_id < num_warps) ? s_m[lane_id] : -1.0e10f;
        warp_state.l = (lane_id < num_warps) ? s_l[lane_id] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            SoftmaxState peer;
            peer.m = __shfl_xor_sync(0xffffffff, warp_state.m, offset);
            peer.l = __shfl_xor_sync(0xffffffff, warp_state.l, offset);
            warp_state = combine_states(warp_state, peer);
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        s_m[0] = warp_state.m;
        s_l[0] = warp_state.l;
    }
    __syncthreads();

    float global_m = s_m[0];
    float global_l = s_l[0];

    if (global_l < 1.0e-30f) {
        global_l = 1.0f;
    }

    // -----------------------------------------------------------------------
    // Phase 4: 计算输出（同样带 mask 检查）
    //   被 mask 的位置由于 val 为极大负值，
    //   exp(val - m) = exp(-inf) → 0，输出自然为 0
    // -----------------------------------------------------------------------
    if (mask != nullptr) {
        const __half* mask_row = mask + row_idx * cols;
        for (int i = tid; i < cols; i += BLOCK_SIZE) {
            float val = __half2float(x_row[i]);
            val += __half2float(mask_row[i]);
            float prob = expf(val - global_m) / global_l;
            out_row[i] = __float2half_rn(prob);
        }
    } else {
        for (int i = tid; i < cols; i += BLOCK_SIZE) {
            float val = __half2float(x_row[i]);
            float prob = expf(val - global_m) / global_l;
            out_row[i] = __float2half_rn(prob);
        }
    }
}

// ============================================================================
// Wrapper 函数：从 PyTorch 调用，负责指针提取、kernel launch 和错误检查
// ============================================================================

void run_online_softmax(
    torch::Tensor x,    // [rows, cols] fp16
    torch::Tensor out)  // [rows, cols] fp16
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: [rows, cols]");
    TORCH_CHECK(out.sizes() == x.sizes(),
                "out must have the same shape as x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));

    constexpr int BLOCK_SIZE = 256;
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    online_softmax_kernel<BLOCK_SIZE><<<grid, block>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        rows, cols);

    CUDA_CHECK(cudaGetLastError());
}

void run_masked_online_softmax(
    torch::Tensor x,         // [rows, cols] fp16
    torch::Tensor mask,      // [rows, cols] fp16（同形状 additive mask）
    torch::Tensor out,       // [rows, cols] fp16
    float mask_value)        // mask 中无效位置的值（用于参数说明）
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(mask.scalar_type() == torch::kHalf, "mask must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: [rows, cols]");
    TORCH_CHECK(mask.sizes() == x.sizes(),
                "mask must have the same shape as x");
    TORCH_CHECK(out.sizes() == x.sizes(),
                "out must have the same shape as x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));

    constexpr int BLOCK_SIZE = 256;
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    masked_online_softmax_kernel<BLOCK_SIZE><<<grid, block>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(mask.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        rows, cols, mask_value);

    CUDA_CHECK(cudaGetLastError());
}
