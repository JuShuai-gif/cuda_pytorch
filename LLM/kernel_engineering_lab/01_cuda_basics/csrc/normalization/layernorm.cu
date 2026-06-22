#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// ============================================================================
// LayerNorm（层归一化）
// 用于 BERT / GPT-2 / T5 等模型
// 算法：
//   mean = sum(x) / hidden_dim
//   var  = sum_sq / hidden_dim - mean^2
//   inv_std = rsqrt(var + eps)
//   centered = (x - mean) * inv_std
//   out = centered * weight + bias
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
// Kernel: LayerNorm 前向传播（一次遍历同时计算 sum 和 sum_sq）
// 使用 Var = E[X^2] - E[X]^2 公式，避免两遍遍历
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void layernorm_fwd_kernel(
    const __half* __restrict__ x,         // [rows, hidden_dim]
    const __half* __restrict__ weight,    // [hidden_dim]
    const __half* __restrict__ bias,      // [hidden_dim]
    __half* __restrict__ out,             // [rows, hidden_dim]
    float eps,
    int rows,
    int hidden_dim)
{
    // 每个 block 处理一行数据
    const int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;       // warp 内索引
    const int warp_id = tid >> 5;       // block 内 warp 索引
    const int num_warps = blockDim.x >> 5;

    // 当前行在全局内存中的指针
    const __half* x_row = x + row_idx * hidden_dim;
    __half* out_row = out + row_idx * hidden_dim;

    // -----------------------------------------------------------------------
    // Step 1: 一次遍历同时计算局部 sum 和 sum_sq
    //   使用 float 累加器避免 fp16 精度丢失
    // -----------------------------------------------------------------------
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    // -----------------------------------------------------------------------
    // Step 2: warp 内 reduction——并行归约 sum 和 sum_sq
    // -----------------------------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset);
    }

    // -----------------------------------------------------------------------
    // Step 3: 跨 warp reduction——warp leader 写入 shared memory
    // -----------------------------------------------------------------------
    __shared__ float s_shared[64];  // [0..31] 存 sum, [32..63] 存 sum_sq
    if (lane_id == 0) {
        s_shared[warp_id] = local_sum;
        s_shared[warp_id + 32] = local_sum_sq;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 4: 在第一个 warp 内对 shared memory 中的 warp 结果做最终 reduction
    // -----------------------------------------------------------------------
    // 只有 warp 0 的所有 32 个线程参与最终 reduction
    float sum = 0.0f;
    float sum_sq = 0.0f;
    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? s_shared[lane_id] : 0.0f;
        sum_sq = (lane_id < num_warps) ? s_shared[lane_id + 32] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        s_shared[0] = sum;
        s_shared[1] = sum_sq;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 5: 计算统计量
    //   mean = sum / N
    //   var  = sum_sq / N - mean^2（数值更稳定的方差公式）
    //   inv_std = rsqrt(var + eps)
    // -----------------------------------------------------------------------
    float mean = 0.0f;
    float inv_std = 0.0f;

    if (tid == 0) {
        float inv_n = 1.0f / (float)hidden_dim;
        mean = s_shared[0] * inv_n;

        // Var(X) = E[X^2] - E[X]^2
        float var = s_shared[1] * inv_n - mean * mean;

        // 防止方差为负（floating point 舍入误差可能导致极小负数）
        if (var < 0.0f) {
            var = 0.0f;
        }
        inv_std = rsqrtf(var + eps);

        // 存入 shared memory 供所有线程使用
        s_shared[0] = mean;
        s_shared[1] = inv_std;
    }
    __syncthreads();

    // 广播统计量到所有线程
    mean = s_shared[0];
    inv_std = s_shared[1];

    // -----------------------------------------------------------------------
    // Step 6: 归一化并应用 affine transform
    //   out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i]
    // -----------------------------------------------------------------------
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        float w = __half2float(__ldg(weight + i));
        float b = __half2float(__ldg(bias + i));

        float centered = (val - mean) * inv_std;
        float normalized = centered * w + b;

        out_row[i] = __float2half_rn(normalized);
    }
}

// ============================================================================
// Wrapper 函数
// ============================================================================

void run_layernorm_fwd(
    torch::Tensor x,         // [rows, hidden_dim] fp16
    torch::Tensor weight,    // [hidden_dim] fp16
    torch::Tensor bias,      // [hidden_dim] fp16
    torch::Tensor out,       // [rows, hidden_dim] fp16
    float eps)
{
    // 参数校验
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kHalf, "weight must be fp16");
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "bias must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: [rows, hidden_dim]");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
        "weight must be 1D: [hidden_dim]");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == x.size(1),
        "bias must be 1D: [hidden_dim]");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have same shape as x");

    const int rows = x.size(0);
    const int hidden_dim = x.size(1);

    constexpr int BLOCK_SIZE = 256;
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    layernorm_fwd_kernel<BLOCK_SIZE><<<grid, block>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        eps, rows, hidden_dim);

    CUDA_CHECK(cudaGetLastError());
}
