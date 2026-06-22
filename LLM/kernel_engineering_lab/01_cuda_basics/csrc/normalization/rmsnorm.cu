#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// ============================================================================
// RMSNorm (Root Mean Square Normalization)
// 用于 LLaMA / Mistral / Gemma 等现代 LLM
// 算法：rms = sqrt(mean(x^2) + eps)，out = x / rms * weight
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
// Kernel 1: 基础 RMSNorm 前向传播
// 每个 thread block 处理一行，使用 warp shuffle 进行高效的 reduction
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void rmsnorm_fwd_kernel(
    const __half* __restrict__ x,         // [rows, hidden_dim]
    const __half* __restrict__ weight,    // [hidden_dim]
    __half* __restrict__ out,             // [rows, hidden_dim]
    float eps,
    int rows,
    int hidden_dim)
{
    // 每个 block 处理一行数据
    const int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;  // 线程在 warp 内的索引
    const int warp_id = tid >> 5;  // warp 在 block 内的索引
    const int num_warps = blockDim.x >> 5;  // block 内的 warp 总数

    // 定位当前行在全局内存中的起始位置
    const __half* x_row = x + row_idx * hidden_dim;
    __half* out_row = out + row_idx * hidden_dim;

    // -----------------------------------------------------------------------
    // Step 1: 每个线程累加自己负责的元素的 x^2
    // -----------------------------------------------------------------------
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        local_sum_sq += val * val;
    }

    // -----------------------------------------------------------------------
    // Step 2: warp 内 reduction（使用 __shfl_xor_sync butterfly 模式）
    // 结束后 warp 内所有线程持有相同的 warp 级 sum
    // -----------------------------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset);
    }

    // -----------------------------------------------------------------------
    // Step 3: 跨 warp reduction——每个 warp 的 thread 0 写入 shared memory
    // -----------------------------------------------------------------------
    __shared__ float s_warp_sum[32];  // 最多 32 个 warp（BLOCK_SIZE=1024）
    if (lane_id == 0) {
        s_warp_sum[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 4: 第一个 warp 的线程将各 warp 结果累加得到全局 sum_sq
    // -----------------------------------------------------------------------
    // 只有 warp 0 的线程参与最终 reduction（所有 32 线程都必须执行 shuffle）
    float sum_sq = 0.0f;
    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        s_warp_sum[0] = sum_sq;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 5: 计算 inv_rms = rsqrt(sum_sq / hidden_dim + eps)
    // -----------------------------------------------------------------------
    float inv_rms = 0.0f;
    if (tid == 0) {
        // 使用 float 精度计算以保证数值稳定性
        //   rms = sqrt(sum_sq / hidden_dim + eps)
        //   inv_rms = rsqrt(sum_sq / hidden_dim + eps)
        // 直接使用 rsqrt 避免先算 sqrt 再取倒数
        inv_rms = rsqrtf(s_warp_sum[0] / (float)hidden_dim + eps);
    }

    // 广播 inv_rms 到 block 内所有线程（通过 shared memory）
    if (tid == 0) {
        s_warp_sum[0] = inv_rms;
    }
    __syncthreads();
    inv_rms = s_warp_sum[0];

    // -----------------------------------------------------------------------
    // Step 6: 每个线程计算 out[i] = x[i] * inv_rms * weight[i]
    // -----------------------------------------------------------------------
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        float w = __half2float(__ldg(weight + i));
        float normalized = val * inv_rms * w;
        out_row[i] = __float2half_rn(normalized);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: 融合残差加法的 RMSNorm
// 先计算 residual + x，再对结果做 RMSNorm，同时输出残差结果和归一化结果
// 用于 transformer block 中的 skip connection + norm，避免中间 tensor 写回显存
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void rmsnorm_residual_fwd_kernel(
    const __half* __restrict__ x,           // [rows, hidden_dim]
    const __half* __restrict__ residual,    // [rows, hidden_dim]
    const __half* __restrict__ weight,      // [hidden_dim]
    __half* __restrict__ out,               // [rows, hidden_dim] 归一化输出
    __half* __restrict__ residual_out,      // [rows, hidden_dim] x + residual 用于后续 skip connection
    float eps,
    int rows,
    int hidden_dim)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    const __half* x_row = x + row_idx * hidden_dim;
    const __half* res_row = residual + row_idx * hidden_dim;
    __half* out_row = out + row_idx * hidden_dim;
    __half* res_out_row = residual_out + row_idx * hidden_dim;

    // -----------------------------------------------------------------------
    // Step 1: 融合残差加法——计算 y = x + residual，同时累加 sum_sq(y)
    // -----------------------------------------------------------------------
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x_val = __half2float(x_row[i]);
        float r_val = __half2float(res_row[i]);
        float y = x_val + r_val;

        // 提前写入残差结果，节省一次遍历
        res_out_row[i] = __float2half_rn(y);

        local_sum_sq += y * y;
    }

    // -----------------------------------------------------------------------
    // Step 2: warp 内 reduction
    // -----------------------------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset);
    }

    // -----------------------------------------------------------------------
    // Step 3: 跨 warp reduction
    // -----------------------------------------------------------------------
    __shared__ float s_warp_sum[32];
    if (lane_id == 0) {
        s_warp_sum[warp_id] = local_sum_sq;
    }
    __syncthreads();

    float sum_sq = 0.0f;
    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        s_warp_sum[0] = sum_sq;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 4: 计算 inv_rms
    // -----------------------------------------------------------------------
    float inv_rms = 0.0f;
    if (tid == 0) {
        inv_rms = rsqrtf(s_warp_sum[0] / (float)hidden_dim + eps);
    }
    if (tid == 0) {
        s_warp_sum[0] = inv_rms;
    }
    __syncthreads();
    inv_rms = s_warp_sum[0];

    // -----------------------------------------------------------------------
    // Step 5: 对 y 做归一化——重新读取残差结果并乘 weight
    //      注：此处重新读取 res_out_row 而非重新计算 y，
    //      因为 res_out_row 刚刚写入，应该仍在 L1/L2 cache 中
    // -----------------------------------------------------------------------
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float y = __half2float(res_out_row[i]);
        float w = __half2float(__ldg(weight + i));
        float normalized = y * inv_rms * w;
        out_row[i] = __float2half_rn(normalized);
    }
}

// ============================================================================
// Wrapper 函数——从 PyTorch 调用，负责指针提取、kernel launch 和错误检查
// ============================================================================

void run_rmsnorm_fwd(
    torch::Tensor x,         // [rows, hidden_dim] fp16
    torch::Tensor weight,    // [hidden_dim] fp16
    torch::Tensor out,       // [rows, hidden_dim] fp16
    float eps)
{
    // 参数校验
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kHalf, "weight must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: [rows, hidden_dim]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D: [hidden_dim]");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have same shape as x");

    const int rows = x.size(0);
    const int hidden_dim = x.size(1);

    constexpr int BLOCK_SIZE = 256;
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    rmsnorm_fwd_kernel<BLOCK_SIZE><<<grid, block>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        eps, rows, hidden_dim);

    CUDA_CHECK(cudaGetLastError());
}

void run_rmsnorm_residual_fwd(
    torch::Tensor x,              // [rows, hidden_dim] fp16
    torch::Tensor residual,       // [rows, hidden_dim] fp16
    torch::Tensor weight,         // [hidden_dim] fp16
    torch::Tensor out,            // [rows, hidden_dim] fp16 归一化输出
    torch::Tensor residual_out,   // [rows, hidden_dim] fp16 x + residual
    float eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(residual_out.is_cuda(), "residual_out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(residual.scalar_type() == torch::kHalf, "residual must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kHalf, "weight must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(residual_out.scalar_type() == torch::kHalf, "residual_out must be fp16");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have same shape");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have same shape as x");
    TORCH_CHECK(residual_out.sizes() == x.sizes(), "residual_out must have same shape as x");

    const int rows = x.size(0);
    const int hidden_dim = x.size(1);

    constexpr int BLOCK_SIZE = 256;
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    rmsnorm_residual_fwd_kernel<BLOCK_SIZE><<<grid, block>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(residual_out.data_ptr<torch::Half>()),
        eps, rows, hidden_dim);

    CUDA_CHECK(cudaGetLastError());
}
