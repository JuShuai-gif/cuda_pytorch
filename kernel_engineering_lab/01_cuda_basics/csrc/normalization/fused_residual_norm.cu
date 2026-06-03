#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// ============================================================================
// 融合残差 + LayerNorm 的前向传播
//
// 工业背景：Transformer 中最常见的模式是：
//   x = x + f(x)       // 残差连接
//   x = LayerNorm(x)   // 层归一化
// 融合这两个操作可以：
//   1. 消除残差结果的中间 tensor（避免写入显存再读回）
//   2. 将两次全局内存读写合并为一次
//   3. 大幅减少显存带宽占用
//
// 算法：
//   1. y = x + residual                              （融合残差）
//   2. mean = sum(y) / hidden_dim                    （计算均值）
//   3. var = sum_sq(y) / hidden_dim - mean^2          （计算方差）
//   4. inv_std = rsqrt(var + eps)                    （标准差倒数）
//   5. centered = (y - mean) * inv_std               （中心化缩放）
//   6. out = centered * weight + bias                 （仿射变换）
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
// Kernel: 融合残差加法的 LayerNorm
// 一次 kernel launch 完成 y = x + residual 和 LayerNorm(y) 两个操作
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void fused_residual_layernorm_kernel(
    const __half* __restrict__ x,           // [rows, hidden_dim]
    const __half* __restrict__ residual,    // [rows, hidden_dim] skip connection 输入
    const __half* __restrict__ weight,      // [hidden_dim]
    const __half* __restrict__ bias,        // [hidden_dim]
    __half* __restrict__ out,               // [rows, hidden_dim] 归一化输出
    float eps,
    int rows,
    int hidden_dim)
{
    // 每个 block 处理一行
    const int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    // 当前行指针
    const __half* x_row = x + row_idx * hidden_dim;
    const __half* res_row = residual + row_idx * hidden_dim;
    __half* out_row = out + row_idx * hidden_dim;

    // -----------------------------------------------------------------------
    // Step 1: 融合残差加法 + 同时累加 sum 和 sum_sq
    //   在一个循环中完成：
    //     y = x[i] + residual[i]
    //     sum += y
    //     sum_sq += y^2
    //   这样做的意义：y 不需要写回全局内存，直接用于后续计算
    // -----------------------------------------------------------------------
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        // 融合残差：直接在寄存器中完成加法
        float x_val = __half2float(x_row[i]);
        float r_val = __half2float(res_row[i]);
        float y = x_val + r_val;

        // 同时累加统计量
        local_sum += y;
        local_sum_sq += y * y;
    }

    // -----------------------------------------------------------------------
    // Step 2: warp 内并行 reduction（对 sum 和 sum_sq 同时做）
    // -----------------------------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset);
    }

    // -----------------------------------------------------------------------
    // Step 3: 跨 warp reduction
    // -----------------------------------------------------------------------
    __shared__ float s_shared[64];  // [0..31] sum, [32..63] sum_sq
    if (lane_id == 0) {
        s_shared[warp_id] = local_sum;
        s_shared[warp_id + 32] = local_sum_sq;
    }
    __syncthreads();

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
    // Step 4: 计算统计量 (mean, inv_std) 并广播
    // -----------------------------------------------------------------------
    float mean = 0.0f;
    float inv_std = 0.0f;

    if (tid == 0) {
        float inv_n = 1.0f / (float)hidden_dim;
        mean = s_shared[0] * inv_n;

        // Var(X) = E[X^2] - E[X]^2
        float var = s_shared[1] * inv_n - mean * mean;

        // 防御：floating point 舍入可能产生极小负值
        if (var < 0.0f) {
            var = 0.0f;
        }
        inv_std = rsqrtf(var + eps);

        s_shared[0] = mean;
        s_shared[1] = inv_std;
    }
    __syncthreads();

    mean = s_shared[0];
    inv_std = s_shared[1];

    // -----------------------------------------------------------------------
    // Step 5: 归一化 + affine transform（重新计算 y = x + residual）
    //   注：这里没有把 y 缓存到 shared memory（hidden_dim 可能很大），
    //   而是重新从全局内存读取 x 和 residual，重新计算 y。
    //   这在 GPU 上是合理的，因为计算（ADD）的延迟远低于全局内存访问，
    //   而 shared memory 有限无法缓存整个 hidden_dim。
    //   不过 x 和 residual 通常在 L2 cache 中，因为刚在 Step 1 读取过。
    // -----------------------------------------------------------------------
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        // 重新读取并计算 y = x + residual
        float x_val = __half2float(x_row[i]);
        float r_val = __half2float(res_row[i]);
        float y = x_val + r_val;

        // 用预加载的 weight 和 bias 做 affine transform
        float w = __half2float(__ldg(weight + i));
        float b = __half2float(__ldg(bias + i));

        float centered = (y - mean) * inv_std;
        float normalized = centered * w + b;

        out_row[i] = __float2half_rn(normalized);
    }
}

// ============================================================================
// Wrapper 函数
// ============================================================================

void run_fused_residual_layernorm_fwd(
    torch::Tensor x,          // [rows, hidden_dim] fp16
    torch::Tensor residual,   // [rows, hidden_dim] fp16
    torch::Tensor weight,     // [hidden_dim] fp16
    torch::Tensor bias,       // [hidden_dim] fp16
    torch::Tensor out,        // [rows, hidden_dim] fp16 归一化输出
    float eps)
{
    // 参数校验
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(residual.scalar_type() == torch::kHalf, "residual must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kHalf, "weight must be fp16");
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "bias must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.sizes() == residual.sizes(),
        "x and residual must have same shape");
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

    fused_residual_layernorm_kernel<BLOCK_SIZE><<<grid, block>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        eps, rows, hidden_dim);

    CUDA_CHECK(cudaGetLastError());
}
