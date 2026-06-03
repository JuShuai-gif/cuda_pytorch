#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>

// ============================================================================
// 融合 bias + activation 的 CUDA kernel
//
// 工业背景：matmul 后接 bias add 再接 activation 是 transformer 中最常见的模式：
//   h = W @ x + bias      ← 线性层
//   h = activation(h)     ← 激活函数
// 融合这两个操作可以消除中间 tensor h 的显存读写，减少 I/O 开销。
//
// 支持三种激活函数的融合：ReLU、GELU、SiLU
// 所有中间计算使用 float，只在写入时转换为 fp16
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

// ============================================================================
// Kernel 1: 融合 bias + ReLU
//   对每个元素：out[row, col] = ReLU(x[row, col] + bias[col])
//   其中 ReLU(z) = max(0, z)
// ============================================================================
__global__ void fused_bias_relu_kernel(
    const __half* __restrict__ x,     // [rows, hidden_dim]
    const __half* __restrict__ bias,  // [hidden_dim]
    __half* __restrict__ out,         // [rows, hidden_dim]
    int rows,
    int hidden_dim)
{
    const int n_elements = rows * hidden_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int col = idx % hidden_dim;
    float val = __half2float(x[idx]) + __half2float(__ldg(bias + col));
    float result = fmaxf(0.0f, val);

    out[idx] = __float2half_rn(result);
}

// ============================================================================
// Kernel 2: 融合 bias + GELU（tanh 近似）
//   对每个元素：out[row, col] = GELU(x[row, col] + bias[col])
//   GELU(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
// ============================================================================
__global__ void fused_bias_gelu_kernel(
    const __half* __restrict__ x,     // [rows, hidden_dim]
    const __half* __restrict__ bias,  // [hidden_dim]
    __half* __restrict__ out,         // [rows, hidden_dim]
    int rows,
    int hidden_dim)
{
    const int n_elements = rows * hidden_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int col = idx % hidden_dim;
    float val = __half2float(x[idx]) + __half2float(__ldg(bias + col));

    // GELU tanh 近似：0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    constexpr float sqrt_2_over_pi = 0.79788456f;
    constexpr float coeff = 0.044715f;

    float val_cu = val * val * val;
    float inner = sqrt_2_over_pi * (val + coeff * val_cu);
    float tanh_inner = tanhf(inner);
    float result = 0.5f * val * (1.0f + tanh_inner);

    out[idx] = __float2half_rn(result);
}

// ============================================================================
// Kernel 3: 融合 bias + SiLU
//   对每个元素：out[row, col] = SiLU(x[row, col] + bias[col])
//   SiLU(z) = z * sigmoid(z)，其中 sigmoid(z) = 1 / (1 + exp(-z))
// ============================================================================
__global__ void fused_bias_silu_kernel(
    const __half* __restrict__ x,     // [rows, hidden_dim]
    const __half* __restrict__ bias,  // [hidden_dim]
    __half* __restrict__ out,         // [rows, hidden_dim]
    int rows,
    int hidden_dim)
{
    const int n_elements = rows * hidden_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int col = idx % hidden_dim;
    float val = __half2float(x[idx]) + __half2float(__ldg(bias + col));

    // SiLU(z) = z * sigmoid(z)
    // 数值稳定的 sigmoid 实现：z>=0 时用 1/(1+exp(-z))，z<0 时用 exp(z)/(1+exp(z))
    float sig;
    if (val >= 0.0f) {
        sig = 1.0f / (1.0f + expf(-val));
    } else {
        float exp_val = expf(val);
        sig = exp_val / (1.0f + exp_val);
    }
    float result = val * sig;

    out[idx] = __float2half_rn(result);
}

// ============================================================================
// Wrapper 函数：处理 PyTorch tensor 到 CUDA 指针的转换、kernel launch 及错误检查
//
// 每个函数分享相同的参数布局：
//   x         - [rows, hidden_dim] fp16，matmul 的输出
//   bias      - [hidden_dim] fp16，逐列的偏置值
//   out       - [rows, hidden_dim] fp16，融合后的输出（原地或单独显存）
// ============================================================================

void run_fused_bias_relu(
    torch::Tensor x,         // [rows, hidden_dim] fp16
    torch::Tensor bias,      // [hidden_dim] fp16
    torch::Tensor out)       // [rows, hidden_dim] fp16
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "bias must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: [rows, hidden_dim]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D: [hidden_dim]");
    TORCH_CHECK(bias.size(0) == x.size(1),
                "bias size must equal hidden_dim");
    TORCH_CHECK(out.sizes() == x.sizes(),
                "out must have the same shape as x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int rows = static_cast<int>(x.size(0));
    const int hidden_dim = static_cast<int>(x.size(1));
    const int n_elements = rows * hidden_dim;

    constexpr int BLOCK_SIZE = 256;
    const int grid = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fused_bias_relu_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        rows, hidden_dim);

    CUDA_CHECK(cudaGetLastError());
}

void run_fused_bias_gelu(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor out)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "bias must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: [rows, hidden_dim]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D: [hidden_dim]");
    TORCH_CHECK(bias.size(0) == x.size(1),
                "bias size must equal hidden_dim");
    TORCH_CHECK(out.sizes() == x.sizes(),
                "out must have the same shape as x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int rows = static_cast<int>(x.size(0));
    const int hidden_dim = static_cast<int>(x.size(1));
    const int n_elements = rows * hidden_dim;

    constexpr int BLOCK_SIZE = 256;
    const int grid = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fused_bias_gelu_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        rows, hidden_dim);

    CUDA_CHECK(cudaGetLastError());
}

void run_fused_bias_silu(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor out)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "bias must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: [rows, hidden_dim]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D: [hidden_dim]");
    TORCH_CHECK(bias.size(0) == x.size(1),
                "bias size must equal hidden_dim");
    TORCH_CHECK(out.sizes() == x.sizes(),
                "out must have the same shape as x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int rows = static_cast<int>(x.size(0));
    const int hidden_dim = static_cast<int>(x.size(1));
    const int n_elements = rows * hidden_dim;

    constexpr int BLOCK_SIZE = 256;
    const int grid = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fused_bias_silu_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        rows, hidden_dim);

    CUDA_CHECK(cudaGetLastError());
}
