#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>

// ============================================================================
// GELU (Gaussian Error Linear Unit) 激活函数
// 用于 BERT / GPT-2 / ViT / OpenAI GPT 系列
// 使用 tanh 近似形式（工业生产标准，相比精确 erf 实现速度更快、精度损失极小）：
//   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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

// tanh 近似 GELU 公式中的常量
// 使用 constexpr 由编译器内联优化，相比 __constant__ memory 延迟更低
constexpr float kGeluSqrt2OverPi = 0.7978845608f; // sqrt(2/pi)
constexpr float kGeluCoeff      = 0.044715f;       // 三阶项系数

// ============================================================================
// Kernel 1: GELU 前向传播（tanh 近似）
//   公式：GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   每个线程处理一个元素，所有中间计算使用 float 精度
// ============================================================================
__global__ void gelu_fwd_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float val = __half2float(x[idx]);

    // 计算 inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    float val_sq = val * val;
    float val_cu = val_sq * val;
    float inner = kGeluSqrt2OverPi * (val + kGeluCoeff * val_cu);

    float tanh_inner = tanhf(inner);

    // GELU(x) = 0.5 * x * (1 + tanh(inner))
    float result = 0.5f * val * (1.0f + tanh_inner);

    out[idx] = __float2half_rn(result);
}

// ============================================================================
// Kernel 2: GELU 反向传播（tanh 近似）
//   推导（链式法则）：
//     令 inner = a * (x + b * x^3)，其中 a = sqrt(2/pi), b = 0.044715
//     令 t = tanh(inner)
//     GELU(x) = 0.5 * x * (1 + t)
//
//     d/dx GELU(x) = 0.5 * (1 + t) + 0.5 * x * d(tanh(inner))/dx
//     d(tanh(inner))/dx = sech^2(inner) * d(inner)/dx
//                       = (1 - tanh^2(inner)) * a * (1 + 3*b*x^2)
//
//     所以：
//     d_GELU/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * a * (1 + 3*b*x^2)
//
//   梯度反向传播：
//     grad_in[i] = grad_out[i] * d_GELU(x[i])/dx
// ============================================================================
__global__ void gelu_bwd_kernel(
    const __half* __restrict__ grad_out,
    const __half* __restrict__ x,
    __half* __restrict__ grad_in,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float val = __half2float(x[idx]);
    float go = __half2float(grad_out[idx]);

    // 计算 inner 和 tanh
    float val_sq = val * val;
    float val_cu = val_sq * val;
    float inner = kGeluSqrt2OverPi * (val + kGeluCoeff * val_cu);
    float tanh_inner = tanhf(inner);

    // sech^2(inner) = 1 - tanh^2(inner)
    float sech2 = 1.0f - tanh_inner * tanh_inner;

    // d_inner/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    float d_inner = kGeluSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * val_sq);

    // d_GELU/dx = 0.5*(1+tanh) + 0.5*x*sech^2*d_inner
    float d_gelu = 0.5f * (1.0f + tanh_inner) + 0.5f * val * sech2 * d_inner;

    float result = go * d_gelu;

    grad_in[idx] = __float2half_rn(result);
}

// ============================================================================
// Wrapper 函数：处理 PyTorch tensor 到 CUDA 指针的转换、kernel launch 及错误检查
// ============================================================================

void run_gelu_fwd(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(x.sizes() == out.sizes(),
                "x and out must have the same shape");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int n_elements = static_cast<int>(x.numel());
    constexpr int BLOCK_SIZE = 256;

    gelu_fwd_kernel<<<(n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        n_elements);

    CUDA_CHECK(cudaGetLastError());
}

void run_gelu_bwd(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor grad_in)
{
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(grad_in.is_cuda(), "grad_in must be a CUDA tensor");
    TORCH_CHECK(grad_out.scalar_type() == torch::kHalf, "grad_out must be fp16");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(grad_in.scalar_type() == torch::kHalf, "grad_in must be fp16");
    TORCH_CHECK(grad_out.sizes() == x.sizes(),
                "grad_out and x must have the same shape");
    TORCH_CHECK(grad_in.sizes() == x.sizes(),
                "grad_in and x must have the same shape");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(grad_in.is_contiguous(), "grad_in must be contiguous");

    const int n_elements = static_cast<int>(x.numel());
    constexpr int BLOCK_SIZE = 256;

    gelu_bwd_kernel<<<(n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(grad_out.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(grad_in.data_ptr<torch::Half>()),
        n_elements);

    CUDA_CHECK(cudaGetLastError());
}
