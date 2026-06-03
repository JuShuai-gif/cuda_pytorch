#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>

// ============================================================================
// SiLU / SwiGLU 激活函数
// SiLU (Sigmoid Linear Unit) = x * sigmoid(x)，其中 sigmoid(x) = 1/(1+exp(-x))
// SwiGLU = gate * SiLU(up)，用于 LLaMA / Mistral / Gemma 的 FFN 层
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

// ---------------------------------------------------------------------------
// 辅助函数：数值稳定的 sigmoid 实现
// 对于正数 x：sigmoid(x) = 1 / (1 + exp(-x))
// 对于负数 x：sigmoid(x) = exp(x) / (1 + exp(x)) 避免 exp(-x) overflow
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

// ============================================================================
// Kernel 1: SiLU 前向传播
//   out[i] = x[i] * sigmoid(x[i])
//   每个线程处理一个元素，所有中间计算使用 float 精度
// ============================================================================
__global__ void silu_fwd_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float val = __half2float(x[idx]);
    float sig = sigmoid_stable(val);
    float result = val * sig;

    out[idx] = __float2half_rn(result);
}

// ============================================================================
// Kernel 2: SiLU 反向传播
//   d_SiLU(x)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//   推导：
//     d/dx (x * sigma(x)) = sigma(x) + x * sigma'(x)
//     sigma'(x) = sigma(x) * (1 - sigma(x))
//     合并：d_SiLU(x)/dx = sigma(x) * (1 + x * (1 - sigma(x)))
//   梯度反向传播到输入：
//     grad_in[i] = grad_out[i] * d_SiLU(x[i])/dx
// ============================================================================
__global__ void silu_bwd_kernel(
    const __half* __restrict__ grad_out,
    const __half* __restrict__ x,
    __half* __restrict__ grad_in,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float val = __half2float(x[idx]);
    float go = __half2float(grad_out[idx]);

    float sig = sigmoid_stable(val);
    // d_SiLU(x)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float d_silu = sig * (1.0f + val * (1.0f - sig));
    float result = go * d_silu;

    grad_in[idx] = __float2half_rn(result);
}

// ============================================================================
// Kernel 3: SwiGLU 前向传播
//   SwiGLU(x_gate, x_up) = x_gate * SiLU(x_up)
//   工业应用（LLaMA / Mistral / Gemma）：
//     h = W_gate @ x    ← gate projection
//     u = W_up @ x      ← up projection（由 SiLU 处理）
//     out = h * silu(u) ← element-wise 乘
//   三个 projection（gate/up/down）的合并写为单独的 CUDA kernel
// ============================================================================
__global__ void swiglu_fwd_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float gate_val = __half2float(gate[idx]);
    float up_val = __half2float(up[idx]);

    float sig_up = sigmoid_stable(up_val);
    float silu_up = up_val * sig_up;
    float result = gate_val * silu_up;

    out[idx] = __float2half_rn(result);
}

// ============================================================================
// Wrapper 函数：处理 PyTorch tensor 到 CUDA 指针的转换、kernel launch 及错误检查
// ============================================================================

void run_silu_fwd(torch::Tensor x, torch::Tensor out) {
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

    silu_fwd_kernel<<<(n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        n_elements);

    CUDA_CHECK(cudaGetLastError());
}

void run_silu_bwd(
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

    silu_bwd_kernel<<<(n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(grad_out.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(grad_in.data_ptr<torch::Half>()),
        n_elements);

    CUDA_CHECK(cudaGetLastError());
}

void run_swiglu_fwd(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor out)
{
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(gate.scalar_type() == torch::kHalf, "gate must be fp16");
    TORCH_CHECK(up.scalar_type() == torch::kHalf, "up must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    TORCH_CHECK(gate.sizes() == out.sizes(),
                "gate and out must have the same shape");
    TORCH_CHECK(up.sizes() == out.sizes(),
                "up and out must have the same shape");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(), "up must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int n_elements = static_cast<int>(gate.numel());
    constexpr int BLOCK_SIZE = 256;

    swiglu_fwd_kernel<<<(n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(gate.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(up.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<torch::Half>()),
        n_elements);

    CUDA_CHECK(cudaGetLastError());
}
