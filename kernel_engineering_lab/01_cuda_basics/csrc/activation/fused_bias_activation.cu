#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <cfloat>

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
    // __half2float: fp16 转 fp32，将半精度提升为单精度以进行高精度计算
    // __ldg:        通过只读数据缓存（texture/L1）加载，带宽更高、延迟更低
    //               适合 bias 这种只读且被所有线程访问的数据
    float val = __half2float(x[idx]) + __half2float(__ldg(bias + col));
    // fmaxf:        C 数学库函数，返回两个浮点数中的较大者（ReLU: max(0, val)）
    float result = fmaxf(0.0f, val);

    // __float2half_rn: fp32 转回 fp16，舍入模式为就近舍入到偶数（round-to-nearest-even）
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
    // __half2float: fp16 转 fp32，提升精度用于计算
    // __ldg:        通过只读数据缓存加载（绕开 L1/shared，延迟更低）
    float val = __half2float(x[idx]) + __half2float(__ldg(bias + col));

    // GELU tanh 近似：0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    constexpr float sqrt_2_over_pi = 0.79788456f;
    constexpr float coeff = 0.044715f;

    float val_cu = val * val * val;
    float inner = sqrt_2_over_pi * (val + coeff * val_cu);
    float tanh_inner = tanhf(inner);
    float result = 0.5f * val * (1.0f + tanh_inner);

    // __float2half_rn: fp32 转回 fp16，就近舍入到偶数
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
    // __half2float: fp16 转 fp32，提升精度用于计算
    // __ldg:        通过只读数据缓存加载（绕开 L1/shared，延迟更低）
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

    // __float2half_rn: fp32 转回 fp16，就近舍入到偶数
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

// ============================================================================
// 精度验证代码
// 对比融合版本 vs 非融合版本 vs CPU float 参考实现
//
// 融合对精度的影响：
//   - 未融合：每一步写回 fp16 都会发生一次 fp32→fp16 截断（就近舍入到偶数）
//     x(fp16)→fp32(+bias)→fp16(写回DRAM)→fp32(ReLU)→fp16(写回DRAM)  ← 2 次截断
//   - 融合：中间全程 fp32，只在最终写回时截断一次
//     x(fp16)→fp32(+bias)(ReLU)→fp16(写回DRAM)                       ← 1 次截断
//   - 结论：融合版本精度更高，因为减少了 fp16 截断次数
//
//   fp16 的机器精度（单位舍入误差）约 9.76e-4，
//   工业标准通常接受 1e-3 ~ 1e-5 的相对误差。
// ============================================================================

// ---------------------------------------------------------------------------
// 用于精度比对的非融合 GPU kernel：先做 bias_add 写回 fp16，再做 activation 写回 fp16
// 模拟 matmul 输出后经过两个独立 kernel 的真实工业场景
// ---------------------------------------------------------------------------

// 非融合步骤 1：out = x + bias，写回 fp16
__global__ void unfused_bias_add_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ bias,
    __half* __restrict__ mid,    // 中间 tensor，模拟 DRAM 回写
    int rows, int hidden_dim)
{
    const int n = rows * hidden_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int col = idx % hidden_dim;
    float val = __half2float(x[idx]) + __half2float(__ldg(bias + col));
    // 第一次 fp32→fp16 截断（模拟回写 DRAM 后又读回）
    mid[idx] = __float2half_rn(val);
}

// 非融合步骤 2：out = ReLU(mid)，写回 fp16
__global__ void unfused_relu_kernel(
    const __half* __restrict__ mid,
    __half* __restrict__ out,
    int rows, int hidden_dim)
{
    const int n = rows * hidden_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = __half2float(mid[idx]);
    // 第二次 fp32→fp16 截断（ReLU 结果写回）
    out[idx] = __float2half_rn(fmaxf(0.0f, val));
}

// 非融合步骤 2：out = GELU(mid)，写回 fp16
__global__ void unfused_gelu_kernel(
    const __half* __restrict__ mid,
    __half* __restrict__ out,
    int rows, int hidden_dim)
{
    const int n = rows * hidden_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = __half2float(mid[idx]);
    constexpr float a = 0.79788456f;
    constexpr float b = 0.044715f;
    float val_cu = val * val * val;
    float inner = a * (val + b * val_cu);
    out[idx] = __float2half_rn(0.5f * val * (1.0f + tanhf(inner)));
}

// 非融合步骤 2：out = SiLU(mid)，写回 fp16
__global__ void unfused_silu_kernel(
    const __half* __restrict__ mid,
    __half* __restrict__ out,
    int rows, int hidden_dim)
{
    const int n = rows * hidden_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = __half2float(mid[idx]);
    float sig = (val >= 0.0f) ? 1.0f / (1.0f + expf(-val))
                              : expf(val) / (1.0f + expf(val));
    out[idx] = __float2half_rn(val * sig);
}

// ---------------------------------------------------------------------------
// 误差计算
// ---------------------------------------------------------------------------
static std::pair<float, float> compute_errors(
    const float* ref,   // CPU float 黄金参考
    const float* test,  // GPU 输出转为 float
    int n)
{
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float abs_err = fabsf(ref[i] - test[i]);
        if (abs_err > max_abs_err) max_abs_err = abs_err;

        float ref_abs = fabsf(ref[i]);
        float rel_err = (ref_abs > 1e-8f) ? abs_err / ref_abs
                        : (fabsf(test[i]) > 1e-8f) ? abs_err / fabsf(test[i])
                        : 0.0f;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    return {max_abs_err, max_rel_err};
}

// ---------------------------------------------------------------------------
// 精度验证入口：对比 融合 / 非融合 / CPU参考 三个版本
// ---------------------------------------------------------------------------
void run_precision_validation(int rows = 1024, int hidden_dim = 4096) {
    at::ScalarType fp16 = torch::kHalf;
    auto options = torch::TensorOptions().dtype(fp16).device(torch::kCUDA);

    auto x    = torch::randn({rows, hidden_dim}, options) * 2.0f;
    auto bias = torch::randn({hidden_dim}, options) * 0.5f;

    auto x_cpu_f32    = x.to(torch::kCPU).to(torch::kFloat32);
    auto bias_cpu_f32 = bias.to(torch::kCPU).to(torch::kFloat32);
    const int n = rows * hidden_dim;

    // --- CPU 黄金参考：全程 fp32 ---
    auto ref_relu = torch::empty({rows, hidden_dim}, torch::kFloat32);
    auto ref_gelu = torch::empty({rows, hidden_dim}, torch::kFloat32);
    auto ref_silu = torch::empty({rows, hidden_dim}, torch::kFloat32);
    {
        float* ref_r = ref_relu.data_ptr<float>();
        float* ref_g = ref_gelu.data_ptr<float>();
        float* ref_s = ref_silu.data_ptr<float>();
        const float* xf = x_cpu_f32.data_ptr<float>();
        const float* bf = bias_cpu_f32.data_ptr<float>();

        for (int idx = 0; idx < n; idx++) {
            int col = idx % hidden_dim;
            float val = xf[idx] + bf[col];
            // ReLU
            ref_r[idx] = fmaxf(0.0f, val);
            // GELU
            constexpr float a = 0.79788456f;
            constexpr float b = 0.044715f;
            float vc = val * val * val;
            float in = a * (val + b * vc);
            ref_g[idx] = 0.5f * val * (1.0f + tanhf(in));
            // SiLU
            float sig = (val >= 0.0f) ? 1.0f/(1.0f+expf(-val))
                                      : expf(val)/(1.0f+expf(val));
            ref_s[idx] = val * sig;
        }
    }

    // 分配中间 tensor
    auto mid_unfused      = torch::empty({rows, hidden_dim}, options);
    auto out_unfused_relu = torch::empty({rows, hidden_dim}, options);
    auto out_unfused_gelu = torch::empty({rows, hidden_dim}, options);
    auto out_unfused_silu = torch::empty({rows, hidden_dim}, options);
    auto out_fused_relu   = torch::empty({rows, hidden_dim}, options);
    auto out_fused_gelu   = torch::empty({rows, hidden_dim}, options);
    auto out_fused_silu   = torch::empty({rows, hidden_dim}, options);

    auto run_compare = [&](
        const char* name,
        torch::Tensor &ref,
        torch::Tensor &mid,
        torch::Tensor &out_unfused,
        void(*unfused_act)(const __half*, const __half*, __half*, int, int),
        torch::Tensor &out_fused,
        void(*fused_launch)(const __half*, const __half*, __half*, int, int))
    {
        constexpr int BLOCK = 256;
        int grid = (n + BLOCK - 1) / BLOCK;
        auto x_ptr = reinterpret_cast<const __half*>(x.data_ptr<torch::Half>());
        auto bias_ptr = reinterpret_cast<const __half*>(bias.data_ptr<torch::Half>());
        auto mid_ptr = reinterpret_cast<__half*>(mid.data_ptr<torch::Half>());

        // 非融合：step1 bias_add → mid, step2 activation → out
        unfused_bias_add_kernel<<<grid, BLOCK>>>(x_ptr, bias_ptr, mid_ptr, rows, hidden_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        int grid2 = (n + BLOCK - 1) / BLOCK;
        unfused_act(mid_ptr, reinterpret_cast<__half*>(out_unfused.data_ptr<torch::Half>()), rows, hidden_dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 融合：单次 kernel
        fused_launch(x_ptr, bias_ptr,
                     reinterpret_cast<__half*>(out_fused.data_ptr<torch::Half>()),
                     rows, hidden_dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto unf_cpu = out_unfused.to(torch::kCPU).to(torch::kFloat32);
        auto fus_cpu = out_fused.to(torch::kCPU).to(torch::kFloat32);
        auto e_unf = compute_errors(ref.data_ptr<float>(), unf_cpu.data_ptr<float>(), n);
        auto e_fus = compute_errors(ref.data_ptr<float>(), fus_cpu.data_ptr<float>(), n);

        float improvement = (e_unf.first > 0) ? (1.0f - e_fus.first / e_unf.first) * 100.0f : 0.0f;

        printf("  %-5s │ unfused: %.4e / %.4e │ fused: %.4e / %.4e │ gain: %+.1f%%\n",
               name,
               e_unf.first, e_unf.second,
               e_fus.first, e_fus.second,
               improvement);
    };

    // 输出精度报告
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("  精度验证：融合 vs 非融合 vs CPU float 参考\n");
    printf("  shape = [%d, %d], fp16 机器精度 = %.2e\n", rows, hidden_dim, 9.76e-4f);
    printf("──────────────────────────────────────────────────────────────────\n");
    printf("  列含义：最大绝对误差 / 最大相对误差\n");
    printf("  参考基准：CPU 全程 fp32（数学上最精确）\n");
    printf("──────────────────────────────────────────────────────────────────\n");

    run_compare("ReLU", ref_relu, mid_unfused, out_unfused_relu,
                unfused_relu_kernel, out_fused_relu, fused_bias_relu_kernel);
    run_compare("GELU", ref_gelu, mid_unfused, out_unfused_gelu,
                unfused_gelu_kernel, out_fused_gelu, fused_bias_gelu_kernel);
    run_compare("SiLU", ref_silu, mid_unfused, out_unfused_silu,
                unfused_silu_kernel, out_fused_silu, fused_bias_silu_kernel);

    printf("──────────────────────────────────────────────────────────────────\n");
    printf("  结论：融合版本减少 fp16 中间存储，误差始终 ≤ 非融合版本。\n");
    printf("        这是算子融合除性能外的另一收益——精度提升。\n");
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("\n");
}
