#include <torch/extension.h>

// ============================================================================
// 前向声明：所有 CUDA kernel 的 host 端 launch 函数
// ============================================================================

// attention 模块
void run_flash_attention_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    float scale,
    bool causal);

void run_paged_attention(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    torch::Tensor O,
    float scale);

std::vector<torch::Tensor> allocate_kv_cache(
    int num_blocks, int block_size, int num_heads, int head_dim);

// normalization 模块
void run_rmsnorm_fwd(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    float eps);

void run_rmsnorm_residual_fwd(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor out,
    torch::Tensor residual_out,
    float eps);

void run_layernorm_fwd(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    float eps);

void run_fused_residual_layernorm_fwd(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    float eps);

// activation 模块
void run_silu_fwd(torch::Tensor x, torch::Tensor out);
void run_swiglu_fwd(torch::Tensor gate, torch::Tensor up, torch::Tensor out);
void run_gelu_fwd(torch::Tensor x, torch::Tensor out);
void run_fused_bias_relu(torch::Tensor x, torch::Tensor bias, torch::Tensor out);
void run_fused_bias_gelu(torch::Tensor x, torch::Tensor bias, torch::Tensor out);
void run_fused_bias_silu(torch::Tensor x, torch::Tensor bias, torch::Tensor out);

// softmax 模块
void run_online_softmax(torch::Tensor x, torch::Tensor out);
void run_masked_online_softmax(
    torch::Tensor x, torch::Tensor mask, torch::Tensor out, float mask_value);

// matmul 模块
void run_tiled_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void run_batched_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C);

// convolution 模块
void run_direct_conv2d_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride_h, int stride_w, int pad_h, int pad_w);
void run_im2col_conv2d_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride_h, int stride_w, int pad_h, int pad_w);

// reduction 模块（已有）
torch::Tensor launch_warp_reduce_sum(const torch::Tensor& input);
torch::Tensor launch_full_warp_reduction(const torch::Tensor& input);
torch::Tensor launch_naive_reduce_sum(const torch::Tensor& input);

// basic 模块（已有）
torch::Tensor launch_vector_add_cuda(const torch::Tensor& a, const torch::Tensor& b);
torch::Tensor launch_reduce_sum_cuda(const torch::Tensor& input);


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// ============================================================================
// Torch Autograd Function：向量加法
// 将 CUDA kernel 包装为标准的 PyTorch 算子
// 支持自动微分（加法的反向传播是恒等映射）
// ============================================================================
class VectorAddFunction : public torch::autograd::Function<VectorAddFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& a,
        const torch::Tensor& b
    ) {
        ctx->saved_data["a_shape"] = a.sizes();
        ctx->saved_data["b_shape"] = b.sizes();
        return launch_vector_add_cuda(a, b);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // d(a+b)/da = 1，d(a+b)/db = 1，所以梯度直接传递
        auto grad = grad_outputs[0];
        torch::Tensor grad_a = grad;
        torch::Tensor grad_b = grad;

        // 广播梯度以匹配原始输入形状
        auto a_shape = ctx->saved_data["a_shape"].toIntVector();
        auto b_shape = ctx->saved_data["b_shape"].toIntVector();

        while (static_cast<int64_t>(grad_a.dim()) < static_cast<int64_t>(a_shape.size())) {
            grad_a = grad_a.sum(0, false);
        }
        while (static_cast<int64_t>(grad_b.dim()) < static_cast<int64_t>(b_shape.size())) {
            grad_b = grad_b.sum(0, false);
        }

        return {grad_a, grad_b};
    }
};


// ============================================================================
// Torch Autograd Function：标量求和 reduction
// ============================================================================
class ReduceSumFunction : public torch::autograd::Function<ReduceSumFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input
    ) {
        ctx->saved_data["input_shape"] = input.sizes();
        return launch_reduce_sum_cuda(input);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // d(sum(x))/dx = 1，将标量梯度广播回输入形状
        auto input_shape = ctx->saved_data["input_shape"].toIntVector();
        auto grad = grad_outputs[0].expand(input_shape);
        return {grad};
    }
};


// ============================================================================
// Torch Autograd Function：warp shuffle reduction
// ============================================================================
class WarpReduceSumFunction : public torch::autograd::Function<WarpReduceSumFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input
    ) {
        ctx->saved_data["input_shape"] = input.sizes();
        return launch_warp_reduce_sum(input);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // d(sum(x))/dx = 1，将标量梯度广播回输入形状
        auto input_shape = ctx->saved_data["input_shape"].toIntVector();
        auto grad = grad_outputs[0].expand(input_shape);
        return {grad};
    }
};


// ============================================================================
// Python API 便捷包装函数
// ============================================================================

// --- 基础算子 ---
torch::Tensor vector_add(const torch::Tensor& a, const torch::Tensor& b) {
    return VectorAddFunction::apply(a, b);
}

torch::Tensor reduce_sum(const torch::Tensor& input) {
    return ReduceSumFunction::apply(input);
}

// --- Reduction ---
torch::Tensor warp_reduce_sum(const torch::Tensor& input) {
    return WarpReduceSumFunction::apply(input);
}

torch::Tensor full_warp_reduction(const torch::Tensor& input) {
    return launch_full_warp_reduction(input);
}

torch::Tensor naive_reduce_sum(const torch::Tensor& input) {
    return launch_naive_reduce_sum(input);
}

// --- Attention ---
// FlashAttention 包装函数（原地计算：结果写入 O）
void flash_attention_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    float scale,
    bool causal)
{
    run_flash_attention_fwd(Q, K, V, O, scale, causal);
}

// PagedAttention 包装函数（原地计算：结果写入 O）
void paged_attention(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    torch::Tensor O,
    float scale)
{
    run_paged_attention(Q, K_cache, V_cache, block_tables, context_lens, O, scale);
}

// KV cache 分配包装
std::vector<torch::Tensor> kv_cache_allocate(
    int num_blocks, int block_size, int num_heads, int head_dim)
{
    return allocate_kv_cache(num_blocks, block_size, num_heads, head_dim);
}

// --- Normalization ---
// RMSNorm 包装函数（原地：结果写入 out）
void rmsnorm_fwd_wrapper(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    float eps)
{
    run_rmsnorm_fwd(x, weight, out, eps);
}

// 融合残差 + RMSNorm 包装函数
void rmsnorm_residual_fwd_wrapper(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor out,
    torch::Tensor residual_out,
    float eps)
{
    run_rmsnorm_residual_fwd(x, residual, weight, out, residual_out, eps);
}

// LayerNorm 包装函数（原地：结果写入 out）
void layernorm_fwd_wrapper(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    float eps)
{
    run_layernorm_fwd(x, weight, bias, out, eps);
}

// 融合残差 + LayerNorm 包装函数
void fused_residual_layernorm_wrapper(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    float eps)
{
    run_fused_residual_layernorm_fwd(x, residual, weight, bias, out, eps);
}

// --- Activation ---
void silu_fwd_wrapper(torch::Tensor x, torch::Tensor out) {
    run_silu_fwd(x, out);
}

void swiglu_fwd_wrapper(torch::Tensor gate, torch::Tensor up, torch::Tensor out) {
    run_swiglu_fwd(gate, up, out);
}

void gelu_fwd_wrapper(torch::Tensor x, torch::Tensor out) {
    run_gelu_fwd(x, out);
}

// --- 融合 bias + activation ---
void fused_bias_relu_wrapper(
    torch::Tensor x, torch::Tensor bias, torch::Tensor out)
{
    run_fused_bias_relu(x, bias, out);
}

void fused_bias_gelu_wrapper(
    torch::Tensor x, torch::Tensor bias, torch::Tensor out)
{
    run_fused_bias_gelu(x, bias, out);
}

void fused_bias_silu_wrapper(
    torch::Tensor x, torch::Tensor bias, torch::Tensor out)
{
    run_fused_bias_silu(x, bias, out);
}

// --- Softmax ---
void online_softmax_wrapper(torch::Tensor x, torch::Tensor out) {
    run_online_softmax(x, out);
}

void masked_online_softmax_wrapper(
    torch::Tensor x, torch::Tensor mask, torch::Tensor out, float mask_value)
{
    run_masked_online_softmax(x, mask, out, mask_value);
}


// --- Matmul ---
// 单 batch tiled matmul 包装函数（原地计算：结果写入 C）
void tiled_matmul_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    run_tiled_matmul(A, B, C);
}

// 批量 tiled matmul 包装函数（原地计算：结果写入 C）
void batched_matmul_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    run_batched_matmul(A, B, C);
}

// --- Convolution ---
// 直接卷积包装函数（原地计算：结果写入 output）
void direct_conv2d_wrapper(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride_h, int stride_w, int pad_h, int pad_w)
{
    run_direct_conv2d_fwd(input, weight, bias, output, stride_h, stride_w, pad_h, pad_w);
}

// im2col + GEMM 卷积包装函数
void im2col_conv2d_wrapper(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride_h, int stride_w, int pad_h, int pad_w)
{
    run_im2col_conv2d_fwd(input, weight, bias, output, stride_h, stride_w, pad_h, pad_w);
}


// ============================================================================
// Pybind11 模块定义：将所有 kernel 暴露给 Python
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "自定义 CUDA kernel 集合：Attention、Normalization、Activation、Softmax、Reduction、Matmul、Convolution";

    // --- 基础算子 ---
    m.def("vector_add", &vector_add,
          "元素级向量加法 (CUDA kernel)。\n"
          "参数:\n"
          "    a (Tensor): 第一个输入张量 (float32, CUDA)\n"
          "    b (Tensor): 第二个输入张量 (float32, CUDA)\n"
          "返回:\n"
          "    Tensor: a + b (float32, CUDA)");

    m.def("reduce_sum", &reduce_sum,
          "所有元素求和 reduction (CUDA kernel)。\n"
          "参数:\n"
          "    input (Tensor): 输入张量 (float32, CUDA, contiguous)\n"
          "返回:\n"
          "    Tensor: 标量和 (float32, CUDA)");

    // --- Reduction 算子 ---
    m.def("warp_reduce_sum", &warp_reduce_sum,
          "warp shuffle reduction：使用 __shfl_down_sync 的 warp 级求和。\n"
          "相比 naive reduction，减少 shared memory bank conflict，延迟更低。\n"
          "参数:\n"
          "    input (Tensor): 输入张量 (float32, CUDA, contiguous)\n"
          "返回:\n"
          "    Tensor: 标量和 (float32, CUDA)");

    m.def("full_warp_reduction", &full_warp_reduction,
          "纯 warp reduction（每个 block 输出一个部分和，CPU 端完成最终求和）。\n"
          "参数:\n"
          "    input (Tensor): 输入张量 (float32, CUDA, contiguous)\n"
          "返回:\n"
          "    Tensor: 标量和 (float32, CUDA)");

    m.def("naive_reduce_sum", &naive_reduce_sum,
          "naive shared memory reduction（用于性能对比）。\n"
          "参数:\n"
          "    input (Tensor): 输入张量 (float32, CUDA, contiguous)\n"
          "返回:\n"
          "    Tensor: 标量和 (float32, CUDA)");

    // --- Attention 算子 ---
    m.def("flash_attention_fwd", &flash_attention_fwd,
          "FlashAttention 前向传播 (Dao et al. 2022)。\n"
          "IO-Aware 精确注意力，避免物化完整的 N×N 注意力矩阵。\n"
          "参数:\n"
          "    Q (Tensor): [batch, n_heads, seq_len, head_dim] (float16, CUDA)\n"
          "    K (Tensor): [batch, n_heads, seq_len, head_dim] (float16, CUDA)\n"
          "    V (Tensor): [batch, n_heads, seq_len, head_dim] (float16, CUDA)\n"
          "    O (Tensor): [batch, n_heads, seq_len, head_dim] (float16, CUDA, 原地输出)\n"
          "    scale (float): 缩放因子，通常为 1/sqrt(head_dim)\n"
          "    causal (bool): 是否使用 causal mask");

    m.def("paged_attention", &paged_attention,
          "PagedAttention (vLLM 风格) 自回归解码注意力。\n"
          "Q 为单个 token，K/V 按固定大小块存储，通过 block_table 映射。\n"
          "参数:\n"
          "    Q (Tensor): [num_heads, head_dim] (float16, CUDA)\n"
          "    K_cache (Tensor): [num_blocks, block_size, num_heads, head_dim] (float16, CUDA)\n"
          "    V_cache (Tensor): [num_blocks, block_size, num_heads, head_dim] (float16, CUDA)\n"
          "    block_tables (Tensor): [batch_size, max_blocks_per_seq] (int32, CUDA)\n"
          "    context_lens (Tensor): [batch_size] (int32, CUDA)\n"
          "    O (Tensor): [num_heads, head_dim] (float16, CUDA, 原地输出)\n"
          "    scale (float): 缩放因子，通常为 1/sqrt(head_dim)");

    m.def("allocate_kv_cache", &kv_cache_allocate,
          "为 PagedAttention 分配 KV cache。\n"
          "参数:\n"
          "    num_blocks (int): 物理块总数\n"
          "    block_size (int): 每个块的 token 数\n"
          "    num_heads (int): 注意力头数\n"
          "    head_dim (int): 每个头的维度\n"
          "返回:\n"
          "    tuple(Tensor, Tensor): (K_cache, V_cache)，形状均为\n"
          "    [num_blocks, block_size, num_heads, head_dim] (float16, CUDA)");

    // --- Normalization 算子 ---
    m.def("rmsnorm_fwd", &rmsnorm_fwd_wrapper,
          "RMSNorm 前向传播 (CUDA)。\n"
          "用于 LLaMA / Mistral / Gemma 等现代 LLM。\n"
          "参数:\n"
          "    x (Tensor): [rows, hidden_dim] (float16, CUDA)\n"
          "    weight (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    out (Tensor): [rows, hidden_dim] (float16, CUDA, 原地输出)\n"
          "    eps (float): epsilon 参数");

    m.def("rmsnorm_residual_fwd", &rmsnorm_residual_fwd_wrapper,
          "融合残差 + RMSNorm 前向传播 (CUDA)。\n"
          "同时计算 residual_out = x + residual 和归一化结果 out。\n"
          "参数:\n"
          "    x (Tensor): [rows, hidden_dim] (float16, CUDA)\n"
          "    residual (Tensor): [rows, hidden_dim] (float16, CUDA)\n"
          "    weight (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    out (Tensor): [rows, hidden_dim] (float16, CUDA, 归一化输出)\n"
          "    residual_out (Tensor): [rows, hidden_dim] (float16, CUDA, x + residual)\n"
          "    eps (float): epsilon 参数");

    m.def("layernorm_fwd", &layernorm_fwd_wrapper,
          "LayerNorm 前向传播 (CUDA)。\n"
          "参数:\n"
          "    x (Tensor): [rows, hidden_dim] (float16, CUDA)\n"
          "    weight (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    bias (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    out (Tensor): [rows, hidden_dim] (float16, CUDA, 原地输出)\n"
          "    eps (float): epsilon 参数");

    m.def("fused_residual_layernorm", &fused_residual_layernorm_wrapper,
          "融合残差 + LayerNorm 前向传播 (CUDA)。\n"
          "消除 intermediate tensor 的显存读写，减少带宽开销。\n"
          "参数:\n"
          "    x (Tensor): [rows, hidden_dim] (float16, CUDA)\n"
          "    residual (Tensor): [rows, hidden_dim] (float16, CUDA)\n"
          "    weight (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    bias (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    out (Tensor): [rows, hidden_dim] (float16, CUDA, 归一化输出)\n"
          "    eps (float): epsilon 参数");

    // --- Activation 算子 ---
    m.def("silu_fwd", &silu_fwd_wrapper,
          "SiLU 激活函数前向传播 (CUDA)。\n"
          "SiLU(x) = x * sigmoid(x)\n"
          "参数:\n"
          "    x (Tensor): 任意形状 (float16, CUDA, contiguous)\n"
          "    out (Tensor): 与 x 形状相同 (float16, CUDA, 原地输出)");

    m.def("swiglu_fwd", &swiglu_fwd_wrapper,
          "SwiGLU 激活函数前向传播 (CUDA)。\n"
          "SwiGLU(gate, up) = gate * SiLU(up)\n"
          "参数:\n"
          "    gate (Tensor): 任意形状 (float16, CUDA, contiguous)\n"
          "    up (Tensor): 与 gate 形状相同 (float16, CUDA, contiguous)\n"
          "    out (Tensor): 与 gate 形状相同 (float16, CUDA, 原地输出)");

    m.def("gelu_fwd", &gelu_fwd_wrapper,
          "GELU 激活函数前向传播 (CUDA)。\n"
          "GELU(x) = x * Phi(x)，使用 tanh 近似。\n"
          "参数:\n"
          "    x (Tensor): 任意形状 (float16, CUDA, contiguous)\n"
          "    out (Tensor): 与 x 形状相同 (float16, CUDA, 原地输出)");

    // --- 融合 bias + activation 算子 ---
    m.def("fused_bias_relu", &fused_bias_relu_wrapper,
          "融合 bias + ReLU 前向传播 (CUDA)。\n"
          "out = ReLU(x + bias)，消除中间张量读写。\n"
          "参数:\n"
          "    x (Tensor): [rows, hidden_dim] (float16, CUDA, contiguous)\n"
          "    bias (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    out (Tensor): [rows, hidden_dim] (float16, CUDA, 原地输出)");

    m.def("fused_bias_gelu", &fused_bias_gelu_wrapper,
          "融合 bias + GELU 前向传播 (CUDA)。\n"
          "out = GELU(x + bias)，消除中间张量读写。\n"
          "参数:\n"
          "    x (Tensor): [rows, hidden_dim] (float16, CUDA, contiguous)\n"
          "    bias (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    out (Tensor): [rows, hidden_dim] (float16, CUDA, 原地输出)");

    m.def("fused_bias_silu", &fused_bias_silu_wrapper,
          "融合 bias + SiLU 前向传播 (CUDA)。\n"
          "out = SiLU(x + bias)，消除中间张量读写。\n"
          "参数:\n"
          "    x (Tensor): [rows, hidden_dim] (float16, CUDA, contiguous)\n"
          "    bias (Tensor): [hidden_dim] (float16, CUDA)\n"
          "    out (Tensor): [rows, hidden_dim] (float16, CUDA, 原地输出)");

    // --- Softmax 算子 ---
    m.def("online_softmax", &online_softmax_wrapper,
          "Online safe softmax 前向传播 (CUDA)。\n"
          "使用 online 算法在单次遍历中计算，避免多次读取输入。\n"
          "参数:\n"
          "    x (Tensor): [rows, cols] (float16, CUDA, contiguous)\n"
          "    out (Tensor): [rows, cols] (float16, CUDA, 原地输出)");

    m.def("masked_online_softmax", &masked_online_softmax_wrapper,
          "带 mask 的 online safe softmax 前向传播 (CUDA)。\n"
          "将 mask 为 mask_value 的位置设为 -inf 后再做 softmax。\n"
          "参数:\n"
          "    x (Tensor): [rows, cols] (float16, CUDA, contiguous)\n"
          "    mask (Tensor): [rows, cols] (float16, CUDA, contiguous)\n"
          "    out (Tensor): [rows, cols] (float16, CUDA, 原地输出)\n"
          "    mask_value (float): mask 中需要遮蔽的值");

    // --- Matmul 算子 ---
    m.def("tiled_matmul", &tiled_matmul_wrapper,
          "Tiled GEMM 前向传播 (CUDA) — C = A @ B。\n"
          "使用 shared memory 缓存 tile 减少 global memory 带宽。\n"
          "每个 block 计算一个 BM×BN 的输出 tile，支持自适应 tile 大小。\n"
          "参数:\n"
          "    A (Tensor): [M, K] (float16, CUDA)\n"
          "    B (Tensor): [K, N] (float16, CUDA)\n"
          "    C (Tensor): [M, N] (float16, CUDA, 原地输出)");

    m.def("batched_matmul", &batched_matmul_wrapper,
          "批量 Tiled GEMM 前向传播 (CUDA) — C[b] = A[b] @ B[b]。\n"
          "为每个 batch 独立调用 tiled GEMM kernel，适合 LLM 批量推理。\n"
          "参数:\n"
          "    A (Tensor): [B, M, K] (float16, CUDA)\n"
           "    B (Tensor): [B, K, N] (float16, CUDA)\n"
           "    C (Tensor): [B, M, N] (float16, CUDA, 原地输出)");

    // --- Convolution 算子 ---
    m.def("direct_conv2d", &direct_conv2d_wrapper,
          "直接卷积前向传播 (CUDA) — 每个线程计算一个输出元素。\n"
          "适合小 kernel（3x3），便于理解卷积计算过程。\n"
          "参数:\n"
          "    input (Tensor): [N, C_in, H, W] (float16, CUDA)\n"
          "    weight (Tensor): [C_out, C_in, KH, KW] (float16, CUDA)\n"
          "    bias (Tensor): [C_out] (float16, CUDA)\n"
          "    output (Tensor): [N, C_out, H_out, W_out] (float16, CUDA, 原地输出)\n"
          "    stride_h, stride_w (int): 步长\n"
          "    pad_h, pad_w (int): padding");

    m.def("im2col_conv2d", &im2col_conv2d_wrapper,
          "im2col + tiled GEMM 卷积前向传播 (CUDA)。\n"
          "工业标准方法：先展开图像为列矩阵，再用 GEMM 计算卷积。\n"
          "cuDNN 在多数配置下使用此策略。\n"
          "参数: 同 direct_conv2d");

    // 版本信息
    m.attr("__version__") = TOSTRING(PROJECT_VERSION);
}
