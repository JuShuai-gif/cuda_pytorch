import time
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# 加载 CUDA 扩展
lib = load(
    name="dot_product_lib",
    sources=["dot_product.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_with_stream(func, a: torch.Tensor, b: torch.Tensor, stream: torch.cuda.Stream = None):
    """
    统一处理显式/隐式 CUDA 流。
    - func: 执行的 dot_prod 函数
    - stream: 可选 CUDA 流
    """
    if stream is None:
        # 不传流时使用默认流
        return func(a, b)
    else:
        # 显式流传参
        return func(a, b, stream)


# 测试数据
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

# 1. 默认流（隐式）
out_default = run_with_stream(lib.dot_prod_f16x2_f32, a, b)
print("Default stream output:", out_default.item())

# 2. 上下文流
ctx_stream = torch.cuda.Stream()
with torch.cuda.stream(ctx_stream):
    out_ctx = run_with_stream(lib.dot_prod_f16x2_f32, a, b)
torch.cuda.synchronize()
print("Context stream output:", out_ctx.item())

# 3. 显式流
explicit_stream = torch.cuda.Stream()
out_explicit = run_with_stream(lib.dot_prod_f16x2_f32, a, b, explicit_stream)
explicit_stream.synchronize()
print("Explicit stream output:", out_explicit.item())