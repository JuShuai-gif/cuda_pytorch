import time

import torch
# 这个是 JIT 编译方式
from torch.utils.cpp_extension import load

# 禁用梯度计算，减少基准测试时的额外开销
torch.set_grad_enabled(False)

# 使用 JIT 方式加载 CUDA 源码并编译为 Python 模块
lib = load(
    name="dot_product_lib",
    sources=["dot_product.cu"], # 对应的 CUDA 源代码文件
    extra_cuda_cflags=[
        "-O3",  # 最高级优化
        # 以下四个选项用于解除对半精度（Half/BF16）运算和转换的限制
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",     # 允许在 CUDA 核函数中使用更灵活的 constexpr
        "--expt-extended-lambda",       # 允许在 GPU 代码中使用 Lambda 表达式
        "--use_fast_math",              # 启用快速数学运算（牺牲极小精度换取大幅速度提升）
    ],
    extra_cflags=["-std=c++17"],        # 使用 C++17 标准
)


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    warmup: int = 10,   # 预热次数，消除 GPU 调频和初始加载的影响
    iters: int = 1000,  # 正式测试次数
):
    # 1. 预热阶段
    for i in range(warmup):
        out = perf_func(a, b)  # warmup

    # 2. 同步并计时
    torch.cuda.synchronize()    # 关键：等待所有 GPU 异步任务完成
    start = time.time()
    for i in range(iters):
        out = perf_func(a, b)
    torch.cuda.synchronize()    # 再次同步，确保计时结束时 GPU 任务已完成
    end = time.time()
    total_time = (end - start) * 1000  # 转换为毫秒(ms)
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.item()

    # 根据数据类型格式化打印输出
    if tag.startswith("i8"):
        print(f"{out_info:>17}: {out_val:<15}, time:{mean_time:.8f}ms")
    else:
        print(f"{out_info:>17}: {out_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time

# 测试不同的矩阵/向量维度组合
Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 80)
    print(" " * 25 + f"S={S}, K={K}")

    # 生成测试数据并搬运到 GPU
    a = torch.randn((S * K)).cuda().float()
    b = torch.randn((S * K)).cuda().float()

    # 测试 FP32 (单精度) 下的各种核函数实现
    run_benchmark(lib.dot_prod_f32_f32, a, b, "f32f32") # 基础实现
    run_benchmark(lib.dot_prod_f32x4_f32, a, b, "f32x4f32") # float4 向量化访存
    run_benchmark(torch.dot, a, b, "f32f32_th") # PyTorch 官方原生实现
    
    # 测试 FP16 (半精度) 下的实现
    print("-" * 80)
    a_f16 = a.half()
    b_f16 = b.half()
    run_benchmark(lib.dot_prod_f16_f32, a_f16, b_f16, "f16f32") # 半精度输入，全精度累加
    run_benchmark(lib.dot_prod_f16x2_f32, a_f16, b_f16, "f16x2f32") # half2 向量化
    run_benchmark(lib.dot_prod_f16x8_pack_f32, a_f16, b_f16, "f16x8packf32") # 更宽的打包访存
    run_benchmark(torch.dot, a_f16, b_f16, "f16f16_th") # PyTorch 官方原生实现
    print("-" * 80)
