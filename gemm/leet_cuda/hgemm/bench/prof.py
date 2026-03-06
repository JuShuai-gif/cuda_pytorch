"""
@file prof.py
@brief HGEMM (Half Precision Matrix Multiplication) 性能基准测试脚本

本脚本用于测试和比较不同HGEMM实现的性能。
支持测试矩阵尺寸: M, N, K 可配置
支持多种GPU矩阵乘法实现对比

依赖:
- PyTorch (用于GPU张量操作和基准测试)
- CUDA扩展 (可选，用于加载自定义CUDA kernel)

使用方法:
1. 确保已编译CUDA扩展 (hgemm.cu -> hgemm.so)
2. 运行脚本进行性能测试
3. 查看各实现的运行时间和输出值
"""

import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

# 禁用梯度计算 - 推理时不需要梯度，节省内存和计算资源
# 对于矩阵乘法基准测试，梯度计算是多余的
torch.set_grad_enabled(False)


# =============================================================================
# 注释: CUDA扩展加载 (可选功能)
# =============================================================================
# 以下代码用于编译和加载自定义CUDA kernel
# 如果需要测试自定义HGEMM实现，取消注释并确保hgemm.cu存在
#
# 参数说明:
# - name: 加载的库名称，用于后续调用
# - sources: CUDA源文件列表
# - extra_cuda_cflags: CUDA编译选项
#   * "-O3": 最高级别优化
#   * "-U__CUDA_NO_HALF_OPERATORS__": 启用half精度运算符
#   * "-U__CUDA_NO_HALF_CONVERSIONS__": 启用half精度转换
#   * "-U__CUDA_NO_HALF2_OPERATORS__": 启用half2向量类型
#   * "-U__CUDA_NO_BFLOAT16_CONVERSIONS__": 启用bfloat16转换
#   * "--expt-relaxed-constexpr": 允许constexpr表达式求值
#   * "--expt-extended-lambda": 允许extended lambda
#   * "--use_fast_math": 使用快速数学运算(低精度)
# - extra_cflags: C++编译选项
# =============================================================================

# lib = load(name='hgemm_lib', 
#            sources=['hgemm.cu'], 
#            extra_cuda_cflags=[
#                "-O3",
#                 "-U__CUDA_NO_HALF_OPERATORS__",
#                 "-U__CUDA_NO_HALF_CONVERSIONS__",
#                 "-U__CUDA_NO_HALF2_OPERATORS__",
#                 "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
#                 "--expt-relaxed-constexpr",
#                 "--expt-extended-lambda",
#                 "--use_fast_math"
#             ], 
#            extra_cflags=['-std=c++17'])


def run_benchmark(
    perf_func: callable, 
    a: torch.Tensor, b: torch.Tensor,
    tag: str, out: Optional[torch.Tensor] = None, 
    warmup: int = 1, iters: int = 10,
    show_all: bool = False):
    """
    @brief 运行性能基准测试的通用函数
    
    @param perf_func 要测试的矩阵乘法函数
                     签名: func(a, b) -> c 或 func(a, b, out) -> None
    @param a 输入矩阵A，形状为(M, K)，类型为half/float16
    @param b 输入矩阵B，形状为(K, N)，类型为half/float16
    @param tag 测试标签，用于输出标识
    @param out 预分配的输出矩阵，可选。如果提供则复用该矩阵，避免频繁分配
    @param warmup 预热迭代次数，让GPU预热并达到稳定状态
    @param iters 正式测试迭代次数，用于计算平均时间
    @param show_all 是否打印完整输出张量
    
    @return tuple (out_clone, mean_time) 输出张量的克隆和平均执行时间(毫秒)
    
    测试流程:
    1. 如果提供了out，先将其填充为0 (可选)
    2. 执行warmup次预热迭代，让GPU达到稳定状态
    3. 同步CUDA设备，确保之前操作完成
    4. 记录开始时间
    5. 执行iters次正式测试迭代
    6. 同步CUDA设备，确保所有操作完成
    7. 记录结束时间，计算平均时间
    8. 打印输出信息和性能统计
    """
    
    # 初始化输出张量
    # 如果提供了out，先将其填充为0，确保初始状态干净
    if out is not None: 
        out.fill_(0)      
    
    # 预热阶段
    # 执行warmup次迭代，让GPU编译器进行JIT编译和优化
    # 第一次调用CUDA kernel时会有编译开销，需要预热来排除干扰
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b) 
    
    # 同步CUDA设备
    # 确保所有之前的CUDA操作都已完成，避免时间测量不准确
    torch.cuda.synchronize()
    
    # 记录测试开始时间
    start = time.time()
    
    # 正式测试阶段
    # 执行iters次迭代并计算平均时间
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b) 
    
    # 再次同步，确保所有测试迭代都已完成
    torch.cuda.synchronize()
    end = time.time()
    
    # 计算总时间和平均时间
    total_time = (end - start) * 1000  # 转换为毫秒
    mean_time = total_time / iters      # 每次迭代的平均时间
    
    # 准备输出信息
    out_info = f"out_{tag}"
    
    # 获取输出张量的前3个元素用于验证
    # flatten()展平为一维，detach()分离计算图，cpu()移到CPU，numpy()转numpy数组
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    
    # 格式化输出值，保留8位小数，保证可读性
    out_val = [round(v, 8) for v in out_val]
    
    # 对齐格式化，每个值占12个字符宽度
    out_val = [f"{v:<12}" for v in out_val]
    
    # 打印结果: 标签(右对齐32字符) + 输出值 + 平均时间
    print(f"{out_info:>32}: {out_val}, time:{mean_time:.6f}ms")
    
    # 可选: 打印完整输出张量
    if show_all: 
        print(out)
    
    # 返回输出张量的克隆和平均时间
    # 返回克隆是为了防止后续操作修改输出
    return out.clone(), mean_time


# =============================================================================
# 测试参数配置
# =============================================================================
# 可选的测试矩阵尺寸列表 (已注释)
# Ms = [1024, 2048, 4096]  # 矩阵A的行数，矩阵C的行数
# Ns = [1024, 2048, 4096]  # 矩阵B的列数，矩阵C的列数
# Ks = [256,  512,  1024]  # 矩阵A的列数，矩阵B的行数

# 当前测试配置: 单一大矩阵
# M = 4096: 矩阵A和C的行数
# N = 4096: 矩阵B和C的列数  
# K = 1024: 矩阵A的列数，矩阵B的行数
# 
# 矩阵维度关系:
# A: (M, K) = (4096, 1024)
# B: (K, N) = (1024, 4096)
# C: (M, N) = (4096, 4096)
#
# 计算量: M × N × K = 4096 × 4096 × 1024 ≈ 17.2 GFLOPs (乘加运算)

Ms = [4096]
Ns = [4096]
Ks = [1024]

# 生成所有MNK组合
# 结果: [(4096, 4096, 1024)] - 只测试一个配置
MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]


# =============================================================================
# 主测试循环
# =============================================================================
for (M, N, K) in MNKs:
    # 打印分隔线，长度110个字符
    print("-" * 110)
    
    # 打印当前测试配置，在前面添加45个空格以居中
    print(" " * 45 + f"M={M}, N={N}, K={K}")
    
    # 创建随机输入矩阵
    # torch.randn: 生成标准正态分布的随机数
    # .cuda(): 将张量移到GPU设备上
    # .half(): 转换为半精度浮点数 (FP16/bfloat16，取决于硬件)
    # .contiguous(): 确保内存连续，方便高效访问
    a = torch.randn((M, K)).cuda().half().contiguous() 
    b = torch.randn((K, N)).cuda().half().contiguous() 
    c = torch.randn((M, N)).cuda().half().contiguous() 
    
    # =============================================================================
    # 可用的HGEMM实现 (已注释，需要先编译CUDA扩展)
    # =============================================================================
    # 以下是各种自定义HGEMM实现的函数指针
    # 每个实现都有不同的优化策略:
    #
    # hgemm_naive_f16:           朴素实现，直接全局内存访问
    # hgemm_sliced_k_f16:        K方向分片优化
    # hgemm_t_4x4_sliced_k_f16x4_pack_bcf:       4x4线程块，4元素打包，bank conflict free
    # hgemm_t_4x4_sliced_k_f16x4_pack_bcf_offset: 4x4线程块，带offset的bank conflict free版本
    # hgemm_t_8x8_sliced_k_f16x4:                8x8线程块，K方向分片
    # hgemm_t_8x8_sliced_k_f16x4_bcf:           8x8线程块，bank conflict free
    # hgemm_t_8x8_sliced_k_f16x4_pack:           8x8线程块，4元素打包
    # hgemm_t_8x8_sliced_k_f16x4_pack_bcf:      8x8线程块，4元素打包，bank conflict free
    # hgemm_t_8x8_sliced_k_f16x4_pack_bcf_offset: 8x8线程块，带offset优化
    # hgemm_t_8x8_sliced_k_f16x8_pack_bcf:      8x8线程块，8元素打包
    # hgemm_t_8x8_sliced_k_f16x8_pack_bcf_offset: 8x8线程块，8元素打包，带offset
    # hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf:  8x8线程块，8元素打包，双缓冲
    
    # run_benchmark(lib.hgemm_naive_f16,                                     a, b, "f16",                   c)
    # run_benchmark(lib.hgemm_sliced_k_f16,                                  a, b, "f16(sk)",               c)
    # run_benchmark(lib.hgemm_t_4x4_sliced_k_f16x4_pack_bcf,                 a, b, "f16x4pack(t4x4bcf)",    c)
    # run_benchmark(lib.hgemm_t_4x4_sliced_k_f16x4_pack_bcf_offset,          a, b, "f16x4pack(t4x4offset)", c)
    # run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4,                          a, b, "f16x4(t8x8sk)",         c)
    # run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4_bcf,                      a, b, "f16x4(t8x8bcf)",        c)
    # run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4_pack,                     a, b, "f16x4pack(t8x8sk)",     c)
    # run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4_pack_bcf,                 a, b, "f16x4pack(bcf)",        c)
    # run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4_pack_bcf_offset,          a, b, "f16x4pack(bcf+offset)", c)
    # run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf,                 a, b, "f16x8pack(bcf)",        c)
    # run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf_offset,          a, b, "f16x8pack(bcf+offset)", c)
    # run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf,            a, b, "f16x8pack(dbuf)",       c)
    
    # 运行PyTorch内置的FP16矩阵乘法作为基准参考
    # partial: 偏函数，绑定torch.matmul的out参数
    # 这使用了PyTorch/cuBLAS的高度优化实现
    run_benchmark(partial(torch.matmul, out=c),                            a, b, "f16_th")
    
    # 打印结束分隔线
    print("-" * 110)

