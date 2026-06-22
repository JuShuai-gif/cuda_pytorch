#!/usr/bin/env python3
"""
GPU Kernel 数值精度分析与验证工具集

提供 GPU Kernel 工程师日常需要的精度分析功能：
- ULP 误差计算（最精确的浮点误差度量）
- 完整的精度分析报告（PrecisionReport）
- 多 Kernel 实现精度对比
- Kahan 补偿求和实现
- 浮点基础性质演示
- Reduction 精度曲线分析

所有注释采用中文，函数文档采用中文。
"""

import torch
import numpy as np
from typing import Callable, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class PrecisionReport:
    """
    精度分析报告

    包含被测 kernel 相对于 golden reference 的所有精度指标。

    属性:
        max_abs_error: 最大绝对误差
        max_rel_error: 最大相对误差
        max_ulp_error: 最大 ULP 误差（整数）
        rmse: 均方根误差
        p50_error: 中位数误差（50% 分位）
        p99_error: 99% 分位误差
        p999_error: 99.9% 分位误差
        nan_count: 输出中的 NaN 数量
        inf_count: 输出中的 Inf 数量
        zero_ref_count: 参考值接近 0 的元素数（影响相对误差计算）
        error_histogram: 误差分布的直方图数据 (bins, values)
        passed: 是否通过了指定的精度阈值
        actual_dtype: 被测输出的 dtype
        shape: 被测输出的 shape
        num_elements: 元素总数
    """

    max_abs_error: float
    max_rel_error: float
    max_ulp_error: int
    rmse: float
    p50_error: float
    p99_error: float
    p999_error: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    zero_ref_count: int = 0
    error_histogram: Optional[Tuple[np.ndarray, np.ndarray]] = None
    passed: bool = False
    actual_dtype: str = ""
    shape: List[int] = field(default_factory=list)
    num_elements: int = 0


# ============================================================================
# ULP 误差计算
# ============================================================================


def float_to_bits(x: torch.Tensor) -> torch.Tensor:
    """
    将 float32 tensor 转换为 IEEE 754 整数位模式。

    返回 int32 表示的位模式，可直接用于 ULP 计算。
    """
    return x.view(torch.int32)


def bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    """
    将 IEEE 754 整数位模式转换回 float32 tensor。
    """
    return bits.view(torch.float32)


def compute_ulp_error(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    计算 ULP（Units in the Last Place）误差。

    ULP 是最精确的浮点误差度量方式。1 ULP 表示两个浮点数在 IEEE 754
    表示中相差一个最小精度单位。

    算法说明:
        对于两个正数，ULP 误差 = |int32(a) - int32(b)|
        （因为 IEEE 754 的整数表示对于正数是单调递增的）

        对于符号不同的两个数，需要特殊处理：
        - 如果 a > 0, b < 0: 需要跨越符号边界计算
        - 当 actual_bits == ref_bits 时误差为 0

    精度判定标准:
        0 ULP:     完全相同
        1-2 ULP:   轻微舍入差异，优秀
        2-4 ULP:   fp32 可接受范围
        4-16 ULP:  fp16 可接受范围
        16+:       需要调查

    Args:
        actual: 被测 kernel 的输出（任意浮点类型）
        reference: 参考标准输出（建议 fp64）

    Returns:
        每个元素的 ULP 误差（int64 tensor，确保不溢出）
    """
    # 统一转换为 fp32 进行 ULP 比较
    a = actual.float()
    r = reference.float()

    actual_bits = a.view(torch.int32).long()
    ref_bits = r.view(torch.int32).long()

    # 处理符号不同的情况
    # 当符号不同时，IEEE 754 的整数表示不是线性的，
    # 需要从负数的位模式转到正数的位模式（或者反过来）
    sign_mask = torch.iinfo(torch.int32).max + 1  # 0x80000000 for int32

    # 将负数位模式映射到正数空间（通过翻转最高位实现单调性）
    actual_mapped = torch.where(
        actual_bits < 0,
        sign_mask - actual_bits,  # 映射负数到正数空间
        actual_bits,
    )
    ref_mapped = torch.where(ref_bits < 0, sign_mask - ref_bits, ref_bits)

    # ULP 误差 = 映射后位模式的绝对差
    ulp = (actual_mapped - ref_mapped).abs()

    # 任何 NaN 或 Inf 的 ULP 设为可在 int32 表示的最大值
    nan_inf_sentinel = torch.tensor(0x7FFFFFFF, dtype=torch.long)  # INT32_MAX
    ulp = torch.where(torch.isnan(a) | torch.isnan(r), nan_inf_sentinel, ulp)
    ulp = torch.where(torch.isinf(a) | torch.isinf(r), nan_inf_sentinel, ulp)

    return ulp.int()


# ============================================================================
# 精度分析
# ============================================================================


def analyze_precision(
    actual: torch.Tensor,
    reference: torch.Tensor,
    name: str = "kernel",
    rtol: float = 1e-3,
    atol: float = 1e-5,
    max_allowed_ulp: int = 16,
    verbose: bool = True,
) -> PrecisionReport:
    """
    综合分析 kernel 输出的数值精度。

    这是 kernel 工程师最常用的函数——给定被测 kernel 输出和 fp64/标准实现输出，
    生成一份完整的精度分析报告，涵盖绝对误差、相对误差、ULP 误差、统计分位数、
    异常值检测等全部指标。

    使用示例:
        >>> actual = my_kernel(input_tensor)
        >>> ref = torch_reference(input_tensor.double()).float()
        >>> report = analyze_precision(actual, ref, name="my_fused_kernel")
        >>> print(f"Passed: {report.passed}, max ULP: {report.max_ulp_error}")

    Args:
        actual: 被测 kernel 的输出（fp16/bf16/fp32 均可）
        reference: 参考输出（建议 fp64 转 fp32，或标准库实现）
        name: kernel 名称，用于报告标题
        rtol: 相对误差阈值
        atol: 绝对误差阈值
        max_allowed_ulp: 最大允许的 ULP 误差
        verbose: 是否打印详细的文本报告

    Returns:
        PrecisionReport 对象，包含所有精度指标
    """
    # 统一转换为 fp32 以便比较
    actual_f32 = actual.detach().float()
    ref_f32 = reference.detach().float()

    # 确保形状一致
    assert actual_f32.shape == ref_f32.shape, (
        f"形状不一致: actual {actual_f32.shape} vs ref {ref_f32.shape}"
    )

    n = actual_f32.numel()
    if n == 0:
        raise ValueError("输入 tensor 为空")

    # --- 基础误差计算 ---
    abs_diff = (actual_f32 - ref_f32).abs()
    max_abs = abs_diff.max().item()

    # 相对误差（对 ref 接近 0 的情况使用修正除数）
    ref_abs = ref_f32.abs()
    safe_divisor = torch.where(ref_abs > atol, ref_abs, torch.tensor(atol))
    rel_diff = torch.where(
        ref_abs > atol,
        abs_diff / safe_divisor,
        abs_diff / atol,  # 当 ref 接近 0 时使用绝对误差标准化
    )
    max_rel = rel_diff.max().item()

    # --- ULP 误差 ---
    ulp_values = compute_ulp_error(actual_f32, ref_f32)
    # 过滤掉 Inf ULP（NaN/Inf 单独统计）
    finite_mask = ulp_values < torch.iinfo(torch.int32).max
    ulp_finite = ulp_values[finite_mask]
    if ulp_finite.numel() > 0:
        max_ulp = ulp_finite.max().item()
    else:
        max_ulp = torch.iinfo(torch.int32).max

    # --- 统计分位数 ---
    sorted_abs = abs_diff.flatten().sort().values
    rmse_val = torch.sqrt((abs_diff**2).mean()).item()
    p50 = sorted_abs[n // 2].item()
    p99_idx = min(int(n * 0.99), n - 1)
    p999_idx = min(int(n * 0.999), n - 1)
    p99 = sorted_abs[p99_idx].item()
    p999 = sorted_abs[p999_idx].item()

    # --- 异常值检测 ---
    nan_count = torch.isnan(actual_f32).sum().item()
    inf_count = torch.isinf(actual_f32).sum().item()
    zero_ref = (ref_abs <= atol).sum().item()

    # --- 误差直方图 ---
    abs_np = sorted_abs.cpu().numpy()
    hist_bins = np.logspace(
        np.log10(max(atol, abs_np[abs_np > 0].min() if (abs_np > 0).any() else atol)),
        np.log10(max(abs_np.max(), 1.0)),
        50,
    )
    hist_vals, _ = np.histogram(abs_np, bins=hist_bins)

    # --- 通过判定 ---
    # 使用 ULP、相对误差、绝对误差的综合判定
    passed_ulp = max_ulp <= max_allowed_ulp
    passed_tol = (max_rel <= rtol) and (max_abs <= max(atol, 1e-3))
    passed = passed_ulp or passed_tol

    # --- 输出报告 ---
    if verbose:
        print(f"\n{'=' * 62}")
        print(f"  精度分析报告: {name}")
        print(f"{'=' * 62}")
        print(f"  dtype:         {actual.dtype}")
        print(f"  shape:         {list(actual.shape)}")
        print(f"  元素总数:      {n:,}")
        print(f"{'-' * 42}")
        print(f"  最大绝对误差:  {max_abs:.4e}")
        print(f"  最大相对误差:  {max_rel:.4e}")
        print(f"  最大 ULP 误差: {max_ulp}")
        print(f"  RMSE:          {rmse_val:.4e}")
        print(f"  P50 误差:      {p50:.4e}")
        print(f"  P99 误差:      {p99:.4e}")
        print(f"  P999 误差:     {p999:.4e}")
        print(f"{'-' * 42}")
        print(f"  NaN 数量:      {nan_count}")
        print(f"  Inf 数量:      {inf_count}")
        print(f"  参考值为0个数: {zero_ref}")
        print(f"{'-' * 42}")
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] rtol={rtol}, atol={atol}, max_ulp={max_allowed_ulp}")
        print(f"{'=' * 62}\n")

    return PrecisionReport(
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        max_ulp_error=int(max_ulp),
        rmse=rmse_val,
        p50_error=p50,
        p99_error=p99,
        p999_error=p999,
        nan_count=int(nan_count),
        inf_count=int(inf_count),
        zero_ref_count=int(zero_ref),
        error_histogram=(hist_bins, hist_vals),
        passed=passed,
        actual_dtype=str(actual.dtype),
        shape=list(actual.shape),
        num_elements=n,
    )


def quick_check(
    actual: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    快速精度检查（一行调用）。

    Returns:
        True 如果精度在容忍范围内，False 否则。
    """
    report = analyze_precision(actual, reference, verbose=False, rtol=rtol, atol=atol)
    return report.passed


# ============================================================================
# 多实现对比
# ============================================================================


def compare_implementations(
    implementations: Dict[str, Callable],
    reference_fn: Callable,
    input_generator: Callable,
    n_runs: int = 10,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Dict[str, float]]:
    """
    对比多个 kernel 实现的数值精度。

    对每个实现运行多次（每次使用新生成的随机输入），收集各次运行
    的精度指标，最终汇总为统计报告。

    使用示例:
        >>> def impl_a(x): return my_kernel_v1(x)
        >>> def impl_b(x): return my_kernel_v2(x)
        >>> def ref(x): return torch.softmax(x.double(), dim=-1).float()
        >>> def generator(): return torch.randn(4096, 64, device='cuda')
        >>> results = compare_implementations(
        ...     {"v1": impl_a, "v2": impl_b},
        ...     reference_fn=ref,
        ...     input_generator=generator,
        ...     n_runs=20
        ... )

    Args:
        implementations: {名称: kernel函数} 字典
        reference_fn: 参考实现函数
        input_generator: 每次运行生成新输入的函数
        n_runs: 每个实现的运行次数
        rtol: 相对误差阈值
        atol: 绝对误差阈值

    Returns:
        {实现名称: {统计指标}} 的字典
    """
    results = {}

    for name, fn in implementations.items():
        max_rel_errors = []
        max_ulp_errors = []
        rmses = []
        passed_count = 0

        for i in range(n_runs):
            inp = input_generator()
            actual = fn(inp)
            ref = reference_fn(inp)

            report = analyze_precision(
                actual, ref, name=f"{name}_run{i}", rtol=rtol, atol=atol, verbose=False
            )

            max_rel_errors.append(report.max_rel_error)
            max_ulp_errors.append(float(report.max_ulp_error))
            rmses.append(report.rmse)

            if report.passed:
                passed_count += 1

        results[name] = {
            "mean_rel_error": float(np.mean(max_rel_errors)),
            "max_rel_error": float(np.max(max_rel_errors)),
            "min_rel_error": float(np.min(max_rel_errors)),
            "mean_ulp_error": float(np.mean(max_ulp_errors)),
            "mean_rmse": float(np.mean(rmses)),
            "pass_rate": passed_count / n_runs,
            "n_runs": n_runs,
        }

    return results


# ============================================================================
# Kahan 补偿求和
# ============================================================================


def kahan_sum_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Kahan 补偿求和算法（CPU 版本）。

    算法原理:
        维护一个"补偿项" c，追踪每次加法中丢失的低位信息。
        在下一次加法中，将丢失的低位加回来（从输入中减去补偿项再相加）。

        伪代码:
            s = 0.0; c = 0.0
            for x in input:
                y = x - c          # 修正输入：补偿上次丢失的低位
                t = s + y          # 临时相加
                c = (t - s) - y    # 恢复丢失的低位（精确的！）
                s = t

    为什么 (t - s) - y 是精确的?
        在浮点加法 s+y 中，由于指数对齐，低位可能被丢弃。
        但是 (t - s) 会精确地恢复 s 和 t 之间的差值。
        (t - s) - y 就是丢失的那部分低位——这是 Kahan 的巧妙之处。

    精度: O(1) 截断误差，与 N 无关。
    代价: 每次加法多 4 次浮点操作。

    Args:
        tensor: 输入 tensor (任意形状)

    Returns:
        补偿求和的结果 (标量 tensor)
    """
    s = 0.0
    c = 0.0
    for x in tensor.flatten().tolist():
        y = float(x) - c
        t = s + y
        c = (t - s) - y
        s = t
    return torch.tensor(s)


def kahan_sum_torch(tensor: torch.Tensor) -> torch.Tensor:
    """
    Kahan 补偿求和算法（PyTorch 向量化版本）。

    使用 PyTorch 的向量化操作实现 Kahan 求和。
    注意：这个实现是"伪向量化"的——实际上仍是顺序处理每个元素，
    因为 Kahan 算法本质上是顺序的（补偿项必须从一个迭代传递到下一个）。

    Args:
        tensor: 输入 tensor

    Returns:
        补偿求和的结果
    """
    s = torch.tensor(0.0, dtype=torch.float64)
    c = torch.tensor(0.0, dtype=torch.float64)
    flat = tensor.flatten().double()

    for i in range(flat.numel()):
        x = flat[i]
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t

    return s.float()


# ============================================================================
# 求和精度对比
# ============================================================================


def compare_summation_methods(
    n: int = 1000000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    对比多种求和方法的数值精度。

    构造一个"最坏情况"的输入：大量极小值 + 少量极大值。
    这种分布对 naive 求和最具挑战性——大值会淹没小值。

    Args:
        n: 输入元素数
        seed: 随机种子
        verbose: 是否打印结果

    Returns:
        {方法名: 相对于 fp64 的绝对误差} 字典
    """
    torch.manual_seed(seed)

    # 构造最坏情况：大量小值 + 少量大值
    x_small = torch.randn(n - 10, dtype=torch.float64) * 1e-8
    x_large = torch.randn(10, dtype=torch.float64) * 1e8
    x = torch.cat([x_small, x_large])[torch.randperm(n)]  # 打乱顺序

    # fp64 参考值
    ref = x.sum()

    # 1. 顺序累加 (naive)
    naive = x.float().sum()

    # 2. 从小到大排序后累加
    x_f32 = x.float()
    sorted_x = x_f32.sort().values
    sorted_sum = sorted_x.sum()

    # 3. 从大到小排序后累加
    sorted_desc_sum = x_f32.sort(descending=True).values.sum()

    # 4. Kahan 补偿求和
    kahan = kahan_sum_cpu(x_f32)

    # 5. 分层归约（用两次 sum 模拟——GPU 的树形 reduction 本质上是 pairwise）
    half = n // 2
    pairwise_a = x_f32[:half].sum()
    pairwise_b = x_f32[half:].sum()
    pairwise_sum = pairwise_a + pairwise_b

    errors = {
        "naive (顺序累加)": abs(naive.float() - ref.float()).item(),
        "sorted (从小到大)": abs(sorted_sum.float() - ref.float()).item(),
        "sorted (从大到小)": abs(sorted_desc_sum.float() - ref.float()).item(),
        "Kahan 补偿求和": abs(kahan.float() - ref.float()).item(),
        "pairwise (分层归约)": abs(pairwise_sum.float() - ref.float()).item(),
    }

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  求和精度对比 (N={n:,})")
        print(f"{'=' * 55}")
        print(f"  fp64 参考值:     {ref.item():.15e}")
        print(f"{'-' * 45}")
        for method, err in errors.items():
            rel = err / (abs(ref.item()) + 1e-30)
            print(f"  {method:<25s} 绝对误差={err:.2e}  相对误差={rel:.2e}")
        print(f"{'=' * 55}\n")

    return errors


# ============================================================================
# Reduction 精度随维度变化曲线
# ============================================================================


def benchmark_reduction_precision(
    dims: Optional[List[int]] = None,
    n_samples: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    测量 reduction 精度随维度增长的变化。

    对不同的内积维度 K，测量 fp32 和 fp16 累加下的 reduction 误差。
    误差以"每维度归一化误差"（error/dim）的形式报告，便于跨维度比较。

    Args:
        dims: 要测试的维度列表，默认 [64, 128, ..., 65536]
        n_samples: 每个维度的测试样本数
        verbose: 是否打印结果表

    Returns:
        {dtype: {dim: error}} 字典
    """
    if dims is None:
        dims = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    result = {"fp32": {}, "fp16": {}, "bf16": {}}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Reduction 精度随维度变化 (N_samples={n_samples})")
        print(f"{'=' * 70}")
        print(f"  {'Dim':>8}   {'fp32 err/dim':>14}   {'fp16 err/dim':>14}   {'bf16 err/dim':>14}")
        print(f"  {'-' * 8}   {'-' * 14}   {'-' * 14}   {'-' * 14}")

    for d in dims:
        fp32_errors = []
        fp16_errors = []
        bf16_errors = []

        for _ in range(n_samples):
            # 生成随机输入
            x = torch.randn(100, d, dtype=torch.float32)
            ref = x.double().sum(dim=-1)  # fp64 参考

            # fp32 累加，fp32 输入
            fp32 = x.sum(dim=-1)
            fp32_errors.append((fp32 - ref.float()).abs().max().item() / d)

            # fp16 输入，直接 sum（自动提升到 fp32 累加？不一定！）
            # 需要显式在 fp16 下做 sum——使用 fp16 intermediate
            x_fp16 = x.half()
            # PyTorch 的 sum 会自动在 fp32 下累加
            # 为了模拟纯 fp16 累加，需要手动实现
            sum_fp16 = torch.zeros(x_fp16.shape[0], dtype=torch.float16)
            for i in range(x_fp16.shape[1]):
                sum_fp16 += x_fp16[:, i]
            fp16_errors.append((sum_fp16.float() - ref.float()).abs().max().item() / d)

            # bf16 累加
            x_bf16 = x.bfloat16()
            sum_bf16 = torch.zeros(x_bf16.shape[0], dtype=torch.bfloat16)
            for i in range(x_bf16.shape[1]):
                sum_bf16 += x_bf16[:, i]
            bf16_errors.append((sum_bf16.float() - ref.float()).abs().max().item() / d)

        result["fp32"][d] = float(np.mean(fp32_errors))
        result["fp16"][d] = float(np.mean(fp16_errors))
        result["bf16"][d] = float(np.mean(bf16_errors))

        if verbose:
            print(
                f"  {d:>8}    {result['fp32'][d]:>14.4e}    {result['fp16'][d]:>14.4e}    {result['bf16'][d]:>14.4e}"
            )

    if verbose:
        print(f"{'=' * 70}\n")

    return result


# ============================================================================
# 浮点基础性质演示
# ============================================================================


def demonstrate_float_properties(verbose: bool = True) -> None:
    """
    演示浮点数的基础性质：
    - fp16/fp32/bf16 的范围和精度
    - 溢出行为
    - 灾难性抵消
    - 非结合性
    - subnormal 行为
    """
    if not verbose:
        return

    print(f"\n{'=' * 60}")
    print(f"  浮点精度基础性质演示")
    print(f"{'=' * 60}\n")

    # --- 1. 各精度的范围 ---
    print("--- 1. 各精度范围 ---")
    for name, dtype in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        info = torch.finfo(dtype)
        decimal_digits = int(-torch.log10(torch.tensor(info.eps)).item())
        print(
            f"  {name}: min={info.min:.4e}, max={info.max:.4e}, eps={info.eps:.4e}, "
            f"tiny={info.tiny:.4e}, 十进制位数≈{decimal_digits}"
        )

    # --- 2. fp16 溢出演示 ---
    print("\n--- 2. fp16 溢出演示 ---")
    fp16_max = torch.tensor(65504.0, dtype=torch.float16)
    print(f"  fp16 最大值: {fp16_max.item()}")
    x = torch.tensor(65504.0, dtype=torch.float16)
    y = torch.tensor(10.0, dtype=torch.float16)
    overflow_result = (x.float() + y.float()).half()
    print(f"  fp16: 65504 + 10 = {overflow_result.item()}  → 溢出!")

    x2 = torch.tensor(12.0, dtype=torch.float16)
    exp_result = torch.exp(x2.float()).half()  # exp(12) ≈ 162754 > 65504
    print(
        f"  fp16: exp(12.0) = {exp_result.item()}  → 溢出! (正确值 ≈ {torch.tensor(12.0).exp().item():.1f})"
    )

    # --- 3. bf16 vs fp16 范围对比 ---
    print("\n--- 3. bf16 vs fp16 范围对比 ---")
    large_val = torch.tensor(100000.0)
    print(f"  bf16 可表示 {large_val.bfloat16().item()}  (不会溢出)")
    try:
        print(f"  fp16 可表示 {large_val.half().item()}  (溢出!)")
    except Exception:
        print(f"  fp16 无法表示 {100000.0}  (溢出!)")

    # --- 4. 浮点加法非结合性 ---
    print("\n--- 4. 浮点加法非结合性 ---")
    a = torch.tensor(1e10, dtype=torch.float32)
    b = torch.tensor(-1e10, dtype=torch.float32)
    c = torch.tensor(1.0, dtype=torch.float32)
    result1 = (a + b) + c
    result2 = a + (b + c)
    print(f"  a={a.item():.0f}, b={b.item():.0f}, c={c.item()}")
    print(f"  (a+b)+c = {result1.item():.6f}  {'✓' if abs(result1.item() - 1.0) < 0.1 else '✗'}")
    print(f"  a+(b+c) = {result2.item():.6f}  {'✓' if abs(result2.item() - 1.0) < 0.1 else '✗'}")
    print(f"  正确值   = 1.0")

    # --- 5. 灾难性抵消 ---
    print("\n--- 5. 灾难性抵消 (catastrophic cancellation) ---")
    x_fp16 = torch.tensor(1.0, dtype=torch.float16)
    large_fp16 = torch.tensor(65504.0, dtype=torch.float16)
    cancelled = (large_fp16 + x_fp16) - large_fp16
    print(f"  fp16: (65504 + 1) - 65504 = {cancelled.item()}  (应为 1.0)")
    print(f"  fp32: {(large_fp16.float() + x_fp16.float() - large_fp16.float()).item():.1f}")

    # --- 6. 分配律不成立 ---
    print("\n--- 6. 分配律不成立: (x+y)*z ≠ x*z + y*z ---")
    x_val, y_val, z_val = 0.1, 0.2, 0.3
    method1 = (torch.tensor(x_val) + torch.tensor(y_val)) * torch.tensor(z_val)
    method2 = torch.tensor(x_val) * torch.tensor(z_val) + torch.tensor(y_val) * torch.tensor(z_val)
    print(f"  ({x_val}+{y_val})*{z_val} = {method1.item():.18f}")
    print(f"  {x_val}*{z_val}+{y_val}*{z_val} = {method2.item():.18f}  ← 更精确")
    print(f"  正确值 = {(x_val + y_val) * z_val:.18f}")

    # --- 7. subnormal 数 ---
    print("\n--- 7. Subnormal 数 ---")
    fp32_tiny = torch.finfo(torch.float32).tiny
    print(f"  fp32 最小正规数: {fp32_tiny:.4e}")
    below_normal = torch.tensor(fp32_tiny / 2.0)
    print(f"  fp32: {fp32_tiny:.4e} / 2 = {below_normal.item():.15e}  (subnormal)")
    print(f"  fp16 最小正规数: {torch.finfo(torch.float16).tiny:.4e}")

    # --- 8. 机器精度演示 ---
    print("\n--- 8. 机器精度演示 ---")
    for name, dtype in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        eps = torch.finfo(dtype).eps
        val = torch.tensor(1.0, dtype=dtype)
        val_plus = val + torch.tensor(eps / 2, dtype=dtype)
        print(
            f"  {name}: 1.0 + ε/2 = {val_plus.float().item():.10f}  {'≠ 1' if val_plus.float().item() != 1.0 else '= 1 (无法区分!)'}"
        )

    print(f"\n{'=' * 60}\n")


# ============================================================================
# 快速测试
# ============================================================================


def run_self_test() -> bool:
    """
    模块自测：验证所有工具函数正常工作。
    """
    print("运行精度工具自测...")

    # 测试 ULP 计算
    a = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0])
    b = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0])
    ulp = compute_ulp_error(a, b)
    assert (ulp == 0).all(), f"相同值应 ULP=0, got {ulp}"

    # 有微小差异
    a2 = torch.tensor([1.0])
    b2 = torch.tensor([1.0 + 1e-7])
    ulp2 = compute_ulp_error(a2, b2)
    assert ulp2.item() > 0, "不同值应 ULP>0"

    # 测试 PrecisionReport
    x = torch.randn(1000)
    y = x + torch.randn(1000) * 1e-6
    report = analyze_precision(y, x, name="self_test", verbose=False)
    assert report.max_abs_error > 0
    assert report.rmse > 0
    assert report.num_elements == 1000

    # 测试 Kahan sum
    x = torch.randn(10000) * 1e-6
    x = torch.cat([x, torch.tensor([1e10])])
    naive = x.float().sum()
    kahan = kahan_sum_cpu(x.float())
    ref = x.double().sum()
    # Kahan 应该比 naive 更接近 reference
    err_naive = abs(naive - ref.float()).item()
    err_kahan = abs(kahan - ref.float()).item()
    # Kahan 应该显著优于 naive
    assert err_kahan <= err_naive * 1.1, (
        f"Kahan ({err_kahan:.2e}) should not be significantly worse than naive ({err_naive:.2e})"
    )

    print("  所有自测通过!\n")
    return True


# ============================================================================
# 独立运行入口
# ============================================================================

if __name__ == "__main__":
    demonstrate_float_properties()
    compare_summation_methods(n=500000)
    benchmark_reduction_precision(dims=[256, 1024, 4096, 16384])
    run_self_test()
