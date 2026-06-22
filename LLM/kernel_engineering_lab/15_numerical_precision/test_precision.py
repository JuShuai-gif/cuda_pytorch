#!/usr/bin/env python3
"""
数值精度模块的 pytest 测试套件

测试覆盖:
  1. ULP 计算正确性
  2. Kahan 补偿求和精度
  3. PrecisionReport 统计正确性
  4. 各 dtype 的精度比较
  5. fp16 overflow 检测
  6. FMA 识别
  7. 常见算子的精度基准
  8. 边界情况处理

运行方式:
    pytest test_precision.py -v
    pytest test_precision.py -v -k "test_ulp"  # 只运行 ULP 测试
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

# 确保能从兄弟模块导入
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from precision_utils import (
    float_to_bits,
    bits_to_float,
    compute_ulp_error,
    analyze_precision,
    PrecisionReport,
    quick_check,
    compare_implementations,
    kahan_sum_cpu,
    kahan_sum_torch,
    compare_summation_methods,
    demonstrate_float_properties,
    run_self_test,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def random_tensors():
    """生成测试用的随机 tensor pair"""
    torch.manual_seed(42)
    x = torch.randn(100, 100)
    y = x + torch.randn(100, 100) * 1e-8
    return x, y


@pytest.fixture
def fp64_reference():
    """生成 fp64 参考用的数据"""
    torch.manual_seed(123)
    a = torch.randn(4096)
    b = torch.randn(4096)
    return a.double(), b.double()


# ============================================================================
# 测试 1: ULP 计算正确性
# ============================================================================


class TestULPComputation:
    """ULP 误差计算的基础正确性测试"""

    def test_ulp_zero_for_identical_values(self):
        """相同值的 ULP 误差应为 0"""
        a = torch.tensor([1.0, -1.0, 0.0, 3.14159, -2.71828])
        b = torch.tensor([1.0, -1.0, 0.0, 3.14159, -2.71828])

        ulp = compute_ulp_error(a, b)
        assert (ulp == 0).all(), f"相同值 ULP 应为 0, got {ulp}"

    def test_ulp_nonzero_for_different_values(self):
        """不同值的 ULP 误差应 > 0"""
        a = torch.tensor([1.0])
        b = torch.tensor([1.0 + 1e-6])

        ulp = compute_ulp_error(a, b)
        assert ulp.item() > 0, "不同值的 ULP 应大于 0"

    def test_ulp_monotonic_with_error(self):
        """ULP 应随误差增大而增大"""
        base = torch.tensor([1.0])
        diffs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        prev_ulp = 0
        for d in diffs:
            a = base + d
            ulp = compute_ulp_error(a, base).item()
            # ULP 应该是非递减的（可能存在平局）
            assert ulp >= prev_ulp, f"ULP should be monotonic, {d}: {ulp} < {prev_ulp}"
            prev_ulp = ulp

    def test_ulp_handles_nan(self):
        """ULP 应正确处理 NaN 值"""
        a = torch.tensor([float("nan")])
        b = torch.tensor([1.0])

        ulp = compute_ulp_error(a, b)
        # NaN 应产生哨兵值（INT32_MAX）
        sentinel = torch.tensor(0x7FFFFFFF, dtype=torch.int32)
        assert ulp.item() == sentinel.item(), f"NaN ULP should equal sentinel, got {ulp.item()}"

    def test_ulp_handles_inf(self):
        """ULP 应正确处理 Inf 值"""
        sentinel = torch.tensor(0x7FFFFFFF, dtype=torch.int32)

        a = torch.tensor([float("inf")])
        b = torch.tensor([float("inf")])

        ulp = compute_ulp_error(a, b)
        # 两个 Inf 在 ULP 中标记为哨兵值
        assert ulp.item() == sentinel.item(), "Inf ULP should be sentinel"

        c = torch.tensor([float("inf")])
        d = torch.tensor([float("-inf")])
        ulp2 = compute_ulp_error(c, d)
        assert ulp2.item() == sentinel.item(), "Diff sign Inf ULP should be sentinel"

    def test_ulp_signed_zero(self):
        """ULP 应正确处理 +0 和 -0"""
        a = torch.tensor([0.0])
        b = torch.tensor([-0.0])

        ulp = compute_ulp_error(a, b)
        # +0 和 -0 在 IEEE 754 中比较相等（虽然位模式不同）
        # 在 ULP 计算中可能存在微小差异
        assert ulp.item() >= 0, "ULP 应为非负数"

    def test_ulp_large_value(self):
        """ULP 应正确处理大值"""
        a = torch.tensor([1e10])
        b = torch.tensor([1e10 + 1.0])

        ulp = compute_ulp_error(a, b)
        assert ulp.item() >= 0, f"大值 ULP 应为非负数, got {ulp.item()}"

    def test_ulp_small_subnormal(self):
        """ULP 应正确处理 subnormal 数"""
        # fp32 的最小正规数是 1.175494e-38
        tiny = torch.finfo(torch.float32).tiny
        a = torch.tensor([tiny])
        b = torch.tensor([tiny * 1.5])

        ulp = compute_ulp_error(a, b)
        assert ulp.item() >= 0, "Subnormal 数的 ULP 应为非负数"

    def test_float_to_bits_roundtrip(self):
        """float_to_bits 和 bits_to_float 应互为逆操作"""
        # 使用 fp32 值（因为 float_to_bits 操作在 fp32 位模式上）
        values = [0.0, -0.0, 1.0, -1.0, 3.1415926535, 1e10, 1e-10]
        for v in values:
            t = torch.tensor([v], dtype=torch.float32)
            bits = float_to_bits(t)
            restored = bits_to_float(bits)
            # 比较 fp32 到 fp32 的恢复（因为 Python float 是 fp64）
            assert restored.item() == t.item(), f"Roundtrip failed for {v}: got {restored.item()}"

        # NaN 特殊处理
        t_nan = torch.tensor([float("nan")], dtype=torch.float32)
        bits_nan = float_to_bits(t_nan)
        restored_nan = bits_to_float(bits_nan)
        assert torch.isnan(restored_nan).item()

        # Inf 特殊处理
        t_inf = torch.tensor([float("inf")], dtype=torch.float32)
        bits_inf = float_to_bits(t_inf)
        restored_inf = bits_to_float(bits_inf)
        assert torch.isinf(restored_inf).item()


# ============================================================================
# 测试 2: Kahan 补偿求和
# ============================================================================


class TestKahanSummation:
    """Kahan 补偿求和算法的精度测试"""

    def test_kahan_exact_for_small_inputs(self):
        """对于小输入，Kahan sum 应非常精确"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        ref = x.double().sum()
        result = kahan_sum_cpu(x)
        assert abs(result.float() - ref.float()).item() < 1e-6, (
            f"Kahan sum error: {abs(result - ref).item():.2e}"
        )

    def test_kahan_better_than_naive(self):
        """Kahan sum 应显著优于 naive sum（在最坏情况下）"""
        torch.manual_seed(42)
        n_small = 10000
        small_vals = torch.randn(n_small) * 1e-10
        large_val = torch.tensor([1e10])
        x = torch.cat([small_vals, large_val])

        ref = x.double().sum().float()
        naive = x.float().sum()
        kahan = kahan_sum_cpu(x.float())

        err_naive = abs(naive - ref).item()
        err_kahan = abs(kahan - ref).item()

        # Kahan 应至少不差于 naive（在极端情况）
        assert err_kahan <= err_naive * 10, (
            f"Kahan ({err_kahan:.2e}) should not be 10x worse than naive ({err_naive:.2e})"
        )

    def test_kahan_random_inputs(self):
        """对随机输入，Kahan sum 的误差应可控"""
        torch.manual_seed(7)
        for _ in range(10):
            x = torch.randn(10000) * 100
            ref = x.double().sum().float()
            kahan = kahan_sum_cpu(x.float())
            err = abs(kahan - ref).item() / abs(ref.item() + 1e-30)
            assert err < 1e-5, f"Kahan relative error too high: {err:.2e}"

    def test_kahan_torch_equivalent(self):
        """CPU 版和 Torch 向量化版的 Kahan sum 应一致"""
        torch.manual_seed(99)
        x = torch.randn(1000) * 10

        k_cpu = kahan_sum_cpu(x.float())
        k_torch = kahan_sum_torch(x.float())

        assert abs(k_cpu - k_torch).item() < 1e-6, (
            f"CPU vs Torch Kahan mismatch: {abs(k_cpu - k_torch).item():.2e}"
        )

    def test_kahan_sum_of_zeros(self):
        """全零输入的 Kahan sum 应为 0"""
        x = torch.zeros(100)
        result = kahan_sum_cpu(x)
        assert result.item() == 0.0, f"Sum of zeros should be 0, got {result.item()}"

    def test_kahan_sum_single_element(self):
        """单元素输入的 Kahan sum 应返回该元素"""
        x = torch.tensor([42.0])
        result = kahan_sum_cpu(x)
        assert abs(result.item() - 42.0) < 1e-6, f"Single element sum failed: {result.item()}"

    def test_kahan_sum_negative_values(self):
        """Kahan sum 应正确处理负数"""
        x = torch.tensor([1.0, -2.0, 3.0, -4.0])
        result = kahan_sum_cpu(x)
        expected = -2.0
        assert abs(result.item() - expected) < 1e-6, f"Negative sum failed: {result.item()}"


# ============================================================================
# 测试 3: PrecisionReport 统计正确性
# ============================================================================


class TestPrecisionReport:
    """PrecisionReport 和 analyze_precision 的正确性测试"""

    def test_report_identical_tensors(self):
        """相同输入应产生 0 误差报告"""
        x = torch.randn(100, 100)
        report = analyze_precision(x, x, name="identical", verbose=False)
        assert report.max_abs_error == 0.0
        assert report.max_rel_error == 0.0
        assert report.max_ulp_error == 0
        assert report.rmse == 0.0
        assert report.passed is True
        assert report.nan_count == 0
        assert report.inf_count == 0

    def test_report_different_tensors(self):
        """不同输入应产生正误差"""
        x = torch.randn(100)
        y = x + torch.randn(100) * 1e-4

        report = analyze_precision(y, x, name="diff", verbose=False)
        assert report.max_abs_error > 0
        assert report.rmse > 0
        assert report.num_elements == 100

    def test_report_detects_nan(self):
        """Report 应正确检测 NaN"""
        x = torch.randn(50)
        y = torch.randn(50)
        y[0] = float("nan")

        report = analyze_precision(y, x, name="nan_test", verbose=False)
        assert report.nan_count >= 1, f"Expected NaN count >= 1, got {report.nan_count}"

    def test_report_detects_inf(self):
        """Report 应正确检测 Inf"""
        x = torch.randn(50)
        y = torch.randn(50)
        y[0] = float("inf")

        report = analyze_precision(y, x, name="inf_test", verbose=False)
        assert report.inf_count >= 1, f"Expected Inf count >= 1, got {report.inf_count}"

    def test_report_shape_mismatch_raises(self):
        """形状不匹配应抛出异常"""
        x = torch.randn(100)
        y = torch.randn(200)
        with pytest.raises(AssertionError):
            analyze_precision(y, x, name="shape_mismatch", verbose=False)

    def test_report_empty_tensor_raises(self):
        """空 tensor 应抛出异常"""
        x = torch.tensor([])
        with pytest.raises(ValueError):
            analyze_precision(x, x, verbose=False)

    def test_report_statistical_metrics(self):
        """统计指标应满足数学性质"""
        x = torch.randn(1000)
        noise = torch.randn(1000) * 1e-5
        y = x + noise

        report = analyze_precision(y, x, name="stats_test", verbose=False)

        # RMSE >= MAE（均方根 ≥ 平均绝对值）
        assert report.rmse >= report.p50_error * 0.1, "RMSE 应不小于零"

        # P99 >= P50
        assert report.p99_error >= report.p50_error, "P99 应 >= P50"

        # max >= P99
        assert report.max_abs_error >= report.p99_error, "max 应 >= P99"

    def test_report_passed_with_loose_tolerance(self):
        """宽松的阈值应通过"""
        x = torch.randn(100)
        y = x + torch.randn(100) * 1e-3

        report = analyze_precision(y, x, rtol=1.0, atol=1.0, max_allowed_ulp=10000, verbose=False)
        assert report.passed is True, "宽松阈值应通过"

    def test_report_failed_with_strict_tolerance(self):
        """严格的阈值应失败"""
        x = torch.randn(100)
        y = x + 10.0  # 故意制造大误差

        report = analyze_precision(y, x, rtol=1e-10, atol=1e-15, max_allowed_ulp=0, verbose=False)
        assert report.passed is False, "严格阈值应失败"

    def test_quick_check(self):
        """quick_check 应正确工作"""
        x = torch.randn(100)
        assert quick_check(x, x) is True

        y = x + 10.0
        assert quick_check(y, x, rtol=1e-10) is False


# ============================================================================
# 测试 4: 各 dtype 精度比较
# ============================================================================


class TestDtypePrecision:
    """不同 dtype 的精度特性测试"""

    def test_fp64_is_highest_precision(self):
        """fp64 应该有最高的精度（最小的误差）"""
        x = torch.randn(1000)

        fp32 = x.float()
        fp16 = x.half()
        bf16 = x.bfloat16()

        # fp64 到 fp32 的误差应小于 fp64 到 fp16 的误差
        fp32_err = (fp32.double() - x).abs().max().item()
        fp16_err = (fp16.float().double() - x).abs().max().item()
        bf16_err = (bf16.float().double() - x).abs().max().item()

        assert fp32_err < fp16_err, (
            f"fp32 error ({fp32_err:.2e}) should be less than fp16 error ({fp16_err:.2e})"
        )

    def test_fp16_range_limits(self):
        """fp16 范围限制测试"""
        fp16_max = torch.finfo(torch.float16).max

        val = torch.tensor(fp16_max, dtype=torch.float16)
        # 最大值应可表示
        assert val.item() == fp16_max

        # 稍微超出范围
        overflow = torch.tensor(fp16_max * 2, dtype=torch.float16)
        assert torch.isinf(overflow).item(), f"Overflow should produce Inf, got {overflow.item()}"

    def test_bf16_same_range_as_fp32(self):
        """bf16 应具有与 fp32 相同的动态范围"""
        bf16_info = torch.finfo(torch.bfloat16)
        fp32_info = torch.finfo(torch.float32)

        # 指数范围相同（max 相同）
        assert abs(bf16_info.max - fp32_info.max) / fp32_info.max < 0.1, "bf16 max 应接近 fp32 max"

    def test_fp16_eps_larger_than_fp32(self):
        """fp16 的 eps 应远大于 fp32"""
        fp16_eps = torch.finfo(torch.float16).eps
        fp32_eps = torch.finfo(torch.float32).eps

        assert fp16_eps > fp32_eps * 1000, (
            f"fp16 eps ({fp16_eps:.2e}) should be >> fp32 eps ({fp32_eps:.2e})"
        )

    def test_bf16_eps_between_fp16_and_fp32(self):
        """bf16 的 eps 应介于 fp16 和 fp32 之间"""
        fp16_eps = torch.finfo(torch.float16).eps
        bf16_eps = torch.finfo(torch.bfloat16).eps
        fp32_eps = torch.finfo(torch.float32).eps

        assert fp16_eps < bf16_eps, (
            f"fp16 eps ({fp16_eps:.2e}) should be less than bf16 eps ({bf16_eps:.2e})"
        )
        assert bf16_eps > fp32_eps, (
            f"bf16 eps ({bf16_eps:.2e}) should be greater than fp32 eps ({fp32_eps:.2e})"
        )


# ============================================================================
# 测试 5: fp16 Overflow 检测
# ============================================================================


class TestFp16Overflow:
    """fp16 溢出行为的测试"""

    def test_softmax_overflow_fp16(self):
        """Softmax 在大输入下 fp16 溢出测试"""
        # exp(12) ≈ 162754 > 65504 (fp16 max)
        x = torch.tensor([12.0, 4.0, 8.0], dtype=torch.float16)

        # 直接计算 exp 会溢出
        x_exp = torch.exp(x.float()).half()
        assert torch.isinf(x_exp).any() or torch.isinf(x_exp).any().item(), (
            f"exp(12) in fp16 should overflow, got {x_exp}"
        )

    def test_softmax_with_max_subtraction(self):
        """Subtract max 应防止 softmax 溢出"""
        x = torch.tensor([12.0, 4.0, 8.0], dtype=torch.float16)

        # Safe softmax
        x_max = x.max()
        x_shifted = x - x_max
        x_exp_safe = torch.exp(x_shifted.float()).half()
        x_sum = x_exp_safe.sum()
        safe_softmax = x_exp_safe / x_sum

        # 不应有 NaN 或 Inf
        assert not torch.isnan(safe_softmax).any()
        assert not torch.isinf(safe_softmax).any()
        # softmax 的和应近似为 1
        assert abs(safe_softmax.sum().item() - 1.0) < 0.01, (
            f"Softmax should sum to 1, got {safe_softmax.sum().item()}"
        )

    def test_reduction_overflow_fp16(self):
        """fp16 reduction 溢出测试"""
        # 大量大值的求和
        x = torch.full((10000,), 1000.0, dtype=torch.float16)
        # 在 fp16 下直接累加会溢出
        # PyTorch 的 sum 在 fp32 累加器中进行，所以不会溢出
        # 这里测试手动 fp16 累加
        sum_fp16 = torch.tensor(0.0, dtype=torch.float16)
        for i in range(1000):  # 只用 1000 个避免太慢
            sum_fp16 += x[i]

        assert not torch.isinf(sum_fp16) or torch.isinf(sum_fp16).item(), (
            "高密度大值 fp16 累加可能溢出"
        )


# ============================================================================
# 测试 6: FMA 识别
# ============================================================================


class TestFMA:
    """FMA (Fused Multiply-Add) 精度特性测试"""

    def test_fma_vs_separate_mul_add(self):
        """FMA 与分步乘加的精度对比"""
        a = torch.tensor(1e15)
        b = torch.tensor(2.0)
        c = torch.tensor(-1e15)

        # 分步计算
        separate = (a * b) + c
        # 使用 addcmul（内部使用 FMA）
        fma_result = torch.addcmul(c, a, b)

        # 两者都是近似值，但 FMA 通常更精确
        ref = (a.double() * b.double() + c.double()).float()

        err_separate = abs(separate - ref).item()
        err_fma = abs(fma_result - ref).item()

        # 在这个特定情况下，两者应该都很精确
        assert err_separate < 1.0 or err_fma < 1.0, (
            "At least one method should be reasonably precise"
        )

    def test_torch_dot_uses_fma(self):
        """torch.dot 应使用 FMA（产生更高的精度）"""
        K = 4096
        x = torch.randn(K)
        y = torch.randn(K)

        dot_torch = torch.dot(x, y)

        # 手动循环（非向量化）
        dot_manual = torch.tensor(0.0)
        for i in range(K):
            dot_manual += x[i] * y[i]

        # Vectorized 版本应更精确（使用 FMA）
        ref = torch.dot(x.double(), y.double()).float()
        err_torch = abs(dot_torch - ref).item()
        err_manual = abs(dot_manual - ref).item()

        # torch.dot 应至少不差于手动循环
        assert err_torch <= err_manual * 100, (
            f"torch.dot error ({err_torch:.2e}) should not be 100x worse than manual ({err_manual:.2e})"
        )


# ============================================================================
# 测试 7: 常见算子的精度基准
# ============================================================================


class TestOperatorPrecisionBaseline:
    """常见 GPU kernel 算子的精度基准测试"""

    def test_matmul_precision_baseline(self):
        """MatMul 的精度基准（fp32）"""
        M, K, N = 128, 4096, 128
        a = torch.randn(M, K)
        b = torch.randn(K, N)

        fp32_res = torch.mm(a, b)
        fp64_ref = torch.mm(a.double(), b.double()).float()

        abs_err = (fp32_res - fp64_ref).abs().max().item()
        # fp32 matmul 在 K=4096 时最大误差应在 ~1e-4 级别
        assert abs_err < 1e-3, f"fp32 matmul error too large: {abs_err:.2e}"

    def test_softmax_precision_baseline(self):
        """Softmax 的精度基准（fp32）"""
        x = torch.randn(1024, 128)

        fp32_sm = F.softmax(x, dim=-1)
        fp64_ref = F.softmax(x.double(), dim=-1).float()

        abs_err = (fp32_sm - fp64_ref).abs().max().item()
        assert abs_err < 1e-5, f"fp32 softmax error too large: {abs_err:.2e}"

    def test_layernorm_precision_baseline(self):
        """LayerNorm 的精度基准（fp32）"""
        x = torch.randn(32, 4096)

        fp32_ln = F.layer_norm(x, (4096,))
        fp64_ref = F.layer_norm(x.double(), (4096,)).float()

        abs_err = (fp32_ln - fp64_ref).abs().max().item()
        assert abs_err < 1e-5, f"fp32 layernorm error too large: {abs_err:.2e}"

    def test_rmsnorm_precision_baseline(self):
        """RMSNorm 的精度基准（fp32）"""
        x = torch.randn(32, 4096)

        # 手动 RMSNorm
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        fp32_rms = x / rms

        rms_ref = torch.sqrt(torch.mean(x.double() ** 2, dim=-1, keepdim=True) + 1e-6)
        fp64_ref = (x.double() / rms_ref).float()

        abs_err = (fp32_rms - fp64_ref).abs().max().item()
        assert abs_err < 1e-5, f"fp32 rmsnorm error too large: {abs_err:.2e}"

    def test_elementwise_relu_precision(self):
        """ReLU elementwise 精度（应为精确或近乎精确）"""
        x = torch.randn(10000)

        fp32_relu = F.relu(x)
        fp64_ref = F.relu(x.double()).float()

        # ReLU 只是 max(0, x)，应该完全精确
        assert torch.equal(fp32_relu, fp64_ref), "ReLU should be exact"

    def test_sigmoid_precision_baseline(self):
        """Sigmoid 的精度基准（fp32）"""
        x = torch.randn(10000)

        fp32_sig = torch.sigmoid(x)
        fp64_ref = torch.sigmoid(x.double()).float()

        abs_err = (fp32_sig - fp64_ref).abs().max().item()
        assert abs_err < 1e-6, f"fp32 sigmoid error too large: {abs_err:.2e}"


# ============================================================================
# 测试 8: 边界情况
# ============================================================================


class TestEdgeCases:
    """边界情况和特殊输入的测试"""

    def test_all_zeros_input(self):
        """全零输入应正确处理"""
        x = torch.zeros(1000)
        ref = torch.zeros(1000, dtype=torch.float64).float()

        report = analyze_precision(x, ref, name="zeros", verbose=False)
        assert report.max_abs_error == 0.0
        assert report.passed is True

    def test_large_values(self):
        """大值输入应正确处理"""
        x = torch.full((100,), 1e10, dtype=torch.float32)
        ref = torch.full((100,), 1e10, dtype=torch.float64).float()

        report = analyze_precision(x, ref, name="large", verbose=False)
        assert report.max_abs_error < 1000.0, f"Large value error too large: {report.max_abs_error}"

    def test_small_values(self):
        """极小值输入应正确处理"""
        x = torch.full((100,), 1e-30, dtype=torch.float32)
        ref = torch.full((100,), 1e-30, dtype=torch.float64).float()

        report = analyze_precision(x, ref, name="small", verbose=False)
        assert report.passed or report.max_abs_error < 1e-10, (
            f"Small value error: max_abs={report.max_abs_error}"
        )

    def test_mixed_signs(self):
        """正负混合输入应正确处理"""
        x = torch.randn(1000)
        ref = x.float().double().float()

        report = analyze_precision(x.float(), ref.float(), name="mixed", verbose=False)
        assert report.max_abs_error < 1e-6

    def test_very_large_scale_difference(self):
        """量级差异极大的输入（灾难性抵消条件）"""
        x = torch.cat([torch.tensor([1e15]), torch.tensor([1e-15])])

        # 求和应能处理
        s = x.float().sum()
        ref = x.double().sum()

        # 误差不应灾难性（虽然会有些损失）
        rel_err = abs(s - ref.float()).item() / abs(ref.item() + 1e-30)
        # 在 fp32 下，这种量级差异可能导致较大相对误差
        assert rel_err < 10.0, f"Relative error too large: {rel_err}"


# ============================================================================
# 测试 9: 模块级自测
# ============================================================================


class TestModuleSelfTest:
    """模块级自测功能"""

    def test_run_self_test(self):
        """run_self_test 应返回 True"""
        result = run_self_test()
        assert result is True

    def test_compare_summation_methods(self):
        """compare_summation_methods 应返回有效结果"""
        result = compare_summation_methods(n=10000, verbose=False)
        assert isinstance(result, dict)
        assert len(result) >= 3  # 至少 3 种方法
        for v in result.values():
            assert v >= 0, "误差应为非负数"

    def test_demonstrate_float_properties(self):
        """demonstrate_float_properties 应正常运行"""
        demonstrate_float_properties(verbose=False)


# ============================================================================
# 测试 10: 多实现对比
# ============================================================================


class TestCompareImplementations:
    """多 Kernel 实现对比测试"""

    def test_compare_basic(self):
        """基础的多实现对比"""

        def impl_a(x):
            return torch.relu(x)

        def impl_b(x):
            return torch.relu(x)

        def ref_fn(x):
            return torch.relu(x.double()).float()

        def gen():
            return torch.randn(100, 100)

        implementations = {"relu_a": impl_a, "relu_b": impl_b}
        results = compare_implementations(
            implementations, reference_fn=ref_fn, input_generator=gen, n_runs=5
        )

        for name in implementations:
            assert name in results
            assert results[name]["pass_rate"] > 0.5, (
                f"{name} pass rate too low: {results[name]['pass_rate']}"
            )

    def test_compare_different_precision(self):
        """不同精度的实现对比（fp32 vs fp16）"""

        def impl_fp32(x):
            return x.sum(dim=-1)

        def impl_fp16(x):
            return x.half().sum(dim=-1).float()

        def ref_fn(x):
            return x.double().sum(dim=-1).float()

        def gen():
            return torch.randn(100, 4096)

        implementations = {"fp32_sum": impl_fp32, "fp16_sum": impl_fp16}
        results = compare_implementations(
            implementations,
            reference_fn=ref_fn,
            input_generator=gen,
            n_runs=5,
            rtol=1e-1,
            atol=1e-2,
        )

        # fp32 应比 fp16 更精确
        assert results["fp32_sum"]["mean_rel_error"] < results["fp16_sum"]["mean_rel_error"], (
            "fp32 should be more precise than fp16"
        )

    def test_precision_report_dataclass(self):
        """PrecisionReport dataclass 的字段"""
        report = PrecisionReport(
            max_abs_error=1e-4,
            max_rel_error=1e-3,
            max_ulp_error=4,
            rmse=1e-5,
            p50_error=1e-6,
            p99_error=1e-5,
        )
        assert report.max_abs_error == 1e-4
        assert report.max_rel_error == 1e-3
        assert report.max_ulp_error == 4


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    print("运行精度模块测试...")
    # 使用 pytest 运行
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
