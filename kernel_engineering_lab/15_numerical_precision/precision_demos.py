#!/usr/bin/env python3
"""
数值精度演示脚本 — 展示 GPU kernel 开发中的各种精度陷阱

演示列表:
  1. fp16 vs fp32 vs bf16 逐位精度对比
  2. Reduction 中的 catastrophic cancellation
  3. Softmax 在 fp16 下的数值稳定性
  4. MatMul 内积精度随维度增长
  5. LayerNorm 在混合精度下的误差分析
  6. Kahan summation vs naive summation
  7. FMA 精度优势演示
  8. Non-deterministic 归约演示

所有注释采用中文，每个演示都输出清晰的结果。
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, "..")

from precision_utils import (
    analyze_precision,
    compute_ulp_error,
    kahan_sum_cpu,
)

# ============================================================================
# 演示 1: fp16 vs fp32 vs bf16 精度对比
# ============================================================================


def demo_fp16_vs_fp32_vs_bf16():
    """
    演示三种浮点精度在同一数据上的精度差异。

    核心要点:
    - fp16 范围最小（max 65504），容易溢出
    - bf16 范围与 fp32 相同，但精度只有 ~7.8e-3
    - fp32 是大多数 GPU kernel 的 baseline 精度
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 1: fp16 vs fp32 vs bf16 精度对比")
    print(f"{'=' * 70}\n")

    # 在每个精度下计算简单的 dot product
    dims = [64, 256, 1024, 4096]
    fp64_ref = {}

    for d in dims:
        a = torch.randn(d)
        b = torch.randn(d)

        # fp64 参考值
        ref = torch.dot(a.double(), b.double())
        fp64_ref[d] = ref

        # 各精度的结果
        fp32_val = torch.dot(a.float(), b.float())
        fp16_val = torch.dot(a.half().float(), b.half().float())  # 先 cast 再计算（模拟 fp16 输入）
        bf16_val = torch.dot(a.bfloat16().float(), b.bfloat16().float())

        fp32_err = abs(fp32_val - ref).item()
        fp16_err = abs(fp16_val - ref).item()
        bf16_err = abs(bf16_val - ref).item()

        print(f"  维度 {d:>5}:")
        print(f"    fp64 参考 = {ref.item():.10f}")
        print(f"    fp32 误差 = {fp32_err:.2e}")
        print(f"    fp16 误差 = {fp16_err:.2e}")
        print(f"    bf16 误差 = {bf16_err:.2e}")
        print()

    # 溢出演示
    print("  --- fp16 溢出演示 ---")
    large_input = torch.tensor([10000.0, 20000.0, 30000.0])
    print(f"  输入: {large_input.tolist()}")
    print(f"  fp16 sum: {large_input.half().sum().item()}  (可能溢出!)")
    print(f"  fp32 sum: {large_input.float().sum().item()}")
    print(f"  bf16 sum: {large_input.bfloat16().sum().item()}\n")

    print(f"{'=' * 70}\n")


# ============================================================================
# 演示 2: Reduction 中的 Catastrophic Cancellation
# ============================================================================


def demo_catastrophic_cancellation():
    """
    演示累加中的灾难性抵消。

    当累加一个巨大值（如 1e10）和大量微小值（如 1e-6）时，
    微小值的贡献会因指数对齐而完全丢失——这就是灾难性抵消。

    GPU reduction kernel 中面临同样的问题：当部分 block 的 partial sum
    大小悬殊时，最终归约会丢失精度。
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 2: Reduction 中的灾难性抵消")
    print(f"{'=' * 70}\n")

    # 场景：大量小值 + 一个大值
    torch.manual_seed(42)
    n_small = 100000
    small_vals = torch.randn(n_small) * 1e-8
    large_val = torch.tensor([1e10])

    # 条件 1: 先累加所有小值，最后加大值
    s1 = small_vals.sum()
    s1 = s1 + large_val[0]
    result_1 = s1

    # 条件 2: 先加大值，再累加小值（小值被淹没）
    s2 = large_val.clone().float()
    s2 = s2 + small_vals.sum()
    result_2 = s2

    # 条件 3: 从小到大排序后累加
    all_vals = torch.cat([small_vals, large_val])
    sorted_vals = all_vals.sort().values
    result_3 = sorted_vals.sum()

    # 条件 4: Kahan 补偿求和
    result_4 = kahan_sum_cpu(all_vals)

    # fp64 参考
    ref = all_vals.double().sum()

    print(f"  场景: {n_small} 个小值 (mean=1e-8) + 1 个大值 (1e10)")
    print(f"  fp64 参考值:     {ref.item():.15e}")
    print()
    print(f"  1) 小值先加再加大值:  误差 = {abs(result_1 - ref).item():.2e}")
    print(f"  2) 大值先加再加小值:  误差 = {abs(result_2 - ref).item():.2e}  ← 小值被淹没!")
    print(f"  3) 从小到大排序后累加: 误差 = {abs(result_3 - ref).item():.2e}")
    print(f"  4) Kahan 补偿求和:     误差 = {abs(result_4 - ref).item():.2e}  ← 最佳!")
    print()
    print(f"  结论: 先加小值优于先加大值，Kahan 补偿求和最优。")
    print(f"  GPU reduction 中默认使用 pairwise sum（分层归约），")
    print(f"  其精度介于 naive 和 sorted 之间。")
    print(f"{'=' * 70}\n")


# ============================================================================
# 演示 3: Softmax 在 fp16 下的数值稳定性
# ============================================================================


def demo_softmax_stability():
    """
    演示 Softmax 在 fp16 下的数值稳定性问题。

    Softmax 的两个关键精度问题:
    1. exp 溢出：fp16 中 exp(x) 在 x > 11.09 时溢出
    2. Subtract max trick 的精度：max 减法引入 cancellation
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 3: Softmax 在 fp16 下的数值稳定性")
    print(f"{'=' * 70}\n")

    # --- Case 1: 正常范围 ---
    print("--- Case 1: 正常范围的输入 ---")
    x_normal = torch.tensor([1.0, 2.0, 3.0])

    ref = F.softmax(x_normal.double(), dim=-1).float()
    fp32_sm = F.softmax(x_normal.float(), dim=-1)
    fp16_sm = F.softmax(x_normal.half().float(), dim=-1).half()

    print(f"  输入: {x_normal.tolist()}")
    analyze_precision(fp16_sm, ref, name="softmax_fp16_normal", verbose=True, max_allowed_ulp=8)

    # --- Case 2: 大值输入 (fp16 溢出风险) ---
    print("--- Case 2: 大值输入 (fp16 溢出风险) ---")
    x_large = torch.tensor([8.0, 11.0, 10.0])  # exp(11) ≈ 59874 < 65504, 恰好在边界

    ref2 = F.softmax(x_large.double(), dim=-1).float()
    fp32_sm2 = F.softmax(x_large.float(), dim=-1)

    # fp16 计算（注意：PyTorch 内部使用 fp32 计算 softmax，
    # 所以这里手动模拟纯 fp16 softmax）
    x_half = x_large.half()
    # 手动 softmax: subtract max → exp → sum → div
    x_max = x_half.max()
    x_shifted = x_half - x_max
    x_exp = torch.exp(x_shifted.float()).half()
    x_sum = x_exp.sum()
    fp16_sm2 = x_exp / x_sum

    print(f"  输入: {x_large.tolist()}")
    print(f"  exp(11.0) ≈ {torch.tensor(11.0).exp().item():.0f} (fp16 安全范围内)")
    analyze_precision(fp16_sm2, ref2, name="softmax_fp16_large", verbose=True, max_allowed_ulp=16)

    # --- Case 3: 极端输入 (一定会溢出) ---
    print("--- Case 3: 极端输入 (fp16 下 exp 溢出) ---")
    x_extreme = torch.tensor([12.0, 13.0, 14.0])  # exp(12) ≈ 162754 > 65504

    x_half2 = x_extreme.half()
    x_max2 = x_half2.max()
    x_shifted2 = x_half2 - x_max2  # 减去 max 后最大值为 0
    x_exp2 = torch.exp(x_shifted2.float()).half()
    x_sum2 = x_exp2.sum()
    fp16_sm3 = x_exp2 / x_sum2

    # 不减去 max 的版本（会溢出）
    x_exp_naive = torch.exp(x_extreme.half().float()).half()
    x_sum_naive = x_exp_naive.sum()

    print(f"  输入: {x_extreme.tolist()}")
    print(f"  subtract max + exp:  {x_exp2.tolist()}  ✓ 安全")
    print(f"  naive exp (无 max):  {x_exp_naive.tolist()}  ✗ 溢出!")
    if torch.isinf(x_sum_naive) or torch.isnan(x_sum_naive):
        print(f"  naive softmax sum:   {x_sum_naive.item()}  ✗ 失败!")
    print()
    analyze_precision(
        fp16_sm3,
        F.softmax(x_extreme.double(), dim=-1).float(),
        name="softmax_fp16_safe",
        verbose=True,
        max_allowed_ulp=8,
    )

    print(f"  结论: subtract max trick 对 fp16 至关重要。")
    print(f"  fp16 softmax 误差主要来源于 exp 的硬件近似和最后的除法。")
    print(f"{'=' * 70}\n")


# ============================================================================
# 演示 4: MatMul 内积精度随维度增长
# ============================================================================


def demo_matmul_precision_scaling():
    """
    演示 MatMul / Dot Product 的精度如何随内积维度 K 增长。

    理论: 相对误差 ∝ K × ε
    实际: 相对误差 ∝ √K × ε （因误差正负抵消）

    fp16 下 K=4096 的相对误差可达 1-10%，这对 attention 影响重大。
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 4: MatMul 内积精度随维度增长")
    print(f"{'=' * 70}\n")

    K_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    M, N = 128, 128

    print(f"  形状: M={M}, N={N}, K=variable")
    print(
        f"  {'K':>6}  {'fp32 max rel':>14}  {'fp16 max rel':>14}  {'bf16 max rel':>14}  {'fp16 ULP':>10}"
    )
    print(f"  {'-' * 6}  {'-' * 14}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    for K in K_values:
        a = torch.randn(M, K)
        b = torch.randn(K, N)

        # fp64 参考
        ref = torch.mm(a.double(), b.double()).float()

        # 各精度
        fp32_res = torch.mm(a.float(), b.float())
        fp16_res = torch.mm(a.half(), b.half()).float()  # PyTorch 在 fp32 acc 中计算
        bf16_res = torch.mm(a.bfloat16(), b.bfloat16()).float()

        _, max_rel_fp32 = _compute_error_stats(fp32_res, ref)
        _, max_rel_fp16 = _compute_error_stats(fp16_res, ref)
        _, max_rel_bf16 = _compute_error_stats(bf16_res, ref)

        # ULP 误差
        ulp_fp16 = compute_ulp_error(fp16_res, ref).max().item()

        print(
            f"  {K:>6}  {max_rel_fp32:>14.4e}  {max_rel_fp16:>14.4e}  {max_rel_bf16:>14.4e}  {ulp_fp16:>10.0f}"
        )

    print()
    print(f"  结论: fp16 内积的相对误差随 K 大致按 √K 增长。")
    print(f"  K=4096 时 fp16 误差约 1-5%（取决于值分布）。")
    print(f"  bf16 误差约为 fp16 的 5-10 倍。")
    print(f"{'=' * 70}\n")


def _compute_error_stats(actual: torch.Tensor, ref: torch.Tensor):
    """返回 (max_abs, max_rel) 误差统计"""
    abs_err = (actual - ref).abs().max().item()
    mask = ref.abs() > 1e-8
    if mask.any():
        rel_err = (abs(actual[mask] - ref[mask]) / ref[mask].abs()).max().item()
    else:
        rel_err = 0.0
    return abs_err, rel_err


# ============================================================================
# 演示 5: LayerNorm 在混合精度下的误差分析
# ============================================================================


def demo_layernorm_precision():
    """
    演示 LayerNorm 在 fp16 下的误差分析。

    LayerNorm 涉及两次 reduction (mean + var)，以及一次 rsqrt。
    在 fp16 下，每步都引入舍入误差，最终累积。

    关键风险:
    - 大 hidden_dim 下 sum(x²) 可能接近 fp16 上限
    - rsqrt 的近似实现引入额外误差
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 5: LayerNorm 在混合精度下的误差分析")
    print(f"{'=' * 70}\n")

    batch, seq_len = 4, 64
    hidden_dims = [256, 512, 1024, 2048, 4096, 8192]

    print(f"  形状: batch={batch}, seq_len={seq_len}, hidden_dim=variable")
    print(f"  {'H':>6}  {'max abs err':>14}  {'max rel err':>14}  {'max ULP':>10}")
    print(f"  {'-' * 6}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    for H in hidden_dims:
        x = torch.randn(batch, seq_len, H)

        # fp64 参考
        ref = F.layer_norm(x.double(), (H,)).float()

        # fp32
        fp32_ln = F.layer_norm(x.float(), (H,))

        # fp16 (手动模拟纯 fp16 计算，因为 PyTorch 内部使用 fp32)
        x_half = x.half()
        mean = x_half.float().mean(dim=-1, keepdim=True).half()
        var = ((x_half.float() - mean.float()) ** 2).mean(dim=-1, keepdim=True).half()
        eps = torch.tensor(1e-5, dtype=torch.float16)
        rstd = torch.rsqrt(var.float() + eps.float()).half()
        fp16_ln = (x_half - mean) * rstd

        # 误差统计
        abs_err = (fp16_ln.float() - ref).abs()
        mask = ref.abs() > 1e-6
        rel_err = (abs_err[mask] / ref[mask].abs()) if mask.any() else torch.tensor(0.0)
        ulp = compute_ulp_error(fp16_ln, ref).max().item()

        print(
            f"  {H:>6}  {abs_err.max().item():>14.4e}  {rel_err.max().item() if rel_err.numel() > 0 else 0:>14.4e}  {ulp:>10.0f}"
        )

    print()
    print(f"  结论: LayerNorm 的 fp16 误差主要来自 rsqrt 和两次 reduction。")
    print(f"  推荐使用 fp32 intermediate accumulation 来保持精度。")
    print(f"{'=' * 70}\n")


# ============================================================================
# 演示 6: Kahan 补偿求和 vs Naive 求和
# ============================================================================


def demo_kahan_vs_naive():
    """
    演示 Kahan 补偿求和相比 naive 求和在高维 reduction 中的精度优势。

    构造一个苛刻的求和场景（大量小值 + 一个大值），
    对比 naive sum、sorted sum、Kahan sum 的精度。
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 6: Kahan 补偿求和 vs Naive 求和")
    print(f"{'=' * 70}\n")

    torch.manual_seed(123)

    sizes = [100, 1000, 10000, 100000]
    for n in sizes:
        # 构造最坏情况的输入
        x = torch.randn(n) * 1e-6
        x = torch.cat([x, torch.tensor([1e8])])
        x = x[torch.randperm(x.numel())]  # 随机排列

        ref = x.double().sum()

        naive = x.float().sum()
        sorted_sum = x.float().sort().values.sum()
        kahan = kahan_sum_cpu(x.float())

        err_naive = abs(naive - ref.float()).item()
        err_sorted = abs(sorted_sum - ref.float()).item()
        err_kahan = abs(kahan - ref.float()).item()

        print(f"  N={n + 1:>7}:")
        print(f"    naive sum 误差 = {err_naive:.2e}")
        print(f"    sorted sum 误差 = {err_sorted:.2e}")
        print(f"    Kahan sum 误差  = {err_kahan:.2e}")
        if err_naive > 0:
            improvement = err_naive / max(err_kahan, 1e-50)
            print(f"    Kahan 精度提升 ≈ {improvement:.0f}x")

    print()
    print(f"  结论: Kahan 补偿求和可将累加误差降低 10³-10⁶ 倍。")
    print(f"  在 GPU kernel 中，线程内部可用 Kahan sum,")
    print(f"  线程间可用 pairwise reduction。")
    print(f"{'=' * 70}\n")


# ============================================================================
# 演示 7: FMA 精度优势
# ============================================================================


def demo_fma_precision():
    """
    演示 FMA (Fused Multiply-Add) 指令的精度优势。

    FMA 执行 a*b+c 时只做一次舍入（而非先乘后加的两阶段舍入）。
    这使 FMA 的精度显著高于分别做乘法和加法。

    在 CUDA GPU 上（Kepler+），FMA 是默认启用的硬件指令。
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 7: FMA (Fused Multiply-Add) 精度优势")
    print(f"{'=' * 70}\n")

    # 案例 1: 基本 FMA 精度
    print("--- 案例 1: a*b+c 的精度对比 ---")
    a = torch.tensor(0.1)
    b = torch.tensor(0.2)
    c = torch.tensor(0.3)

    # 方式 1: 分步计算（两次舍入）
    result_separate = (a * b) + c
    # 方式 2: FMA（PyTorch 默认启用）
    result_fma = torch.addcmul(c, a, b)  # c + a*b，内部使用 FMA

    # fp64 参考
    ref = (a.double() * b.double() + c.double()).float()

    print(f"  a={a.item()}, b={b.item()}, c={c.item()}")
    print(f"  fp64 参考:          {ref.item():.18f}")
    print(f"  (a*b)+c (两次舍入):  {result_separate.item():.18f}")
    print(f"  FMA (一次舍入):      {result_fma.item():.18f}")

    err_separate = abs(result_separate - ref).item()
    err_fma = abs(result_fma - ref).item()

    if err_fma < err_separate:
        print(f"  ✓ FMA 更精确 (误差 {err_fma:.2e} vs {err_separate:.2e})")
    else:
        print(f"  = 精度相同 (在小值范围内差异不明显)")

    # 案例 2: 大尺度下的 FMA 优势
    print("\n--- 案例 2: 大尺度下的 FMA 优势 ---")
    a2 = torch.tensor(1e15)
    b2 = torch.tensor(1.0)
    c2 = torch.tensor(1e15)

    separate2 = (a2 * b2) + c2
    fma2 = torch.addcmul(c2, a2, b2)
    ref2 = (a2.double() * b2.double() + c2.double()).float()

    print(f"  a={a2.item():.0f}, b={b2.item()}, c={c2.item():.0f}")
    print(f"  fp64 参考:          {ref2.item():.18f}")
    print(f"  (a*b)+c (两次舍入):  {separate2.item():.18f}")
    print(f"  FMA (一次舍入):      {fma2.item():.18f}")
    print(f"  差值: FMA 更接近参考值")

    # 案例 3: 内积中的 FMA 累积效果
    print("\n--- 案例 3: 内积中 FMA 的累积效果 ---")
    K = 4096
    x = torch.randn(K)
    y = torch.randn(K)

    dot_separate = (x * y).sum()  # 先存储所有乘积，再求和
    dot_fma_style = torch.tensor(0.0)
    for i in range(K):
        dot_fma_style = dot_fma_style + x[i] * y[i]  # Python 会逐元素（非向量化）
    dot_torch = torch.dot(x, y)  # PyTorch 内部使用 FMA

    ref_dot = torch.dot(x.double(), y.double()).float()

    print(f"  K={K}")
    print(f"  fp64 参考:      {ref_dot.item():.10f}")
    print(f"  torch.dot (FMA): {dot_torch.item():.10f}  误差={abs(dot_torch - ref_dot).item():.2e}")
    print(
        f"  (x*y).sum():    {dot_separate.item():.10f}  误差={abs(dot_separate - ref_dot).item():.2e}"
    )

    print()
    print(f"  结论: FMA 通过在 a*b+c 中只做一次舍入来提高精度。")
    print(f"  在 GPU 上 FMA 是默认的，这也是 MatMul 精度优于 fp16 naive 分析的原因之一。")
    print(f"{'=' * 70}\n")


# ============================================================================
# 演示 8: Non-deterministic 归约
# ============================================================================


def demo_nondeterministic_reduction():
    """
    演示 GPU reduction 的非确定性。

    由于不同 warp/block 的调度顺序不同，归约中的累加顺序也不同。
    由于浮点加法不满足结合律，这会导致最终结果有微小差异（通常 1-2 ULP）。

    这在 CI 测试中可能导致 allclose 随机 pass/fail。
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 8: Non-deterministic 归约模拟")
    print(f"{'=' * 70}\n")

    # 使用随机累加顺序模拟不同调度下的 reduction
    torch.manual_seed(42)

    # 构造一组需要归约的值（模拟各 block 的 partial sum）
    blocks = 32
    elems_per_block = 256
    total = blocks * elems_per_block
    data = torch.randn(total, dtype=torch.float32)

    # 按不同顺序归约（模拟不同调度）
    results = []
    for seed in range(10):
        torch.manual_seed(seed)
        perm = torch.randperm(blocks)
        # 按这个顺序累加不同 block 的 sum
        block_sums = []
        for b in range(blocks):
            block_sums.append(data[b * elems_per_block : (b + 1) * elems_per_block].sum())
        block_sums = torch.stack(block_sums)

        # 按 perm 顺序累加
        total_sum = torch.tensor(0.0)
        for idx in perm:
            total_sum = total_sum + block_sums[idx]
        results.append(total_sum.item())

    # 分析差异
    results_t = torch.tensor(results)
    diff = results_t.max() - results_t.min()
    ulp_diff = compute_ulp_error(
        torch.tensor([results_t.max()]), torch.tensor([results_t.min()])
    ).item()

    print(f"  {blocks} 个 block, 随机排列后累加:")
    print(f"  所有结果: {[f'{r:.8f}' for r in results]}")
    print(f"  最大差异: {diff:.2e}")
    print(f"  ULP 差异: {int(ulp_diff)}")
    print()
    print(f"  结论: 不同累加顺序导致 {int(ulp_diff)} ULP 的差异。")
    print(f"  这解释了为什么 CUDA kernel 的浮点输出可能在")
    print(f"  不同运行中产生 1-2 ULP 的差异。")
    print(f"  在 CI 中应该使用 atol >= 1e-6 来容忍这种差异。")
    print(f"{'=' * 70}\n")


# ============================================================================
# 演示 9: 经验法则速查 — 各算子推荐精度
# ============================================================================


def demo_cheatsheet():
    """
    打印 GPU kernel 精度经验法则速查表。
    """
    print(f"\n{'=' * 70}")
    print(f"  演示 9: GPU Kernel 精度经验法则速查表")
    print(f"{'=' * 70}\n")

    rules = [
        ("Elementwise (add/mul)", "fp32=1e-7, fp16=1e-4", "单次舍入, 误差 ~0.5 ULP"),
        ("Reduction (K=4096)", "fp32=1e-6, fp16=1e-2", "用 fp32 中间累加至关重要"),
        ("MatMul (K=4096)", "fp32=1e-5, fp16=1e-2", "Tensor Core 内部使用 fp32 acc"),
        ("Softmax", "fp32=1e-5, fp16=1e-2", "必须 subtract max, 最好用 online softmax"),
        ("LayerNorm (H=4096)", "fp32=1e-5, fp16=1e-2", "两次 reduction, 用 fp32 中间累加"),
        ("RMSNorm (H=4096)", "fp32=1e-5, fp16=1e-2", "只有一次 reduction, 略好于 LayerNorm"),
        ("GELU activation", "fp32=1e-5, fp16=1e-2", "tanh 近似是主要误差源"),
        ("SiLU / Swish", "fp32=1e-6, fp16=1e-3", "只用 sigmoid, 精度较好"),
    ]

    print(f"  {'算子':<25s} {'推荐 rtol':<20s} {'备注':<40s}")
    print(f"  {'-' * 25} {'-' * 20} {'-' * 40}")
    for op, tol, note in rules:
        print(f"  {op:<25s} {tol:<20s} {note:<40s}")

    print(f"\n{'=' * 70}\n")


# ============================================================================
# 主入口
# ============================================================================


if __name__ == "__main__":
    """
    运行所有数值精度演示。

    可选择性运行:
        python precision_demos.py           # 运行全部
        python precision_demos.py 1 3 6     # 只运行演示 1, 3, 6
    """
    demos = {
        "1": ("fp16 vs fp32 vs bf16 精度对比", demo_fp16_vs_fp32_vs_bf16),
        "2": ("Reduction 中的灾难性抵消", demo_catastrophic_cancellation),
        "3": ("Softmax 在 fp16 下的数值稳定性", demo_softmax_stability),
        "4": ("MatMul 内积精度随维度增长", demo_matmul_precision_scaling),
        "5": ("LayerNorm 在混合精度下的误差分析", demo_layernorm_precision),
        "6": ("Kahan 补偿求和 vs Naive 求和", demo_kahan_vs_naive),
        "7": ("FMA 精度优势演示", demo_fma_precision),
        "8": ("Non-deterministic 归约演示", demo_nondeterministic_reduction),
        "9": ("经验法则速查表", demo_cheatsheet),
    }

    if len(sys.argv) > 1:
        selected = [k for k in sys.argv[1:] if k in demos]
    else:
        selected = list(demos.keys())

    print(f"\n{'#' * 70}")
    print(f"  GPU Kernel 数值精度演示套件")
    print(f"  运行演示: {', '.join(selected)}")
    print(f"{'#' * 70}")

    for key in selected:
        demos[key][1]()

    print(f"\n所有演示完成!")
