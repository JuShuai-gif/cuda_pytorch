/**
 * avx512_rounding_demo.cpp -- AVX-512 嵌入式舍入 & SAE 演示
 *
 * 演示 AVX-512F 嵌入式舍入与 SAE：
 *   1. SAE (抑制所有异常) - 无需修改 MXCSR
 *   2. 四种舍入模式: 最近、向下、向上、截断
 *   3. 区间算术 (下界 + 上界)
 *   4. 使用 SAE 的 Kahan 求和
 *
 * 需要: -mavx512f
 */

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __AVX512F__
#error "This file requires -mavx512f compiler flag"
#endif

/* ================================================================
 * 1. 四种舍入模式下的加法
 * ================================================================ */

__attribute__((noinline))
static void add_with_rounding_modes_avx512(
    const float* a, const float* b, float* nearest, float* floor_v,
    float* ceil_v, float* trunc_v, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);

        /* 最近偶数 (默认，等同于 _mm512_add_ps) */
        _mm512_storeu_ps(nearest + i, _mm512_add_round_ps(va, vb,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        /* 向下取整 (趋向 -∞) */
        _mm512_storeu_ps(floor_v + i, _mm512_add_round_ps(va, vb,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));

        /* 向上取整 (趋向 +∞) */
        _mm512_storeu_ps(ceil_v + i, _mm512_add_round_ps(va, vb,
            _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));

        /* 截断 (趋向 0) */
        _mm512_storeu_ps(trunc_v + i, _mm512_add_round_ps(va, vb,
            _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    }
    for (; i < n; i++) {
        float s = a[i] + b[i];
        nearest[i] = s;
        floor_v[i] = floorf(s);
        ceil_v[i]  = ceilf(s);
        trunc_v[i] = truncf(s);
    }
}

/* ================================================================
 * 2. 区间算术: 计算下界 + 上界
 * ================================================================ */

__attribute__((noinline))
static void interval_add_avx512(const float* a, const float* b,
                                 float* lo, float* hi, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);

        /* 下界: 趋向 -∞ 舍入 */
        _mm512_storeu_ps(lo + i, _mm512_add_round_ps(va, vb,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));

        /* 上界: 趋向 +∞ 舍入 */
        _mm512_storeu_ps(hi + i, _mm512_add_round_ps(va, vb,
            _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
    }
    for (; i < n; i++) {
        float s = a[i] + b[i];
        lo[i] = nextafterf(s, -INFINITY);
        hi[i] = nextafterf(s, INFINITY);
    }
}

/* 验证区间性质: lo <= exact <= hi */
__attribute__((noinline))
static int check_interval(const float* lo, const float* hi,
                           const float* exact, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (lo[i] > exact[i] || hi[i] < exact[i]) return 0;
    }
    return 1;
}

/* ================================================================
 * 3. 使用 SAE 的 Kahan 求和 (无需修改 MXCSR)
 *
 * 标准 Kahan 求和需要计算补偿项。
 * 使用 SAE 可以逐指令控制舍入，无需修改全局 MXCSR 寄存器。
 * ================================================================ */

__attribute__((noinline))
static float kahan_sum_avx512_sae(const float* x, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    __m512 c   = _mm512_setzero_ps();  /* 丢失的低位补偿 */
    const int round_near = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);

        /* y = v - c  (补偿输入) */
        __m512 y = _mm512_sub_round_ps(v, c, round_near);

        /* t = sum + y */
        __m512 t = _mm512_add_round_ps(sum, y, round_near);

        /* c = (t - sum) - y  (补偿更新) */
        __m512 t_minus_sum = _mm512_sub_round_ps(t, sum, round_near);
        c = _mm512_sub_round_ps(t_minus_sum, y, round_near);

        sum = t;
    }

    float total = _mm512_reduce_add_ps(sum);
    for (; i < n; i++) {
        float y = x[i] - _mm_cvtss_f32(_mm512_castps512_ps128(c));
        float t = total + y;
        c = _mm512_set1_ps((t - total) - y);
        total = t;
    }
    return total;
}

__attribute__((noinline))
static float naive_sum(const float* x, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += x[i];
    return s;
}

/* ================================================================
 * 4. MXCSR 舍入模式对比
 * ================================================================ */

static void demonstrate_mxcsr_rounding(void) {
    printf("--- Embedded Rounding Demonstration ---\n");

    float a[16], b[16];
    for (int i = 0; i < 16; i++) {
        a[i] = (float)i + 0.3f;  /* 带小数的值，用于舍入测试 */
        b[i] = 0.5f;
    }

    float nearest[16], floor_v[16], ceil_v[16], trunc_v[16];
    add_with_rounding_modes_avx512(a, b, nearest, floor_v, ceil_v, trunc_v, 16);

    printf("  i   a       b       nearest   floor     ceil      trunc\n");
    printf("  --- ------- ------- --------- --------- --------- ---------\n");
    for (int i = 0; i < 4; i++) {
        printf("  %d  %7.3f %7.3f %9.4f %9.4f %9.4f %9.4f\n",
               i, a[i], b[i], nearest[i], floor_v[i], ceil_v[i], trunc_v[i]);
    }
    printf("  ...\n\n");

    /* 验证舍入一致性 */
    int ok = 1;
    for (int i = 0; i < 16; i++) {
        double exact = (double)a[i] + (double)b[i];
        /* floor ≤ nearest ≤ ceil */
        if (floor_v[i] > nearest[i] || nearest[i] > ceil_v[i]) ok = 0;
        /* floor ≤ exact ≤ ceil */
        if (floor_v[i] > exact || ceil_v[i] < exact) ok = 0;
    }
    printf("  [%s] Rounding mode consistency check\n",
           ok ? "PASS" : "FAIL");
}

/* ================================================================
 * 基准测试基础设施
 * ================================================================ */

static const size_t N = 1000000;
static float* g_x  = NULL;
static float g_sum = 0.0f;

__attribute__((noinline)) static void bn_naive()   { g_sum = naive_sum(g_x, N); }
__attribute__((noinline)) static void bn_kahan()   { g_sum = kahan_sum_avx512_sae(g_x, N); }

/* ================================================================
 * 主函数
 * ================================================================ */

int main() {
    cpu_print_features();

    if (!cpu_has_avx512f()) {
        printf("AVX-512F not supported. Exiting.\n");
        return 0;
    }

    printf("\n=== AVX-512 Embedded Rounding & SAE Demo ===\n");
    printf("ISA: AVX-512F (_mm512_add_round_ps, etc.)\n");
    printf("SAE = Suppress All Exceptions + specify rounding mode\n\n");

    /* 舍入模式演示 */
    demonstrate_mxcsr_rounding();

    /* 区间算术测试 */
    printf("--- Interval Arithmetic ---\n");
    {
        const size_t Nt = 1000;
        float* a  = ALIGNED_ALLOC(float, Nt, 64);
        float* b  = ALIGNED_ALLOC(float, Nt, 64);
        float* lo = ALIGNED_ALLOC(float, Nt, 64);
        float* hi = ALIGNED_ALLOC(float, Nt, 64);
        float* exact = ALIGNED_ALLOC(float, Nt, 64);

        rand_xorshift64_seed(42);
        fill_random_f32(a, Nt);
        rand_xorshift64_seed(99);
        fill_random_f32(b, Nt);

        interval_add_avx512(a, b, lo, hi, Nt);
        for (size_t i = 0; i < Nt; i++) exact[i] = a[i] + b[i];

        int ok = check_interval(lo, hi, exact, Nt);
        printf("  [%s] Interval property: lo[i] <= exact[i] <= hi[i] "
               "(%zu elements)\n", ok ? "PASS" : "FAIL", Nt);

        ALIGNED_FREE(a); ALIGNED_FREE(b); ALIGNED_FREE(lo);
        ALIGNED_FREE(hi); ALIGNED_FREE(exact);
    }

    /* Kahan 求和测试 */
    printf("\n--- Kahan Summation with SAE ---\n");
    {
        g_x = ALIGNED_ALLOC(float, N, 64);

        /* 生成不同数量级的数值 */
        rand_xorshift64_seed(42);
        for (size_t i = 0; i < N; i++) {
            g_x[i] = 1.0f + (float)(rand_xorshift64_next() % 1000) * 1e-6f;
        }

        float s_naive = naive_sum(g_x, N);
        float s_kahan = kahan_sum_avx512_sae(g_x, N);

        printf("  Naive sum:     %.10f\n", s_naive);
        printf("  Kahan (SAE):   %.10f\n", s_kahan);
        printf("  Difference:    %.2e\n", fabsf(s_kahan - s_naive));

        /* 基准测试 */
        benchmark_result_t results[2];
        memset(results, 0, sizeof(results));

        BENCH_COMPUTE(bn_naive(), N, N * sizeof(float), 30, results[0]);
        results[0].name = "sum (naive)";

        BENCH_COMPUTE(bn_kahan(), N, N * sizeof(float), 30, results[1]);
        results[1].name = "sum (Kahan SAE)";

        bench_report(results, 2);

        ALIGNED_FREE(g_x);
    }

    printf("--- SAE Key Points ---\n");
    printf("  SAE controls rounding + exception masking per-instruction.\n");
    printf("  No need to modify global MXCSR (thread-safe).\n");
    printf("  Rounding modes: NEAREST, NEG_INF(floor), POS_INF(ceil), ZERO(trunc)\n");
    printf("  Use cases:\n");
    printf("    - Interval arithmetic: bound exact results\n");
    printf("    - High-precision summation (Kahan without MXCSR change)\n");
    printf("    - ML quantization simulation (truncate mode)\n");
    printf("  Cost: zero extra cycles vs normal add (same uop, just EVEX bit set)\n");

    return 0;
}
