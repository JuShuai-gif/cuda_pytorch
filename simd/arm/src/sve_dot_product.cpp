/**
 * sve_dot_product.cpp -- Dot Product using ARM SVE
 *
 * Computes dot(a, b) = sum(a[i] * b[i]) using SVE with:
 *
 *   1. svmul + svmla (predicated multiply-accumulate into vector accumulator)
 *   2. Tree reduction: split accumulation across multiple vector registers
 *      to better utilize SVE's wide execution units
 *   3. svadda folding reduction (SVE2+)
 *
 * Key SVE advantage over NEON: tail elements handled by predicate --
 * no scalar fallback loop needed. The svwhilelt predicate automatically
 * masks in only valid elements for the last partial vector.
 *
 * N = 1000000
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#include <arm_neon.h>

extern "C" {
#include "../../common/benchmark.h"
#include "../../common/cpu_features.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
}

static const int N_DOT_SVE = 1000000;

// ============================================================================
// Scalar dot product
// ============================================================================

static float scalar_dot(const float* __restrict a,
                         const float* __restrict b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ============================================================================
// NEON dot product (for comparison)
// ============================================================================
//
// 4 accumulators, 16 elements per iteration. Horizontal reduce at end.
// Scalar tail for non-multiple-of-4 remainder.

static float neon_dot(const float* __restrict a,
                       const float* __restrict b, int n) {
    float32x4_t vacc0 = vdupq_n_f32(0.0f);
    float32x4_t vacc1 = vdupq_n_f32(0.0f);
    float32x4_t vacc2 = vdupq_n_f32(0.0f);
    float32x4_t vacc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 15 < n; i += 16) {
        vacc0 = vmlaq_f32(vacc0, vld1q_f32(&a[i]),      vld1q_f32(&b[i]));
        vacc1 = vmlaq_f32(vacc1, vld1q_f32(&a[i + 4]),  vld1q_f32(&b[i + 4]));
        vacc2 = vmlaq_f32(vacc2, vld1q_f32(&a[i + 8]),  vld1q_f32(&b[i + 8]));
        vacc3 = vmlaq_f32(vacc3, vld1q_f32(&a[i + 12]), vld1q_f32(&b[i + 12]));
    }

    for (; i + 3 < n; i += 4) {
        vacc0 = vmlaq_f32(vacc0, vld1q_f32(&a[i]), vld1q_f32(&b[i]));
    }

    vacc0 = vaddq_f32(vaddq_f32(vacc0, vacc1), vaddq_f32(vacc2, vacc3));
    float sum = vaddvq_f32(vacc0);

    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// ============================================================================
// SVE dot product: single accumulator (simple predicated loop)
// ============================================================================
//
// Accumulate a[i]*b[i] into one SVE vector register using svmla_f32_m.
// At the end, svaddv horizontally reduces. Because SVE handles the tail
// via predicate, there is zero scalar tail code.

#ifdef __ARM_FEATURE_SVE
static float sve_dot_simple(const float* __restrict a,
                             const float* __restrict b, int n) {
    svfloat32_t vsum = svdup_f32(0.0f);

    int i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        // Predicated load of a and b; inactive lanes are zero
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);

        // Multiply-accumulate: vsum += va * vb (active lanes only)
        // inactive lanes of vsum are unchanged because of 'm' (merge) form
        vsum = svmla_f32_m(pg, vsum, va, vb);

        i += svcntw();
    }

    // Horizontal reduction: sum all elements in vsum
    float result = svaddv_f32(svptrue_b32(), vsum);
    return result;
}

// ============================================================================
// SVE dot product: tree reduction (4 accumulators)
// ============================================================================
//
// Strategy: Use 4 independent SVE vector accumulators (vsum0..vsum3).
// Process 4 vectors per outer loop iteration to hide svmla latency.
// This "tree reduction" pattern:
//   - Lets the CPU execute 4 independent svmla chains in parallel
//   - Improves instruction-level parallelism on wide out-of-order cores
//   - Particularly beneficial for SVE widths >= 256 bits
//
// At the end, we combine the 4 accumulators pairwise and svaddv the result.

static float sve_dot_tree(const float* __restrict a,
                           const float* __restrict b, int n) {
    svfloat32_t vsum0 = svdup_f32(0.0f);
    svfloat32_t vsum1 = svdup_f32(0.0f);
    svfloat32_t vsum2 = svdup_f32(0.0f);
    svfloat32_t vsum3 = svdup_f32(0.0f);

    uint64_t vl = svcntw();
    // Process in blocks of 4 * vl to feed 4 independent accumulators
    uint64_t stride = vl * 4;
    int i = 0;

    // Process full blocks (all accumulators get full vectors)
    for (; (uint64_t)i + stride <= (uint64_t)n; i += (int)stride) {
        vsum0 = svmla_f32_x(svptrue_b32(), vsum0,
                            svld1(svptrue_b32(), &a[i]),
                            svld1(svptrue_b32(), &b[i]));
        vsum1 = svmla_f32_x(svptrue_b32(), vsum1,
                            svld1(svptrue_b32(), &a[i + (int)vl]),
                            svld1(svptrue_b32(), &b[i + (int)vl]));
        vsum2 = svmla_f32_x(svptrue_b32(), vsum2,
                            svld1(svptrue_b32(), &a[i + 2 * (int)vl]),
                            svld1(svptrue_b32(), &b[i + 2 * (int)vl]));
        vsum3 = svmla_f32_x(svptrue_b32(), vsum3,
                            svld1(svptrue_b32(), &a[i + 3 * (int)vl]),
                            svld1(svptrue_b32(), &b[i + 3 * (int)vl]));
    }

    // Process remaining elements with single accumulator + predicate
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        vsum0 = svmla_f32_m(pg, vsum0,
                            svld1(pg, &a[i]),
                            svld1(pg, &b[i]));
        i += svcntw();
    }

    // Tree reduction: combine 4 accumulators -> 2 -> 1
    vsum0 = svadd_f32_x(svptrue_b32(), vsum0, vsum1);
    vsum2 = svadd_f32_x(svptrue_b32(), vsum2, vsum3);
    vsum0 = svadd_f32_x(svptrue_b32(), vsum0, vsum2);

    float result = svaddv_f32(svptrue_b32(), vsum0);
    return result;
}
#endif // __ARM_FEATURE_SVE

// ============================================================================
// Benchmark wrappers
// ============================================================================

static float* g_dp_a = nullptr;
static float* g_dp_b = nullptr;
static int    g_dp_n = 0;
static volatile float g_dp_result = 0.0f;

__attribute__((noinline))
static void bench_scalar_dot() { g_dp_result = scalar_dot(g_dp_a, g_dp_b, g_dp_n); }
__attribute__((noinline))
static void bench_neon_dot()   { g_dp_result = neon_dot(g_dp_a, g_dp_b, g_dp_n); }

#ifdef __ARM_FEATURE_SVE
__attribute__((noinline))
static void bench_sve_simple() { g_dp_result = sve_dot_simple(g_dp_a, g_dp_b, g_dp_n); }
__attribute__((noinline))
static void bench_sve_tree()   { g_dp_result = sve_dot_tree(g_dp_a, g_dp_b, g_dp_n); }
#endif

// ============================================================================
// Main
// ============================================================================

int main() {
    cpu_print_features();

    int n = N_DOT_SVE;
    g_dp_n = n;
    g_dp_a = ALIGNED_ALLOC(float, n, 64);
    g_dp_b = ALIGNED_ALLOC(float, n, 64);

    fill_random_f32(g_dp_a, n);
    fill_random_f32(g_dp_b, n);

    // --- Correctness ---
    printf("\n=== Correctness Checks ===\n");
    float ref = scalar_dot(g_dp_a, g_dp_b, n);
    printf("  Scalar reference dot product: %.6f\n", (double)ref);

    float neon_val = neon_dot(g_dp_a, g_dp_b, n);
    CHECK_NEAR(neon_val, ref, 5e-1f, "NEON dot product vs scalar");

#ifdef __ARM_FEATURE_SVE
    if (cpu_has_sve()) {
        float sve_simple = sve_dot_simple(g_dp_a, g_dp_b, n);
        CHECK_NEAR(sve_simple, ref, 5e-1f,
                   "SVE dot product (simple) vs scalar");

        float sve_tree = sve_dot_tree(g_dp_a, g_dp_b, n);
        CHECK_NEAR(sve_tree, ref, 5e-1f,
                   "SVE dot product (tree reduction) vs scalar");

        int sve_width = svcntw();
        printf("\n=== SVE Dot Product Analysis ===\n");
        printf("  Vector width: %d x f32 lanes\n", sve_width);
        printf("  Simple:     %d iterations, 1 accumulator\n",
               (n + sve_width - 1) / sve_width);
        printf("  Tree (4x):  %d full-block iters + %d tail iters\n",
               (int)((uint64_t)n / (svcntw() * 4)),
               (int)(((uint64_t)n % (svcntw() * 4) + svcntw() - 1) / svcntw()));
        printf("  NEON:       4 accumulators, ~%d iters\n",
               (n + 15) / 16 * 4);
        printf("  NO scalar tail code in SVE paths (predicate handles it)\n");
    }
#endif

    // --- Benchmark ---
    printf("\n=== Benchmark: Dot Product (N=%d) ===\n", n);
    size_t bytes = (size_t)n * 2 * sizeof(float); // read a,b

#ifdef __ARM_FEATURE_SVE
    int num_r = cpu_has_sve() ? 4 : 2;
    benchmark_result_t results[4];
    int ri = 0;

    BENCH_COMPUTE(bench_scalar_dot(), n, bytes, 30, results[ri]);
    results[ri++].name = "scalar dot";

    BENCH_COMPUTE(bench_neon_dot(), n, bytes, 30, results[ri]);
    results[ri++].name = "NEON dot (4 acc)";

    if (cpu_has_sve()) {
        BENCH_COMPUTE(bench_sve_simple(), n, bytes, 30, results[ri]);
        results[ri++].name = "SVE dot (simple)";

        BENCH_COMPUTE(bench_sve_tree(), n, bytes, 30, results[ri]);
        results[ri++].name = "SVE dot (tree 4x)";
    }

    bench_report(results, (size_t)ri);

    printf("\n=== Key Insights ===\n");
    printf("  SVE eliminates all scalar tail code: the predicate in the\n");
    printf("  last iteration masks out-of-bounds elements automatically.\n");
    printf("  This is a major advantage over NEON where you need a separate\n");
    printf("  scalar loop for the last (n %% 4) elements.\n");
    printf("\n");
    printf("  Tree reduction (4 accumulators) improves ILP by allowing\n");
    printf("  the CPU to execute 4 independent svmla chains concurrently.\n");
    printf("  For 256-bit SVE, this means processing 32 floats per outer\n");
    printf("  iteration with 4 independent dependency chains.\n");
#else
    benchmark_result_t results[2];
    BENCH_COMPUTE(bench_scalar_dot(), n, bytes, 30, results[0]);
    results[0].name = "scalar dot";
    BENCH_COMPUTE(bench_neon_dot(), n, bytes, 30, results[1]);
    results[1].name = "NEON dot (4 acc)";
    bench_report(results, 2);
    printf("  SVE code not compiled in (use -march=armv8-a+sve)\n");
#endif

    printf("\n  Checksum: %.6f\n", (double)g_dp_result);

    ALIGNED_FREE(g_dp_a);
    ALIGNED_FREE(g_dp_b);
    return 0;
}
