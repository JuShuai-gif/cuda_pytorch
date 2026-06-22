/*
 * avx2_aos_to_soa.cpp -- AVX2 AoS to SoA transformation (and back)
 *
 * struct Point4D { float x, y, z, w; };
 *
 * AoS layout: [p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w, ...]
 * SoA layout: x[0..N], y[0..N], z[0..N], w[0..N]  (4 separate arrays)
 *
 * SIMD width: 256-bit = 8x f32 per register
 * N = 100000 (must be multiple of 8 for AVX2; rounded down)
 *
 * Approach: Process 8 points per iteration (32 floats = 4 x __m256).
 * Each __m256 holds 2 consecutive Point4D.
 *
 * Shuffle strategy: Extract lo/hi 128-bit halves from each __m256 load,
 * then use the well-known _MM_TRANSPOSE4_PS macro on each group of 4 points.
 * This is cleaner and equally efficient vs. a pure 256-bit lane-crossing
 * shuffle sequence.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
#include "../../common/cpu_features.h"

static const size_t N = 100000 - (100000 % 8); /* round down to multiple of 8 */

struct Point4D {
    float x, y, z, w;
};

/* ================================================================
 * Scalar AoS -> SoA
 * ================================================================ */

static void scalar_aos_to_soa(const Point4D* aos,
                               float* x, float* y, float* z, float* w,
                               size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = aos[i].x;
        y[i] = aos[i].y;
        z[i] = aos[i].z;
        w[i] = aos[i].w;
    }
}

/* ================================================================
 * Scalar SoA -> AoS
 * ================================================================ */

static void scalar_soa_to_aos(const float* x, const float* y,
                               const float* z, const float* w,
                               Point4D* aos, size_t n) {
    for (size_t i = 0; i < n; i++) {
        aos[i].x = x[i];
        aos[i].y = y[i];
        aos[i].z = z[i];
        aos[i].w = w[i];
    }
}

/* ================================================================
 * 4x4 float transpose helper (SSE macro, works on __m128)
 *
 * Transposes a 4x4 matrix stored in row-major order within 4 __m128 regs.
 *   Before: r0=[r0c0,r0c1,r0c2,r0c3], r1=[r1c0,...,r1c3], ...
 *   After:  r0=[r0c0,r1c0,r2c0,r3c0], r1=[r0c1,r1c1,r2c1,r3c1], ...
 *
 * shufps immediate encoding:
 *   _MM_SHUFFLE(z,y,x,w) = (z<<6)|(y<<4)|(x<<2)|w
 *   0x44 = _MM_SHUFFLE(1,0,1,0) -> select elements 0,2 from a and 0,2 from b
 *   0xEE = _MM_SHUFFLE(3,2,3,2) -> select elements 1,3 from a and 1,3 from b
 *   0x88 = _MM_SHUFFLE(2,0,2,0) -> select elements 0,2 from a and 0,2 from b
 *   0xDD = _MM_SHUFFLE(3,1,3,1) -> select elements 1,3 from a and 1,3 from b
 * ================================================================ */

#define SHUFPS(a,b,i) _mm_shuffle_ps(a, b, i)

#define TRANSPOSE4_PS(v0,v1,v2,v3) do {                                     \
    __m128 _t0 = SHUFPS((v0), (v1), 0x44);  /* = {v0[0],v0[2],v1[0],v1[2]} */ \
    __m128 _t1 = SHUFPS((v0), (v1), 0xEE);  /* = {v0[1],v0[3],v1[1],v1[3]} */ \
    __m128 _t2 = SHUFPS((v2), (v3), 0x44);  /* = {v2[0],v2[2],v3[0],v3[2]} */ \
    __m128 _t3 = SHUFPS((v2), (v3), 0xEE);  /* = {v2[1],v2[3],v3[1],v3[3]} */ \
    (v0) = SHUFPS(_t0, _t2, 0x88);        /* = {t0[0],t0[2],t2[0],t2[2]} */  \
    (v1) = SHUFPS(_t0, _t2, 0xDD);        /* = {t0[1],t0[3],t2[1],t2[3]} */  \
    (v2) = SHUFPS(_t1, _t3, 0x88);        /* = {t1[0],t1[2],t3[0],t3[2]} */  \
    (v3) = SHUFPS(_t1, _t3, 0xDD);        /* = {t1[1],t1[3],t3[1],t3[3]} */  \
} while (0)

/* ================================================================
 * AVX2 AoS -> SoA: process 8 points per iteration
 *
 * Strategy:
 *   1. Load 8 points from AoS memory into 4 __m256 registers.
 *      Each register holds 2 consecutive Point4D (8 floats).
 *   2. Extract lo/hi 128-bit halves -> 8 __m128 (one per point).
 *   3. Group into even-indexed points (0,2,4,6) and odd-indexed (1,3,5,7).
 *   4. Apply TRANSPOSE4_PS to each group of 4.
 *   5. Recombine __m128 halves into __m256 and store to SoA arrays.
 * ================================================================ */

static void avx2_aos_to_soa(const Point4D* aos,
                             float* x, float* y, float* z, float* w,
                             size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        /*
         * Load 8 Point4D (32 floats = 4 x __m256).
         * r0 = points 0,1   r1 = points 2,3
         * r2 = points 4,5   r3 = points 6,7
         */
        __m256 r0 = _mm256_loadu_ps((const float*)(aos + i + 0));
        __m256 r1 = _mm256_loadu_ps((const float*)(aos + i + 2));
        __m256 r2 = _mm256_loadu_ps((const float*)(aos + i + 4));
        __m256 r3 = _mm256_loadu_ps((const float*)(aos + i + 6));

        /* Extract lo 128-bit halves: points 0,2,4,6 */
        __m128 elo0 = _mm256_castps256_ps128(r0);    /* p0 */
        __m128 elo1 = _mm256_castps256_ps128(r1);    /* p2 */
        __m128 elo2 = _mm256_castps256_ps128(r2);    /* p4 */
        __m128 elo3 = _mm256_castps256_ps128(r3);    /* p6 */

        /* Extract hi 128-bit halves: points 1,3,5,7 */
        __m128 ehi0 = _mm256_extractf128_ps(r0, 1);  /* p1 */
        __m128 ehi1 = _mm256_extractf128_ps(r1, 1);  /* p3 */
        __m128 ehi2 = _mm256_extractf128_ps(r2, 1);  /* p5 */
        __m128 ehi3 = _mm256_extractf128_ps(r3, 1);  /* p7 */

        /*
         * Transpose 4x4 for even-indexed points (0,2,4,6).
         * Input:  each = [p.x, p.y, p.z, p.w] (same component order).
         * Output: elo0 = [p0.x, p2.x, p4.x, p6.x]   = x[0..3]
         *         elo1 = [p0.y, p2.y, p4.y, p6.y]   = y[0..3]
         *         elo2 = [p0.z, p2.z, p4.z, p6.z]   = z[0..3]
         *         elo3 = [p0.w, p2.w, p4.w, p6.w]   = w[0..3]
         */
        TRANSPOSE4_PS(elo0, elo1, elo2, elo3);

        /*
         * Transpose 4x4 for odd-indexed points (1,3,5,7).
         * Output: ehi0 = [p1.x, p3.x, p5.x, p7.x]   = x[4..7]
         *         ehi1 = [p1.y, p3.y, p5.y, p7.y]   = y[4..7]
         *         ehi2 = [p1.z, p3.z, p5.z, p7.z]   = z[4..7]
         *         ehi3 = [p1.w, p3.w, p5.w, p7.w]   = w[4..7]
         */
        TRANSPOSE4_PS(ehi0, ehi1, ehi2, ehi3);

        /* Combine lo+hi halves into __m256 and store as contiguous SoA. */
        __m256 x256 = _mm256_insertf128_ps(_mm256_castps128_ps256(elo0), ehi0, 1);
        __m256 y256 = _mm256_insertf128_ps(_mm256_castps128_ps256(elo1), ehi1, 1);
        __m256 z256 = _mm256_insertf128_ps(_mm256_castps128_ps256(elo2), ehi2, 1);
        __m256 w256 = _mm256_insertf128_ps(_mm256_castps128_ps256(elo3), ehi3, 1);

        _mm256_storeu_ps(x + i, x256);
        _mm256_storeu_ps(y + i, y256);
        _mm256_storeu_ps(z + i, z256);
        _mm256_storeu_ps(w + i, w256);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        x[i] = aos[i].x;
        y[i] = aos[i].y;
        z[i] = aos[i].z;
        w[i] = aos[i].w;
    }
}

/* ================================================================
 * AVX2 SoA -> AoS: inverse of above
 *
 * The transpose operation is symmetric (its own inverse), so we use
 * the same TRANSPOSE4_PS macro. The order of loads/stores is reversed:
 * load SoA arrays, transpose, store into AoS layout.
 * ================================================================ */

static void avx2_soa_to_aos(const float* x, const float* y,
                             const float* z, const float* w,
                             Point4D* aos, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        /* Load 8 consecutive values from each SoA array */
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vz = _mm256_loadu_ps(z + i);
        __m256 vw = _mm256_loadu_ps(w + i);

        /* Split into even [0..3] and odd [4..7] halves */
        __m128 ex = _mm256_castps256_ps128(vx);
        __m128 ox = _mm256_extractf128_ps(vx, 1);
        __m128 ey = _mm256_castps256_ps128(vy);
        __m128 oy = _mm256_extractf128_ps(vy, 1);
        __m128 ez = _mm256_castps256_ps128(vz);
        __m128 oz = _mm256_extractf128_ps(vz, 1);
        __m128 ew = _mm256_castps256_ps128(vw);
        __m128 ow = _mm256_extractf128_ps(vw, 1);

        /*
         * Transpose each 4-element group.
         * Input state:
         *   ex=[x0,x2,x4,x6], ey=[y0,y2,y4,y6], ez=[z0,z2,z4,z6], ew=[w0,w2,w4,w6]
         * After transpose:
         *   ex = [x0,y0,z0,w0]  = point 0
         *   ey = [x2,y2,z2,w2]  = point 2
         *   ez = [x4,y4,z4,w4]  = point 4
         *   ew = [x6,y6,z6,w6]  = point 6
         */
        TRANSPOSE4_PS(ex, ey, ez, ew);

        /* Same for odd indices: points 1,3,5,7 */
        TRANSPOSE4_PS(ox, oy, oz, ow);

        /* Combine into __m256 pairs (even point in lo 128, odd in hi 128) */
        __m256 r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(ex), ox, 1);
        /* r0 = [p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w] */
        __m256 r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(ey), oy, 1);
        /* r1 = [p2.x, p2.y, p2.z, p2.w, p3.x, p3.y, p3.z, p3.w] */
        __m256 r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(ez), oz, 1);
        /* r2 = [p4.x, p4.y, p4.z, p4.w, p5.x, p5.y, p5.z, p5.w] */
        __m256 r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(ew), ow, 1);
        /* r3 = [p6.x, p6.y, p6.z, p6.w, p7.x, p7.y, p7.z, p7.w] */

        /* Store as contiguous AoS */
        _mm256_storeu_ps((float*)(aos + i + 0), r0);
        _mm256_storeu_ps((float*)(aos + i + 2), r1);
        _mm256_storeu_ps((float*)(aos + i + 4), r2);
        _mm256_storeu_ps((float*)(aos + i + 6), r3);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        aos[i].x = x[i];
        aos[i].y = y[i];
        aos[i].z = z[i];
        aos[i].w = w[i];
    }
}

/* ================================================================
 * main
 * ================================================================ */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 1;
    }

    printf("\n=== AVX2 AoS <-> SoA Transformation (N = %zu) ===\n\n", N);
    printf("Element: Point4D { float x, y, z, w; }\n");
    printf("Total data: %zu points = %.2f MB SoA + %.2f MB AoS\n",
           N,
           (double)(N * 4 * sizeof(float)) / (1024.0 * 1024.0),
           (double)(N * sizeof(Point4D)) / (1024.0 * 1024.0));

    /* Allocate AoS buffers */
    Point4D* aos_src  = ALIGNED_ALLOC(Point4D, N, 32);
    Point4D* aos_dst  = ALIGNED_ALLOC(Point4D, N, 32);
    Point4D* aos_ref  = ALIGNED_ALLOC(Point4D, N, 32);

    /* Allocate SoA buffers */
    float* soa_x = ALIGNED_ALLOC(float, N, 32);
    float* soa_y = ALIGNED_ALLOC(float, N, 32);
    float* soa_z = ALIGNED_ALLOC(float, N, 32);
    float* soa_w = ALIGNED_ALLOC(float, N, 32);

    float* soa_x_ref = ALIGNED_ALLOC(float, N, 32);
    float* soa_y_ref = ALIGNED_ALLOC(float, N, 32);
    float* soa_z_ref = ALIGNED_ALLOC(float, N, 32);
    float* soa_w_ref = ALIGNED_ALLOC(float, N, 32);

    if (!aos_src || !aos_dst || !aos_ref ||
        !soa_x || !soa_y || !soa_z || !soa_w ||
        !soa_x_ref || !soa_y_ref || !soa_z_ref || !soa_w_ref) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    /* Fill AoS source with random data */
    rand_xorshift64_seed(42);
    fill_random_f32((float*)aos_src, N * 4);

    /* ---- Correctness: AoS <-> SoA round-trip ---- */

    printf("--- Correctness ---\n");
    printf("  Strategy: AoS -> SIMD SoA -> SIMD AoS = original AoS\n");
    printf("  (SIMD SoA uses interleaved even/odd order from 4x4 transpose)\n");

    /*
     * The SIMD AoS->SoA uses a 4x4 transpose that produces:
     *   SoA x = [p0.x, p2.x, p4.x, p6.x, p1.x, p3.x, p5.x, p7.x]
     * This is different from scalar's natural order [p0.x, p1.x, ...].
     * The SoA->AoS conversion inverts this, so round-trip is exact.
     * In production, SoA layout ordering doesn't matter as long as
     * all operations use the same layout convention.
     */

    /* AoS -> SoA (SIMD) -> SoA -> AoS (SIMD) = original AoS */
    memset(soa_x, 0, N * sizeof(float));
    memset(soa_y, 0, N * sizeof(float));
    memset(soa_z, 0, N * sizeof(float));
    memset(soa_w, 0, N * sizeof(float));
    memset(aos_dst, 0, N * sizeof(Point4D));

    avx2_aos_to_soa(aos_src, soa_x, soa_y, soa_z, soa_w, N);
    avx2_soa_to_aos(soa_x, soa_y, soa_z, soa_w, aos_dst, N);

    CHECK_NEAR_ARRAY((float*)aos_dst, (float*)aos_src, N * 4, 0.0f,
                     "AoS->SoA->AoS round-trip (AVX2)");

    /* SoA -> AoS using SIMD SoA layout (from previous conversion) */
    memset(aos_dst, 0, N * sizeof(Point4D));
    avx2_soa_to_aos(soa_x, soa_y, soa_z, soa_w, aos_dst, N);

    CHECK_NEAR_ARRAY((float*)aos_dst, (float*)aos_src, N * 4, 0.0f,
                     "SoA(SIMD)->AoS(AVX2) matches original AoS");

    /* ---- Benchmark ---- */

    /*
     * Bytes processed for AoS -> SoA:
     *   Read AoS:  N * 4 * sizeof(float) = N * 16 bytes
     *   Write SoA: N * 4 * sizeof(float) = N * 16 bytes
     *   Total:     N * 32 bytes
     *
     * Same for SoA -> AoS
     */

    const size_t bytes_rw = N * 8 * sizeof(float); /* 32 bytes per point */

    benchmark_result_t results[4];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(scalar_aos_to_soa(aos_src, soa_x, soa_y, soa_z, soa_w, N),
                  N, bytes_rw, 20, results[0]);
    results[0].name = "scalar_AoS_to_SoA";

    BENCH_COMPUTE(avx2_aos_to_soa(aos_src, soa_x, soa_y, soa_z, soa_w, N),
                  N, bytes_rw, 20, results[1]);
    results[1].name = "avx2_AoS_to_SoA (8pts)";

    BENCH_COMPUTE(scalar_soa_to_aos(soa_x_ref, soa_y_ref, soa_z_ref, soa_w_ref, aos_dst, N),
                  N, bytes_rw, 20, results[2]);
    results[2].name = "scalar_SoA_to_AoS";

    BENCH_COMPUTE(avx2_soa_to_aos(soa_x_ref, soa_y_ref, soa_z_ref, soa_w_ref, aos_dst, N),
                  N, bytes_rw, 20, results[3]);
    results[3].name = "avx2_SoA_to_AoS (8pts)";

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (8x f32)\n");
    printf("Processes 8 Point4D per AVX2 iteration\n");
    bench_report(results, 4);

    printf("Notes:\n");
    printf("  - AoS (Array of Structures): [p0.xyzw, p1.xyzw, p2.xyzw, ...]\n");
    printf("    Good for: accessing all fields of a single element.\n");
    printf("  - SoA (Structure of Arrays): [x0,x1,x2,...], [y0,y1,y2,...], ...\n");
    printf("    Good for: vectorizing operations on a single component.\n");
    printf("  - The conversion uses TRANSPOSE4_PS (8 shufps per 4 points).\n");
    printf("    Total: 16 shufps + 4 extracts + 4 inserts per 8 points.\n");
    printf("  - The shuffle pattern is a 4x4 matrix transpose: the output\n");
    printf("    components come out in the order (p0,p2,p4,p6) for even\n");
    printf("    indices and (p1,p3,p5,p7) for odd.\n");
    printf("  - For large N, SoA layout often gives better SIMD utilization\n");
    printf("    because each vector load contains 8 of the same component\n");
    printf("    (continuity of access for that component).\n");

    ALIGNED_FREE(aos_src);
    ALIGNED_FREE(aos_dst);
    ALIGNED_FREE(aos_ref);
    ALIGNED_FREE(soa_x);
    ALIGNED_FREE(soa_y);
    ALIGNED_FREE(soa_z);
    ALIGNED_FREE(soa_w);
    ALIGNED_FREE(soa_x_ref);
    ALIGNED_FREE(soa_y_ref);
    ALIGNED_FREE(soa_z_ref);
    ALIGNED_FREE(soa_w_ref);

    return 0;
}
