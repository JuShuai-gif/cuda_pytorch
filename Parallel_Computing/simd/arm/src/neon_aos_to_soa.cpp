#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

// =============================================================================
// neon_aos_to_soa -- AoS <-> SoA conversion with NEON
//   AoS: array of structs { float x, y, z, w; } points[N]
//   SoA: separate arrays  float xs[N], ys[N], zs[N], ws[N]
//
//   Forward  (AoS -> SoA): vld4q_f32 to deinterleave, vst1q to store planar
//   Reverse  (SoA -> AoS): vld1q to load planar, vst4q_f32 to interleave
//   SIMD width: 4 structs per iteration (vld4q loads 4x4 floats interleaved)
//   N = 100000
// =============================================================================

static const size_t N = 100000;
static const int    BENCH_ITERS = 10;

// ---- AoS struct definition ----
#define AOS_ALIGN(x) __attribute__((aligned(x)))

typedef struct {
    float x, y, z, w;
} AOS_ALIGN(16) Point4D;

// ---- scalar AoS -> SoA ----
static void scalar_aos_to_soa(const float* aos, float* xs, float* ys,
                               float* zs, float* ws, size_t n) {
    for (size_t i = 0; i < n; i++) {
        xs[i] = aos[i * 4 + 0];
        ys[i] = aos[i * 4 + 1];
        zs[i] = aos[i * 4 + 2];
        ws[i] = aos[i * 4 + 3];
    }
}

// ---- scalar SoA -> AoS ----
static void scalar_soa_to_aos(const float* xs, const float* ys,
                               const float* zs, const float* ws,
                               float* aos, size_t n) {
    for (size_t i = 0; i < n; i++) {
        aos[i * 4 + 0] = xs[i];
        aos[i * 4 + 1] = ys[i];
        aos[i * 4 + 2] = zs[i];
        aos[i * 4 + 3] = ws[i];
    }
}

// ---- NEON AoS -> SoA: vld4q_f32 deinterleaves 4 structs at once ----
// vld4q_f32 loads from interleaved memory and separates into 4 registers,
// one per component (x, y, z, w). Each register holds 4 values.
static void neon_aos_to_soa(const float* aos, float* xs, float* ys,
                             float* zs, float* ws, size_t n) {
    size_t i = 0;
    // Process 4 structs at a time (16 floats, 4 per component)
    for (; i + 4 <= n; i += 4) {
        float32x4x4_t interleaved = vld4q_f32(aos + i * 4);
        vst1q_f32(xs + i, interleaved.val[0]);
        vst1q_f32(ys + i, interleaved.val[1]);
        vst1q_f32(zs + i, interleaved.val[2]);
        vst1q_f32(ws + i, interleaved.val[3]);
    }
    // Tail: scalar fallback
    for (; i < n; i++) {
        xs[i] = aos[i * 4 + 0];
        ys[i] = aos[i * 4 + 1];
        zs[i] = aos[i * 4 + 2];
        ws[i] = aos[i * 4 + 3];
    }
}

// ---- NEON SoA -> AoS: vst4q_f32 interleaves 4 components into structs ----
static void neon_soa_to_aos(const float* xs, const float* ys,
                             const float* zs, const float* ws,
                             float* aos, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x4_t planes;
        planes.val[0] = vld1q_f32(xs + i);
        planes.val[1] = vld1q_f32(ys + i);
        planes.val[2] = vld1q_f32(zs + i);
        planes.val[3] = vld1q_f32(ws + i);
        vst4q_f32(aos + i * 4, planes);
    }
    for (; i < n; i++) {
        aos[i * 4 + 0] = xs[i];
        aos[i * 4 + 1] = ys[i];
        aos[i * 4 + 2] = zs[i];
        aos[i * 4 + 3] = ws[i];
    }
}

// =============================================================================
// main
// =============================================================================
int main(void) {
    printf("================================================================\n");
    printf("  NEON AoS <-> SoA Conversion (4D points)\n");
    printf("  SIMD width: 4 structs per vld4q_f32 / vst4q_f32\n");
    printf("  N = %zu points\n", N);
    printf("================================================================\n");

    // AoS: interleaved float array of size N*4
    float* aos_in   = ALIGNED_ALLOC(float, N * 4, 16);
    float* aos_out  = ALIGNED_ALLOC(float, N * 4, 16);

    // SoA: separate per-component arrays
    float* xs       = ALIGNED_ALLOC(float, N, 16);
    float* ys       = ALIGNED_ALLOC(float, N, 16);
    float* zs       = ALIGNED_ALLOC(float, N, 16);
    float* ws       = ALIGNED_ALLOC(float, N, 16);

    // Reference SoA arrays for correctness checking
    float* ref_xs   = ALIGNED_ALLOC(float, N, 16);
    float* ref_ys   = ALIGNED_ALLOC(float, N, 16);
    float* ref_zs   = ALIGNED_ALLOC(float, N, 16);
    float* ref_ws   = ALIGNED_ALLOC(float, N, 16);

    CHECK_TRUE(is_aligned(aos_in, 16), "aos_in is 16-byte aligned");
    CHECK_TRUE(is_aligned(xs, 16),    "xs is 16-byte aligned");

    // Fill AoS input with random data
    fill_random_f32(aos_in, N * 4);

    // ---- Correctness: AoS -> SoA ----
    printf("\n-- Correctness: AoS -> SoA --\n");

    memset(ref_xs, 0, N * sizeof(float));
    memset(ref_ys, 0, N * sizeof(float));
    memset(ref_zs, 0, N * sizeof(float));
    memset(ref_ws, 0, N * sizeof(float));
    scalar_aos_to_soa(aos_in, ref_xs, ref_ys, ref_zs, ref_ws, N);

    memset(xs, 0, N * sizeof(float));
    memset(ys, 0, N * sizeof(float));
    memset(zs, 0, N * sizeof(float));
    memset(ws, 0, N * sizeof(float));
    neon_aos_to_soa(aos_in, xs, ys, zs, ws, N);

    CHECK_NEAR_ARRAY(ref_xs, xs, N, 1e-6, "AoS->SoA: x matches");
    CHECK_NEAR_ARRAY(ref_ys, ys, N, 1e-6, "AoS->SoA: y matches");
    CHECK_NEAR_ARRAY(ref_zs, zs, N, 1e-6, "AoS->SoA: z matches");
    CHECK_NEAR_ARRAY(ref_ws, ws, N, 1e-6, "AoS->SoA: w matches");

    // ---- Correctness: SoA -> AoS (round-trip) ----
    printf("\n-- Correctness: SoA -> AoS --\n");

    memset(aos_out, 0, N * 4 * sizeof(float));
    neon_soa_to_aos(xs, ys, zs, ws, aos_out, N);

    // Verify round-trip: aos_out should match original aos_in
    CHECK_NEAR_ARRAY(aos_in, aos_out, N * 4, 1e-6,
                     "SoA->AoS round-trip matches original");

    // ---- Benchmarks ----
    printf("\n-- Benchmarks (%d timed iterations) --\n", BENCH_ITERS);

    // bytes_processed = N*4 floats read + N*4 floats written (AoS->SoA)
    //                 = same for SoA->AoS
    size_t bytes = N * 8 * sizeof(float); // 4 read + 4 write

    // ---- AoS -> SoA ----
    benchmark_result_t res_fwd[2];
    BENCH_COMPUTE(scalar_aos_to_soa(aos_in, xs, ys, zs, ws, N),
                  N, bytes, BENCH_ITERS, res_fwd[0]);
    res_fwd[0].name = "scalar_aos_to_soa";

    BENCH_COMPUTE(neon_aos_to_soa(aos_in, xs, ys, zs, ws, N),
                  N, bytes, BENCH_ITERS, res_fwd[1]);
    res_fwd[1].name = "neon_aos_to_soa (vld4q)";

    printf("\n>>> AoS -> SoA (deinterleave)\n");
    bench_report(res_fwd, 2);

    // ---- SoA -> AoS ----
    benchmark_result_t res_rev[2];
    BENCH_COMPUTE(scalar_soa_to_aos(xs, ys, zs, ws, aos_out, N),
                  N, bytes, BENCH_ITERS, res_rev[0]);
    res_rev[0].name = "scalar_soa_to_aos";

    BENCH_COMPUTE(neon_soa_to_aos(xs, ys, zs, ws, aos_out, N),
                  N, bytes, BENCH_ITERS, res_rev[1]);
    res_rev[1].name = "neon_soa_to_aos (vst4q)";

    printf("\n>>> SoA -> AoS (interleave)\n");
    bench_report(res_rev, 2);

    // ---- Summary ----
    printf("Summary:\n");
    printf("  AoS->SoA NEON speedup: %.2fx  (vld4q_f32 deinterleave)\n",
           res_fwd[0].elapsed_ns / res_fwd[1].elapsed_ns);
    printf("  SoA->AoS NEON speedup: %.2fx  (vst4q_f32 interleave)\n",
           res_rev[0].elapsed_ns / res_rev[1].elapsed_ns);
    printf("  vld4q/vst4q handle the shuffle in hardware, avoiding\n");
    printf("  manual unpack/pack in the scalar loop.\n");

    ALIGNED_FREE(aos_in);
    ALIGNED_FREE(aos_out);
    ALIGNED_FREE(xs);
    ALIGNED_FREE(ys);
    ALIGNED_FREE(zs);
    ALIGNED_FREE(ws);
    ALIGNED_FREE(ref_xs);
    ALIGNED_FREE(ref_ys);
    ALIGNED_FREE(ref_zs);
    ALIGNED_FREE(ref_ws);

    printf("\nAll tests passed.\n");
    return 0;
}
