/**
 * AVX2 1D Convolution with Register Ring Buffer
 *
 * Demonstrates:
 *   - 1D convolution: out[i] = sum(kernel[j] * input[i+j])
 *   - Kernel sizes 3 and 5
 *   - "Register ring buffer" technique: maintain sliding window in registers,
 *     shifting by loading only 1 new value per output position instead of 3
 *   - N = 1000000, kernel = {0.25, 0.5, 0.25} (Gaussian blur-like)
 *   - Edge handling: zero-padding
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------------ */
/*  Scalar 1D convolution                                                   */
/* ------------------------------------------------------------------------ */
void conv1d_scalar(const float *input, const float *kernel, float *output,
                   int N, int K) {
    const int pad = K / 2;
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            int idx = i + j - pad;
            float val = (idx >= 0 && idx < N) ? input[idx] : 0.0f;
            sum += kernel[j] * val;
        }
        output[i] = sum;
    }
}

/* ------------------------------------------------------------------------ */
/*  AVX2 1D convolution - naive (3 loads per output)                        */
/* ------------------------------------------------------------------------ */
void conv1d_avx2_naive(const float *input, const float *kernel, float *output,
                       int N, int K) {
    const int pad = K / 2;
    __m256 k0 = _mm256_set1_ps(kernel[0]);
    __m256 k1 = _mm256_set1_ps(kernel[1]);
    __m256 k2 = _mm256_set1_ps(kernel[2]);
    (void)K;

    for (int i = 0; i < N; i++) {
        __m256 sum = _mm256_setzero_ps();

        /* element j=0 */
        int i0 = i - pad;
        float v0 = (i0 >= 0 && i0 < N) ? input[i0] : 0.0f;
        sum = _mm256_fmadd_ps(k0, _mm256_set1_ps(v0), sum);

        /* element j=1 */
        float v1 = (i >= 0 && i < N) ? input[i] : 0.0f;
        sum = _mm256_fmadd_ps(k1, _mm256_set1_ps(v1), sum);

        /* element j=2 */
        int i2 = i + pad;
        float v2 = (i2 >= 0 && i2 < N) ? input[i2] : 0.0f;
        sum = _mm256_fmadd_ps(k2, _mm256_set1_ps(v2), sum);

        output[i] = _mm256_cvtss_f32(sum);
    }
}

/* ------------------------------------------------------------------------ */
/*  AVX2 1D convolution - register ring buffer (kernel=3)                   */
/*                                                                          */
/*  Idea: Instead of loading all 3 elements per output position, we         */
/*  maintain a sliding window of register values: {prev, curr, next}.       */
/*  When we advance to the next output position, "next" becomes "curr",     */
/*  "curr" becomes "prev", and we load only the new "next".                 */
/*                                                                          */
/*  For each output i we compute: k0*prev + k1*curr + k2*next               */
/*  Then shift: prev=curr, curr=next, load new next.                        */
/* ------------------------------------------------------------------------ */
void conv1d_avx2_ringbuf(const float *input, const float *kernel, float *output,
                         int N) {
    if (N < 1) return;

    __m256 k0 = _mm256_set1_ps(kernel[0]);
    __m256 k1 = _mm256_set1_ps(kernel[1]);
    __m256 k2 = _mm256_set1_ps(kernel[2]);

    /* Initialize the sliding window: prev, curr, next */
    /* For i=0: we need input[-1] (zero), input[0], input[1] */

    /* prev starts as 0 (zero-padding for i=-1) */
    float prev = 0.0f;
    float curr = input[0];
    float next = (N > 1) ? input[1] : 0.0f;

    /* Output i=0 */
    {
        __m256 vp = _mm256_set1_ps(prev);
        __m256 vc = _mm256_set1_ps(curr);
        __m256 vn = _mm256_set1_ps(next);
        __m256 sum = _mm256_setzero_ps();
        sum = _mm256_fmadd_ps(k0, vp, sum);
        sum = _mm256_fmadd_ps(k1, vc, sum);
        sum = _mm256_fmadd_ps(k2, vn, sum);
        output[0] = _mm256_cvtss_f32(sum);
    }

    /* Shift window and process remaining i */
    for (int i = 1; i < N; i++) {
        /* Shift the ring: prev <- curr, curr <- next, load new next */
        prev = curr;
        curr = next;
        next = (i + 1 < N) ? input[i + 1] : 0.0f;

        __m256 vp = _mm256_set1_ps(prev);
        __m256 vc = _mm256_set1_ps(curr);
        __m256 vn = _mm256_set1_ps(next);
        __m256 sum = _mm256_setzero_ps();
        sum = _mm256_fmadd_ps(k0, vp, sum);
        sum = _mm256_fmadd_ps(k1, vc, sum);
        sum = _mm256_fmadd_ps(k2, vn, sum);
        output[i] = _mm256_cvtss_f32(sum);
    }
}

/* ------------------------------------------------------------------------ */
/*  AVX2 1D convolution for kernel=5 (scalar inner, but SIMD outer)         */
/* ------------------------------------------------------------------------ */
void conv1d_avx2_k5(const float *input, const float *kernel, float *output,
                    int N) {
    const int pad = 2; /* K/2 = 2 */
    for (int i = 0; i < N; i++) {
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < 5; j++) {
            int idx = i + j - pad;
            float val = (idx >= 0 && idx < N) ? input[idx] : 0.0f;
            sum = _mm256_fmadd_ps(_mm256_set1_ps(kernel[j]),
                                   _mm256_set1_ps(val), sum);
        }
        output[i] = _mm256_cvtss_f32(sum);
    }
}

/* ------------------------------------------------------------------------ */
/*  Verification                                                            */
/* ------------------------------------------------------------------------ */
static int verify(const float *a, const float *b, int N, float tol) {
    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
        if (err > tol) {
            if (err_count < 3)
                fprintf(stderr, "  mismatch at %d: %f vs %f (err=%e)\n",
                        i, a[i], b[i], err);
            err_count++;
        }
    }
    if (err_count > 0) {
        fprintf(stderr, "  max_err=%e, total mismatches=%d\n", max_err, err_count);
        return 0;
    }
    printf("  max_err=%e, all good\n", max_err);
    return 1;
}

/* ------------------------------------------------------------------------ */
/*  Benchmark helper                                                        */
/* ------------------------------------------------------------------------ */
static double benchmark(void (*fn)(const float*, const float*, float*, int, int),
                        const float *input, const float *kernel, float *output,
                        int N, int K, int iters) {
    double start = get_time_sec();
    for (int iter = 0; iter < iters; iter++) {
        fn(input, kernel, output, N, K);
    }
    double elapsed = get_time_sec() - start;
    return elapsed / iters;
}

static double benchmark_simple(void (*fn)(const float*, const float*, float*, int),
                               const float *input, const float *kernel, float *output,
                               int N, int iters) {
    double start = get_time_sec();
    for (int iter = 0; iter < iters; iter++) {
        fn(input, kernel, output, N);
    }
    double elapsed = get_time_sec() - start;
    return elapsed / iters;
}

/* ------------------------------------------------------------------------ */
/*  Main                                                                    */
/* ------------------------------------------------------------------------ */
int main() {
    const int N = 1000000;
    const int K3 = 3;
    const int K5 = 5;

    float kernel3[3] = {0.25f, 0.5f, 0.25f};
    float kernel5[5] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};

    printf("=== AVX2 1D Convolution with Register Ring Buffer ===\n");
    printf("N = %d\n", N);
    printf("Kernel (K=3) = {%.2f, %.2f, %.2f}\n",
           kernel3[0], kernel3[1], kernel3[2]);
    printf("Kernel (K=5) = {%.2f, %.2f, %.2f, %.2f, %.2f}\n",
           kernel5[0], kernel5[1], kernel5[2], kernel5[3], kernel5[4]);

    /* Allocate */
    float *input  = (float*)aligned_alloc(32, N * sizeof(float));
    float *output = (float*)aligned_alloc(32, N * sizeof(float));
    float *ref    = (float*)aligned_alloc(32, N * sizeof(float));

    /* Fill input with some values */
    for (int i = 0; i < N; i++) {
        input[i] = (float)(i % 1000) * 0.001f + ((float)i * 0.000001f);
    }

    printf("\n--- Kernel = 3 ---\n");

    /* Scalar reference */
    memset(ref, 0, N * sizeof(float));
    conv1d_scalar(input, kernel3, ref, N, K3);

    /* Naive AVX2 */
    memset(output, 0, N * sizeof(float));
    conv1d_avx2_naive(input, kernel3, output, N, K3);
    printf("Naive AVX2 check: ");
    verify(output, ref, N, 1e-6f);

    /* Ring buffer AVX2 */
    memset(output, 0, N * sizeof(float));
    conv1d_avx2_ringbuf(input, kernel3, output, N);
    printf("Ringbuf AVX2 check: ");
    verify(output, ref, N, 1e-6f);

    /* Benchmark kernel=3 */
    int iters = 50;
    printf("\nPerformance (kernel=3, %d iterations):\n", iters);

    double t_scalar = benchmark(conv1d_scalar, input, kernel3, output, N, K3, iters);
    printf("  Scalar:        %8.3f ms\n", t_scalar * 1000);

    double t_naive = benchmark(conv1d_avx2_naive, input, kernel3, output, N, K3, iters);
    printf("  Naive AVX2:    %8.3f ms  (%.2fx)\n", t_naive * 1000, t_scalar / t_naive);

    double t_ring = benchmark_simple(conv1d_avx2_ringbuf, input, kernel3, output, N, iters);
    printf("  Ringbuf AVX2:  %8.3f ms  (%.2fx vs scalar, %.2fx vs naive)\n",
           t_ring * 1000, t_scalar / t_ring, t_naive / t_ring);

    double bandwidth = (double)N / t_ring / 1e9;
    printf("  Ringbuf throughput: %.3f GB/s (input reads)\n", bandwidth * sizeof(float));

    /* --- Kernel = 5 --- */
    printf("\n--- Kernel = 5 ---\n");

    memset(ref, 0, N * sizeof(float));
    conv1d_scalar(input, kernel5, ref, N, K5);

    memset(output, 0, N * sizeof(float));
    conv1d_avx2_k5(input, kernel5, output, N);
    printf("AVX2 K5 check: ");
    verify(output, ref, N, 1e-6f);

    double t_k5 = benchmark_simple(conv1d_avx2_k5, input, kernel5, output, N, iters);
    double t_scalar_k5 = benchmark(conv1d_scalar, input, kernel5, output, N, K5, iters);
    printf("  Scalar (K=5):  %8.3f ms\n", t_scalar_k5 * 1000);
    printf("  AVX2 (K=5):    %8.3f ms  (%.2fx speedup)\n",
           t_k5 * 1000, t_scalar_k5 / t_k5);

    printf("\n--- Register Ring Buffer Explanation ---\n");
    printf("For kernel=3, the ring buffer maintains prev/curr/next in scalars,\n");
    printf("reducing from 3 loads per output position to 1 load per position.\n");
    printf("This is the 1D analog of using shifted register windows for 2D convolution.\n");

    free(input);
    free(output);
    free(ref);

    return 0;
}
