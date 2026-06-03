#pragma once

#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * benchmark.h -- Lightweight benchmarking framework for SIMD kernels.
 *
 * Provides BENCH_COMPUTE and BENCH_MEMORY macros that:
 *   1. Warm up the function with a few iterations.
 *   2. Run `iters` iterations, timing each with clock_gettime.
 *   3. Compute minimum elapsed time (to reduce noise).
 *   4. Compute ns/element, GB/s, speedup, and store results.
 *
 * bench_report() prints a formatted table of all collected results.
 *
 * Usage example:
 *
 *   benchmark_result_t results[3];
 *   BENCH_COMPUTE(scalar_add(c, a, b, n), n, n * 3 * sizeof(float), 50, results[0]);
 *   results[0].name = "scalar_add";
 *   BENCH_COMPUTE(sse_add(c, a, b, n), n, n * 3 * sizeof(float), 50, results[1]);
 *   results[1].name = "sse_add";
 *   BENCH_COMPUTE(avx2_add(c, a, b, n), n, n * 3 * sizeof(float), 50, results[2]);
 *   results[2].name = "avx2_add";
 *   bench_report(results, 3);
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- data types ---- */

typedef struct {
    const char* name;
    double elapsed_ns;
    double ns_per_element;
    double gb_per_sec;
    double speedup;
    size_t iterations;
    size_t num_elements;
} benchmark_result_t;

/* ---- BENCH_COMPUTE / BENCH_MEMORY macros ---- */

/*
 * Parameters:
 *   func_call       - void-returning function call expression
 *   nelem           - number of elements processed per call
 *   bytes           - total bytes read + written per call (for GB/s calc)
 *   iters           - number of timed iterations (after warmup)
 *   result          - benchmark_result_t lvalue to fill
 *
 * The result->name is initialized to ""; user must set it after the macro.
 * result->speedup is initialized to 0; bench_report() computes it.
 */

#define BENCH_COMPUTE(func_call, nelem, bytes, iters, result) do {               \
    double _bc_min_ns = 1e18;                                                    \
    for (int _bc_w = 0; _bc_w < 3; ++_bc_w) { func_call; }                      \
    for (int _bc_i = 0; _bc_i < (iters); ++_bc_i) {                             \
        double _bc_t0 = get_time_ns();                                           \
        func_call;                                                               \
        double _bc_el = get_time_ns() - _bc_t0;                                  \
        if (_bc_el < _bc_min_ns) _bc_min_ns = _bc_el;                           \
    }                                                                            \
    (result).elapsed_ns    = _bc_min_ns;                                         \
    (result).iterations    = (size_t)(iters);                                    \
    (result).num_elements  = (size_t)(nelem);                                    \
    (result).name          = "";                                                 \
    (result).ns_per_element = ((nelem) > 0)                                      \
        ? (_bc_min_ns / (double)(nelem)) : 0.0;                                  \
    (result).speedup        = 0.0;                                               \
    (result).gb_per_sec     = (_bc_min_ns > 0.0)                                 \
        ? ((double)(bytes) / _bc_min_ns) : 0.0;                                  \
} while (0)

#define BENCH_MEMORY(func_call, nelem, bytes, iters, result) \
    BENCH_COMPUTE(func_call, nelem, bytes, iters, result)

/* ---- result printing ---- */

/*
 * bench_report -- Print a formatted table of benchmark results.
 *
 * The first result (index 0) is treated as the scalar baseline.
 * speedup is computed relative to it.
 */

static inline void bench_report(const benchmark_result_t* results,
                                size_t num_results) {
    if (num_results == 0) return;

    double baseline_ns = results[0].elapsed_ns;
    double baseline_ns_per = results[0].ns_per_element;

    printf("\n");
    printf("%-3s %-28s %12s %12s %12s %8s\n",
           "", "Name", "ns/element", "GB/s", "Speedup", "Iters");
    printf("%-3s %-28s %12s %12s %12s %8s\n",
           "", "----------------------------", "------------",
           "------------", "------------", "--------");

    for (size_t i = 0; i < num_results; ++i) {
        const benchmark_result_t* r = &results[i];

        double speedup = 0.0;
        double baseline_for_cmp = (baseline_ns_per > 0.0)
            ? baseline_ns_per : baseline_ns;
        double result_for_cmp = (r->ns_per_element > 0.0)
            ? r->ns_per_element : r->elapsed_ns;
        if (baseline_for_cmp > 0.0 && result_for_cmp > 0.0) {
            speedup = baseline_for_cmp / result_for_cmp;
        }

        char speedup_str[32];
        char ns_per_el_str[32];
        char gb_per_sec_str[32];

        if (i == 0) {
            snprintf(speedup_str, sizeof(speedup_str), "1.00x (baseline)");
        } else if (speedup >= 0.995) {
            snprintf(speedup_str, sizeof(speedup_str), "%.2fx", speedup);
        } else if (speedup > 0.0) {
            snprintf(speedup_str, sizeof(speedup_str), "%.3fx", speedup);
        } else {
            snprintf(speedup_str, sizeof(speedup_str), "N/A");
        }

        if (r->ns_per_element >= 100.0) {
            snprintf(ns_per_el_str, sizeof(ns_per_el_str), "%.1f",
                     r->ns_per_element);
        } else if (r->ns_per_element >= 1.0) {
            snprintf(ns_per_el_str, sizeof(ns_per_el_str), "%.3f",
                     r->ns_per_element);
        } else if (r->ns_per_element >= 0.001) {
            snprintf(ns_per_el_str, sizeof(ns_per_el_str), "%.5f",
                     r->ns_per_element);
        } else {
            snprintf(ns_per_el_str, sizeof(ns_per_el_str), "%.8f",
                     r->ns_per_element);
        }

        if (r->gb_per_sec >= 100.0) {
            snprintf(gb_per_sec_str, sizeof(gb_per_sec_str), "%.1f",
                     r->gb_per_sec);
        } else if (r->gb_per_sec >= 1.0) {
            snprintf(gb_per_sec_str, sizeof(gb_per_sec_str), "%.3f",
                     r->gb_per_sec);
        } else if (r->gb_per_sec >= 0.001) {
            snprintf(gb_per_sec_str, sizeof(gb_per_sec_str), "%.5f",
                     r->gb_per_sec);
        } else {
            snprintf(gb_per_sec_str, sizeof(gb_per_sec_str), "%.8f",
                     r->gb_per_sec);
        }

        printf("%3zu %-28s %12s %12s %12s %8zu\n",
               i,
               r->name ? r->name : "(unnamed)",
               ns_per_el_str,
               gb_per_sec_str,
               speedup_str,
               r->iterations);
    }
    printf("\n");
}

#ifdef __cplusplus
}
#endif
