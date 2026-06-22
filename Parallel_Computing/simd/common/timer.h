#pragma once

/*
 * timer.h -- High-precision monotonic timer for benchmarking.
 *
 * Requires _POSIX_C_SOURCE >= 199309L for clock_gettime.
 * Define it before including this header, or compile with -std=gnu11.
 */

#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200112L
#define _POSIX_C_SOURCE 200112L
#endif

#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline double get_time_ns(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#elif defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static inline double get_time_us(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#elif defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

typedef struct {
    double start_ns;
} scoped_timer_t;

static inline scoped_timer_t timer_start(void) {
    scoped_timer_t t;
    t.start_ns = get_time_ns();
    return t;
}

static inline double timer_elapsed_ns(scoped_timer_t* t) {
    return get_time_ns() - t->start_ns;
}

static inline double timer_elapsed_us(scoped_timer_t* t) {
    return (get_time_ns() - t->start_ns) / 1e3;
}

#ifdef __cplusplus
}
#endif
