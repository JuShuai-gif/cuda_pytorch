#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * random_data.h -- Deterministic pseudo-random data generation for benchmarks.
 *
 * Uses xorshift64* for fast, reproducible random numbers.
 * All functions produce the same sequence given the default seed (42).
 * Call rand_xorshift64_seed() to reset the sequence.
 *
 * Thread safety: NOT thread-safe (uses internal static state).
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- xorshift64* RNG (shared state) ---- */

static uint64_t* _get_state_ptr(void) {
    static uint64_t state = 42;
    return &state;
}

static inline void rand_xorshift64_seed(uint64_t seed) {
    *_get_state_ptr() = seed;
}

static inline uint64_t rand_xorshift64_next(void) {
    uint64_t* state = _get_state_ptr();
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(0x2545F4914F6CDD1D);
}

/* ---- Fill functions ---- */

static inline void fill_random_f32(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint64_t r = rand_xorshift64_next();
        data[i] = ((float)(r & 0xFFFFFFFFu) / (float)0xFFFFFFFFu) * 2.0f - 1.0f;
    }
}

static inline void fill_random_i32(int32_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint64_t r = rand_xorshift64_next();
        int64_t v = (int64_t)(r % 2001) - 1000;
        data[i] = (int32_t)v;
    }
}

static inline void fill_random_u8(uint8_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = (uint8_t)(rand_xorshift64_next() & 0xFF);
    }
}

static inline void fill_random_i8(int8_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = (int8_t)((rand_xorshift64_next() & 0xFF) - 128);
    }
}

static inline void fill_random_f16(void* data, size_t n) {
    uint16_t* out = (uint16_t*)data;
    for (size_t i = 0; i < n; ++i) {
        uint64_t r = rand_xorshift64_next();
        float f = ((float)(r & 0xFFFFFFFFu) / (float)0xFFFFFFFFu) * 2.0f - 1.0f;

        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exp  = (int32_t)((bits >> 23) & 0xFFu) - 127;
        uint32_t mant = (bits >> 13) & 0x3FFu;

        if (exp > 15) {
            exp = 15;
            mant = 0x3FFu;
        } else if (exp < -14) {
            exp = -14;
            mant = 0;
        }
        out[i] = (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | mant);
    }
}

static inline void fill_range_f32(float* data, size_t n, float lo, float hi) {
    float range = hi - lo;
    for (size_t i = 0; i < n; ++i) {
        uint64_t r = rand_xorshift64_next();
        float t = (float)(r & 0xFFFFFFFFu) / (float)0xFFFFFFFFu;
        data[i] = lo + t * range;
    }
}

static inline void fill_constant_f32(float* data, size_t n, float val) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = val;
    }
}

#ifdef __cplusplus
}
#endif
