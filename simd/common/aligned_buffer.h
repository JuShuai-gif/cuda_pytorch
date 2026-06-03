#pragma once

/*
 * aligned_buffer.h -- Aligned memory allocation utilities.
 *
 * Requires _POSIX_C_SOURCE >= 200112L for posix_memalign on POSIX.
 * Uses _aligned_malloc / _aligned_free on Windows.
 */

#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200112L
#define _POSIX_C_SOURCE 200112L
#endif

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = NULL;
#if defined(_WIN32) || defined(_WIN64)
    ptr = _aligned_malloc(size, alignment);
    if (ptr == NULL) {
        fprintf(stderr, "aligned_alloc_wrapper: _aligned_malloc failed "
                "(alignment=%zu, size=%zu)\n", alignment, size);
        abort();
    }
#else
    int rc = posix_memalign(&ptr, alignment, size);
    if (rc != 0 || ptr == NULL) {
        fprintf(stderr, "aligned_alloc_wrapper: posix_memalign failed "
                "(alignment=%zu, size=%zu, rc=%d)\n", alignment, size, rc);
        abort();
    }
#endif
    return ptr;
}

static inline void aligned_free(void* ptr) {
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline int is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

#define ALIGNED_ALLOC(type, count, alignment) \
    ((type*)aligned_alloc_wrapper((alignment), (count) * sizeof(type)))

#define ALIGNED_FREE(ptr) aligned_free(ptr)

#ifdef __cplusplus
}
#endif
