#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Error checking macros
#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t error = call;                                                                        \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

// Common function declarations
void clear_l2();
bool is_close(float a, float b, float rtol = 1e-5, float atol = 1e-8);
float benchmark_sum_reduction(float (*func)(float*, int), float* data, unsigned int length, int warmup = 25,
                              int reps = 100);
float sequential_sum_reduction(float* data, int length);
