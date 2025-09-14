#pragma once
#ifndef CONV2D_FUNCTIONS_H
#define CONV2D_FUNCTIONS_H

#include "conv2d_kernels.cuh"  // Include this since we need BLOCK_SIZE

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void conv2d_with_constant_memory(float* M, float* F, float* P, int r, int height, int width);
void conv2d_torch_with_tiled_convolution(float* M, float* F, float* P, int r, int height, int width);
void conv2d_with_tiled_convolution_with_l2_caching(float* M, float* F, float* P, int r, int height, int width);

#endif
