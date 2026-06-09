#pragma once

#include <cuda_runtime.h>

__global__ void compute_kernel(float *data, int offset, int size, float scale);

void demo_cuda_streams();
