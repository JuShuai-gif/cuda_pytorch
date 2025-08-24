#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <vector>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void embedding_f32_kernel(const int *idx, float *weight, float *output, int n, int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void embedding_f32x4_kernel(const int *idx, float *weight, float *output, int n, int emb_size) {
    int tx = threadIdx.x * 4;
    int bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
    output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
    output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
    output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
}
