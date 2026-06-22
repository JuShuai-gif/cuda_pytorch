#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) {
    return *(half2 *)&x;
}

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) {
    return *(uint32_t *)&x;
}

__forceinline__ __device__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ float ptx_log2(float x) {
    float y;
    asm volatile("lg2.approx.ftx.f32 %0,%1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ half2 ptx_exp2(half2 x) {
    uint32_t y_u32;
    uint32_t x_u32 = half2_as_uint32(x);
    asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
    return uint32_as_half2(y_u32);
}

__forceinline__ __device__ half ptx_exp2(half x) {
    ushort y_u16;
    asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
    return __ushort_as_half(y_u16);
}

__forceinline__ __device__ float ptx_rcp(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
    float y;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(y)
                 : "f"(x), "r"(lane_mask));
    return y;
}

__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
    return __shfl_xor_sync(0xffffffff, x, lane_mask);
}

__forceinline__ __device__ float rsqrt(float x) {
    float y;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ float tanh(float x) {
    float y;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ half2 tanh(half2 x) {
    uint32_t y_u32;
    uint32_t x_u32 = half2_as_uint32(x);
    asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
    return uint32_as_half2(y_u32);
}

__forceinline__ __device__ half tanh(half x) {
    ushort y_u16;
    asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
    return __ushort_as_half(y_u16);
}

}  // namespace math
}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CUH_
