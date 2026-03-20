#ifndef VEC_DTYPES_CUH_
#define VEC_DTYPES_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <cuda_runtime.h>

#include <type_traits>

namespace flashinfer {

#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
#define FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
#endif

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

__device__ __forceinline__ void st_global_release(int4 const &val, int4 *addr) {
    asm volatile("st.release.global.sys.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y),
                 "r"(val.z), "r"(val.w), "l"(addr));
}

__device__ __forceinline__ int4 ld_global_acquire(int4 *addr) {
    int4 val;
    asm volatile("ld.acquire.global.sys.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr));
    return val;
}

__device__ __forceinline__ void st_global_volatile(int4 const& val,int4* addr){
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y),
               "r"(val.z), "r"(val.w), "l"(addr));    
}

__device__ __forceinline__ int4 ld_global_volatile(int4* addr){
    int4 val;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  return val;
}

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 < 120200) && \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
// CUDA version < 12.2 and GPU architecture < 80
FLASHINFER_INLINE __nv_bfloat162 make_bfloat162(const __nv_bfloat16 x,const __nv_bfloat16 y){
    __nv_bfloat162 t;
    t.x = x;
    t.y = y;
    return t;
}

FLASHINFER_INLINE __nv_bfloat16 __hmul(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  __nv_bfloat16 val;
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    val = __float2bfloat16(__fmaf_ieee_rn(fa,fb,-0.0f));
    return val;
}















} // namespace flashinfer
