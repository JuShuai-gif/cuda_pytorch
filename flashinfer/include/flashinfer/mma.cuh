#ifndef FLASHINFER_MMA_CUH_
#define FLASHINFER_MMA_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace flashinfer{
namespace mma{
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FLASHINFER_MMA_F8F8F32_M16N8K32_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
#define FLASHINFER_STMATRIX_M8N8X4_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED
#define FLASHINFER_MMA_F16F16F16_M16N8K16_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED
#define FLASHINFER_MMA_F16F16F16_M16N8K8_ENABLED
#define FLASHINFER_LDMATRIX_M8N8X4_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define FLASHINFER_RUNTIME_ASSERT(x) __brkpt()
#else
#define FLASHINFER_RUNTIME_ASSERT(x) assert(0 && x)
#endif

enum class MMAMode{
    kInit = 0U,
    kInplaceUpdate = 1U,
};

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R,T* smem_ptr){
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}










}




}

















































#endif  // FLASHINFER_MMA_CUH_