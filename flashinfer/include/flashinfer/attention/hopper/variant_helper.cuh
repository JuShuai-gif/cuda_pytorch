#ifndef FLASHINFER_ATTENTION_HOPPER_VARIANT_HELPER_H
#define FLASHINFER_ATTENTION_HOPPER_VARIANT_HELPER_H

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {

#define REGISTER_QUERY_TRANSFORM(params, q, ...)                                            \
  template <typename MainloopParams, typename T>                                            \
  __device__ __forceinline__ T QueryTransform(const MainloopParams& params, void* q_smem) { \
    __VA_ARGS__                                                                             \
  }

#define REGISTER_KEY_TRANSFORM(params, k, ...)                                            \
  template <typename MainloopParams, typename T>                                          \
  __device__ __forceinline__ T KeyTransform(const MainloopParams& params, void* k_smem) { \
    __VA_ARGS__                                                                           \
  }

#define REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, \
                                  kv_head_idx, ...)                                       \
  template <typename MainloopParams, typename T>                                          \
  __device__ __forceinline__ T LogitsTransform(                                           \
      const MainloopParams& params, T logits, uint32_t batch_idx, uint32_t qo_idx,        \
      uint32_t kv_idx, uint32_t qo_head_idx, uint32_t kv_head_idx) {                      \
    __VA_ARGS__                                                                           \
  }

#define REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, ...) \
  template <typename MainloopParams>                                                           \
  __device__ __forceinline__ bool LogitsMask(const MainloopParams& params, uint32_t batch_idx, \
                                             uint32_t qo_idx, uint32_t kv_idx,                 \
                                             uint32_t qo_head_idx, uint32_t kv_head_idx) {     \
    __VA_ARGS__                                                                                \
  }

struct AttentionVariantBase {
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                            { return logits; })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                       { return true; })
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_VARIANT_HELPER_H
