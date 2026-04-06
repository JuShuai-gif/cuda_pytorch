#ifndef FLASHINFER_ATTENTION_MASK_CUH_
#define FLASHINFER_ATTENTION_MASK_CUH_

namespace flashinfer {

enum class MaskMode {
  kNone = 0U,    // No mask
  kCausal = 1U,  // Causal mask
  kCustom = 2U,  // Custom mask
  kMultiItemScoring = 3U,
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_MASK_CUH_