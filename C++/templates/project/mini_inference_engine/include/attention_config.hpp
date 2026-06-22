#pragma once

#include <cstdint>

namespace mini_inference {

// ============================================================================
// AttentionConfig - FlashAttention 风格的编译期配置
// ============================================================================
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: FlashAttention 的 Tile 策略                        │
// │                                                                  │
// │  Q (seq_len × head_dim)    K^T (head_dim × seq_len)             │
// │  ┌─────────────────┐       ┌─────────────────┐                   │
// │  │ Q_tile 0         │       │ K^T_tile 0       │                   │
// │  │ (Br × d)         │   ×   │ (d × Bc)         │                   │
// │  ├─────────────────┤       ├─────────────────┤                   │
// │  │ Q_tile 1         │       │ K^T_tile 1       │                   │
// │  │ (Br × d)         │       │ (d × Bc)         │                   │
// │  ├─────────────────┤       ├─────────────────┤                   │
// │  │ ...              │       │ ...              │                   │
// │  └─────────────────┘       └─────────────────┘                   │
// │                                                                  │
// │  Algorithm:                                                      │
// │  1. Outer loop: iterate over K,V tiles (loaded to SRAM)         │
// │  2. Inner loop: for each Q tile, compute QK^T (in SRAM)        │
// │  3. Softmax rescaling (online softmax, keeps running stats)    │
// │  4. Compute PV = softmax(QK^T) × V (in SRAM)                  │
// │  5. Write output to HBM                                        │
// │                                                                  │
// │  Key insight: FlashAttention 通过 tiling 和 recomputation      │
// │  避免了 O(N^2) 的 attention matrix 写回 HBM。                   │
// │  传统 attention: O(N^2) memory → 序列长度瓶颈                    │
// │  FlashAttention: O(N) memory → 可以处理长序列                    │
// └──────────────────────────────────────────────────────────────────┘
//
// WHY 编译期配置:
//   FlashAttention 的 tile 大小 Directly 影响 SRAM 使用。
//   SRAM = Q_tile(Br×d) + K_tile(Bc×d) + V_tile(Bc×d) + S(Br×Bc) + O(Br×d)
//
//   SM80 A100: 164KB SRAM
//     Br=128, Bc=128, d=64: 2×(128×64×2) + 2×(128×64×2) + (128×128×4) + (128×64×4)
//     = 32KB + 32KB + 64KB + 32KB = 160KB ✓ (刚好在预算内)
//
//   SM90 H100: 228KB SRAM
//     Br=256, Bc=128, d=128: 更大 tile，更高效率
//
// 这些计算全部在编译期完成 → 编译器分配最优 SRAM 布局

// ============================================================================
// AttentionConfig 主配置
// ============================================================================

template <
    int HeadDim_,           // head dimension (通常 64 或 128)
    int Br_,                // Q tile size (rows)
    int Bc_,                // KV tile size (rows)
    int NumWarps_,          // 每个 block 的 warp 数
    bool Causal_ = false,   // 因果 mask (decoder-only 模型)
    bool UseDropout_ = false,
    float DropoutProb_ = 0.0f,
    bool UseAlibi_ = false  // ALiBi 位置编码
>
struct AttentionConfig {
  // ── 编译期常量 ──
  static constexpr int kHeadDim = HeadDim_;
  static constexpr int kBr = Br_;
  static constexpr int kBc = Bc_;
  static constexpr int kNumWarps = NumWarps_;
  static constexpr int kNumThreads = NumWarps_ * 32;

  static constexpr bool kCausal = Causal_;
  static constexpr bool kUseDropout = UseDropout_;
  static constexpr bool kUseAlibi = UseAlibi_;

  // ── SRAM 预算计算 (以 FP16 为例) ──
  static constexpr int kElementSize = 2; // FP16

  // Q tile: Br × d 个 half
  static constexpr int kSmemQ = kBr * kHeadDim * kElementSize;

  // K tile: Bc × d 个 half
  static constexpr int kSmemK = kBc * kHeadDim * kElementSize;

  // V tile: Bc × d 个 half
  static constexpr int kSmemV = kBc * kHeadDim * kElementSize;

  // S = QK^T: Br × Bc 个 float (softmax 需要 FP32 精度)
  static constexpr int kSmemS = kBr * kBc * 4; // float

  // O (output accumulator): Br × d 个 float
  static constexpr int kSmemO = kBr * kHeadDim * 4; // float

  // 额外: m (running max), l (running sum) 用于 online softmax
  static constexpr int kSmemM = kBr * 4;
  static constexpr int kSmemL = kBr * 4;

  // 总 SRAM 使用
  static constexpr int kSmemTotal =
      kSmemQ + kSmemK + kSmemV + kSmemS + kSmemO + kSmemM + kSmemL;

  // ── 缩放因子 ──
  // softmax_scale = 1/√d (标准 attention)
  static constexpr float kSoftmaxScale = 1.0f / static_cast<float>(kHeadDim);

  // ── 架构兼容性 ──
  // 检查是否适合 SM80 (164KB SRAM)
  static constexpr bool kFitsSm80 = (kSmemTotal <= 164 * 1024);

  // 检查是否适合 SM90 (228KB SRAM)
  static constexpr bool kFitsSm90 = (kSmemTotal <= 228 * 1024);
};

// ============================================================================
// 预定义 Attention 配置 (受 FlashAttention-2 论文启发)
// ============================================================================

// SM80 A100 最优配置 (d=64)
using AttnConfigSm80D64 = AttentionConfig<64, 128, 128, 4>;

// SM80 A100 最优配置 (d=128)
using AttnConfigSm80D128 = AttentionConfig<128, 128, 64, 8>;

// SM90 H100 最优配置 (d=64, 更大的 tile)
using AttnConfigSm90D64 = AttentionConfig<64, 256, 128, 8>;

// SM90 H100 最优配置 (d=128)
using AttnConfigSm90D128 = AttentionConfig<128, 256, 64, 16>;

// Causal attention (GPT decoder 需要)
using AttnConfigCausalSm80D128 = AttentionConfig<128, 128, 64, 8, true>;

// ALiBi attention (MosaicML 使用)
using AttnConfigAlibiSm80D64 = AttentionConfig<64, 128, 128, 4, false, false, 0.0f, true>;

// ============================================================================
// AttentionConfig 选择器 - 根据 head_dim 和架构自动选择
// ============================================================================

template <int HeadDim, int Arch>
struct DefaultAttentionConfig;

// SM80 A100
template <>
struct DefaultAttentionConfig<64, 80> {
  using type = AttnConfigSm80D64;
};

template <>
struct DefaultAttentionConfig<128, 80> {
  using type = AttnConfigSm80D128;
};

// SM90 H100
template <>
struct DefaultAttentionConfig<64, 90> {
  using type = AttnConfigSm90D64;
};

template <>
struct DefaultAttentionConfig<128, 90> {
  using type = AttnConfigSm90D128;
};

template <int HeadDim, int Arch>
using default_attention_config_t = typename DefaultAttentionConfig<HeadDim, Arch>::type;

} // namespace mini_inference
