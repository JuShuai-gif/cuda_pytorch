#pragma once

#include <cstdint>
#include <cmath>

namespace mini_inference {

// ============================================================================
// LayerNormConfig - Layer Normalization 编译期配置
// ============================================================================
//
// LayerNorm 是 Transformer 推理中调用频率最高的 normalization 操作之一。
// 每个 Transformer block 至少有 2 次 LayerNorm (pre-attn + pre-ffn)。
// 对于 DeepSeek、LLaMA 等大模型，每层 forward 至少:
//   1. Pre-Attention LayerNorm
//   2. Post-Attention LayerNorm (有的架构省略)
//   3. Pre-FFN LayerNorm
//   4. Final LayerNorm
//
// 高频调用 → 必须极致优化 → 编译期配置。使用 Welford 算法在线计算
// mean 和 variance (在 warp-level 做 reduction)。

// ============================================================================
// LayerNormConfig 主配置
// ============================================================================

template <
    int NormalizedShape_,      // 归一化维度 (通常 head_dim 或 hidden_dim)
    float Epsilon_ = 1e-5f,    // 数值稳定性常数
    int WarpsPerBlock_ = 4,    // 每个 block 的 warp 数
    int ElementsPerThread_ = 4,// 每个线程处理的元素数 (向量化)
    bool UseAffine_ = true,    // 是否使用 γ (scale) 和 β (shift)
    bool UseRmsNorm_ = false   // 是否为 RMSNorm (不计算 mean，只计算 RMS)
>
struct LayerNormConfig {
  static constexpr int kNormalizedShape = NormalizedShape_;
  static constexpr float kEpsilon = Epsilon_;
  static constexpr int kWarpsPerBlock = WarpsPerBlock_;
  static constexpr int kElementsPerThread = ElementsPerThread_;
  static constexpr int kThreadsPerBlock = WarpsPerBlock_ * 32;

  static constexpr bool kUseAffine = UseAffine_;
  static constexpr bool kUseRmsNorm = UseRmsNorm_;

  // ── 编译期验证 ──
  // 每个线程至少处理 kElementsPerThread 个元素
  static constexpr int kTotalElementsPerBlock =
      kThreadsPerBlock * kElementsPerThread;

  static_assert(kTotalElementsPerBlock >= kNormalizedShape,
                "NormalizedShape too large for given threads and elements/thread");

  // ── Shared Memory 需求 ──
  // 用于 warp-level reduction 的共享内存
  // 每个 warp 需要一个 float 用于 partial mean/variance
  static constexpr int kSmemSize =
      kWarpsPerBlock * 2 * sizeof(float); // mean + variance per warp

  // ── 性能模型 ──
  // 计算每个元素的工作量 (用于性能预估)
  // LayerNorm: y = (x - μ) / √(σ² + ε) * γ + β
  //   - 读取 x: 1 read
  //   - 计算 μ: 1 pass over data
  //   - 计算 σ²: 1 pass over data (或与 μ 并行)
  //   - 写出 y: 1 write (如果原地更新)

  // RMSNorm: y = x / √(E[x²] + ε) * γ
  //   - 更简单，省略了 mean 的计算和减法
  //   - LLM 中逐渐流行 (LLaMA, T5 使用)
};

// ============================================================================
// 预定义 LayerNorm 配置
// ============================================================================

// 标准 LayerNorm: hidden_dim=4096 (LLaMA-7B)
using LnConfig4096 = LayerNormConfig<4096, 1e-5f, 8, 16>;

// 标准 LayerNorm: hidden_dim=8192 (LLaMA-70B)
using LnConfig8192 = LayerNormConfig<8192, 1e-5f, 16, 16>;

// RMSNorm: hidden_dim=4096 (LLaMA 系列广泛使用)
using RmsNormConfig4096 = LayerNormConfig<4096, 1e-5f, 8, 16, true, true>;

// RMSNorm: hidden_dim=7168 (DeepSeek-V2)
using RmsNormConfig7168 = LayerNormConfig<7168, 1e-6f, 16, 16, true, true>;

// ============================================================================
// LayerNorm 运行时的辅助信息
// ============================================================================

struct LayerNormParams {
  const void* gamma = nullptr;  // 缩放因子 (γ)
  const void* beta = nullptr;   // 偏移因子 (β)
  int rows = 0;                 // batch_size × seq_len
  int cols = 0;                 // normalized_shape (即 hidden_dim)

  // 运行时验证
  bool is_valid() const {
    return rows > 0 && cols > 0;
  }
};

// ============================================================================
// LayerNormType 枚举 (用于运行时选择特化版本)
// ============================================================================

enum class LayerNormType : int {
  kStandardLayerNorm = 0,
  kRMSNorm = 1,
  kGroupNorm = 2,     // 未来可扩展
  kInstanceNorm = 3,   // 未来可扩展
};

// 编译期: LayerNormType → 最合适的配置
template <LayerNormType LnType, int HiddenDim>
struct DefaultLayerNormConfig;

template <int HiddenDim>
struct DefaultLayerNormConfig<LayerNormType::kStandardLayerNorm, HiddenDim> {
  using type = LayerNormConfig<HiddenDim, 1e-5f, 8, 16, true, false>;
};

template <int HiddenDim>
struct DefaultLayerNormConfig<LayerNormType::kRMSNorm, HiddenDim> {
  using type = LayerNormConfig<HiddenDim, 1e-5f, 8, 16, true, true>;
};

} // namespace mini_inference
