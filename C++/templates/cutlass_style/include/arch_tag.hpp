#pragma once

#include <cstdint>
#include <type_traits>

namespace cutlass_style {

// ============================================================================
// GPU 架构标签 - 编译期架构识别
// ============================================================================
//
// WHY 需要编译期架构分派:
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: GPU 架构与可用指令的关系                           │
// │                                                                  │
// │  SM70 (Volta)  ──┬── Tensor Core 1.0: mma.sync (FP16 only)     │
// │                  │── Shared Memory: 96KB configurable            │
// │                  │── Max Threads/SM: 2048                        │
// │                  │                                              │
// │  SM75 (Turing)  ──┬── Tensor Core 2.0: INT8, INT4              │
// │                  │── Shared Memory: 64KB                         │
// │                  │── 统一 L1/Shared Memory                       │
// │                  │                                              │
// │  SM80 (Ampere)  ──┬── Tensor Core 3.0: TF32, BF16, FP64       │
// │  (A100)          │── Shared Memory: 164KB                       │
// │                  │── 异步拷贝: cp.async                          │
// │                  │── Sparse Tensor Core                          │
// │                  │                                              │
// │  SM90 (Hopper)  ──┬── Tensor Core 4.0: FP8 (E4M3, E5M2)       │
// │  (H100)          │── TMA: Tensor Memory Accelerator             │
// │                  │── wgmma: Warp Group MMA                      │
// │                  │── Distributed Shared Memory                   │
// └──────────────────────────────────────────────────────────────────┘
//
// 如果架构检测放在运行时:
//   if (arch == SM80) {
//       kernel_sm80<<<...>>>(...);  // SM80 PTX
//   } else if (arch == SM90) {
//       kernel_sm90<<<...>>>(...);  // SM90 PTX
//   }
//   → 所有架构的代码都编译进二进制 → 二进制膨胀 3-4 倍
//   → SM90 代码 (含 wgmma) 在 SM70 上无法编译 (指令不存在)
//
// 如果架构检测放在编译期:
//   using Kernel = KernelSelector<ArchTag>::Kernel;  // 只编译一种架构
//   → 二进制只包含当前架构的代码
//   → 可以使用该架构独有的 PTX 指令
//   → 编译错误（如果用了不支持的指令）在编译期暴露
//
// 类比: ArchTag 相当于 C 预处理的 #ifdef __CUDA_ARCH__ >= 800
//       但 ArchTag 是类型系统的一部分，享受类型安全和 SFINAE 的好处。

struct Sm70 {
  static constexpr int compute_capability = 70;
  static constexpr const char* name = "Volta (V100)";

  // 架构特性检测 (编译期)
  static constexpr bool has_tensor_core = true;
  static constexpr bool has_fp16_tensor = true;
  static constexpr bool has_int8_tensor = false;
  static constexpr bool has_bf16_tensor = false;
  static constexpr bool has_tf32_tensor = false;
  static constexpr bool has_fp8_tensor = false;
  static constexpr bool has_cp_async = false;
  static constexpr bool has_tma = false;
  static constexpr bool has_wgmma = false;

  // Shared memory 配置
  static constexpr int max_shared_memory_per_sm = 96 * 1024;   // 96 KB
  static constexpr int max_shared_memory_per_block = 48 * 1024; // 48 KB

  // 寄存器限制
  static constexpr int max_registers_per_thread = 255;
  static constexpr int max_registers_per_sm = 65536;

  // Warp 调度器数量
  static constexpr int num_warp_schedulers = 4;
};

struct Sm75 {
  static constexpr int compute_capability = 75;
  static constexpr const char* name = "Turing (T4/RTX 2080)";

  static constexpr bool has_tensor_core = true;
  static constexpr bool has_fp16_tensor = true;
  static constexpr bool has_int8_tensor = true;
  static constexpr bool has_bf16_tensor = false;
  static constexpr bool has_tf32_tensor = false;
  static constexpr bool has_fp8_tensor = false;
  static constexpr bool has_cp_async = false;
  static constexpr bool has_tma = false;
  static constexpr bool has_wgmma = false;

  static constexpr int max_shared_memory_per_sm = 64 * 1024;
  static constexpr int max_shared_memory_per_block = 48 * 1024;
  static constexpr int max_registers_per_thread = 255;
  static constexpr int max_registers_per_sm = 65536;
  static constexpr int num_warp_schedulers = 4;
};

struct Sm80 {
  static constexpr int compute_capability = 80;
  static constexpr const char* name = "Ampere (A100/A6000)";

  static constexpr bool has_tensor_core = true;
  static constexpr bool has_fp16_tensor = true;
  static constexpr bool has_int8_tensor = true;
  static constexpr bool has_bf16_tensor = true;
  static constexpr bool has_tf32_tensor = true;
  static constexpr bool has_fp8_tensor = false;
  static constexpr bool has_cp_async = true;
  static constexpr bool has_tma = false;
  static constexpr bool has_wgmma = false;

  static constexpr int max_shared_memory_per_sm = 164 * 1024;  // 164 KB (A100)
  static constexpr int max_shared_memory_per_block = 48 * 1024;
  static constexpr int max_registers_per_thread = 255;
  static constexpr int max_registers_per_sm = 65536;
  static constexpr int num_warp_schedulers = 4;
};

struct Sm90 {
  static constexpr int compute_capability = 90;
  static constexpr const char* name = "Hopper (H100/H200)";

  static constexpr bool has_tensor_core = true;
  static constexpr bool has_fp16_tensor = true;
  static constexpr bool has_int8_tensor = true;
  static constexpr bool has_bf16_tensor = true;
  static constexpr bool has_tf32_tensor = true;
  static constexpr bool has_fp8_tensor = true;       // E4M3, E5M2
  static constexpr bool has_cp_async = true;
  static constexpr bool has_tma = true;              // Tensor Memory Accelerator
  static constexpr bool has_wgmma = true;            // Warp Group MMA

  static constexpr int max_shared_memory_per_sm = 228 * 1024;  // 228 KB (H100)
  static constexpr int max_shared_memory_per_block = 48 * 1024;
  static constexpr int max_registers_per_thread = 255;
  static constexpr int max_registers_per_sm = 65536;
  static constexpr int num_warp_schedulers = 4;
};

// ============================================================================
// 编译期架构检测工具
// ============================================================================

// is_sm80_v<T>: 是否是 SM80 架构
template <typename ArchTag>
inline constexpr bool is_sm70_v = (ArchTag::compute_capability == 70);

template <typename ArchTag>
inline constexpr bool is_sm75_v = (ArchTag::compute_capability == 75);

template <typename ArchTag>
inline constexpr bool is_sm80_v = (ArchTag::compute_capability == 80);

template <typename ArchTag>
inline constexpr bool is_sm90_v = (ArchTag::compute_capability >= 90);

template <typename ArchTag>
inline constexpr bool is_sm80_plus_v = (ArchTag::compute_capability >= 80);

template <typename ArchTag>
inline constexpr bool is_sm90_plus_v = (ArchTag::compute_capability >= 90);

// ============================================================================
// 架构版本比较 (用于编译期条件选择)
// ============================================================================

template <typename ArchA, typename ArchB>
inline constexpr bool is_same_arch_v =
    (ArchA::compute_capability == ArchB::compute_capability);

template <typename ArchA, typename ArchB>
inline constexpr bool is_newer_arch_v =
    (ArchA::compute_capability > ArchB::compute_capability);

// ============================================================================
// 所有支持的架构列表
// ============================================================================

template <typename...>
struct SupportedArchitectures;

// 默认所有架构都注册 (实际项目中通过构建系统控制)
using AllArchitectures = SupportedArchitectures<Sm70, Sm75, Sm80, Sm90>;

} // namespace cutlass_style
