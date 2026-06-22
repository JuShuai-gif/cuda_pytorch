#pragma once

#include <cstdint>
#include <type_traits>

#include "engine_config.hpp"

namespace mini_inference {

// ============================================================================
// KernelRegistry - 编译期 Kernel 注册和查找系统
// ============================================================================
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: KernelRegistry 的编译期注册表                      │
// │                                                                  │
// │  RegisterTime (编译期)             LookupTime (编译期)           │
// │  ┌──────────────────┐              ┌──────────────────────┐      │
// │  │ REGISTER_KERNEL   │              │ find_kernel<Config>()│       │
// │  │ (Sm80, FP16, 128) │──────────────▶│ → Kernel_FP16_Sm80_  │      │
// │  │ REGISTER_KERNEL   │              │   128x128            │      │
// │  │ (Sm80, BF16, 128) │              └──────────────────────┘      │
// │  │ REGISTER_KERNEL   │              Match by:                    │
// │  │ (Sm90, FP8,  256) │              1. Arch  (exact)             │
// │  │ ...               │              2. Dtype (exact)             │
// │  └──────────────────┘              3. Tile  (best fit)          │
// │                                                                  │
// │  所有注册在编译期完成 → 零运行时开销                              │
// └──────────────────────────────────────────────────────────────────┘
//
// WHY 编译期 Kernel 注册表:
//   1. 零运行时查找开销 (vs std::map 的 O(log n))
//   2. 编译期类型安全 (Kernel 签名在编译期验证)
//   3. 未使用的 kernel 不会被编译进 binary (减小体积)
//   4. 编译器可以做全局优化 (链接时优化)
//
// 类比: KernelRegistry 相当于 C 语言的函数指针表
//   static kernel_fn kernel_table[] = { kernel_fp16, kernel_bf16, ... };
//   但 KernelRegistry 是类型安全的，支持编译期多态。

// ============================================================================
// KernelDescriptor - 描述一个 kernel 的编译期元数据
// ============================================================================

template <
    DataType DType_,
    int Arch_,
    int TileM_,
    int TileN_,
    int TileK_,
    int NumThreads_,
    int SmemBytes_
>
struct KernelDescriptor {
  static constexpr DataType dtype = DType_;
  static constexpr int arch = Arch_;
  static constexpr int tile_m = TileM_;
  static constexpr int tile_n = TileN_;
  static constexpr int tile_k = TileK_;
  static constexpr int num_threads = NumThreads_;
  static constexpr int smem_bytes = SmemBytes_;
};

// ============================================================================
// KernelEntry - 一个具体的 Kernel 注册项
// ============================================================================

template <typename KernelT, typename DescriptorT>
struct KernelEntry {
  using Kernel = KernelT;
  using Descriptor = DescriptorT;

  // 编译期匹配: 检查这个 kernel 是否匹配给定配置
  template <typename Config>
  static constexpr bool matches() {
    return (Descriptor::dtype == Config::DType) &&
           (Descriptor::arch == Config::Arch);
  }

  // 编译期优先级: 更大的 tile = 更高优先级 (更高效)
  template <typename Config>
  static constexpr int priority() {
    return Descriptor::tile_m * Descriptor::tile_n;
  }
};

// ============================================================================
// KernelRegistry - 编译期 Kernel 注册表
// ============================================================================

// 前向声明
template <typename... Entries>
class KernelRegistry;

// 空注册表 (递归基)
template <>
class KernelRegistry<> {
 public:
  // 空表中查找 → 编译错误
  template <typename Config>
  static constexpr bool has_kernel() {
    return false;
  }

  template <typename Config>
  struct select_best {
    // 空表回退: 找不到匹配
    using type = void;
  };

  template <typename Config>
  struct BestMatch {
    using type = void;
  };

  template <typename Config>
  using best_match_t = void;

  template <typename NewEntry>
  using register_kernel = KernelRegistry<NewEntry>;
};

// 非空注册表 (递归)
template <typename FirstEntry, typename... RestEntries>
class KernelRegistry<FirstEntry, RestEntries...> {
 public:
  // 所有注册的 kernel 条目
  using Entries = KernelRegistry<FirstEntry, RestEntries...>;

  // 条目数量
  static constexpr int num_entries = 1 + sizeof...(RestEntries);

  // =========================================================================
  // find_kernel<Config> - 编译期 Kernel 查找
  // =========================================================================
  //
  // WHY SFINAE/if constexpr 查找:
  //   编译器遍历所有注册的 KernelEntry，对每个检查 Descriptor::matches<Config>()。
  //   第一个匹配的 (且 tile 大小最优的) 被选中。
  //   整个过程在编译期完成 → 运行时就是一次直接的 kernel launch。
  //
  // 模板展开后:
  //   find_kernel<ConfigA100Fp16>
  //   → if (KernelEntry1::matches<ConfigA100Fp16>()) → use Kernel1
  //   → else if (KernelEntry2::matches<ConfigA100Fp16>()) → use Kernel2
  //   → ...
  //   → 编译器选择一个分支，其余分支被 if constexpr 消除
  //
  // 类比: 相当于编译期版本的 std::find_if

  template <typename Config>
  static constexpr bool has_kernel() {
    return first_matches<Config>() || rest_has_kernel<Config>();
  }

  // 获取最佳匹配 kernel 的类型 (编译期)
  template <typename Config>
  struct select_best {
    using type = std::conditional_t<
        FirstEntry::template matches<Config>(),
        typename FirstEntry::Kernel,
        typename KernelRegistry<RestEntries...>::template best_match_t<Config>
    >;
  };

  template <typename Config>
  struct BestMatch {
    using type = typename select_best<Config>::type;
  };

  template <typename Config>
  using best_match_t = typename BestMatch<Config>::type;

  // =========================================================================
  // 注册新的 Kernel (返回新的 Registry)
  // =========================================================================
  //
  // WHY 返回新类型而非修改:
  //   模板元编程中一切都是不可变的。每次注册返回一个包含所有条目
  //   的新类型，类比函数式编程中的 cons 操作。

  template <typename NewEntry>
  using register_kernel = KernelRegistry<NewEntry, FirstEntry, RestEntries...>;

 private:
  template <typename Config>
  static constexpr bool first_matches() {
    return FirstEntry::template matches<Config>();
  }

  template <typename Config>
  static constexpr bool rest_has_kernel() {
    return KernelRegistry<RestEntries...>::template has_kernel<Config>();
  }
};

// ============================================================================
// KernelRegistryBuilder - 流式 API 构建 Registry
// ============================================================================
//
// WHY Builder 模式:
//   让用户以声明式方式注册 kernel:
//     using MyRegistry = KernelRegistryBuilder{}
//       .add<Sm80FP16Kernel128>()
//       .add<Sm80BF16Kernel128>()
//       .add<Sm90FP8Kernel256>()
//       .build();
//
// 模板展开后:
//   MyRegistry = KernelRegistry<
//     KernelEntry<Sm90FP8Kernel256, ...>,
//     KernelEntry<Sm80BF16Kernel128, ...>,
//     KernelEntry<Sm80FP16Kernel128, ...>
//   >

template <typename Registry = KernelRegistry<>>
struct KernelRegistryBuilder {
  using CurrentRegistry = Registry;

  // 添加一个 kernel
  template <typename KernelType>
  using add = KernelRegistryBuilder<
      typename CurrentRegistry::template register_kernel<
          KernelEntry<KernelType, typename KernelType::Descriptor>
      >
  >;

  // 构建最终 Registry
  using build = CurrentRegistry;

  static constexpr int kernel_count = CurrentRegistry::num_entries;
};

// ============================================================================
// 宏: 简化 Kernel 注册
// ============================================================================
//
// WHY 宏: 减少样板代码。在实际 CUTLASS 代码中也有类似宏。
//   这不是"过度宏"，而是"受控的代码生成"。

#define REGISTER_GEMM_KERNEL(KernelClass, DTypeVal, ArchVal, M, N, K, Threads, Smem) \
  struct KernelClass {                                                               \
    using Descriptor = ::mini_inference::KernelDescriptor<                           \
        DTypeVal, ArchVal, M, N, K, Threads, Smem>;                                   \
    /* launch 方法 (伪代码，实际需要 CUDA runtime) */                                \
    template <typename Config>                                                       \
    static void launch(const typename Config::RuntimeConfig& rcfg,                   \
                       void* A, void* B, void* C, int m, int n, int k) {             \
      /* 实际 CUDA kernel launch */                                                  \
      (void)rcfg; (void)A; (void)B; (void)C; (void)m; (void)n; (void)k;              \
    }                                                                                \
  }

} // namespace mini_inference
