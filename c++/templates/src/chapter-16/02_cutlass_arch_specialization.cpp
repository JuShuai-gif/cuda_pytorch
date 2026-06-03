// ============================================================================
// 02_cutlass_arch_specialization.cpp - 模拟 CUTLASS ArchTag
//                                       特化系统
// ============================================================================
//
// 目的:
//   CUTLASS 使用复杂的架构标签（ArchTag）系统
//   来为不同 GPU 代次（SM70/SM75/SM80/SM90）特化 kernel。
//   每个 ArchTag 是一个空结构体，触发配置模板、
//   指令选择器和内存布局的
//   不同偏特化。
//
// CUTLASS ARCHTAG 系统:
//   - ArchTag 是一个空结构体（例如 arch::Sm80）
//   - 它被用作配置结构体中的模板参数
//   - 对 ArchTag 的偏特化选择:
//     * Tile 大小（新架构上更大）
//     * 指令集（SM80+ 使用 mma.sync，SM90+ 使用 wgmma）
//     * 共享内存布局（swizzle 模式）
//     * 流水线阶段数（更快 GPU 上更多阶段）
//     * Epilogue tile 大小
//
// 关键优势:
//   - 编译期选择: 没有运行时分发开销
//   - 类型安全: 不会意外在 SM70 上使用 SM90 特性
//   - 可扩展性: 添加新 ArchTag 无需修改已有代码
//   - 单一来源: 相同的 kernel 逻辑，不同的特化
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <cstdint>

// ============================================================================
// 第 1 节: 架构标签定义
// ============================================================================

namespace cutlass_arch {

/// \brief SM70（Volta, V100）的架构标签。
/// 第一代带有 Tensor Core 的架构。
struct Sm70 {
    static constexpr int kComputeCapability  = 70;
    static constexpr int kSharedMemKB        = 96;
    static constexpr int kMaxThreadsPerSM    = 2048;
    static constexpr bool kHasTensorCore     = true;
    static constexpr bool kHasMMA            = true;   // mma.sync
    static constexpr bool kHasAsyncCopy      = false;  // cp.async 在 SM80 才引入
    static constexpr bool kHasTMA            = false;  // 仅 SM90
    static constexpr bool kHasWGMMA          = false;  // 仅 SM90
    static constexpr const char* kArchName   = "Volta (SM70)";
    static constexpr const char* kGpuName    = "V100";
};

/// \brief SM75（Turing, T4/RTX 20xx）的架构标签。
/// 增加了整数点积指令。
struct Sm75 {
    static constexpr int kComputeCapability  = 75;
    static constexpr int kSharedMemKB        = 64;   // T4 有 64KB 可配置
    static constexpr int kMaxThreadsPerSM    = 1024;
    static constexpr bool kHasTensorCore     = true;
    static constexpr bool kHasMMA            = true;
    static constexpr bool kHasAsyncCopy      = false;
    static constexpr bool kHasTMA            = false;
    static constexpr bool kHasWGMMA          = false;
    static constexpr bool kHasIntegerDotProd = true;  // Turing 特有
    static constexpr const char* kArchName   = "Turing (SM75)";
    static constexpr const char* kGpuName    = "T4";
};

/// \brief SM80（Ampere, A100/A6000）的架构标签。
/// 增加了异步拷贝、更大的共享内存、结构化稀疏性。
struct Sm80 {
    static constexpr int kComputeCapability  = 80;
    static constexpr int kSharedMemKB        = 163;
    static constexpr int kMaxThreadsPerSM    = 2048;
    static constexpr bool kHasTensorCore     = true;
    static constexpr bool kHasMMA            = true;
    static constexpr bool kHasAsyncCopy      = true;
    static constexpr bool kHasTMA            = false;
    static constexpr bool kHasWGMMA          = false;
    static constexpr bool kHasSparseMMA      = true;  // 2:4 结构化稀疏性
    static constexpr const char* kArchName   = "Ampere (SM80)";
    static constexpr const char* kGpuName    = "A100";
};

/// \brief SM90（Hopper, H100）的架构标签。
/// 增加了 TMA（Tensor 内存加速器）和 WGMMA（Warp Group MMA）。
struct Sm90 {
    static constexpr int kComputeCapability  = 90;
    static constexpr int kSharedMemKB        = 227;
    static constexpr int kMaxThreadsPerSM    = 2048;
    static constexpr bool kHasTensorCore     = true;
    static constexpr bool kHasMMA            = true;
    static constexpr bool kHasAsyncCopy      = true;
    static constexpr bool kHasTMA            = true;
    static constexpr bool kHasWGMMA          = true;
    static constexpr bool kHasSparseMMA      = true;
    static constexpr const char* kArchName   = "Hopper (SM90)";
    static constexpr const char* kGpuName    = "H100";
};

/// \brief SM100（Blackwell, B200）的架构标签。
/// 下一代 — 前瞻性模拟。
struct Sm100 {
    static constexpr int kComputeCapability  = 100;
    static constexpr int kSharedMemKB        = 256;
    static constexpr int kMaxThreadsPerSM    = 2048;
    static constexpr bool kHasTensorCore     = true;
    static constexpr bool kHasMMA            = true;
    static constexpr bool kHasAsyncCopy      = true;
    static constexpr bool kHasTMA            = true;
    static constexpr bool kHasWGMMA          = true;
    static constexpr bool kHasSparseMMA      = true;
    static constexpr bool kHasFP4            = true;  // 4 位浮点
    static constexpr const char* kArchName   = "Blackwell (SM100)";
    static constexpr const char* kGpuName    = "B200";
};

} // namespace cutlass_arch

// ============================================================================
// 第 2 节: 按架构特化的配置
// ============================================================================
//
// GemmTileConfig 被按 ArchTag 偏特化。
// 每个特化提供该架构的最优 tile 大小
// 和流水线配置。

/// \brief 主模板（无特化） — 回退配置。
template <typename ArchTag, typename = void>
struct GemmTileConfig {
    static constexpr int kThreadblockM = 64;
    static constexpr int kThreadblockN = 64;
    static constexpr int kThreadblockK = 8;
    static constexpr int kWarpM        = 16;
    static constexpr int kWarpN        = 16;
    static constexpr int kWarpK        = 8;
    static constexpr int kStages       = 2;
    static constexpr int kThreads      = 128;
    static constexpr const char* config_name() { return "通用-回退"; }
};

/// \brief SM70 特化 — Volta 优化的 tile 大小。
template <typename ArchTag>
struct GemmTileConfig<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, cutlass_arch::Sm70>>>
{
    static constexpr int kThreadblockM = 128;
    static constexpr int kThreadblockN = 128;
    static constexpr int kThreadblockK = 16;
    static constexpr int kWarpM        = 32;
    static constexpr int kWarpN        = 32;
    static constexpr int kWarpK        = 16;
    static constexpr int kStages       = 2;
    static constexpr int kThreads      = 256;
    static constexpr const char* config_name() { return "SM70-Volta"; }
};

/// \brief SM75 特化 — Turing 优化。
template <typename ArchTag>
struct GemmTileConfig<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, cutlass_arch::Sm75>>>
{
    static constexpr int kThreadblockM = 128;
    static constexpr int kThreadblockN = 128;
    static constexpr int kThreadblockK = 32;
    static constexpr int kWarpM        = 32;
    static constexpr int kWarpN        = 32;
    static constexpr int kWarpK        = 16;
    static constexpr int kStages       = 3;
    static constexpr int kThreads      = 256;
    static constexpr const char* config_name() { return "SM75-Turing"; }
};

/// \brief SM80 特化 — 带异步拷贝的 Ampere。
template <typename ArchTag>
struct GemmTileConfig<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, cutlass_arch::Sm80>>>
{
    static constexpr int kThreadblockM = 256;
    static constexpr int kThreadblockN = 128;
    static constexpr int kThreadblockK = 32;
    static constexpr int kWarpM        = 64;
    static constexpr int kWarpN        = 64;
    static constexpr int kWarpK        = 32;
    static constexpr int kStages       = 4;  // 更多流水线阶段用于异步拷贝
    static constexpr int kThreads      = 256;
    static constexpr const char* config_name() { return "SM80-Ampere"; }
};

/// \brief SM90 特化 — 带 TMA 的 Hopper。
template <typename ArchTag>
struct GemmTileConfig<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, cutlass_arch::Sm90>>>
{
    static constexpr int kThreadblockM = 256;
    static constexpr int kThreadblockN = 256;  // 使用 TMA 加倍 N
    static constexpr int kThreadblockK = 64;   // 使用 TMA 加深 K
    static constexpr int kWarpM        = 64;
    static constexpr int kWarpN        = 64;
    static constexpr int kWarpK        = 32;
    static constexpr int kStages       = 5;   // 更深的流水线
    static constexpr int kThreads      = 256;
    static constexpr const char* config_name() { return "SM90-Hopper-TMA"; }
};

/// \brief SM100 特化 — Blackwell（前瞻性）。
template <typename ArchTag>
struct GemmTileConfig<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, cutlass_arch::Sm100>>>
{
    static constexpr int kThreadblockM = 512;
    static constexpr int kThreadblockN = 256;
    static constexpr int kThreadblockK = 128;
    static constexpr int kWarpM        = 128;
    static constexpr int kWarpN        = 64;
    static constexpr int kWarpK        = 64;
    static constexpr int kStages       = 6;
    static constexpr int kThreads      = 512;
    static constexpr const char* config_name() { return "SM100-Blackwell"; }
};

// ============================================================================
// 第 3 节: 通过 ArchTag 特化进行指令选择
// ============================================================================

/// \brief 基于 ArchTag 选择合适的 MMA 指令。
/// 不同架构使用不同的矩阵乘加
/// 指令格式。

template <typename ArchTag, typename = void>
struct MmaInstructionSelector {
    static constexpr const char* instruction() { return "FMA（标量）"; }
    static constexpr int kMmaM = 1;
    static constexpr int kMmaN = 1;
    static constexpr int kMmaK = 1;
};

/// \brief SM70/SM75: mma.sync.aligned.m8n8k4（Volta/Turing Tensor Core）。
template <typename ArchTag>
struct MmaInstructionSelector<ArchTag,
    std::enable_if_t<
        std::is_same_v<ArchTag, cutlass_arch::Sm70> ||
        std::is_same_v<ArchTag, cutlass_arch::Sm75>
    >>
{
    static constexpr const char* instruction() { return "mma.sync.aligned.m8n8k4"; }
    static constexpr int kMmaM = 8;
    static constexpr int kMmaN = 8;
    static constexpr int kMmaK = 4;
};

/// \brief SM80: mma.sync.aligned.m16n8k16（Ampere Tensor Core）。
template <typename ArchTag>
struct MmaInstructionSelector<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, cutlass_arch::Sm80>>>
{
    static constexpr const char* instruction() { return "mma.sync.aligned.m16n8k16"; }
    static constexpr int kMmaM = 16;
    static constexpr int kMmaN = 8;
    static constexpr int kMmaK = 16;
};

/// \brief SM90/SM100: wgmma.mma_async（Hopper Warp Group MMA）。
template <typename ArchTag>
struct MmaInstructionSelector<ArchTag,
    std::enable_if_t<
        std::is_same_v<ArchTag, cutlass_arch::Sm90> ||
        std::is_same_v<ArchTag, cutlass_arch::Sm100>
    >>
{
    static constexpr const char* instruction() { return "wgmma.mma_async.sync.aligned.m64n64k16"; }
    static constexpr int kMmaM = 64;
    static constexpr int kMmaN = 64;
    static constexpr int kMmaK = 16;
};

// ============================================================================
// 第 4 节: 共享内存布局特化
// ============================================================================

/// \brief 共享内存 swizzle 模式 — 防止 bank 冲突。
/// 不同架构有不同的共享内存 bank 大小。

template <typename ArchTag, typename = void>
struct SharedMemoryConfig {
    static constexpr int kBankSize    = 4;   // 每 bank 4 字节
    static constexpr int kSwizzleMode = 0;   // 无 swizzle
    static constexpr const char* mode_name() { return "无Swizzle"; }
};

template <typename ArchTag>
struct SharedMemoryConfig<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, cutlass_arch::Sm80>>>
{
    static constexpr int kBankSize    = 4;
    static constexpr int kSwizzleMode = 1;   // 32 字节交错
    static constexpr const char* mode_name() { return "SM80-Swizzle-32B"; }
};

template <typename ArchTag>
struct SharedMemoryConfig<ArchTag,
    std::enable_if_t<
        std::is_same_v<ArchTag, cutlass_arch::Sm90> ||
        std::is_same_v<ArchTag, cutlass_arch::Sm100>
    >>
{
    static constexpr int kBankSize    = 4;
    static constexpr int kSwizzleMode = 2;   // 128 字节交错
    static constexpr const char* mode_name() { return "SM90-Swizzle-128B"; }
};

// ============================================================================
// 第 5 节: 统一架构查询接口
// ============================================================================

/// \brief 编译期架构查询。
/// 将所有架构特定配置聚合到一个接口中。
template <typename ArchTag>
struct ArchitectureTraits {
    using TileConfig   = GemmTileConfig<ArchTag>;
    using MmaSelector  = MmaInstructionSelector<ArchTag>;
    using SharedMemory = SharedMemoryConfig<ArchTag>;

    /// \brief 打印全面的架构摘要。
    static void describe() {
        std::cout << "=== 架构: " << ArchTag::kArchName
                  << " (" << ArchTag::kGpuName << ") ===\n";
        std::cout << "  计算能力:         " << ArchTag::kComputeCapability << "\n";
        std::cout << "  共享内存:          " << ArchTag::kSharedMemKB << " KB\n";
        std::cout << "  Tensor Core:        "
                  << (ArchTag::kHasTensorCore ? "是" : "否") << "\n";
        std::cout << "  异步拷贝:          "
                  << (ArchTag::kHasAsyncCopy ? "是" : "否") << "\n";
        std::cout << "  TMA:                "
                  << (ArchTag::kHasTMA ? "是" : "否") << "\n";
        std::cout << "  WGMMA:              "
                  << (ArchTag::kHasWGMMA ? "是" : "否") << "\n";
        std::cout << "  Tile 配置:          " << TileConfig::config_name()
                  << " (" << TileConfig::kThreadblockM << "x"
                  << TileConfig::kThreadblockN << "x"
                  << TileConfig::kThreadblockK << ", "
                  << TileConfig::kStages << " 阶段)\n";
        std::cout << "  MMA 指令:          " << MmaSelector::instruction()
                  << " (" << MmaSelector::kMmaM << "x"
                  << MmaSelector::kMmaN << "x" << MmaSelector::kMmaK << ")\n";
        std::cout << "  共享内存布局:       " << SharedMemory::mode_name() << "\n";
        std::cout << "  每 SM 最大线程数:   " << ArchTag::kMaxThreadsPerSM << "\n";
    }
};

// ============================================================================
// 第 6 节: 编译期架构检测与分发
// ============================================================================

/// \brief 模拟目标架构的编译期检测。
/// 在真实 CUTLASS 中，这使用 __CUDA_ARCH__ 或类似宏。
/// 为了模拟，我们通过模板参数选择。
template <typename ArchTag>
void configure_and_report() {
    ArchitectureTraits<ArchTag>::describe();
}

/// \brief 编译期配置聚合器。
/// 给定一个 ArchTag，组装完整的 kernel 配置。
template <typename ArchTag>
struct KernelConfiguration {
    using Traits = ArchitectureTraits<ArchTag>;

    static constexpr int kThreadblockM = Traits::TileConfig::kThreadblockM;
    static constexpr int kThreadblockN = Traits::TileConfig::kThreadblockN;
    static constexpr int kThreadblockK = Traits::TileConfig::kThreadblockK;
    static constexpr int kStages       = Traits::TileConfig::kStages;
    static constexpr const char* kMmaInstruction = Traits::MmaSelector::instruction();

    /// \brief 验证此架构的配置是否有效。
    static constexpr bool validate() {
        // 所有 tile 必须为正
        if (kThreadblockM <= 0 || kThreadblockN <= 0 || kThreadblockK <= 0)
            return false;
        // Tile 维度必须是 MMA 指令维度的倍数
        if (kThreadblockM % Traits::MmaSelector::kMmaM != 0) return false;
        if (kThreadblockN % Traits::MmaSelector::kMmaN != 0) return false;
        if (kThreadblockK % Traits::MmaSelector::kMmaK != 0) return false;
        return true;
    }
};

// ============================================================================
// 第 7 节: 编译期验证
// ============================================================================

// 验证所有配置有效
static_assert(GemmTileConfig<cutlass_arch::Sm70>::kThreadblockM > 0);
static_assert(GemmTileConfig<cutlass_arch::Sm80>::kStages == 4);
static_assert(GemmTileConfig<cutlass_arch::Sm90>::kThreadblockN == 256);

// 验证 MMA 指令选择
static_assert(MmaInstructionSelector<cutlass_arch::Sm80>::kMmaM == 16);
static_assert(MmaInstructionSelector<cutlass_arch::Sm90>::kMmaK == 16);

// 验证共享内存配置
static_assert(SharedMemoryConfig<cutlass_arch::Sm80>::kSwizzleMode == 1);
static_assert(SharedMemoryConfig<cutlass_arch::Sm90>::kSwizzleMode == 2);

// 验证架构标签唯一性
static_assert(!std::is_same_v<cutlass_arch::Sm80, cutlass_arch::Sm90>);
static_assert(!std::is_same_v<cutlass_arch::Sm70, cutlass_arch::Sm100>);

// 验证已知架构的 kernel 配置
static_assert(KernelConfiguration<cutlass_arch::Sm70>::validate());
static_assert(KernelConfiguration<cutlass_arch::Sm80>::validate());
static_assert(KernelConfiguration<cutlass_arch::Sm90>::validate());

// 验证 Sm100 使用的 tile 比 Sm80 的大
static_assert(
    GemmTileConfig<cutlass_arch::Sm100>::kThreadblockK >
    GemmTileConfig<cutlass_arch::Sm80>::kThreadblockK
);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== CUTLASS ArchTag 特化系统 ===\n\n";

    // 第 5 节: 架构特征
    configure_and_report<cutlass_arch::Sm70>();
    std::cout << "\n";
    configure_and_report<cutlass_arch::Sm75>();
    std::cout << "\n";
    configure_and_report<cutlass_arch::Sm80>();
    std::cout << "\n";
    configure_and_report<cutlass_arch::Sm90>();
    std::cout << "\n";
    configure_and_report<cutlass_arch::Sm100>();

    // 第 6 节: Kernel 配置验证
    std::cout << "\n--- 配置验证 ---\n";
    std::cout << std::boolalpha;
    std::cout << "SM70 配置有效:  "
              << KernelConfiguration<cutlass_arch::Sm70>::validate() << "\n";
    std::cout << "SM80 配置有效:  "
              << KernelConfiguration<cutlass_arch::Sm80>::validate() << "\n";
    std::cout << "SM90 配置有效:  "
              << KernelConfiguration<cutlass_arch::Sm90>::validate() << "\n";
    std::cout << "SM100 配置有效: "
              << KernelConfiguration<cutlass_arch::Sm100>::validate() << "\n";

    std::cout << "\nArchTag 特化系统演示完成。\n";
    return 0;
}
