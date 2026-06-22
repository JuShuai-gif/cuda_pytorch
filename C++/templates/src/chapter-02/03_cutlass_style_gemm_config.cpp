// =============================================================================
// 第 02.3 章：CUTLASS 风格的 GemmConfiguration 模板类
//
// CUTLASS 使用一组丰富的编译期配置参数（以模板类型编码）来组织
// GEMM（通用矩阵乘法）内核：
//
//   GemmConfiguration<
//     TileShape<M, N, K>,           // 线程块 tile 维度
//     Mainloop<...>,                // 主循环策略（全局内存 -> 共享内存）
//     Epilogue<..., OutputOp>       // 尾声策略（共享内存 -> 全局内存 + bias/激活函数）
//   >
//
// 本文件模拟该设计，包含：
//   1. 策略标签类型（Mainloop、Epilogue、OutputOp）
//   2. GemmConfiguration 主模板
//   3. 不同策略组合的偏特化
//   4. 通过 static_assert 进行编译期验证
//   5. 通过 trait 提取参数
//
// 编译：g++ -std=c++20 -o 03_cutlass_style_gemm_config 03_cutlass_style_gemm_config.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// 1. Tile 形状（非类型 + 类型参数）
// =============================================================================
// 在真实 CUTLASS 中：template <int M_, int N_, int K_> struct GemmShape {};

template <int M_, int N_, int K_>
struct GemmShape {
  static constexpr int M = M_;  // C 的行数 / 线程块 tile
  static constexpr int N = N_;  // C 的列数 / 线程块 tile
  static constexpr int K = K_;  // 内层维度 tile（K 维度）

  static constexpr int threadblock_elements = M * N;
  static constexpr int total_flops = M * N * K * 2;  // 乘加各一次

  static void print() {
    std::cout << "    GemmShape<M=" << M << ", N=" << N << ", K=" << K << ">"
              << " (flops=" << total_flops << ")" << std::endl;
  }
};

// =============================================================================
// 2. 策略标签（主循环）
// =============================================================================
// 主循环策略控制数据如何从全局内存加载到共享内存，
// 以及 warp 级矩阵乘法如何执行。

// 基线：简单加载，无双缓冲，单阶段流水线
struct MainloopBaseline {
  static constexpr char const* name      = "Baseline";
  static constexpr int  stages           = 1;   // 单缓冲
  static constexpr bool double_buffering = false;
  static constexpr bool swizzle          = false;
};

// 优化版：双缓冲共享内存，swizzle 布局以避免 bank conflict
struct MainloopOptimized {
  static constexpr char const* name      = "Optimized";
  static constexpr int  stages           = 2;   // 双缓冲
  static constexpr bool double_buffering = true;
  static constexpr bool swizzle          = true;
};

// =============================================================================
// 3. 策略标签（尾声）
// =============================================================================
// 尾声策略处理从共享内存到全局内存的最终输出，
// 包括可选融合操作（bias、ReLU 等）。

// --- 输出操作（融合到尾声中）---
struct LinearCombination {
  static constexpr char const* name = "LinearCombination";
  static constexpr bool has_bias    = false;
  static constexpr bool has_relu    = false;
  static void describe() {
    std::cout << "      Op：LinearCombination (D = alpha*A*B)" << std::endl;
  }
};

struct LinearCombinationRelu {
  static constexpr char const* name = "LinearCombinationRelu";
  static constexpr bool has_bias    = false;
  static constexpr bool has_relu    = true;
  static void describe() {
    std::cout << "      Op：LinearCombination+ReLU (D = relu(alpha*A*B))"
              << std::endl;
  }
};

struct LinearCombinationBiasRelu {
  static constexpr char const* name = "LinearCombinationBiasRelu";
  static constexpr bool has_bias    = true;
  static constexpr bool has_relu    = true;
  static void describe() {
    std::cout << "      Op：LinearCombination+Bias+ReLU "
              << "(D = relu(alpha*A*B + C))" << std::endl;
  }
};

// --- 尾声策略 ---
template <typename OutputOp>
struct EpilogueBaseline {
  using output_op = OutputOp;
  static constexpr char const* name = "EpilogueBaseline";
  static constexpr bool vectorized_store = false;
  static constexpr int  vector_width     = 1;
};

template <typename OutputOp>
struct EpilogueVectorized {
  using output_op = OutputOp;
  static constexpr char const* name = "EpilogueVectorized";
  static constexpr bool vectorized_store = true;
  static constexpr int  vector_width     = 4;  // 128 位存储
};

// =============================================================================
// 4. GemmConfiguration -- 主配置类
// =============================================================================
// 这是内核参数化的顶层配置。
// 在 CUTLASS 中，此类针对不同策略组合进行偏特化，以优化代码生成。

template <typename Shape_,
          typename Mainloop_,
          typename Epilogue_,
          typename ElementA_ = float,
          typename ElementB_ = float,
          typename ElementC_ = float>
struct GemmConfiguration {
  using Shape    = Shape_;
  using Mainloop = Mainloop_;
  using Epilogue = Epilogue_;

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;

  // --- 编译期验证 ---
  // 确保 tile 维度合理（CUTLASS 在编译期强制执行 2 的幂
  // 和对齐约束）。
  static_assert(Shape::M > 0 && Shape::N > 0 && Shape::K > 0,
                "Tile 维度必须为正数");
  static_assert(Shape::M % 8 == 0 && Shape::N % 8 == 0,
                "Tile M 和 N 必须是 8 的倍数以支持 warp 级 MMA");
  static_assert(Shape::K % 8 == 0,
                "Tile K 必须是 8 的倍数");

  // --- 推导的编译期配置 ---
  // 每个线程块的 warp 数
  static constexpr int warps_per_tb = (Shape::M * Shape::N) / (32 * 4);  // 简化
  // 所需最小共享内存（字节）
  static constexpr int smem_size =
      (Shape::M * Shape::K + Shape::K * Shape::N) * sizeof(ElementA) *
      Mainloop::stages;

  // --- 运行时描述 ---
  static void describe() {
    std::cout << "GemmConfiguration：" << std::endl;
    Shape::print();
    std::cout << "    Mainloop：" << Mainloop::name
              << " (stages=" << Mainloop::stages
              << ", dbuf=" << Mainloop::double_buffering
              << ", swizzle=" << Mainloop::swizzle << ")" << std::endl;
    std::cout << "    Epilogue：" << Epilogue::name
              << " (vstore=" << Epilogue::vectorized_store
              << ", vwidth=" << Epilogue::vector_width << ")" << std::endl;
    Epilogue::output_op::describe();
    std::cout << "    Warps/线程块：" << warps_per_tb << std::endl;
    std::cout << "    共享内存：" << smem_size << " 字节" << std::endl;
  }
};

// =============================================================================
// 5. 编译期配置工厂（元函数）
// =============================================================================
// 一个辅助元函数，从更简单的参数构造 GemmConfiguration，
// 选择合适的默认值。这模拟了 CUTLASS 的 DefaultGemmConfiguration 辅助。

template <int M, int N, int K,
          typename OutputOp = LinearCombination>
struct DefaultGemmConfig {
  // 根据 K 维度选择主循环策略（启发式）
  using Mainloop = std::conditional_t<
      (K >= 64), MainloopOptimized, MainloopBaseline>;

  // 根据输出操作选择尾声策略
  template <typename Op>
  struct ChooseEpilogue {
    // 如果操作涉及 ReLU 或 bias，使用向量化尾声以提高吞吐量
    using type = std::conditional_t<
        Op::has_relu || Op::has_bias,
        EpilogueVectorized<Op>,
        EpilogueBaseline<Op>>;
  };

  using Epilogue = typename ChooseEpilogue<OutputOp>::type;

  using type = GemmConfiguration<
      GemmShape<M, N, K>,
      Mainloop,
      Epilogue
  >;
};

template <int M, int N, int K, typename OutputOp = LinearCombination>
using default_gemm_config_t =
    typename DefaultGemmConfig<M, N, K, OutputOp>::type;

// =============================================================================
// 6. 从 Config 的运行时分发（模拟内核）
// =============================================================================
// 在真实 CUTLASS 中，GemmConfiguration 作为模板参数传递给内核。
// 这里模拟一个读取其配置的内核。

template <typename Config>
struct SimulatedKernel {
  using Shape    = typename Config::Shape;
  using Mainloop = typename Config::Mainloop;
  using Epilogue = typename Config::Epilogue;

  void launch(int M, int N, int K) const {
    std::cout << "\n  [模拟内核] 启动：" << std::endl;
    Config::describe();

    // 从问题大小和 tile 形状计算 grid 维度
    int grid_m = (M + Shape::M - 1) / Shape::M;
    int grid_n = (N + Shape::N - 1) / Shape::N;
    std::cout << "    Grid：(" << grid_m << ", " << grid_n << ")"
              << " 问题大小 (M=" << M << ", N=" << N << ", K=" << K << ")"
              << std::endl;

    // 模拟工作
    std::cout << "    使用 " << Mainloop::stages << "-阶段流水线"
              << " 和 " << Epilogue::name << " 执行中..." << std::endl;
  }
};

// =============================================================================
// 7. 配置验证元函数
// =============================================================================
// CUTLASS 在编译期验证配置。如果配置无效（例如 tile 太大超出共享内存），
// static_assert 会触发并给出明确的错误消息。

template <typename Config, int SmemLimitBytes = 48 * 1024>
struct ValidateConfig {
  static constexpr bool smem_ok =
      Config::smem_size <= SmemLimitBytes;
  static constexpr bool valid = smem_ok;

  static_assert(smem_ok, "共享内存使用超出 SM 限制");
};

template <typename Config, int Limit = 48 * 1024>
inline constexpr bool validate_config_v =
    ValidateConfig<Config, Limit>::valid;

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 02.3 章：CUTLASS 风格的 GemmConfiguration ===\n" << endl;

  // --- 测试 1：手动配置 ---
  cout << "[测试 1] 手动 GemmConfiguration：" << endl;
  using ManualConfig = GemmConfiguration<
      GemmShape<128, 128, 8>,
      MainloopBaseline,
      EpilogueBaseline<LinearCombination>,
      float, float, float
  >;
  ManualConfig::describe();

  // 编译期检查
  static_assert(ManualConfig::Shape::M == 128);
  static_assert(ManualConfig::Shape::N == 128);
  static_assert(ManualConfig::Mainloop::stages == 1);
  static_assert(!ManualConfig::Epilogue::vectorized_store);
  static_assert(ManualConfig::smem_size > 0);
  static_assert(validate_config_v<ManualConfig>);

  // --- 测试 2：默认配置工厂 ---
  cout << "\n[测试 2] DefaultGemmConfig 工厂：" << endl;
  using DefaultCfg1 = default_gemm_config_t<128, 64, 32>;
  DefaultCfg1::describe();
  static_assert(is_same_v<DefaultCfg1::Mainloop, MainloopBaseline>);

  using DefaultCfg2 = default_gemm_config_t<128, 128, 64, LinearCombinationRelu>;
  DefaultCfg2::describe();
  static_assert(is_same_v<DefaultCfg2::Mainloop, MainloopOptimized>);
  static_assert(DefaultCfg2::Epilogue::vectorized_store);

  // --- 测试 3：模拟内核启动 ---
  cout << "\n[测试 3] 使用不同配置的模拟内核：" << endl;

  SimulatedKernel<ManualConfig> kernel1;
  kernel1.launch(1024, 1024, 64);

  using VecConfig = default_gemm_config_t<64, 64, 32, LinearCombinationBiasRelu>;
  SimulatedKernel<VecConfig> kernel2;
  kernel2.launch(512, 256, 32);

  // --- 测试 4：配置 traits ---
  cout << "\n[测试 4] 从配置中提取 traits：" << endl;
  using Cfg = default_gemm_config_t<256, 128, 32>;

  using ExtractedShape = typename Cfg::Shape;
  cout << "  Tile M：" << ExtractedShape::M << endl;
  cout << "  Tile N：" << ExtractedShape::N << endl;
  cout << "  Tile K：" << ExtractedShape::K << endl;
  cout << "  每个 tile 的 FLOPs 总数：" << ExtractedShape::total_flops << endl;

  using ExtractedMainloop = typename Cfg::Mainloop;
  cout << "  Mainloop 名称：" << ExtractedMainloop::name << endl;

  using ExtractedOutputOp = typename Cfg::Epilogue::output_op;
  cout << "  输出操作名称：" << ExtractedOutputOp::name << endl;

  // --- 测试 5：无效配置检测（已注释——会触发 static_assert 失败）---
  cout << "\n[测试 5] 配置验证：" << endl;
  // 这会失败：M 不能不是 8 的倍数
  // using BadConfig = GemmConfiguration<GemmShape<13, 13, 13>, MainloopBaseline,
  //                                      EpilogueBaseline<LinearCombination>>;
  // static_assert(BadConfig::Shape::M > 0);  // 这是 OK 的
  // GemmConfiguration 内部的 static_assert 会触发

  using GoodConfig = GemmConfiguration<
      GemmShape<128, 128, 8>,
      MainloopOptimized,
      EpilogueVectorized<LinearCombinationRelu>
  >;
  cout << "  GoodConfig 已验证：smem_size=" << GoodConfig::smem_size
       << "，valid=" << validate_config_v<GoodConfig> << endl;
  static_assert(validate_config_v<GoodConfig>);

  // --- 测试 6：多种输出操作 ---
  cout << "\n[测试 6] 不同的输出操作：" << endl;

  using CfgNone  = default_gemm_config_t<64, 64, 32, LinearCombination>;
  using CfgRelu  = default_gemm_config_t<64, 64, 32, LinearCombinationRelu>;
  using CfgBias  = default_gemm_config_t<64, 64, 32, LinearCombinationBiasRelu>;

  cout << "  LinearCombination：" << endl;
  CfgNone::describe();
  cout << "  LinearCombination+ReLU：" << endl;
  CfgRelu::describe();
  cout << "  LinearCombination+Bias+ReLU：" << endl;
  CfgBias::describe();

  static_assert(!CfgNone::Epilogue::vectorized_store);
  static_assert(CfgRelu::Epilogue::vectorized_store);
  static_assert(CfgBias::Epilogue::vectorized_store);

  // --- 测试 7：相同形状、不同策略产生不同类型 ---
  cout << "\n[测试 7] 相同形状，不同策略：" << endl;
  static_assert(!is_same_v<CfgNone, CfgRelu>);
  static_assert(!is_same_v<CfgRelu, CfgBias>);
  cout << "  CfgNone != CfgRelu != CfgBias（如预期，不同类型）"
       << endl;

  cout << "\n所有测试通过！" << endl;
  return 0;
}
