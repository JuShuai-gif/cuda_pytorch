#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cutlass_style {

// ============================================================================
// RowMajor Layout - 行优先存储
// ============================================================================
//
// Layout 系统的核心设计决策: 为什么 Layout 进模板参数而不进运行时？
//
// ┌──────────────────────────────────────────────────────────────┐
// │  Mermaid 架构图: 编译期 Layout vs 运行时 Layout              │
// │                                                              │
// │  运行时 Layout (❌ 传统做法):                                │
// │  ┌─────────────────┐                                         │
// │  │ layout_enum lyt   │──▶ if (lyt == RowMajor) { ... }       │
// │  │ data_ptr         │    else if (lyt == ColMajor) { ... }   │
// │  └─────────────────┘    → 每次访问都有分支                    │
// │                         → 分支预测失败 = 20 cycles            │
// │                         → 无法内联 offset 计算                │
// │                                                              │
// │  编译期 Layout (✅ CUTLASS 做法):                            │
// │  ┌─────────────────┐                                         │
// │  │ TensorRef<T,      │──▶ ldr.global.f32 %r, [%ptr + %off]  │
// │  │  RowMajor>        │    offset = row*ldm + col             │
// │  └─────────────────┘    → 零分支，零开销                      │
// │                         → offset 计算完全内联                 │
// │                         → 编译器可以预取、向量化              │
// └──────────────────────────────────────────────────────────────┘
//
// 类比: Layout 进模板参数，相当于 C 语言的
//   #define INDEX(row, col) ((row)*(ldm)+(col))
// 而不是
//   int index(int row, int col, int ldm, LayoutType layout)
//
// WHY: GPU 上分支的代价极高。一个 warp 中如果线程发散，
//      所有线程都必须等待分支的两条路径都执行完毕。
//      编译期 Layout 完全消除了这种发散。

// ============================================================================
// RowMajor - 行优先 Layout
// ============================================================================
//
// 内存布局 (以 3 行 4 列为例):
//   Logical:  [0,0] [0,1] [0,2] [0,3]
//             [1,0] [1,1] [1,2] [1,3]
//             [2,0] [2,1] [2,2] [2,3]
//
//   Physical: [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] [1,2] [1,3] [2,0] [2,1] [2,2] [2,3]
//              ↑───────────────────↑  ↑───────────────────↑  ↑───────────────────↑
//              Row 0 (stride=4)      Row 1 (stride=4)      Row 2 (stride=4)
//
// C/CUDA 默认 layout。A[i][j] 时 i 变化更慢，j 变化更快。
//
// 模板展开后:
//   RowMajor{}(2, 3, ldm=4) → 编译器直接计算: 2*4 + 3 = 11
//   → 对应一条单周期整数乘加指令: mad.lo.u32 %r, %row, %ldm, %col

struct RowMajor {
  // operator() 返回线性偏移量
  // 类比: 二维笛卡尔坐标 → 一维数组索引的映射函数
  // 编译器可以完全内联这个函数，因为它对每个实例化都是确定的
  constexpr std::size_t operator()(
      std::size_t row,
      std::size_t col,
      std::size_t leading_dimension  // 类比: 矩阵的"宽度"，即每行元素数
  ) const noexcept {
    // WHY 用乘加而非两步骤:
    // GPU 的 mad (multiply-add) 指令可以在一个周期内完成
    // 编译器可以将其映射为: mad.lo.u32 %off, %row, %ldm, %col
    return row * leading_dimension + col;
  }

  // 从线性索引反算行列 (调试用，kernel 中一般不使用)
  static constexpr std::size_t row_from_index(
      std::size_t index,
      std::size_t leading_dimension) noexcept {
    return index / leading_dimension;
  }

  static constexpr std::size_t col_from_index(
      std::size_t index,
      std::size_t leading_dimension) noexcept {
    return index % leading_dimension;
  }

  // Layout 名称 (编译期常量)
  static constexpr const char* name = "RowMajor";

  // 判断是否连续访问 (row 不变时 col 递增，地址连续)
  // 用于编译器自动向量化优化
  static constexpr bool is_row_contiguous = true;
};

// ============================================================================
// ColumnMajor - 列优先 Layout
// ============================================================================
//
// 内存布局 (以 3 行 4 列为例):
//   Logical:  [0,0] [0,1] [0,2] [0,3]
//             [1,0] [1,1] [1,2] [1,3]
//             [2,0] [2,1] [2,2] [2,3]
//
//   Physical: [0,0] [1,0] [2,0] [0,1] [1,1] [2,1] [0,2] [1,2] [2,2] [0,3] [1,3] [2,3]
//              ↑───────────↑  ↑───────────↑  ↑───────────↑  ↑───────────↑
//              Col 0 (stride=3) Col 1 (stride=3) Col 2       Col 3
//
// Fortran/MATLAB 默认 layout。BLAS 库中矩阵乘法偏好列优先。
//
// WHY CUTLASS 同时支持两种 Layout:
//   - PyTorch 默认 RowMajor (C-contiguous)
//   - cuBLAS 默认 ColumnMajor (Fortran-contiguous)
//   - 避免不必要的转置 = 节省 2x 内存和带宽
//
// 模板展开后:
//   ColumnMajor{}(2, 3, ldm=3) → 编译器计算: 3*3 + 2 = 11
//   → 和 RowMajor 不同，col 乘 stride 而非 row
//   → 但编译器生成的指令完全相同: mad.lo.u32 %off, %col, %ldm, %row

struct ColumnMajor {
  constexpr std::size_t operator()(
      std::size_t row,
      std::size_t col,
      std::size_t leading_dimension  // 列优先时 ldm = 行数 (列 stride)
  ) const noexcept {
    return col * leading_dimension + row;
  }

  static constexpr std::size_t row_from_index(
      std::size_t index,
      std::size_t leading_dimension) noexcept {
    return index % leading_dimension;
  }

  static constexpr std::size_t col_from_index(
      std::size_t index,
      std::size_t leading_dimension) noexcept {
    return index / leading_dimension;
  }

  static constexpr const char* name = "ColumnMajor";

  // 列优先时，列内元素地址连续
  static constexpr bool is_row_contiguous = false;
};

// ============================================================================
// NCHW Layout - 图像/特征图 Layout (Batch, Channel, Height, Width)
// ============================================================================
//
// 类比: 一堆扑克牌叠在一起。N=牌堆数, C=每堆牌的张数,
//       H=每张牌的行数, W=每张牌的列数。
//       NCHW 先遍历 Width，再 Height，再 Channel，再 Batch。
//
// 常用于: cuDNN, Caffe, 老版本 PyTorch
//
// WHY CUTLASS 需要支持 NCHW:
//   Conv2D 的 im2col 变换和 GEMM 的多维索引都需要 Layout 抽象。
//   把 Conv 变成 GEMM 后，NCHW→NHWC 就是 Layout 的切换。

struct NCHW {
  // 4D → 1D 索引映射
  constexpr std::size_t operator()(
      std::size_t n,  // batch index
      std::size_t c,  // channel index
      std::size_t h,  // height index
      std::size_t w,  // width index
      std::size_t C,  // number of channels (stride for N)
      std::size_t H,  // height (stride for C)
      std::size_t W   // width (stride for H)
  ) const noexcept {
    // WHY 这个公式:
    // NCHW 意味着最外层是 N，最内层是 W
    // offset = n*C*H*W + c*H*W + h*W + w
    // 编译器可以将 C*H*W 和 H*W 预计算为常量
    return ((n * C + c) * H + h) * W + w;
  }

  static constexpr const char* name = "NCHW";
};

// ============================================================================
// NHWC Layout - 通道最后 Layout
// ============================================================================
//
// 常用于: TensorFlow, TensorRT, 新版 PyTorch (channels_last)
//
// WHY NHWC 在 GPU 上更高效:
//   - GPU Tensor Core 偏好连续的元素访问
//   - NHWC 中同一像素的所有通道紧邻，利于向量化加载
//   - Conv 的 filter 展开时，NHWC 减少 shared memory bank conflict

struct NHWC {
  constexpr std::size_t operator()(
      std::size_t n,  // batch index
      std::size_t h,  // height index
      std::size_t w,  // width index
      std::size_t c,  // channel index
      std::size_t H,  // height (stride for N)
      std::size_t W,  // width (stride for H)
      std::size_t C   // number of channels (stride for W)
  ) const noexcept {
    // NHWC: 最外层 N，最内层 C
    // offset = n*H*W*C + h*W*C + w*C + c
    return ((n * H + h) * W + w) * C + c;
  }

  static constexpr const char* name = "NHWC";
};

// ============================================================================
// Layout Traits - 编译期 Layout 特征提取
// ============================================================================
//
// WHY: SFINAE 和 concept 需要依赖 Layout 特征来做 dispatch。
//      比如: "如果 Layout 是行连续的，用 vectorized load"
//            "如果 Layout 是列连续的，用 warp-level 转置"

template <typename Layout>
struct LayoutTraits {
  // 默认: 二维 layout
  static constexpr int rank = 2;
  static constexpr bool is_row_major = false;
  static constexpr bool is_column_major = false;
};

template <>
struct LayoutTraits<RowMajor> {
  static constexpr int rank = 2;
  static constexpr bool is_row_major = true;
  static constexpr bool is_column_major = false;
};

template <>
struct LayoutTraits<ColumnMajor> {
  static constexpr int rank = 2;
  static constexpr bool is_row_major = false;
  static constexpr bool is_column_major = true;
};

template <>
struct LayoutTraits<NCHW> {
  static constexpr int rank = 4;
  static constexpr bool is_row_major = false;
  static constexpr bool is_column_major = false;
};

template <>
struct LayoutTraits<NHWC> {
  static constexpr int rank = 4;
  static constexpr bool is_row_major = false;
  static constexpr bool is_column_major = false;
};

// ============================================================================
// 编译期 Layout 兼容性检测
// ============================================================================
//
// WHY: 在 dispatch 阶段，需要验证 A 和 B 的 layout 组合是否合法。
//      比如某些 tile 配置只在 RowMajor x ColumnMajor 时最优。

template <typename LayoutA, typename LayoutB>
struct is_layout_compatible {
  // 默认所有组合兼容；特定组合可以特化
  static constexpr bool value = true;
};

// 已知的最优组合 (CUTLASS 内部特殊优化)
template <>
struct is_layout_compatible<RowMajor, ColumnMajor> {
  static constexpr bool value = true;
  static constexpr const char* note =
      "NN GEMM: A row-major, B col-major → C = A*B 无需转置";
};

template <>
struct is_layout_compatible<ColumnMajor, RowMajor> {
  static constexpr bool value = true;
  static constexpr const char* note =
      "TT GEMM: A col-major, B row-major → C = A^T*B^T 无需转置";
};

} // namespace cutlass_style
