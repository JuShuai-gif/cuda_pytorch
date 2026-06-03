#pragma once

#include <cstddef>
#include <type_traits>

#include "layout.hpp"

namespace cutlass_style {

// ============================================================================
// TensorRef<Element, Layout> - 零开销张量引用
// ============================================================================
//
// WHY TensorRef 存在: CUTLASS 需要一种"零开销"的张量描述方式。
//
// 类比: TensorRef 相当于 C++20 的 std::span<T>，但拓展到多维。
//       span 只持有指针+大小，零拷贝，零所有权。
//       TensorRef 持有指针+维度+stride，零拷贝，零所有权。
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: TensorRef 内存布局 vs 逻辑视图                     │
// │                                                                  │
// │  物理内存 (连续):                                                │
// │  [a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23]              │
// │                                                                  │
// │  TensorRef<float, RowMajor> ref(ptr, 3, 4, 4);                  │
// │  ┌───────────────────────┐                                       │
// │  │ ptr_    → 指向 a00    │                                       │
// │  │ rows_   = 3           │ 逻辑视图:                             │
// │  │ cols_   = 4           │ [a00 a01 a02 a03]  row=0, col=0      │
// │  │ stride_ = 4           │ [a10 a11 a12 a13]  row=1, col=1      │
// │  │ Layout  = RowMajor    │ [a20 a21 a22 a23]  row=2, col=2      │
// │  └───────────────────────┘                                       │
// │                                                                  │
// │  ref(1, 2) → 1*4 + 2 = 6 → 访问 a12                             │
// │  → Layout::operator() 被内联 → 单条 mad 指令                    │
// └──────────────────────────────────────────────────────────────────┘
//
// WHY Layout 进模板参数而非运行时:
//   看 ref(1, 2) 的访问路径:
//     - 编译期 Layout: offset = Layout::operator()(1, 2, stride)
//       → 编译器知道是 RowMajor::operator() = 1*stride + 2
//       → 映射为单条 PTX 指令: mad.lo.u32 %r, 1, %stride, 2
//
//     - 运行时 Layout: offset = layout_fn(1, 2, stride)
//       → 编译器不知道 layout_fn 是哪个实现
//       → 需要间接调用/分支 → 10-20 cycles 开销
//       → 阻止编译器做软件流水线、预取等优化
//
//   在 GPU 最内层循环中，每个线程每次迭代至少做 8 次这样的偏移计算。
//   K=32 的 tile × 4 warps × 每 warp 64 次 MMA = 8192 次偏移计算。
//   编译期 Layout 省下的 10 cycles × 8192 = ~82000 cycles ≈ 节省 5% 延迟。

template <typename Element_, typename Layout_>
class TensorRef {
public:
  // Types exposed for metaprogramming
  using Element = Element_;
  using Layout = Layout_;

  // =========================================================================
  // 构造函数
  // =========================================================================
  //
  // 零开销构造: 只存 3 个成员变量 (指针+2个维度+stride)
  // 运行时成本: 4 次寄存器写入 (极微)

  constexpr TensorRef(
      Element* data,
      std::size_t rows,
      std::size_t cols,
      std::size_t leading_dimension = 0  // 0 = 默认使用 cols
  ) noexcept
    : data_(data)
    , rows_(rows)
    , cols_(cols)
    , leading_dimension_(leading_dimension > 0 ? leading_dimension : cols) {}

  // =========================================================================
  // 元素访问 (编译期内联)
  // =========================================================================
  //
  // 模板展开后 (TensorRef<float, RowMajor>):
  //   ref(2, 3) → RowMajor::operator()(2, 3, leading_dimension_)
  //   → 2 * leading_dimension_ + 3
  //   → 编译器内联为单条乘加指令
  //
  // constexpr 标记: 如果实参是编译期常量，整个调用可在编译期求值
  // (用于静态初始化和常量表达式上下文)

  constexpr Element& operator()(std::size_t row, std::size_t col) noexcept {
    return data_[Layout{}(row, col, leading_dimension_)];
  }

  constexpr const Element& operator()(std::size_t row, std::size_t col) const noexcept {
    return data_[Layout{}(row, col, leading_dimension_)];
  }

  // =========================================================================
  // Raw pointer access (used by kernel code for vectorized loads/stores)
  // =========================================================================

  constexpr Element* data() noexcept { return data_; }
  constexpr const Element* data() const noexcept { return data_; }

  // =========================================================================
  // 维度访问
  // =========================================================================

  constexpr std::size_t rows() const noexcept { return rows_; }
  constexpr std::size_t cols() const noexcept { return cols_; }
  constexpr std::size_t leading_dimension() const noexcept { return leading_dimension_; }
  constexpr std::size_t size() const noexcept { return rows_ * cols_; }

  // =========================================================================
  // 子张量视图 (slice)
  // =========================================================================
  //
  // WHY slice: 推理引擎中经常需要处理矩阵的"一部分"
  //   - attention: 取 Q 的第 i 个 head
  //   - layer norm: 取输入的某个 batch
  //   - GEMM epilogue: 取 C 矩阵的子块写回
  //
  // TensorRef 的 slice 是零拷贝的——只修改偏移和维度，不复制数据。
  // 类比: Python 的 arr[2:5, :] 如果是 view 则零拷贝，
  //       如果是 copy 则复制数据。TensorRef 永远零拷贝。

  constexpr TensorRef slice(
      std::size_t row_offset,
      std::size_t col_offset,
      std::size_t slice_rows,
      std::size_t slice_cols) const noexcept {
    return TensorRef(
        data_ + Layout{}(row_offset, col_offset, leading_dimension_),
        slice_rows,
        slice_cols,
        leading_dimension_
    );
  }

  // =========================================================================
  // 比较运算符 (用于测试)
  // =========================================================================

  bool operator==(const TensorRef& other) const noexcept {
    return data_ == other.data_ &&
           rows_ == other.rows_ &&
           cols_ == other.cols_ &&
           leading_dimension_ == other.leading_dimension_;
  }

  bool operator!=(const TensorRef& other) const noexcept {
    return !(*this == other);
  }

private:
  Element* data_;                    // 数据指针 (不拥有所有权)
  std::size_t rows_;                // 行数
  std::size_t cols_;                // 列数
  std::size_t leading_dimension_;   // 主维度步长 (>= cols，用于 padding)
};

// ============================================================================
// 4D TensorRef 特化 (用于 Conv/Attention)
// ============================================================================
//
// WHY 4D: Attention 中的 QKV 是 4D tensor [batch, heads, seq_len, head_dim]
//         NCHW/NHWC 的 Conv 也需要 4D 访问

template <typename Element_, typename Layout_>
class Tensor4DRef {
public:
  using Element = Element_;
  using Layout = Layout_;

  constexpr Tensor4DRef(
      Element* data,
      std::size_t n, std::size_t c, std::size_t h, std::size_t w,
      std::size_t stride_c = 0, std::size_t stride_h = 0, std::size_t stride_w = 0
  ) noexcept
    : data_(data)
    , n_(n), c_(c), h_(h), w_(w)
    , stride_c_(stride_c > 0 ? stride_c : h * w)
    , stride_h_(stride_h > 0 ? stride_h : w)
    , stride_w_(stride_w > 0 ? stride_w : 1) {}

  constexpr Element& operator()(
      std::size_t n, std::size_t c, std::size_t h, std::size_t w) noexcept {
    return data_[Layout{}(n, c, h, w, stride_c_, stride_h_, stride_w_)];
  }

  constexpr const Element& operator()(
      std::size_t n, std::size_t c, std::size_t h, std::size_t w) const noexcept {
    return data_[Layout{}(n, c, h, w, stride_c_, stride_h_, stride_w_)];
  }

  constexpr Element* data() noexcept { return data_; }
  constexpr const Element* data() const noexcept { return data_; }

  constexpr std::size_t n() const noexcept { return n_; }
  constexpr std::size_t c() const noexcept { return c_; }
  constexpr std::size_t h() const noexcept { return h_; }
  constexpr std::size_t w() const noexcept { return w_; }

private:
  Element* data_;
  std::size_t n_, c_, h_, w_;
  std::size_t stride_c_, stride_h_, stride_w_;
};

// ============================================================================
// 便利类型别名 (模仿 PyTorch 命名)
// ============================================================================

template <typename T>
using Tensor2D = TensorRef<T, RowMajor>;  // 2D 默认行优先

template <typename T>
using Tensor4D = Tensor4DRef<T, NCHW>;    // 4D 默认 NCHW

} // namespace cutlass_style
