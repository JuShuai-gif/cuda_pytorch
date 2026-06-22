// =============================================================================
// 第 01.3 章：CUTLASS 风格的布局感知 Max 操作
//
// CUTLASS（CUDA Templates for Linear Algebra Subroutines）使用布局标签
// （RowMajor、ColumnMajor）在编译期编码数据访问模式。
// 本文件模拟该设计：一个根据编译期布局标签选择最优遍历策略的 max 操作。
//
// 演示的关键 CUTLASS 模式：
//   1. 通过模板参数的标签分发（RowMajor、ColumnMajor）
//   2. 静态多态（无虚函数，无运行时 dispatch）
//   3. 通过基类指针转换模式的类型擦除
//   4. 控制算法选择的编译期常量
//
// 编译：g++ -std=c++20 -o 03_cutlass_style_max 03_cutlass_style_max.cpp
// =============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// 1. 布局标签（编译期元数据）
// =============================================================================
// 在 CUTLASS 中，布局标签编码张量元素如何映射到线性内存。
// RowMajor：沿最后一个维度的连续元素（M x N -> 步长为 N）
// ColumnMajor：沿第一个维度的连续元素（M x N -> 步长为 M）

struct RowMajor {};     // 标签：元素按行存储
struct ColumnMajor {};  // 标签：元素按列存储

// ---------------------------------------------------------------------------
// 2. 布局 trait：在编译期将布局标签映射到其属性
// ---------------------------------------------------------------------------
// Trait 类提取编译期信息，类似于 CUTLASS 的布局 trait
// 提供 ::stride、::Index、::LongIndex 等。

template <typename Layout>
struct LayoutTraits;

template <>
struct LayoutTraits<RowMajor> {
  static constexpr char const* name = "RowMajor";
  // 对于 [rows x cols] 的 RowMajor 矩阵：
  // 元素 (r, c) 位于偏移量 r * cols + c
  static inline constexpr int offset(int r, int c, int /*rows*/, int cols) {
    return r * cols + c;
  }
  // "快速维度"是最后一个（cols）
  static constexpr int fast_dim_size(int /*rows*/, int cols) { return cols; }
};

template <>
struct LayoutTraits<ColumnMajor> {
  static constexpr char const* name = "ColumnMajor";
  // 对于 [rows x cols] 的 ColumnMajor 矩阵：
  // 元素 (r, c) 位于偏移量 c * rows + r
  static inline constexpr int offset(int r, int c, int rows, int /*cols*/) {
    return c * rows + r;
  }
  // "快速维度"是第一个（rows）
  static constexpr int fast_dim_size(int rows, int /*cols*/) { return rows; }
};

// =============================================================================
// 3. 布局感知的 max 查找器
// =============================================================================
// find_max 函数根据编译期 Layout 标签选择最优遍历策略。
// RowMajor：按行迭代（对行主序数据缓存友好）。
// ColumnMajor：按列迭代。
//
// 这是对 CUTLASS 方法的模拟：在编译期，标签选择内层循环结构，
// 最大化内存合并。

template <typename Layout, typename T, typename IndexType = int>
T find_max_matrix(T const* data, IndexType rows, IndexType cols,
                  Layout /*layout_tag*/ = Layout{}) {
  using Traits = LayoutTraits<Layout>;
  T max_val = data[0];

  if constexpr (std::is_same_v<Layout, RowMajor>) {
    // 行主序遍历：内层循环遍历 cols -> 局部性好
    for (IndexType r = 0; r < rows; ++r) {
      for (IndexType c = 0; c < cols; ++c) {
        IndexType idx = Traits::offset(r, c, rows, cols);
        if (data[idx] > max_val) max_val = data[idx];
      }
    }
  } else if constexpr (std::is_same_v<Layout, ColumnMajor>) {
    // 列主序遍历：内层循环遍历 rows -> 局部性好
    for (IndexType c = 0; c < cols; ++c) {
      for (IndexType r = 0; r < rows; ++r) {
        IndexType idx = Traits::offset(r, c, rows, cols);
        if (data[idx] > max_val) max_val = data[idx];
      }
    }
  } else {
    // 回退通用遍历（未做缓存优化）
    for (IndexType i = 0; i < rows * cols; ++i) {
      if (data[i] > max_val) max_val = data[i];
    }
  }

  return max_val;
}

// =============================================================================
// 4. CUTLASS 风格的通过手动虚表模拟的类型擦除
// =============================================================================
// 在真实的 CUTLASS 中，不同的内核配置具有不同的类型。
// 一个常见模式是在某个抽象边界处擦除布局类型，
// 使用函数指针或策略对象。
//
// 这里演示一种轻量级的类型擦除：将布局感知的 max 函数存储为
// std::function（或为性能使用裸函数指针），允许运行时选择布局策略，
// 同时保持内层循环完全静态编译。

template <typename T>
class LayoutErasedMaxFinder {
 public:
  // 回调类型：接受数据指针 + 维度，返回最大值
  using FinderFunc = std::function<T(T const*, int, int)>;

  // 从特定布局策略构造
  template <typename Layout>
  static LayoutErasedMaxFinder create_for_layout() {
    // lambda 捕获编译期布局知识，但将其擦除到 std::function 中。
    // 每个实例化都单独编译，具有高效的内层循环。
    LayoutErasedMaxFinder finder;
    finder.finder_ = [](T const* data, int rows, int cols) -> T {
      return find_max_matrix<Layout>(data, rows, cols);
    };
    finder.layout_name_ = LayoutTraits<Layout>::name;
    return finder;
  }

  T find_max(T const* data, int rows, int cols) const {
    return finder_(data, rows, cols);
  }

  std::string const& layout_name() const { return layout_name_; }

 private:
  FinderFunc finder_;
  std::string layout_name_;
};

// =============================================================================
// 5. 静态分发包装器（无类型擦除）
// =============================================================================
// 另一种 CUTLASS 模式：顶层 dispatch 函数接受运行时枚举，
// 通过 switch 或 if-else 链分发到正确的静态模板。
// 这避免了 std::function 的开销。

enum class LayoutEnum { RowMajor = 0, ColumnMajor = 1 };

template <typename T>
T dispatch_max(T const* data, int rows, int cols, LayoutEnum layout) {
  switch (layout) {
    case LayoutEnum::RowMajor:
      return find_max_matrix<RowMajor>(data, rows, cols);
    case LayoutEnum::ColumnMajor:
      return find_max_matrix<ColumnMajor>(data, rows, cols);
    default:
      return T{};
  }
}

// =============================================================================
// 6. 从迭代器类别推导编译期布局（高级）
// =============================================================================
// 在 CUTLASS 中，内存访问模式被编码为类型。我们可以从数据
// 的访问方式推导出"自然"布局。这是一个简化的演示。

template <typename Iterator>
struct DeduceLayout;

// 对于裸指针和 vector 迭代器，默认使用 RowMajor
template <typename Iterator>
struct DeduceLayout {
  using type = RowMajor;  // 大多数迭代器的默认假设
};

// =============================================================================
// 7. 基准测试辅助函数
// =============================================================================

template <typename T>
void fill_matrix(T* data, int rows, int cols, std::string const& layout_name) {
  int count = 0;
  if (layout_name == "RowMajor") {
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        data[r * cols + c] = static_cast<T>(count++);
  } else {
    for (int c = 0; c < cols; ++c)
      for (int r = 0; r < rows; ++r)
        data[c * rows + r] = static_cast<T>(count++);
  }
  // 在某个位置插入一个大值
  data[(rows / 2) * cols + (cols / 2)] = static_cast<T>(99999);
}

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 01.3 章：CUTLASS 风格的布局感知 Max ===\n" << endl;

  constexpr int ROWS = 100, COLS = 200;
  constexpr int TOTAL = ROWS * COLS;

  // 分配数据
  vector<int> row_major_data(TOTAL);
  vector<int> col_major_data(TOTAL);

  fill_matrix<int>(row_major_data.data(), ROWS, COLS, "RowMajor");
  fill_matrix<int>(col_major_data.data(), ROWS, COLS, "ColumnMajor");

  // --- 测试 1：静态分发 ---
  cout << "[测试 1] 静态分发：" << endl;
  int max_rm = find_max_matrix<RowMajor>(row_major_data.data(), ROWS, COLS);
  int max_cm = find_max_matrix<ColumnMajor>(col_major_data.data(), ROWS, COLS);
  cout << "  RowMajor 最大值    = " << max_rm << endl;
  cout << "  ColumnMajor 最大值 = " << max_cm << endl;
  assert(max_rm == 99999);
  assert(max_cm == 99999);

  // --- 测试 2：类型擦除的 max 查找器 ---
  cout << "\n[测试 2] 类型擦除的查找器：" << endl;
  auto rm_finder =
      LayoutErasedMaxFinder<int>::create_for_layout<RowMajor>();
  auto cm_finder =
      LayoutErasedMaxFinder<int>::create_for_layout<ColumnMajor>();

  int max_te_rm = rm_finder.find_max(row_major_data.data(), ROWS, COLS);
  int max_te_cm = cm_finder.find_max(col_major_data.data(), ROWS, COLS);
  cout << "  " << rm_finder.layout_name() << " 最大值 = " << max_te_rm << endl;
  cout << "  " << cm_finder.layout_name() << " 最大值 = " << max_te_cm << endl;
  assert(max_te_rm == 99999);
  assert(max_te_cm == 99999);

  // --- 测试 3：运行时枚举分发 ---
  cout << "\n[测试 3] 运行时枚举分发：" << endl;
  int max_enum_rm =
      dispatch_max(row_major_data.data(), ROWS, COLS, LayoutEnum::RowMajor);
  int max_enum_cm =
      dispatch_max(col_major_data.data(), ROWS, COLS, LayoutEnum::ColumnMajor);
  cout << "  RowMajor 枚举    = " << max_enum_rm << endl;
  cout << "  ColumnMajor 枚举 = " << max_enum_cm << endl;
  assert(max_enum_rm == 99999);
  assert(max_enum_cm == 99999);

  // --- 测试 4：缓存行为演示 ---
  // 对行主序数据使用列主序遍历（缓存不友好）vs
  // 行主序遍历（缓存友好）。真正的 CUTLASS 通过按布局特化内
  // 核来避免不匹配。
  cout << "\n[测试 4] 布局不匹配惩罚演示：" << endl;

  // 正确布局匹配：RowMajor 数据 + RowMajor 遍历
  int matched = find_max_matrix<RowMajor>(row_major_data.data(), ROWS, COLS);
  // 有意不匹配：RowMajor 数据 + ColumnMajor 遍历（仍然正确但因跨步访问更慢）
  int mismatched =
      find_max_matrix<ColumnMajor>(row_major_data.data(), ROWS, COLS);
  cout << "  正确布局匹配：  " << matched << endl;
  cout << "  有意不匹配：  " << mismatched << endl;
  // 两者应得到相同的正确结果，只是性能不同
  assert(matched == 99999);
  assert(mismatched == 99999);

  // --- 测试 5：布局 traits ---
  cout << "\n[测试 5] 编译期布局 traits：" << endl;
  cout << "  RowMajor 名称：" << LayoutTraits<RowMajor>::name << endl;
  cout << "  ColMajor 名称：" << LayoutTraits<ColumnMajor>::name << endl;
  static_assert(
      LayoutTraits<RowMajor>::offset(1, 2, 3, 4) == 1 * 4 + 2,
      "RowMajor 偏移量计算错误");
  static_assert(
      LayoutTraits<ColumnMajor>::offset(1, 2, 3, 4) == 2 * 3 + 1,
      "ColumnMajor 偏移量计算错误");
  cout << "  static_assert 检查通过！" << endl;

  // --- 测试 6：小矩阵边界情况 ---
  cout << "\n[测试 6] 边界情况：" << endl;
  int tiny_data[] = {42};
  int tiny_max = find_max_matrix<RowMajor>(tiny_data, 1, 1);
  cout << "  1x1 矩阵最大值 = " << tiny_max << endl;
  assert(tiny_max == 42);

  int flat_data[] = {1, 5, 3, 9, 2};
  int flat_max = find_max_matrix<RowMajor>(flat_data, 1, 5);
  cout << "  1x5 扁平最大值 = " << flat_max << endl;
  assert(flat_max == 9);

  cout << "\n所有测试通过！" << endl;
  return 0;
}
