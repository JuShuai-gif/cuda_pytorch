// =============================================================================
// 第 07.3 章：CUTLASS 风格 TensorRef 传递设计
//
// 在 CUTLASS 中，TensorRef 是对张量数据的轻量视图
// （指针 + layout + 维度）。设计必须平衡：
//   1. 效率：无不必要拷贝（小 POD 类型传值）
//   2. 安全：无悬空引用
//   3. 灵活性：支持不同的 layout、元素类型、步长
//
// CUTLASS 的 TensorRef 是传值传递的，因为它们很小（通常是指针 +
// layout 描述符 + 维度元组），而且 ABI 的传值效率很高。
//
// 本文实现：
//   1. 简化的 TensorRef 类
//   2. 基于 layout 的偏移计算
//   3. 子张量视图（切片）
//   4. 传值设计原理
//   5. const vs 非 const TensorRef
//   6. TensorRef 作为通用引用包装器
//
// 编译：g++ -std=c++20 -o 03_cutlass_tensor_ref 03_cutlass_tensor_ref.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// 1. Layout 类型（RowMajor / ColumnMajor）
// =============================================================================

struct RowMajorLayout {
  // 计算元素 (r, c) 在给定 leading dimension（沿快速维度的步长）
  // 的矩阵中的线性偏移。
  static constexpr std::size_t offset(std::size_t r, std::size_t c,
                                       std::size_t rows, std::size_t cols,
                                       std::size_t ld) {
    (void)rows;
    (void)cols;
    return r * ld + c;
  }

  static constexpr char const* name = "RowMajor";
  static constexpr bool is_row_major = true;
};

struct ColumnMajorLayout {
  static constexpr std::size_t offset(std::size_t r, std::size_t c,
                                       std::size_t rows, std::size_t cols,
                                       std::size_t ld) {
    (void)rows;
    (void)cols;
    return c * ld + r;
  }

  static constexpr char const* name = "ColumnMajor";
  static constexpr bool is_row_major = false;
};

// =============================================================================
// 2. TensorRef：轻量张量视图
// =============================================================================
// 以 VALUE 方式传递（廉价：3-5 个字）。存储：
//   - 数据指针
//   - Layout 标签（编译期）
//   - 维度（运行期）
//   - Leading dimension / 步长

template <typename T_, typename Layout_>
class TensorRef {
 public:
  using T      = T_;
  using Layout = Layout_;
  using Index  = std::size_t;

  // 默认构造函数：空视图
  TensorRef() : data_(nullptr), rows_(0), cols_(0), ld_(0) {}

  // 从原始指针 + 维度构造
  TensorRef(T* data, Index rows, Index cols, Index ld = 0)
      : data_(data),
        rows_(rows),
        cols_(cols),
        ld_(ld == 0 ? cols : ld) {}

  // 拷贝构造函数（默认 -- TensorRef 拷贝很廉价）
  TensorRef(TensorRef const&) = default;

  // 拷贝赋值（默认）
  TensorRef& operator=(TensorRef const&) = default;

  // --- 访问器 ---

  // 通过基于 layout 的偏移访问元素 (r, c)
  T& at(Index r, Index c) {
    return data_[Layout::offset(r, c, rows_, cols_, ld_)];
  }

  T const& at(Index r, Index c) const {
    return data_[Layout::offset(r, c, rows_, cols_, ld_)];
  }

  // 原始数据指针
  T* data() const { return data_; }

  // 维度
  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  Index leading_dim() const { return ld_; }
  Index total_elements() const { return rows_ * cols_; }

  // Layout 名称
  static constexpr char const* layout_name() { return Layout::name; }

  // --- 子张量（切片/视图） ---
  // 创建一个 TensorRef，表示此张量的一个子矩形。
  // 这是零开销操作（只计算新的指针 + 维度）。

  TensorRef sub_tensor(Index row_start, Index col_start,
                       Index sub_rows, Index sub_cols) const {
    T const* sub_data = &at(row_start, col_start);
    return TensorRef(const_cast<T*>(sub_data), sub_rows, sub_cols, ld_);
  }

  // --- 迭代器风格访问，支持 range-for ---
  // 行访问（用于遍历行）

 private:
  T*    data_;  // 指向第一个元素的指针
  Index rows_;  // 行数
  Index cols_;  // 列数
  Index ld_;    // Leading dimension（连续行/列之间的步长）
};

// =============================================================================
// 3. Const TensorRef（只读视图）
// =============================================================================

template <typename T_, typename Layout_>
class ConstTensorRef {
 public:
  using T      = T_ const;
  using Layout = Layout_;
  using Index  = std::size_t;

  ConstTensorRef() : data_(nullptr), rows_(0), cols_(0), ld_(0) {}

  ConstTensorRef(T const* data, Index rows, Index cols, Index ld = 0)
      : data_(data),
        rows_(rows),
        cols_(cols),
        ld_(ld == 0 ? cols : ld) {}

  // 从非 const TensorRef 隐式转换
  ConstTensorRef(TensorRef<T_, Layout_> const& ref)
      : data_(ref.data()),
        rows_(ref.rows()),
        cols_(ref.cols()),
        ld_(ref.leading_dim()) {}

  T const& at(Index r, Index c) const {
    return data_[Layout::offset(r, c, rows_, cols_, ld_)];
  }

  T const* data() const { return data_; }
  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  Index leading_dim() const { return ld_; }

  ConstTensorRef sub_tensor(Index row_start, Index col_start,
                             Index sub_rows, Index sub_cols) const {
    T const* sub_data = &at(row_start, col_start);
    return ConstTensorRef(sub_data, sub_rows, sub_cols, ld_);
  }

 private:
  T const* data_;
  Index    rows_;
  Index    cols_;
  Index    ld_;
};

// =============================================================================
// 4. 接收 TensorRef 的函数（传值）
// =============================================================================
// 在 CUTLASS 中，kernel 以 VALUE 方式接收 TensorRef，因为：
//   - TensorRef 是可平凡拷贝的（指针 + 3 个 int = 16-24 字节）
//   - 在 CUDA kernel 参数中传值效率高（寄存器）
//   - 没有间接寻址（无双重指针解引用）

template <typename T, typename Layout>
T compute_sum(TensorRef<T, Layout> tensor) {
  T sum = T{};
  for (std::size_t r = 0; r < tensor.rows(); ++r) {
    for (std::size_t c = 0; c < tensor.cols(); ++c) {
      sum += tensor.at(r, c);
    }
  }
  return sum;
}

template <typename T, typename Layout>
void fill_value(TensorRef<T, Layout> tensor, T value) {
  for (std::size_t r = 0; r < tensor.rows(); ++r) {
    for (std::size_t c = 0; c < tensor.cols(); ++c) {
      tensor.at(r, c) = value;
    }
  }
}

// =============================================================================
// 5. 通过 TensorRef 实现基于 Layout 的矩阵转置
// =============================================================================
// 我们通过交换 layout 标签来创建转置"视图"。

template <typename Layout>
struct TransposeLayout;

template <>
struct TransposeLayout<RowMajorLayout> {
  using type = ColumnMajorLayout;
};

template <>
struct TransposeLayout<ColumnMajorLayout> {
  using type = RowMajorLayout;
};

template <typename T, typename Layout>
TensorRef<T, typename TransposeLayout<Layout>::type>
transpose_view(TensorRef<T, Layout> tensor) {
  // 交换行和列，保持数据指针和 leading dimension 不变
  return TensorRef<T, typename TransposeLayout<Layout>::type>(
      tensor.data(), tensor.cols(), tensor.rows(), tensor.leading_dim());
}

// =============================================================================
// 6. 辅助函数：打印张量
// =============================================================================

template <typename T, typename Layout>
void print_tensor(TensorRef<T, Layout> tensor) {
  std::cout << "Tensor[" << Layout::name << "] " << tensor.rows() << "x"
            << tensor.cols() << " (ld=" << tensor.leading_dim() << ")"
            << std::endl;
  for (std::size_t r = 0; r < tensor.rows(); ++r) {
    std::cout << "  ";
    for (std::size_t c = 0; c < tensor.cols(); ++c) {
      std::cout << tensor.at(r, c) << " ";
    }
    std::cout << std::endl;
  }
}

// =============================================================================
// 7. 行访问器（CUTLASS 模式：逐行迭代）
// =============================================================================
// 在 CUTLASS 中，常见模式是获取每行起始位置的指针。

template <typename T, typename Layout>
T* row_start(TensorRef<T, Layout> tensor, std::size_t row) {
  if constexpr (Layout::is_row_major) {
    return tensor.data() + row * tensor.leading_dim();
  } else {
    return tensor.data() + row;  // 列优先：行是连续的
  }
}

// =============================================================================
//                                   MAIN
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 07.3 章：CUTLASS 风格 TensorRef ===\n" << endl;

  // --- 测试 1：RowMajor 张量 ---
  cout << "[Test 1] RowMajor 张量：" << endl;
  vector<int> data_rm(6 * 4);
  for (int i = 0; i < 24; ++i) data_rm[i] = i;

  TensorRef<int, RowMajorLayout> rm_tensor(data_rm.data(), 6, 4);
  cout << "  Layout: " << rm_tensor.layout_name() << endl;
  cout << "  at(0,0)=" << rm_tensor.at(0, 0) << endl;
  cout << "  at(0,1)=" << rm_tensor.at(0, 1) << endl;
  cout << "  at(1,0)=" << rm_tensor.at(1, 0) << endl;

  // RowMajor：第 1 行从偏移 4 开始
  assert(rm_tensor.at(0, 0) == 0);
  assert(rm_tensor.at(0, 1) == 1);
  assert(rm_tensor.at(1, 0) == 4);  // 第 1 行，第 0 列

  print_tensor(rm_tensor);

  // --- 测试 2：ColumnMajor 张量 ---
  cout << "\n[Test 2] ColumnMajor 张量：" << endl;
  // 将相同数据重新解释为 ColumnMajor
  TensorRef<int, ColumnMajorLayout> cm_tensor(data_rm.data(), 6, 4);
  cout << "  Layout: " << cm_tensor.layout_name() << endl;
  cout << "  at(0,0)=" << cm_tensor.at(0, 0) << endl;
  cout << "  at(0,1)=" << cm_tensor.at(0, 1) << endl;
  cout << "  at(1,0)=" << cm_tensor.at(1, 0) << endl;

  // ColumnMajor：第 1 列从 ld = 4 开始，所以 at(0,1) = data[1*4+0] = 4
  assert(cm_tensor.at(0, 1) == 4);

  print_tensor(cm_tensor);

  // --- 测试 3：张量传值（廉价） ---
  cout << "\n[Test 3] 传值：" << endl;
  // TensorRef 小到可以传值
  cout << "  sizeof(TensorRef<int,RowMajor>) = "
       << sizeof(TensorRef<int, RowMajorLayout>) << " bytes" << endl;
  cout << "  sizeof(ConstTensorRef<int,RowMajor>) = "
       << sizeof(ConstTensorRef<int, RowMajorLayout>) << " bytes" << endl;

  int sum_rm = compute_sum(rm_tensor);
  cout << "  RowMajor 求和 = " << sum_rm << endl;
  int sum_cm = compute_sum(cm_tensor);
  cout << "  ColumnMajor 求和 = " << sum_cm << endl;
  // 注意：两个和不同，因为数据是按 RowMajor 布局的；
  // 将其 reinterpret_cast 为 ColumnMajor 会产生不同的遍历顺序。
  // 每个 layout 的求和对于其对数据的"自身"解释来说是正确的。

  // --- 测试 4：fill_value ---
  cout << "\n[Test 4] 通过 TensorRef 填充值：" << endl;
  vector<int> fill_data(3 * 3);
  TensorRef<int, RowMajorLayout> fill_ref(fill_data.data(), 3, 3);
  fill_value(fill_ref, 42);
  print_tensor(fill_ref);
  for (int v : fill_data) assert(v == 42);

  // --- 测试 5：子张量 ---
  cout << "\n[Test 5] 子张量（零拷贝视图）：" << endl;
  vector<int> big_data(5 * 5);
  for (int i = 0; i < 25; ++i) big_data[i] = i;
  TensorRef<int, RowMajorLayout> big_ref(big_data.data(), 5, 5);
  cout << "  原始 5x5：" << endl;
  print_tensor(big_ref);

  auto sub = big_ref.sub_tensor(1, 1, 3, 2);  // 行 1-3，列 1-2
  cout << "  子张量 (row=1,col=1, 3x2)：" << endl;
  print_tensor(sub);

  // 验证子张量元素
  // 原始的第 1 行第 1 列 = 1*5+1=6
  assert(sub.at(0, 0) == 6);
  // 第 1 行第 2 列 = 7
  assert(sub.at(0, 1) == 7);
  // 第 3 行第 1 列 = 3*5+1 = 16
  assert(sub.at(2, 0) == 16);

  // 通过子张量修改 -> 反映到原始张量
  sub.at(0, 0) = 999;
  assert(big_ref.at(1, 1) == 999);
  cout << "  修改 sub(0,0)=999 -> 原始(1,1)=" << big_ref.at(1, 1)
       << endl;

  // --- 测试 6：ConstTensorRef ---
  cout << "\n[Test 6] ConstTensorRef（只读视图）：" << endl;
  ConstTensorRef<int, RowMajorLayout> const_ref(rm_tensor);
  cout << "  const_ref.at(2,2) = " << const_ref.at(2, 2) << endl;

  // 从 TensorRef 到 ConstTensorRef 的隐式转换
  ConstTensorRef<int, RowMajorLayout> auto_const = rm_tensor;
  cout << "  auto_const.at(1,3) = " << auto_const.at(1, 3) << endl;

  // --- 测试 7：转置视图 ---
  cout << "\n[Test 7] 转置视图：" << endl;
  vector<int> trans_data(2 * 3);
  int cnt = 0;
  for (auto& v : trans_data) v = cnt++;
  TensorRef<int, RowMajorLayout> trans_ref(trans_data.data(), 2, 3);
  cout << "  原始 2x3 RowMajor：" << endl;
  print_tensor(trans_ref);

  auto trans_view = transpose_view(trans_ref);
  cout << "  转置视图 (3x2 ColumnMajor)：" << endl;
  print_tensor(trans_view);

  // 验证转置：原始(0,1) 应等于转置后(1,0)
  assert(trans_ref.at(0, 1) == trans_view.at(1, 0));

  // --- 测试 8：带填充的 Leading dimension ---
  cout << "\n[Test 8] 带填充 leading dimension 的张量：" << endl;
  // 一个 3x2 矩阵，按 row-major 存储，ld=4（填充）
  vector<int> padded_data(3 * 4, -1);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 2; ++c)
      padded_data[r * 4 + c] = r * 2 + c;

  TensorRef<int, RowMajorLayout> padded_ref(padded_data.data(), 3, 2, 4);
  cout << "  3x2 RowMajor ld=4：" << endl;
  print_tensor(padded_ref);

  assert(padded_ref.at(0, 0) == 0);
  assert(padded_ref.at(0, 1) == 1);
  assert(padded_ref.at(1, 0) == 2);
  assert(padded_ref.at(1, 1) == 3);
  // 填充元素在索引 2,3, 6,7, 10,11

  // --- 测试 9：通过 row_start() 直接访问行 ---
  cout << "\n[Test 9] 通过 row_start() 访问行：" << endl;
  auto* row0 = row_start(rm_tensor, 0);
  auto* row1 = row_start(rm_tensor, 1);
  cout << "  row0[0] = " << row0[0] << " (应为 0)" << endl;
  cout << "  row1[0] = " << row1[0] << " (应为 4)" << endl;
  assert(row0[0] == 0);
  assert(row1[0] == 4);

  // 对于 ColumnMajor，row_start 给出连续元素
  auto* cm_row0 = row_start(cm_tensor, 0);
  cout << "  cm_row0[0] = " << cm_row0[0] << " (列优先中行是连续的)"
       << endl;

  cout << "\n所有测试通过！" << endl;
  return 0;
}
