// ============================================================================
// 01_explicit_instantiation.cpp - 显式模板实例化
// ============================================================================
//
// 目的：
//   演示显式实例化、extern template 声明，以及它们如何在大型代码库中
//   减少编译期开销。
//
// 关键概念：
//   1. 隐式实例化 —— 编译器在首次使用时生成代码
//   2. 显式实例化 —— 程序员主动请求实例化
//   3. extern template —— 抑制隐式实例化（C++11）
//   4. CUTLASS 等高性能模板库使用的"显式实例化模型"，
//      将声明与定义分离，同时仍支持模板。
//
// 构建：
//   c++ -std=c++20 -c 01_explicit_instantiation.cpp -o /dev/null
//   （静态断言作为编译期测试）
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>

// ---------------------------------------------------------------------------
// 第 1 部分：问题 —— 到处都在隐式实例化
// ---------------------------------------------------------------------------

// 一个典型的工具模板。每个包含此头文件并使用 Vector<T> 的翻译单元
// 都会生成自己的副本，通过链接器去重代价拖慢编译速度并增加二进制体积。
template <typename T>
class Vector {
public:
    explicit Vector(std::size_t n) : size_(n), data_(new T[n]) {}
    ~Vector() { delete[] data_; }

    T const& operator[](std::size_t i) const { return data_[i]; }
    T&       operator[](std::size_t i)       { return data_[i]; }
    std::size_t size() const { return size_; }

private:
    std::size_t size_;
    T*          data_;
};

// 每个使用 Vector<T> 的函数都会触发整个类模板对该 T 的隐式实例化。
// 这很方便但代价高昂。
void use_vector_implicitly() {
    Vector<int>    vi(10);  // 隐式实例化 Vector<int>
    Vector<double> vd(20);  // 隐式实例化 Vector<double>
    std::cout << vi[0] << " " << vd[0] << "\n";
}

// ---------------------------------------------------------------------------
// 第 2 部分：显式实例化
// ---------------------------------------------------------------------------

// 2a. 显式实例化声明（C++11 "extern template"）
//     告诉编译器："不要在这里隐式实例化这个模板特化；
//     它会由另一个翻译单元提供。"
extern template class Vector<float>;

// 2b. 显式实例化定义
//     告诉编译器："就在这里、立刻为这个特化生成代码。"
//     这是履行 extern 承诺的对应定义。
template class Vector<float>;

// 通过这种模式，许多翻译单元可以使用 `extern template`
// 声明，而单个翻译单元提供显式实例化定义。
// 这正是 CUTLASS 管理其数千个 kernel 特化的方式。

// ---------------------------------------------------------------------------
// 第 3 部分：选择性显式实例化以提升性能
// ---------------------------------------------------------------------------

// 具有许多成员函数的模板。用户可能只需要其中一部分。
template <typename T>
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(new T[rows * cols]) {}

    ~Matrix() { delete[] data_; }

    // 内联定义始终被隐式实例化
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // 非内联函数可以有选择地显式实例化
    void fill(T value);
    void transpose();

private:
    std::size_t rows_;
    std::size_t cols_;
    T*          data_;
};

template <typename T>
void Matrix<T>::fill(T value) {
    for (std::size_t i = 0; i < rows_ * cols_; ++i)
        data_[i] = value;
}

template <typename T>
void Matrix<T>::transpose() {
    // 简化版：方阵的就地转置
    if (rows_ != cols_) return;
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = i + 1; j < cols_; ++j)
            std::swap(data_[i * cols_ + j], data_[j * cols_ + i]);
}

// 选择性显式实例化：只实例化我们需要的函数。
// 这可以避免为未使用的成员函数生成代码。
template class Matrix<int>;    // 实例化所有成员
// 在 C++11 中我们可以对单个函数模板进行更细粒度的选择，
// 但类模板的显式实例化始终会实例化所有成员。

// ---------------------------------------------------------------------------
// 第 4 部分：编译期验证
// ---------------------------------------------------------------------------

// 验证 Vector<float> 在显式实例化后确实是一个完整类型。
// 如果模板只有前向声明而没有定义，此处会编译失败。
static_assert(sizeof(Vector<float>) > 0,
    "Vector<float> must be a complete type after explicit instantiation");

// 验证我们的特化是独立的类型
static_assert(!std::is_same_v<Vector<int>, Vector<float>>,
    "Vector<int> and Vector<float> must be distinct types");

// ---------------------------------------------------------------------------
// 第 5 部分："手动头文件"模式
// ---------------------------------------------------------------------------
// 有时库将声明和定义分离到不同的文件中，并在头文件底部
// 包含定义文件。通过宏来控制以同时支持隐式和显式实例化模型。

#ifdef MATRIX_IMPLICIT_INSTANTIATION
// 隐式模式：包含 .inl/.tcc 文件，每个 TU 编译自己的副本
#  include "matrix_impl.inl"
#endif

// 显式模式（默认）：.inl 文件不在这里包含；
// 而是由专门的 .cpp 文件包含并提供显式实例化。

// ---------------------------------------------------------------------------
// 第 6 部分：显式实例化中的静态数据成员
// ---------------------------------------------------------------------------

template <typename T>
class Counter {
public:
    static int count;
    Counter() { ++count; }
};

// 静态成员定义
template <typename T>
int Counter<T>::count = 0;

// 显式实例化同时也会实例化静态数据成员。
// 没有这个，每个 TU 都会得到自己的 Counter<int>::count 副本，
// 违反 ODR。
template class Counter<int>;

// 验证类型是完整的（count 不是 constexpr，所以只检查 sizeof）
static_assert(sizeof(Counter<int>) > 0,
    "Counter<int> must be a complete type");

// ============================================================================
// MAIN：运行期演示
// ============================================================================

int main() {
    // 第 1 部分：隐式实例化实际运作
    Vector<int> vi(5);
    vi[0] = 42;
    std::cout << "Implicit Vector<int>[0] = " << vi[0] << "\n";

    // 第 2 部分：Vector<float> 在上面已被显式实例化
    Vector<float> vf(3);
    vf[0] = 3.14f;
    std::cout << "Explicit Vector<float>[0] = " << vf[0] << "\n";

    // 第 3 部分：Matrix 显式实例化
    Matrix<int> m(2, 2);
    m.fill(7);
    std::cout << "Matrix<int>(2,2).rows() = " << m.rows() << "\n";

    // 第 6 部分：静态计数器
    Counter<int> c1, c2;
    std::cout << "Counter<int>::count = " << Counter<int>::count << "\n";

    std::cout << "All tests passed.\n";
    return 0;
}
