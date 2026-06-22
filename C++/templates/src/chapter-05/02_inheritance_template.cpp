// =============================================================================
// 第 05.2 章：继承模板 -- 依赖基类 & CRTP
//
// 当从模板参数或依赖模板的基类继承时，有特殊规则：
//   1. 依赖基类的成员在阶段 1 不可见（必须使用
//      this-> 或 Base<T>:: 限定）
//   2. CRTP（奇异递归模板模式）：Base<Derived> 其中
//      Derived 继承自 Base<Derived>
//   3. Mixin 类：通过继承向派生类注入行为
//   4. 空基类优化（EBO）
//   5. 无虚函数的多态模板设计
//
// CUTLASS 广泛使用 CRTP。CRTP 实现静态多态：
// 在编译期分发到派生类方法，无虚函数开销。
//
// 编译：g++ -std=c++20 -o 02_inheritance_template 02_inheritance_template.cpp
// =============================================================================

#include <cassert>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// 1. 依赖基类：详细的访问方法
// =============================================================================
// 访问依赖基类成员的三种方式：
//   a) this->member       （推荐：明确表示是成员访问）
//   b) Base<T>::member     （显式，但禁用虚函数分发）
//   c) using Base<T>::member  （将名字引入作用域）

template <typename T>
struct DepBase {
  T storage;
  int counter = 0;

  void increment() { ++counter; }
  int  get_count() const { return counter; }
  void set_storage(T const& v) { storage = v; }
  T    get_storage() const { return storage; }

  using value_type = T;
};

template <typename T>
struct DepDerived : DepBase<T> {
  // 方法 A：using 声明（频繁访问时最方便）
  using typename DepBase<T>::value_type;
  using DepBase<T>::counter;
  using DepBase<T>::increment;

  // 方法 B：一次性访问使用 this->
  void do_work_this() {
    this->increment();
    std::cout << "  this->counter = " << this->counter << std::endl;
  }

  // 方法 C：使用 Base<T>:: 限定（避免虚函数分发）
  void do_work_base_qual() {
    DepBase<T>::increment();
    std::cout << "  DepBase<T>::counter = " << DepBase<T>::counter << std::endl;
  }

  // 使用引入的名字
  void demo_using() {
    increment();  // 通过 using 引入
    ++counter;    // 通过 using 引入
    std::cout << "  counter（通过 using）= " << counter << std::endl;
  }
};

// =============================================================================
// 2. CRTP：奇异递归模板模式
// =============================================================================
// CRTP 允许基类在编译期调用派生类的方法，实现无虚函数的
// 静态多态。
//
// 模式：
//   template <typename Derived>
//   class Base {
//     Derived& self() { return static_cast<Derived&>(*this); }
//     Derived const& self() const { return static_cast<Derived const&>(*this); }
//   };
//   class MyClass : public Base<MyClass> { ... };

// --- CRTP 基类：向任何派生类添加 operator== 和 operator!= ---
// 派生类只需提供 equal_to()。

template <typename Derived>
struct EqualityComparable {
  friend bool operator==(Derived const& a, Derived const& b) {
    return a.equal_to(b);
  }
  friend bool operator!=(Derived const& a, Derived const& b) {
    return !a.equal_to(b);
  }

 protected:
  // 辅助：将 this 转换为 Derived
  Derived&       self() { return static_cast<Derived&>(*this); }
  Derived const& self() const { return static_cast<Derived const&>(*this); }
};

// 使用 CRTP mixin 的具体类
struct Point2D : EqualityComparable<Point2D> {
  double x, y;

  Point2D(double x_, double y_) : x(x_), y(y_) {}

  bool equal_to(Point2D const& other) const {
    return x == other.x && y == other.y;
  }

  friend std::ostream& operator<<(std::ostream& os, Point2D const& p) {
    return os << "(" << p.x << "," << p.y << ")";
  }
};

// --- CRTP 基类：Sized（添加 size() 方法）---
template <typename Derived>
struct Sized {
  std::size_t size() const {
    return static_cast<Derived const*>(this)->size_impl();
  }
};

struct MyVec : Sized<MyVec> {
  std::vector<int> data;

  std::size_t size_impl() const { return data.size(); }
};

// --- CRTP 基类：Incrementable（提供前置/后置递增）---
template <typename Derived>
struct Incrementable {
  Derived& operator++() {        // 前置 ++
    auto& self = static_cast<Derived&>(*this);
    self.inc();
    return self;
  }
  Derived operator++(int) {      // 后置 ++
    auto& self = static_cast<Derived&>(*this);
    Derived tmp = self;
    self.inc();
    return tmp;
  }
};

struct Counter : Incrementable<Counter> {
  int value = 0;
  void inc() { ++value; }
};

// =============================================================================
// 3. 多层级 CRTP（层级结构）
// =============================================================================
// CRTP 可以嵌套：Base<Derived> -> Middle<Derived> -> Derived。
// 这允许组合行为。

template <typename Derived>
struct Logger {
  void log(std::string const& msg) {
    std::cout << "[Logger] " << msg << std::endl;
  }
};

template <typename Derived>
struct Validator : Logger<Derived> {
  bool validate() {
    auto& self = static_cast<Derived&>(*this);
    bool ok = self.is_valid();
    this->log(ok ? "验证通过" : "验证失败");
    return ok;
  }
};

struct DataProcessor : Validator<DataProcessor> {
  int threshold = 10;
  int value     = 20;

  bool is_valid() const { return value >= threshold; }
};

// =============================================================================
// 4. 空基类优化（EBO）
// =============================================================================
// 用作基类的空类不消耗存储（EBO）。
// 这对基于策略的设计很有用。

template <typename T, typename AllocPolicy>
struct Buffer : private AllocPolicy {
  T* data_ = nullptr;

  Buffer() {
    // 使用策略分配
    data_ = static_cast<T*>(AllocPolicy::allocate(sizeof(T) * 10));
  }

  ~Buffer() {
    AllocPolicy::deallocate(data_);
  }
};

struct MallocPolicy {
  static void* allocate(std::size_t n) {
    std::cout << "  MallocPolicy::allocate(" << n << ")" << std::endl;
    return std::malloc(n);
  }
  static void deallocate(void* p) {
    std::cout << "  MallocPolicy::deallocate()" << std::endl;
    std::free(p);
  }
};

// 验证 EBO：如果 AllocPolicy 为空，Buffer 的大小仍是 sizeof(T*)
static_assert(sizeof(Buffer<int, MallocPolicy>) == sizeof(int*),
              "EBO 应使 Buffer 大小与指针相同");

// =============================================================================
// 5. CUTLASS 风格 CRTP：坐标 / TileIterator 模拟
// =============================================================================
// 在 CUTLASS 中，线程级坐标迭代器使用 CRTP 避免虚函数分发。
// 基类定义接口；派生类实现硬件特定的迭代逻辑。

template <typename Derived, typename Index = int>
struct TileIteratorBase {
  using IndexType = Index;

  // 派生类必须实现的接口
  IndexType row() const {
    return static_cast<Derived const*>(this)->row_impl();
  }
  IndexType col() const {
    return static_cast<Derived const*>(this)->col_impl();
  }
  void advance() {
    static_cast<Derived*>(this)->advance_impl();
  }
  bool done() const {
    return static_cast<Derived const*>(this)->done_impl();
  }

  // 所有迭代器继承的通用工具
  IndexType linear_index(IndexType row_stride) const {
    return row() * row_stride + col();
  }
};

// 具体实现：对 tile 的行主序迭代
template <typename Index = int>
struct RowMajorTileIterator
    : TileIteratorBase<RowMajorTileIterator<Index>, Index> {
  Index rows_, cols_;
  Index cur_row_, cur_col_;

  RowMajorTileIterator(Index rows, Index cols)
      : rows_(rows), cols_(cols), cur_row_(0), cur_col_(0) {}

  Index row_impl() const { return cur_row_; }
  Index col_impl() const { return cur_col_; }

  void advance_impl() {
    ++cur_col_;
    if (cur_col_ >= cols_) {
      cur_col_ = 0;
      ++cur_row_;
    }
  }

  bool done_impl() const { return cur_row_ >= rows_; }
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 05.2 章：继承模板 & CRTP ===\n" << endl;

  // --- 测试 1：依赖基类访问方法 ---
  cout << "[测试 1] 依赖基类访问方法：" << endl;
  DepDerived<int> dd;
  dd.set_storage(42);

  cout << "  this-> 风格：" << endl;
  dd.do_work_this();
  cout << "  Base<T>:: 风格：" << endl;
  dd.do_work_base_qual();
  cout << "  using 声明风格：" << endl;
  dd.demo_using();

  assert(dd.get_storage() == 42);

  // --- 测试 2：CRTP EqualityComparable ---
  cout << "\n[测试 2] CRTP EqualityComparable：" << endl;
  Point2D p1(1.0, 2.0), p2(1.0, 2.0), p3(3.0, 4.0);

  cout << "  " << p1 << " == " << p2 << " ? " << (p1 == p2) << endl;
  cout << "  " << p1 << " == " << p3 << " ? " << (p1 == p3) << endl;
  cout << "  " << p1 << " != " << p3 << " ? " << (p1 != p3) << endl;

  assert(p1 == p2);
  assert(p1 != p3);

  // --- 测试 3：CRTP Sized ---
  cout << "\n[测试 3] CRTP Sized：" << endl;
  MyVec mv;
  mv.data = {1, 2, 3, 4, 5};
  cout << "  MyVec::size() = " << mv.size() << endl;
  assert(mv.size() == 5);

  // --- 测试 4：CRTP Incrementable ---
  cout << "\n[测试 4] CRTP Incrementable：" << endl;
  Counter cnt;
  cout << "  cnt.value = " << cnt.value << endl;
  ++cnt;
  cout << "  ++cnt 后：" << cnt.value << endl;
  cnt++;
  cout << "  cnt++ 后：" << cnt.value << endl;
  assert(cnt.value == 2);

  // --- 测试 5：CRTP 层级 ---
  cout << "\n[测试 5] CRTP 多级层级：" << endl;
  DataProcessor dp;
  dp.threshold = 50;  // 验证将失败
  dp.validate();       // 调用 Validator<DataProcessor>::validate() -> Logger::log()
  dp.threshold = 10;   // 现在应该通过
  bool ok = dp.validate();
  assert(ok);

  // --- 测试 6：EBO ---
  cout << "\n[测试 6] 空基类优化：" << endl;
  cout << "  sizeof(int*) = " << sizeof(int*) << endl;
  cout << "  sizeof(Buffer<int, MallocPolicy>) = "
       << sizeof(Buffer<int, MallocPolicy>) << endl;
  cout << "  MallocPolicy 为空，EBO 确保零开销" << endl;

  {
    Buffer<int, MallocPolicy> buf;
    cout << "  Buffer 通过策略分配" << endl;
  }
  cout << "  Buffer 已销毁（策略释放内存）" << endl;

  // --- 测试 7：CUTLASS 风格 TileIterator ---
  cout << "\n[测试 7] CUTLASS 风格 CRTP TileIterator：" << endl;
  RowMajorTileIterator<int> iter(3, 4);  // 3 行 x 4 列 tile
  cout << "  遍历 3x4 tile：" << endl;
  int count = 0;
  while (!iter.done()) {
    cout << "    (" << iter.row() << "," << iter.col()
         << ") linear=" << iter.linear_index(4);
    iter.advance();
    ++count;
    if (!iter.done()) cout << "," << endl;
  }
  cout << endl;
  cout << "  总元素数：" << count << endl;
  assert(count == 12);

  // --- 测试 8：CRTP 无开销 ---
  // 验证 CRTP 不增加任何大小开销
  struct PlainCounter { int value = 0; };
  static_assert(sizeof(Counter) == sizeof(PlainCounter),
                "CRTP Incrementable 零大小开销");

  cout << "\n所有测试通过！" << endl;
  return 0;
}
