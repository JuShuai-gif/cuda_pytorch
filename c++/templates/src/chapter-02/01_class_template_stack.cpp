// =============================================================================
// 第 02.1 章：类模板 -- 完整栈实现
//
// 演示一个完整的、生产质量的 Stack<T> 类模板：
//   - 默认构造函数、拷贝/移动构造函数、析构函数
//   - 核心操作：push、pop、top、empty、size
//   - 用于异常安全的 copy-and-swap 惯用法
//   - emplace 用于原地构造（C++11+）
//   - 迭代器支持（begin/end）
//   - operator<< 的模板友元
//   - 特殊成员函数模板注意事项
//   - 类模板中的静态成员变量
//
// 编译：g++ -std=c++20 -o 01_class_template_stack 01_class_template_stack.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

// =============================================================================
// Stack<T> -- 一个简单的基于动态数组的栈
// =============================================================================
// 每个类模板实例化（例如 Stack<int>、Stack<string>）都是一个完全独立的类型。
// 静态成员是每实例化一份的。成员函数仅在用到时才实例化（惰性实例化）。

template <typename T>
class Stack {
 public:
  // --- 类型别名（惯例）---
  using value_type      = T;
  using size_type       = std::size_t;
  using reference       = T&;
  using const_reference = T const&;
  using pointer         = T*;
  using const_pointer   = T const*;
  using iterator        = T*;
  using const_iterator  = T const*;

  // --- 静态成员（每实例化一份，非全局）---
  // 每个 Stack<T> 有自己的静态成员副本。
  inline static size_type s_instance_count = 0;  // C++17 内联静态成员
  static inline constexpr size_type DEFAULT_CAPACITY = 16;

  // --- 构造函数 / 析构函数 ---

  // 默认构造函数：以默认容量分配内存
  Stack() : data_(nullptr), size_(0), capacity_(0) {
    allocate(DEFAULT_CAPACITY);
    ++s_instance_count;
  }

  // 带显式初始容量的构造函数
  explicit Stack(size_type initial_capacity)
      : data_(nullptr), size_(0), capacity_(0) {
    allocate(initial_capacity);
    ++s_instance_count;
  }

  // 从 initializer_list 构造
  Stack(std::initializer_list<T> init)
      : data_(nullptr), size_(0), capacity_(0) {
    allocate(init.size() > 0 ? init.size() : DEFAULT_CAPACITY);
    for (auto const& elem : init) {
      push(elem);  // 为效率可用 push_back，但 push 也行
    }
    ++s_instance_count;
  }

  // 拷贝构造函数 -- 深拷贝
  Stack(Stack const& other) : data_(nullptr), size_(0), capacity_(0) {
    if (other.size_ > 0) {
      allocate(other.capacity_);
      for (size_type i = 0; i < other.size_; ++i) {
        new (&data_[i]) T(other.data_[i]);  // placement new 用于拷贝
      }
      size_ = other.size_;
    }
    ++s_instance_count;
  }

  // 移动构造函数 -- 窃取资源
  Stack(Stack&& other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    // 将被移动的对象置于有效的空状态
    other.data_     = nullptr;
    other.size_     = 0;
    other.capacity_ = 0;
    ++s_instance_count;
  }

  // 析构函数
  ~Stack() {
    destroy_elements();
    ::operator delete(data_);
    --s_instance_count;
  }

  // --- 赋值运算符 ---

  // copy-and-swap 惯用法：异常安全，自赋值安全
  Stack& operator=(Stack other) noexcept {
    swap(other);
    return *this;
  }

  // --- 核心栈操作 ---

  // Push：向栈顶添加元素。可能触发重新分配。
  void push(T const& value) {
    ensure_capacity();
    new (&data_[size_]) T(value);
    ++size_;
  }

  // 移动语义的 push（右值重载）
  void push(T&& value) {
    ensure_capacity();
    new (&data_[size_]) T(std::move(value));
    ++size_;
  }

  // emplace：在栈顶原地构造元素
  template <typename... Args>
  void emplace(Args&&... args) {
    ensure_capacity();
    new (&data_[size_]) T(std::forward<Args>(args)...);
    ++size_;
  }

  // Pop：移除栈顶元素。不返回它（异常安全）。
  void pop() {
    if (empty()) {
      throw std::out_of_range("Stack::pop()：栈为空");
    }
    data_[size_ - 1].~T();  // 显式析构函数调用
    --size_;
  }

  // Top：访问栈顶元素（const 和非 const 重载）
  reference top() {
    if (empty()) {
      throw std::out_of_range("Stack::top()：栈为空");
    }
    return data_[size_ - 1];
  }

  const_reference top() const {
    if (empty()) {
      throw std::out_of_range("Stack::top()：栈为空");
    }
    return data_[size_ - 1];
  }

  // 容量 / 大小查询
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] size_type size() const noexcept { return size_; }
  [[nodiscard]] size_type capacity() const noexcept { return capacity_; }

  // --- 迭代器支持 ---
  // 从底（最旧）到顶（最新）迭代
  iterator begin() noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cbegin() const noexcept { return data_; }
  const_iterator cend() const noexcept { return data_ + size_; }

  // --- 交换 ---
  void swap(Stack& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

  // --- 静态查询 ---
  static size_type instance_count() { return s_instance_count; }

 private:
  pointer   data_;
  size_type size_;
  size_type capacity_;

  // 分配原始内存（不构造对象）
  void allocate(size_type n) {
    capacity_ = n;
    data_ = static_cast<pointer>(::operator new(n * sizeof(T)));
  }

  // 确保至少有容纳一个以上元素的空间
  void ensure_capacity() {
    if (size_ >= capacity_) {
      // 容量翻倍（几何增长以实现均摊 O(1) 的 push）
      size_type new_cap = (capacity_ == 0) ? DEFAULT_CAPACITY : capacity_ * 2;
      pointer new_data =
          static_cast<pointer>(::operator new(new_cap * sizeof(T)));

      // 将现有元素移动到新存储
      for (size_type i = 0; i < size_; ++i) {
        new (&new_data[i]) T(std::move(data_[i]));
        data_[i].~T();  // 销毁旧元素
      }

      ::operator delete(data_);
      data_     = new_data;
      capacity_ = new_cap;
    }
  }

  // 销毁所有元素（逆序以符合栈语义）
  void destroy_elements() {
    for (size_type i = size_; i > 0; --i) {
      data_[i - 1].~T();
    }
  }

  // 用于打印的模板友元
  template <typename U>
  friend std::ostream& operator<<(std::ostream& os, Stack<U> const& stack);
};

// --- 友元 operator<<（非成员，类模板友元）---
// 必须在类之后定义；作为友元，它可以访问任何 Stack<U> 实例化的私有成员。

template <typename T>
std::ostream& operator<<(std::ostream& os, Stack<T> const& stack) {
  os << "Stack(size=" << stack.size_ << ", capacity=" << stack.capacity_
     << ") [";
  for (std::size_t i = 0; i < stack.size_; ++i) {
    if (i > 0) os << ", ";
    os << stack.data_[i];
  }
  os << "]";
  return os;
}

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 02.1 章：类模板栈 ===\n" << endl;

  // --- 测试 1：基本操作 ---
  cout << "[测试 1] 基本 push/pop/top：" << endl;
  Stack<int> si;
  assert(si.empty());
  cout << "  空栈：" << si << endl;

  si.push(10);
  si.push(20);
  si.push(30);
  cout << "  3 次 push 后：" << si << endl;
  assert(si.size() == 3);
  assert(si.top() == 30);

  si.pop();
  cout << "  pop 后：" << si << endl;
  assert(si.size() == 2);
  assert(si.top() == 20);

  // --- 测试 2：不同类型的实例化是独立的 ---
  cout << "\n[测试 2] 独立实例化：" << endl;
  Stack<double> sd;
  Stack<string> ss;

  sd.push(3.14159);
  ss.push("hello");

  cout << "  int 实例数：" << Stack<int>::instance_count() << endl;
  cout << "  double 实例数：" << Stack<double>::instance_count() << endl;
  cout << "  string 实例数：" << Stack<string>::instance_count() << endl;
  // 每个静态计数器是每实例化一份的
  assert(Stack<int>::instance_count() == 1);
  assert(Stack<double>::instance_count() == 1);
  assert(Stack<string>::instance_count() == 1);

  // --- 测试 3：拷贝和移动 ---
  cout << "\n[测试 3] 拷贝和移动语义：" << endl;

  si.push(40);
  Stack<int> si_copy(si);  // 拷贝
  cout << "  拷贝：" << si_copy << endl;
  assert(si_copy.size() == si.size());
  assert(si_copy.top() == si.top());
  // 修改原对象以证明是深拷贝
  si.push(50);
  cout << "  原对象 push 后：" << si << endl;
  cout << "  拷贝（未变）：    " << si_copy << endl;
  assert(si_copy.top() == 40);

  Stack<int> si_moved(std::move(si));
  cout << "  移动至：" << si_moved << endl;
  cout << "  移动源：" << si << endl;
  assert(si.empty());
  assert(!si_moved.empty());
  // 实例数：原始 si 被移动但未被销毁
  assert(Stack<int>::instance_count() == 3);  // si、si_copy、si_moved

  // --- 测试 4：copy-and-swap 赋值 ---
  cout << "\n[测试 4] 赋值运算符：" << endl;
  Stack<int> si_assign;
  si_assign.push(100);
  si_assign = si_copy;  // 通过 copy-and-swap 进行拷贝赋值
  cout << "  赋值后：" << si_assign << endl;
  assert(si_assign.size() == si_copy.size());

  si_assign = std::move(si_moved);  // 通过 copy-and-swap 进行移动赋值
  cout << "  移动赋值后：" << si_assign << endl;

  // --- 测试 5：emplace ---
  cout << "\n[测试 5] emplace（原地构造）：" << endl;
  struct Point {
    int x, y;
    Point(int x_, int y_) : x(x_), y(y_) {}
    bool operator==(Point const& rhs) const {
      return x == rhs.x && y == rhs.y;
    }
  };

  Stack<Point> sp;
  sp.emplace(3, 4);  // 直接在栈上构造 Point(3,4)
  sp.emplace(5, 6);
  cout << "  Point 栈大小：" << sp.size()
       << ", 栈顶=(" << sp.top().x << "," << sp.top().y << ")" << endl;
  assert(sp.top() == Point(5, 6));

  // --- 测试 6：迭代器支持 ---
  cout << "\n[测试 6] 迭代器：" << endl;
  cout << "  迭代中：";
  for (auto const& val : si_copy) {
    cout << val << " ";
  }
  cout << endl;

  // Range-for 可以工作是因为定义了 begin()/end()
  int sum = 0;
  for (int v : si_copy) sum += v;
  assert(sum == 70);  // 10 + 20 + 40

  // --- 测试 7：initializer_list 构造函数 ---
  cout << "\n[测试 7] 初始化列表：" << endl;
  Stack<int> si_init{1, 2, 3, 4, 5};
  cout << "  " << si_init << endl;
  assert(si_init.size() == 5);
  assert(si_init.top() == 5);

  // --- 测试 8：异常安全 -- 空栈 pop ---
  cout << "\n[测试 8] 空栈 pop 异常：" << endl;
  Stack<int> si_empty;
  try {
    si_empty.pop();
    cout << "  错误：应该抛出异常！" << endl;
    assert(false);
  } catch (std::out_of_range const& e) {
    cout << "  捕获：" << e.what() << endl;
  }

  try {
    si_empty.top();
    cout << "  错误：应该抛出异常！" << endl;
    assert(false);
  } catch (std::out_of_range const& e) {
    cout << "  捕获：" << e.what() << endl;
  }

  cout << "\n所有测试通过！" << endl;
  return 0;
}
