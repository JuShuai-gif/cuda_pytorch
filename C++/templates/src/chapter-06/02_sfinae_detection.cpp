// =============================================================================
// 第 06.2 章：SFINAE 检测 -- void_t、检测惯用法、has_method
//
// SFINAE（替换失败不是错误）实现编译期反射：在不实例化类型的情况下，
// 检测该类型是否具有特定成员、方法或嵌套 typedef。
//
// 主题：
//   1. void_t：SFINAE 检测的瑞士军刀（C++17）
//   2. has_method：检测类型是否有特定的成员函数
//   3. has_nested_type：检测 typedef/value_type
//   4. is_iterable：检测 begin()/end()
//   5. 检测惯用法（std::experimental::is_detected，C++17）
//   6. 复合检测：同时具有 push_back 和 value_type
//   7. C++20 concepts 作为更简单的替代方案
//   8. 实际应用：检测类型是否支持 operator<<
//
// 编译：g++ -std=c++20 -o 02_sfinae_detection 02_sfinae_detection.cpp
// =============================================================================

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// 1. void_t -- C++17 工具（在此实现以供说明）
// =============================================================================
// void_t<Args...> 无论 Args 是什么都会映射到 void。它用于触发 SFINAE：
// 如果任何 Args 是非良构的，整个特化将被丢弃。

template <typename...>
using void_t = void;

// =============================================================================
// 2. has_method：检测成员函数
// =============================================================================

// --- has_size：检测 T 是否有 .size() 方法 ---
template <typename T, typename = void>
struct HasSize : std::false_type {};

template <typename T>
struct HasSize<T, void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_size_v = HasSize<T>::value;

// --- has_push_back：检测 T::push_back(const T::value_type&) ---
template <typename T, typename = void>
struct HasPushBack : std::false_type {};

template <typename T>
struct HasPushBack<
    T, void_t<decltype(std::declval<T>().push_back(
           std::declval<typename T::value_type>()))>> : std::true_type {};

template <typename T>
inline constexpr bool has_push_back_v = HasPushBack<T>::value;

// --- has_clear 带特定签名 ---
template <typename T, typename = void>
struct HasClear : std::false_type {};

template <typename T>
struct HasClear<T, void_t<decltype(std::declval<T>().clear())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_clear_v = HasClear<T>::value;

// =============================================================================
// 3. has_nested_type：检测 Typedef
// =============================================================================

// --- has_value_type ---
template <typename T, typename = void>
struct HasValueType : std::false_type {};

template <typename T>
struct HasValueType<T, void_t<typename T::value_type>> : std::true_type {};

template <typename T>
inline constexpr bool has_value_type_v = HasValueType<T>::value;

// --- has_iterator ---
template <typename T, typename = void>
struct HasIterator : std::false_type {};

template <typename T>
struct HasIterator<T, void_t<typename T::iterator>> : std::true_type {};

template <typename T>
inline constexpr bool has_iterator_v = HasIterator<T>::value;

// =============================================================================
// 4. is_iterable：复合检测（begin + end）
// =============================================================================

template <typename T, typename = void>
struct IsIterable : std::false_type {};

template <typename T>
struct IsIterable<
    T, void_t<decltype(std::begin(std::declval<T&>())),
              decltype(std::end(std::declval<T&>()))>> : std::true_type {};

template <typename T>
inline constexpr bool is_iterable_v = IsIterable<T>::value;

// =============================================================================
// 5. 检测惯用法（std::experimental::is_detected）
// =============================================================================
// 检测惯用法泛化了 SFINAE 检测：无需为每个成员编写单独的 trait，
// 而是将操作参数化。

namespace detail {

// nonesuch：永远无法构造的类型（失败的哨兵）
struct nonesuch {
  nonesuch()                 = delete;
  ~nonesuch()                = delete;
  nonesuch(nonesuch const&) = delete;
  void operator=(nonesuch const&) = delete;
};

// 主模板：替换失败 -> nonesuch
template <typename Default, typename AlwaysVoid,
          template <typename...> class Op, typename... Args>
struct Detector {
  using value_t = std::false_type;
  using type    = Default;
};

// 特化：替换成功
template <typename Default,
          template <typename...> class Op, typename... Args>
struct Detector<Default, void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
  using type    = Op<Args...>;
};

}  // namespace detail

// is_detected：如果 Op<Args...> 有效则为 true
template <template <typename...> class Op, typename... Args>
using is_detected =
    typename detail::Detector<detail::nonesuch, void, Op, Args...>::value_t;

template <template <typename...> class Op, typename... Args>
inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

// detected_t：如果有效则为类型 Op<Args...>，否则为 nonesuch
template <template <typename...> class Op, typename... Args>
using detected_t =
    typename detail::Detector<detail::nonesuch, void, Op, Args...>::type;

// =============================================================================
// 6. 使用检测惯用法：将操作定义为模板别名
// =============================================================================

// 检测：T::value_type
template <typename T>
using value_type_op = typename T::value_type;

// 检测：declval<T>().size()
template <typename T>
using size_method_op = decltype(std::declval<T>().size());

// 检测：declval<T>().empty()
template <typename T>
using empty_method_op = decltype(std::declval<T>().empty());

// 检测：operator<<(ostream, T)
template <typename T>
using streamable_op =
    decltype(std::declval<std::ostream&>() << std::declval<T>());

// =============================================================================
// 7. 使用 SFINAE 选择实现的函数
// =============================================================================

// 如果类型有 .size() 则打印大小
template <typename T>
std::enable_if_t<has_size_v<T>, void> print_info(T const& obj) {
  std::cout << "  size = " << obj.size() << std::endl;
}

template <typename T>
std::enable_if_t<!has_size_v<T>, void> print_info(T const&) {
  std::cout << "  （无 size 方法）" << std::endl;
}

// 如果可清除则清除，否则无操作
template <typename T>
std::enable_if_t<has_clear_v<T>, void> safe_clear(T& obj) {
  std::cout << "  正在清除..." << std::endl;
  obj.clear();
}

template <typename T>
std::enable_if_t<!has_clear_v<T>, void> safe_clear(T&) {
  std::cout << "  （不可清除，跳过）" << std::endl;
}

// =============================================================================
// 8. 实际应用：检测 operator<<
// =============================================================================

template <typename T>
std::enable_if_t<is_detected_v<streamable_op, T>, void>
stream_print(T const& obj) {
  std::cout << "  可流输出：" << obj << std::endl;
}

template <typename T>
std::enable_if_t<!is_detected_v<streamable_op, T>, void>
stream_print(T const&) {
  std::cout << "  （不可流输出）" << std::endl;
}

// 无 operator<< 的自定义类型
struct SecretType {
  int id = 42;
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 06.2 章：SFINAE 检测 ===\n" << endl;

  // --- 测试 1：has_size ---
  cout << "[测试 1] has_size：" << endl;
  static_assert(has_size_v<vector<int>>);
  static_assert(has_size_v<string>);
  static_assert(!has_size_v<int>);
  static_assert(!has_size_v<int*>);
  cout << "  vector<int>：" << has_size_v<vector<int>> << endl;
  cout << "  string：     " << has_size_v<string> << endl;
  cout << "  int：        " << has_size_v<int> << endl;

  // --- 测试 2：has_push_back ---
  cout << "\n[测试 2] has_push_back：" << endl;
  static_assert(has_push_back_v<vector<int>>);
  static_assert(has_push_back_v<list<double>>);
  static_assert(has_push_back_v<string>);
  // static_assert(!has_push_back_v<map<int,int>>); // push_back vs insert
  cout << "  vector<int>：" << has_push_back_v<vector<int>> << endl;
  cout << "  list<double>：" << has_push_back_v<list<double>> << endl;

  // --- 测试 3：has_value_type ---
  cout << "\n[测试 3] has_value_type：" << endl;
  static_assert(has_value_type_v<vector<int>>);
  static_assert(has_value_type_v<map<int, double>>);
  static_assert(!has_value_type_v<int>);
  cout << "  vector<int>：" << has_value_type_v<vector<int>> << endl;
  cout << "  int：        " << has_value_type_v<int> << endl;

  // --- 测试 4：is_iterable ---
  cout << "\n[测试 4] is_iterable：" << endl;
  static_assert(is_iterable_v<vector<int>>);
  static_assert(is_iterable_v<int[5]>);  // 数组有 std::begin
  static_assert(!is_iterable_v<int>);
  cout << "  vector<int>：" << is_iterable_v<vector<int>> << endl;
  cout << "  int[5]：     " << is_iterable_v<int[5]> << endl;
  cout << "  int：        " << is_iterable_v<int> << endl;

  // --- 测试 5：检测惯用法 ---
  cout << "\n[测试 5] 检测惯用法：" << endl;
  cout << "  有 value_type："
       << is_detected_v<value_type_op, vector<int>> << endl;
  cout << "  有 size()：    "
       << is_detected_v<size_method_op, string> << endl;
  cout << "  int 有 size()："
       << is_detected_v<size_method_op, int> << endl;

  static_assert(is_detected_v<value_type_op, vector<int>>);
  static_assert(!is_detected_v<value_type_op, int>);
  static_assert(is_detected_v<size_method_op, vector<int>>);
  static_assert(!is_detected_v<size_method_op, int>);

  // --- 测试 6：SFINAE 分发的 print_info ---
  cout << "\n[测试 6] print_info（SFINAE 分发）：" << endl;
  vector<int> v{1, 2, 3};
  print_info(v);
  int x = 42;
  print_info(x);

  // --- 测试 7：safe_clear ---
  cout << "\n[测试 7] safe_clear：" << endl;
  safe_clear(v);
  cout << "  清除后，v.size() = " << v.size() << endl;
  assert(v.empty());

  safe_clear(x);  // 对 int 无操作
  cout << "  x = " << x << " （未变）" << endl;
  assert(x == 42);

  // --- 测试 8：operator<< 检测 ---
  cout << "\n[测试 8] operator<< 检测：" << endl;
  stream_print(42);
  stream_print(string("hello"));
  stream_print(SecretType{});

  static_assert(is_detected_v<streamable_op, int>);
  static_assert(is_detected_v<streamable_op, string>);
  static_assert(!is_detected_v<streamable_op, SecretType>);

  // --- 测试 9：复合检查 ---
  // 具有 push_back 和 value_type 的自定义类型
  struct MyVec {
    using value_type = int;
    void push_back(int) {}
  };
  static_assert(has_push_back_v<MyVec>);
  static_assert(has_value_type_v<MyVec>);

  cout << "\n所有测试通过！" << endl;
  return 0;
}
