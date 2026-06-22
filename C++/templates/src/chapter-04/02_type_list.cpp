// =============================================================================
// 第 04.2 章：编译期类型列表
//
// 类型列表是以模板参数编码的编译期类型序列。
// 这是许多模板元编程库（Boost.MPL、Brigand、Metal、CUTLASS 的 type_list）的基础。
//
// 实现内容：
//   TypeList<Ts...>           -- 基本类型列表容器
//   at_t<List, N>             -- 访问第 N 个类型
//   index_of<List, T>         -- 查找 T 的索引（未找到返回 -1）
//   push_front<List, T>       -- 前置类型
//   push_back<List, T>        -- 后置类型
//   front_t<List>             -- 第一个类型
//   back_t<List>              -- 最后一个类型
//   size<List>                -- 类型数量
//   empty<List>               -- 列表是否为空？
//   concat<L1, L2>            -- 连接两个列表
//   transform<F, List>        -- 对每个元素应用元函数
//   find_if<Pred, List>       -- 查找第一个满足谓词的类型
//
// 编译：g++ -std=c++20 -o 02_type_list 02_type_list.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

// =============================================================================
// 1. TypeList 定义
// =============================================================================

template <typename... Ts>
struct TypeList {
  static constexpr std::size_t size = sizeof...(Ts);
};

// 空列表别名
using EmptyList = TypeList<>;

// =============================================================================
// 2. at_t：访问列表中第 N 个类型
// =============================================================================

// 主模板：N > 0，剥离头部
template <std::size_t N, typename List>
struct At;

// 递归情况：剥离一个类型
template <std::size_t N, typename T, typename... Ts>
struct At<N, TypeList<T, Ts...>> {
  using type = typename At<N - 1, TypeList<Ts...>>::type;
};

// 基例：N == 0，返回第一个类型
template <typename T, typename... Ts>
struct At<0, TypeList<T, Ts...>> {
  using type = T;
};

template <std::size_t N, typename List>
using at_t = typename At<N, List>::type;

// =============================================================================
// 3. front_t / back_t：第一个和最后一个元素
// =============================================================================

template <typename List>
struct Front;

template <typename T, typename... Ts>
struct Front<TypeList<T, Ts...>> {
  using type = T;
};

template <typename List>
using front_t = typename Front<List>::type;

// Back：递归方式
template <typename List>
struct Back;

template <typename T>
struct Back<TypeList<T>> {
  using type = T;
};

template <typename T, typename... Ts>
struct Back<TypeList<T, Ts...>> {
  using type = typename Back<TypeList<Ts...>>::type;
};

template <typename List>
using back_t = typename Back<List>::type;

// =============================================================================
// 4. index_of：在列表中查找 T 的索引（未找到返回 -1）
// =============================================================================

template <typename List, typename T>
struct IndexOf;

template <typename T, typename... Ts>
struct IndexOf<TypeList<T, Ts...>, T> {
  static constexpr int value = 0;
};

template <typename T, typename U, typename... Ts>
struct IndexOf<TypeList<U, Ts...>, T> {
  static constexpr int value = []() constexpr {
    if constexpr (IndexOf<TypeList<Ts...>, T>::value == -1)
      return -1;
    else
      return 1 + IndexOf<TypeList<Ts...>, T>::value;
  }();
};

template <typename T>
struct IndexOf<TypeList<>, T> {
  static constexpr int value = -1;
};

template <typename List, typename T>
inline constexpr int index_of_v = IndexOf<List, T>::value;

// =============================================================================
// 5. push_front / push_back
// =============================================================================

template <typename List, typename T>
struct PushFront;

template <typename... Ts, typename T>
struct PushFront<TypeList<Ts...>, T> {
  using type = TypeList<T, Ts...>;
};

template <typename List, typename T>
using push_front_t = typename PushFront<List, T>::type;

template <typename List, typename T>
struct PushBack;

template <typename... Ts, typename T>
struct PushBack<TypeList<Ts...>, T> {
  using type = TypeList<Ts..., T>;
};

template <typename List, typename T>
using push_back_t = typename PushBack<List, T>::type;

// =============================================================================
// 6. concat：连接两个类型列表
// =============================================================================

template <typename L1, typename L2>
struct Concat;

template <typename... Ts1, typename... Ts2>
struct Concat<TypeList<Ts1...>, TypeList<Ts2...>> {
  using type = TypeList<Ts1..., Ts2...>;
};

template <typename L1, typename L2>
using concat_t = typename Concat<L1, L2>::type;

// =============================================================================
// 7. transform：对每个元素应用一元元函数
// =============================================================================

template <template <typename> class F, typename List>
struct Transform;

template <template <typename> class F, typename... Ts>
struct Transform<F, TypeList<Ts...>> {
  using type = TypeList<typename F<Ts>::type...>;
};

template <template <typename> class F, typename List>
using transform_t = typename Transform<F, List>::type;

// 示例元函数：添加指针
template <typename T>
struct AddPointer {
  using type = T*;
};

// 示例元函数：移除 const
template <typename T>
struct RemoveConstMF {
  using type = std::remove_const_t<T>;
};

// =============================================================================
// 8. find_if：查找第一个满足谓词的类型
// =============================================================================

template <template <typename> class Pred, typename List, typename Default = void>
struct FindIf;

template <template <typename> class Pred, typename Default>
struct FindIf<Pred, TypeList<>, Default> {
  using type = Default;  // 未找到
};

template <template <typename> class Pred, typename T, typename... Ts,
          typename Default>
struct FindIf<Pred, TypeList<T, Ts...>, Default> {
  using type = std::conditional_t<Pred<T>::value, T,
                                  typename FindIf<Pred, TypeList<Ts...>,
                                                  Default>::type>;
};

template <template <typename> class Pred, typename List,
          typename Default = void>
using find_if_t = typename FindIf<Pred, List, Default>::type;

// =============================================================================
// 9. filter：仅保留满足谓词的类型
// =============================================================================

template <template <typename> class Pred, typename List>
struct Filter;

template <template <typename> class Pred>
struct Filter<Pred, TypeList<>> {
  using type = TypeList<>;
};

template <template <typename> class Pred, typename T, typename... Ts>
struct Filter<Pred, TypeList<T, Ts...>> {
  using tail = typename Filter<Pred, TypeList<Ts...>>::type;
  using type = std::conditional_t<
      Pred<T>::value,
      push_front_t<tail, T>,
      tail>;
};

template <template <typename> class Pred, typename List>
using filter_t = typename Filter<Pred, List>::type;

// =============================================================================
// 10. for_each：在编译期遍历列表并打印类型名称
// =============================================================================

// 获取可读类型名的辅助函数（因编译器而异，尽力而为）
template <typename T>
constexpr std::string_view type_name() {
#if defined(__clang__) || defined(__GNUC__)
  std::string_view p = __PRETTY_FUNCTION__;
  return p;
#else
  return "unknown";
#endif
}

template <typename List>
struct ForEachPrinter;

template <>
struct ForEachPrinter<TypeList<>> {
  static void print() {}
};

template <typename T, typename... Ts>
struct ForEachPrinter<TypeList<T, Ts...>> {
  static void print() {
    std::cout << "  " << typeid(T).name() << std::endl;
    ForEachPrinter<TypeList<Ts...>>::print();
  }
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 04.2 章：编译期类型列表 ===\n" << endl;

  // 定义一些类型列表
  using L1 = TypeList<int, double, float>;
  using L2 = TypeList<char, short, long, long long>;
  using LInts = TypeList<int, unsigned int, long, unsigned long>;

  // --- 测试 1：size ---
  cout << "[测试 1] 大小：" << endl;
  cout << "  L1 size = " << L1::size << endl;
  cout << "  L2 size = " << L2::size << endl;
  cout << "  空 size = " << EmptyList::size << endl;
  static_assert(L1::size == 3);
  static_assert(L2::size == 4);
  static_assert(EmptyList::size == 0);

  // --- 测试 2：at_t ---
  cout << "\n[测试 2] at_t（索引访问）：" << endl;
  static_assert(is_same_v<at_t<0, L1>, int>);
  static_assert(is_same_v<at_t<1, L1>, double>);
  static_assert(is_same_v<at_t<2, L1>, float>);
  static_assert(is_same_v<at_t<3, L2>, long long>);
  cout << "  L1[0] = int, L1[1] = double, L1[2] = float：已确认" << endl;

  // --- 测试 3：front_t / back_t ---
  cout << "\n[测试 3] front_t / back_t：" << endl;
  static_assert(is_same_v<front_t<L1>, int>);
  static_assert(is_same_v<back_t<L1>, float>);
  static_assert(is_same_v<front_t<L2>, char>);
  static_assert(is_same_v<back_t<L2>, long long>);
  cout << "  L1.front = int, L1.back = float：已确认" << endl;

  // --- 测试 4：index_of ---
  cout << "\n[测试 4] index_of：" << endl;
  static_assert(index_of_v<L1, int> == 0);
  static_assert(index_of_v<L1, double> == 1);
  static_assert(index_of_v<L1, float> == 2);
  static_assert(index_of_v<L1, char> == -1);  // 未找到
  cout << "  index_of(L1, int)=0, index_of(L1, char)=-1：已确认" << endl;

  // --- 测试 5：push_front / push_back ---
  cout << "\n[测试 5] push_front / push_back：" << endl;
  using L1Prepend = push_front_t<L1, char>;
  using L1Append  = push_back_t<L1, long>;

  static_assert(is_same_v<front_t<L1Prepend>, char>);
  static_assert(L1Prepend::size == 4);
  static_assert(is_same_v<back_t<L1Append>, long>);
  static_assert(L1Append::size == 4);
  cout << "  push_front：char, int, double, float（" << L1Prepend::size
       << " 个元素）" << endl;
  cout << "  push_back： int, double, float, long（" << L1Append::size
       << " 个元素）" << endl;

  // --- 测试 6：concat ---
  cout << "\n[测试 6] concat：" << endl;
  using L1L2 = concat_t<L1, L2>;
  static_assert(L1L2::size == 7);
  static_assert(is_same_v<at_t<0, L1L2>, int>);
  static_assert(is_same_v<at_t<3, L1L2>, char>);
  static_assert(is_same_v<at_t<6, L1L2>, long long>);
  cout << "  L1 + L2 = " << L1L2::size << " 个元素：已确认" << endl;

  // --- 测试 7：transform ---
  cout << "\n[测试 7] transform（添加指针）：" << endl;
  using L1Ptr = transform_t<AddPointer, L1>;
  static_assert(is_same_v<at_t<0, L1Ptr>, int*>);
  static_assert(is_same_v<at_t<1, L1Ptr>, double*>);
  static_assert(is_same_v<at_t<2, L1Ptr>, float*>);
  cout << "  L1 -> L1Ptr：int*, double*, float*：已确认" << endl;

  // Transform：移除 const
  using LConst = TypeList<int const, double const, char const>;
  using LNoConst = transform_t<RemoveConstMF, LConst>;
  static_assert(is_same_v<at_t<0, LNoConst>, int>);
  static_assert(is_same_v<at_t<1, LNoConst>, double>);
  static_assert(is_same_v<at_t<2, LNoConst>, char>);
  cout << "  const 移除 transform：已确认" << endl;

  // --- 测试 8：find_if ---
  cout << "\n[测试 8] find_if：" << endl;
  // 在 L1（包含 int, double, float）中查找第一个整数类型
  using Found = find_if_t<std::is_integral, L1, void>;
  static_assert(is_same_v<Found, int>);
  cout << "  {int, double, float} 中第一个整数 = int：已确认" << endl;

  // 查找第一个浮点类型
  using FoundFloat = find_if_t<std::is_floating_point, L1, void>;
  static_assert(is_same_v<FoundFloat, double>);
  cout << "  {int, double, float} 中第一个浮点数 = double：已确认" << endl;

  // 未找到
  using NotFound = find_if_t<std::is_pointer, L1, void>;
  static_assert(is_same_v<NotFound, void>);
  cout << "  L1 中没有指针类型 -> void：已确认" << endl;

  // --- 测试 9：filter ---
  cout << "\n[测试 9] filter：" << endl;
  using Mixed = TypeList<int, double, char, float, long>;
  using OnlyIntegrals = filter_t<std::is_integral, Mixed>;
  static_assert(OnlyIntegrals::size == 3);
  static_assert(is_same_v<at_t<0, OnlyIntegrals>, int>);
  static_assert(is_same_v<at_t<1, OnlyIntegrals>, char>);
  static_assert(is_same_v<at_t<2, OnlyIntegrals>, long>);
  cout << "  从 {int,double,char,float,long} 过滤整数 = {int,char,long}（size="
       << OnlyIntegrals::size << "）：已确认" << endl;

  // --- 测试 10：打印类型名称 ---
  cout << "\n[测试 10] ForEachPrinter：" << endl;
  cout << "  L1 类型：" << endl;
  ForEachPrinter<L1>::print();

  cout << "\n所有测试通过！" << endl;
  return 0;
}
