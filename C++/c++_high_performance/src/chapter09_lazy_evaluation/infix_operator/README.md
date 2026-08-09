# infix_operator

用运算符重载实现中缀表达式 `value <in> range`，模拟 Python 的 `in`。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 275-277 页：

- `x <in> range` 被拆成两个普通调用：
  1. `x < in`：`operator<(const T&, const InTag&)` 把左值包装成 `InProxy<T>{x}`；
  2. `proxy > range`：`operator>(const InProxy<T>&, const Range&)` 执行
     `std::find` 判断是否存在；
- `InTag` + `constexpr in` 让语法是 `<in>` 而非 `<InTag{}>`；
- 与管道运算符不同，代理持有的是**左**操作数。

## 构建与运行

```bash
cmake --build build --target ch09_infix_example ch09_infix_tests -j
./build/chapter09_lazy_evaluation/ch09_infix_example
./build/chapter09_lazy_evaluation/ch09_infix_tests
```

## 关键点

- `InProxy` 持有 `const T&`，只存活于表达式内；
- 全局重载 `operator<`/`operator>` 是"hack 式"写法，会污染命名空间；
- 书上原话：更多是给 Python 粉朋友展示，生产代码慎用。

## 注意

- `constexpr static auto in` 在头文件中应改为 `inline`（C++17）或放到
  一个 cpp 里，避免 ODR 问题；本示例单文件演示无碍。
