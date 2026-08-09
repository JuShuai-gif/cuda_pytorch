# pipe_operator

用管道运算符 `range | contains(value)` 模拟扩展方法。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 274-275 页：

- C++ 没有扩展方法，但可重载 `operator|` 实现近乎等价的语法：
  `numbers | contains(2)` 代替 `std::find(numbers.begin(), numbers.end(), 2) != numbers.end()`；
- `contains(value)` 是一个工厂函数，返回持有右操作数的 `ContainsProxy<T>`；
- 重载 `operator|(const Range&, const ContainsProxy<T>&)` 识别管道表达式并执行查找；
- 对任何有 `begin()/end()` 的范围、任何元素类型通用。

## 构建与运行

```bash
cmake --build build --target ch09_pipe_example ch09_pipe_tests -j
./build/chapter09_lazy_evaluation/ch09_pipe_example
./build/chapter09_lazy_evaluation/ch09_pipe_tests
```

## 关键点

- `ContainsProxy` 持有 `const T&`：只存活于表达式内，被 `operator|` 立即消费；
- 工厂函数消除显式类型书写：不用 `ContainsProxy<int>{2}` 而用 `contains(2)`；
- Range V3 / Fit 库把该思想推广到任意适配器（filtered、transformed…）。

## 注意

- 全局 `operator|` 可能与其他库冲突，生产代码需谨慎；
- 这是教学性"创造性运算符重载"，书中明确提示多数人会反对在生产中使用。
