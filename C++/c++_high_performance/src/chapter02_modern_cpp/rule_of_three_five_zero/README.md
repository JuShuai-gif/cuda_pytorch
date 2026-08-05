# rule_of_three_five_zero

Rule of Three / Five / Zero。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 67-77 页：

- **Rule of Three**：管理资源的类需要拷贝构造、拷贝赋值、析构三个函数；
- **Rule of Five**：加上移动构造、移动赋值（应 `noexcept`）；
- **Rule of Zero**：现代 C++ 用 RAII 成员（vector/string/unique_ptr）让特殊成员
  全部隐式正确，无需手写。手写特殊成员应是例外而非常态。

`= default` 可强制生成；用户声明析构函数会抑制隐式移动构造/赋值生成。

## 构建与运行

```bash
cmake --build build --target ch02_rule_of_example ch02_rule_of_tests
./build/chapter02_modern_cpp/ch02_rule_of_example
./build/chapter02_modern_cpp/ch02_rule_of_tests
```

## 关键点

- `RuleOfFive`：copy-and-swap 赋值、`noexcept` 移动、正确释放；
- 移动后源对象 `size()==0`（有效但未指定状态）；
- `RuleOfZero`：成员都是 RAII 类型，编译器生成的拷贝/移动/析构自动正确；
- tests 验证：向量扩容用移动而非复制、构造/析构计数相等、移动后源为空。

## 结论

- 优先 Rule of Zero：把资源封装进库类，应用代码不手写特殊成员；
- 必须手写时用 Rule of Five 且移动标记 `noexcept`；
- 空析构函数陷阱见 `noexcept_move` 实验。
