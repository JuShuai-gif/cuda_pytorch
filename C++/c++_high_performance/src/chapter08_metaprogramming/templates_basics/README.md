# templates_basics

模板函数/类、非类型模板参数、static_assert。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 210-214 页：

- 模板为每种实例化**生成**一份常规 C++ 代码（`pow_n<float>`/`pow_n<int>` 各自独立）；
- **非类型模板参数**（`int N`）让每个 N 生成一个独立函数；
- `static_assert` 在编译期拒绝非法模板参数（优于运行期 assert）。

## 构建与运行

```bash
cmake --build build --target ch08_templates_example ch08_templates_tests
./build/chapter08_metaprogramming/ch08_templates_example
./build/chapter08_metaprogramming/ch08_templates_tests
```

## 关键点

- `const_pow_n<T,N>` 编译期检查 `N >= 0`；
- 模板类 `Rectangle<T>` 对 float/int 通用；
- 错误（如负数 N）在**编译期**报错而非运行期崩溃。
