# any_demo

`std::any` 的用法与性能。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 83-84 页：

- `std::any` 可持有**任意类型**的值（运行时类型擦除）；
- 读取必须用 `std::any_cast<T>`，类型不符抛 `std::bad_any_cast`；
- **性能**：`std::any` 堆分配其持有的值（实现被鼓励用 SBO 存小对象），
  `any_cast` 比直接类型访问慢得多。

## 构建与运行

```bash
cmake --build build --target ch02_any_example ch02_any_benchmark ch02_any_tests
./build/chapter02_modern_cpp/ch02_any_example
./build/chapter02_modern_cpp/ch02_any_tests
./build/chapter02_modern_cpp/ch02_any_benchmark
```

## 结果解释（GCC 13.3 Release）

- `sizeof(std::any)=16`（libstdc++ SBO 装不下 32 字节的 `std::string`，字符串需堆分配）；
- Benchmark：`std::any_cast<int>` 读取约 **1.4 ns**，直接 `int` 约 0.1 ns，
  相差约 **11 倍**（含 `any` 构造与类型检查）；
- 错误类型 `any_cast` 抛异常（`tests` 验证）。

## 结论

- `std::any` 是"动态类型值"的兜底方案，类型安全由运行时检查保证；
- 热路径、已知类型集合请用 `std::variant`（编译期类型安全，见 Chapter 8）；
- 与 `boost::any` 的 `any_cast_unsafe`（错误类型 UB）相比，`std::any_cast` 更安全。
