# rotate_example

手写 for 循环 vs `std::rotate`。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 156-158 页：把前 n 个元素移到末尾。三种方案对比：

| 方案 | 问题 |
|---|---|
| for 循环 + 迭代器 | 若 vector 扩容，迭代器失效 → 崩溃/异常 |
| for 循环 + 索引 | 安全，但 `std::next(c.begin(), i)` 在 list 上是 O(i) → 整体 O(n²) |
| `std::rotate` | O(n)、不分配、可用于定长容器（`std::array`/C 数组） |

## 构建与运行

```bash
cmake --build build --target ch06_rotate_example ch06_rotate_tests
./build/chapter06_algorithms/ch06_rotate_example
./build/chapter06_algorithms/ch06_rotate_tests
```

## 关键点

- 三种方案结果一致（tests 验证）；
- `std::rotate` 对 C 数组同样有效（本书强调的"算法操作迭代器"）；
- 演示了"先找现成 STL 算法，再手写"的工程习惯。

## 注意

- "unsafe" 版本在 list 上运行安全（list 不重新分配），但在 vector 上
  扩容时会崩溃——本实验只在 list 上演示以保证安全。
