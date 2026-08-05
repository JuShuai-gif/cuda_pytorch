# algorithm_basics

STL 算法基础概念。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 144-148 页：

- **算法只操作迭代器**，不操作容器（`contains()` 可用于任何容器）；
- **算法不改变容器大小**：`std::remove`/`std::unique` 只是把元素移到末尾
  并返回新 end 迭代器，须配合 `erase` 真正删除；
- 2D Grid 可用 1D vector + 迭代器对暴露行（`get_row` 返回迭代器对）。

## 构建与运行

```bash
cmake --build build --target ch06_algorithm_basics_example ch06_algorithm_basics_tests
./build/chapter06_algorithms/ch06_algorithm_basics_example
./build/chapter06_algorithms/ch06_algorithm_basics_tests
```

## 关键点

- `contains()` 模板对 vector/list 通用；
- `std::remove` + `erase` 惯用法（PDF 147 页）；
- `std::unique` + `erase` 惯用法（PDF 147-148 页）。
