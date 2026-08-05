# linear_range

浮点线性范围迭代器。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 136-142 页：

- **问题**：`for (float t = 0; t <= 1; t += 0.1f)` 因 0.1 无法精确表示，
  循环到不了 1.0；
- **解法**：改用"起点 + 步长 × 索引"计算，迭代**索引**而非累加值；
- **封装**：`LinearRangeIterator<T>`（双向迭代器，值即时生成）+
  `LinearRange<T>`（begin/end）+ `make_linear_range()`（类型推导）。

关键设计（PDF 139 页）：迭代器只存 `start_`、`step_size_`、`idx_`，
`operator*` 返回 `start_ + step_size_ * idx_`（计算值）。因为值不存储，
`reference = T`（而非 T&），`pointer = void`。

## 构建与运行

```bash
cmake --build build --target ch05_linear_range_example ch05_linear_range_tests
./build/chapter05_iterators/ch05_linear_range_example
./build/chapter05_iterators/ch05_linear_range_tests
```

## 输出

```
0..1 in 11 values: 0.0 0.1 ... 0.9 1.0     <- 精确到达 1.0
0..1 in 4 values:  0.00 0.33 0.67 1.00
1..0 in 4 values:  1.00 0.67 0.33 0.00
```

## 关键点

- 用索引迭代避免浮点累积误差（PDF 138 页"Using the index is advantageous"）；
- 迭代器/范围是一等公民：可用于 range-for、`std::copy`、`std::set` 等；
- C++17 类模板实参推导（CTAD）使 `LinearRange{0.0f, 1.0f, 4}` 无需显式类型，
  `make_linear_range` 只是锦上添花（PDF 142 页）。

## 注意

- 该迭代器不能用于需要 `T&` 引用的算法（如写入），因为值是计算出来的；
- 本书用 C++17 关联类型方式定义 traits（比 C++11 的 `std::iterator` 继承
  更简单，`std::iterator` 已废弃）。
