# parallel_for

并行化索引式 for 循环。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 338-339 页：

- 索引式 for 循环没有 STL 算法等价物（range-based 对应 `for_each`）；
- 用 Chapter 5 的 `LinearRange` 生成下标范围，再
  `std::for_each(policy, r.begin(), r.end(), f)` 并行执行；
- 包装成 `parallel_for(policy, first, last, f)`，可接受任意执行策略。

## 构建与运行

```bash
cmake --build build --target ch11_parallel_for_example ch11_parallel_for_tests -j
./build/chapter11_parallel_stl/ch11_parallel_for_example
./build/chapter11_parallel_stl/ch11_parallel_for_tests
```

## 关键点

- 复用 `src/chapter05_iterators/linear_range/linear_range.hpp`（跨章复用）；
- 索引循环的体必须**元素独立**（每个下标只读写自己那份数据）；
- example 同时演示 `par` 与 `par_unseq`。

## 注意

- `LinearRange` 的迭代器是 bidirectional，满足 `for_each` 的 ForwardIterator
  要求；
- 逻辑上仍建议优先用算法而非裸索引循环（可读性与可并行性）。
