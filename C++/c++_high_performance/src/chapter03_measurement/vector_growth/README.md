# vector_growth

摊销复杂度与 `reserve`。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 92-94 页：`vector::push_back` 是**均摊 O(1)**。内部数组满时扩容，
扩容本身 O(n)，但通过**指数增长**（翻倍）使昂贵操作只发生 log₂(n) 次，
整段序列的均摊成本为 O(1)。

书中推导：对大小为 n 的 vector，需扩容 log₂(n) 次，第 i 次移动 2^i 个元素，
等比数列求和后除以 m 得 O(1)。

## 构建与运行

```bash
cmake --build build --target ch03_vector_growth_example \
      ch03_vector_growth_benchmark ch03_vector_growth_tests
./build/chapter03_measurement/ch03_vector_growth_example
./build/chapter03_measurement/ch03_vector_growth_tests
./build/chapter03_measurement/ch03_vector_growth_benchmark
```

## 结果解释

example 输出（libstdc++ 翻倍策略）：

```
after  1 inserts: capacity =    1 (moves: 0)
after  2 inserts: capacity =    2 (moves: 1)
after  3 inserts: capacity =    4 (moves: 3)
...
after 513 inserts: capacity = 1024 (moves: 1023)
size=1024 capacity=1024 reallocations=11
total moves = 1023, moves/size ≈ 1.0   (均摊 ~O(1))
```

11 次扩容 = log₂(1024)，总移动 1023 ≈ size，均摊每次 push_back ~1 次移动。

Benchmark（100k 元素构建）：`reserve` 版本比不 reserve 快约 **3.5 倍**
（本环境实测；差距来自避免扩容搬移 + 分配次数减少）。

## 结论

- `push_back` 均摊 O(1)，但偶尔触发扩容是真实的 stall；
- 已知大致规模时 `reserve()` 消除扩容，是有价值的低风险优化；
- 扩容搬移成本随元素复制/移动成本增大而增大（见 Chapter 2 noexcept_move）。
