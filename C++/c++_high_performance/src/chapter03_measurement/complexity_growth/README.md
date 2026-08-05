# complexity_growth

不同渐近复杂度的增长曲线。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 86-89 页：三种搜索算法在不同输入规模（n=10, 1000, 100000）下的耗时。

- 线性搜索 `int`：O(n)；
- 线性搜索 `Point`：O(n)，常数更大（比较两个整数）；
- 二分搜索 `int`（要求有序）：O(log n)。

PDF 第 89 页表格（作者机器）：线性搜索 n=10→0.04ms，n=1000→4.7ms，
n=100000→458ms；二分搜索 0.03→0.08→0.16ms。

## 构建与运行

```bash
cmake --build build --target ch03_complexity_benchmark ch03_complexity_tests
./build/chapter03_measurement/ch03_complexity_tests
./build/chapter03_measurement/ch03_complexity_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，单次搜索 ns）

| n | linear/int | linear/Point | binary/int |
|---|---|---|---|
| 10 | 2.24 | 2.26 | 1.71 |
| 1000 | 194.6 | 196.1 | 3.77 |
| 100000 | 18794 | 19371 | 8.42 |

- 线性搜索：n×100 → 耗时×~100（线性关系）；
- 二分搜索：n×100 → 耗时仅×~2.2（log₂(100)≈6.6 次比较）；
- Point 比 int 慢一点但同一量级（书中亦如此）。

## 重要发现（书中代码的 bug）

书中 PDF 第 88 页的二分搜索使用闭区间 `low <= high` 配合无符号 `size_t`：
当 key 小于所有元素时 `high = mid - 1` 会**下溢**为 `SIZE_MAX`，导致越界访问。
本项目改为半开区间 `[low, high)`，对无符号索引是安全的。这正是
"不要照抄代码、要理解算法"的例证。

## 结论

- 输入规模大时，算法复杂度（增长速率）决定一切，远超常数因子；
- 书中结论："在确认算法和数据结构正确之前，不要浪费时间调优代码"（PDF 89 页）。
