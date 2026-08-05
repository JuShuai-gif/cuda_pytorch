# priority_queue

用优先队列实现"只取前 m 个最大"。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 117-120 页：搜索命中按 rank 排序，只需返回前 m 个。三种方案：
- `std::sort` 全部排序：O(n log n)；
- `std::partial_sort`：O(n log m)（但需要随机访问迭代器）；
- **`std::priority_queue`**：只需前向迭代器即可，O(n log m) 时间、O(m) 内存。

书中实现：最小堆保存当前最高的 m 个；新元素若比堆顶（最小）大则替换。

## 构建与运行

```bash
cmake --build build --target ch04_priority_queue_example
./build/chapter04_data_structures/ch04_priority_queue_example
```

## 输出

```
top-10 ranks: 1000.0 1000.0 ... 999.9 ...
top-m matches full sort: yes
```

程序用 10 万随机 Hit 求 top-10，并与全排序结果比对验证正确性。

## 复杂度

- 时间 O(n × log m)（n 个元素 × 每次 O(log m) 堆操作）；
- 空间 O(m)（堆中最多 m 个）。

## 结论

- 优先队列是"部分排序"的高效替代，尤其当迭代器不支持随机访问时；
- 需理解 `priority_queue` 默认是**最大堆**，用 `greater` 比较器变最小堆。
