# hash_policy

哈希质量、桶数与 load_factor。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 114-117 页：

- 哈希表 = 桶数组 + **分离链接**（separate chaining）解决冲突；
- `load_factor` = 元素数 / 桶数；超过 `max_load_factor` 触发 **rehash**；
- 等值对象必须同哈希；`==` 判等，哈希决定桶，桶内线性扫描；
- **坏哈希函数**（如恒返 47）合法但让查找退化为 O(n)；
- 哈希组合应包含所有参与 `==` 的字段（书中 `boost::hash_combine` 例子）。

## 构建与运行

```bash
cmake --build build --target ch04_hash_policy_benchmark ch04_hash_policy_tests
./build/chapter04_data_structures/ch04_hash_policy_tests
./build/chapter04_data_structures/ch04_hash_policy_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX）

- 10 万次插入：桶数 172933，rehash 14 次（桶翻倍策略）；
- 好哈希：100 万键、105 万桶、**最大桶大小 1**（分布理想）；
- 坏哈希：10 万键全部挤进 bucket[0]；
- 查找：坏哈希 vs 好哈希 **慢约 47500 倍**（O(n) vs O(1)）。

## 结论

- 哈希质量对性能影响巨大：分布不均 → 单桶退化为 O(n)；
- 自定义哈希应均匀、包含全部判等字段；
- `reserve`/`rehash` 可控制 rehash 频率与 load_factor。
