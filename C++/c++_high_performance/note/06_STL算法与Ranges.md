# 06 STL 算法与 Ranges

> 对应 PDF Chapter 6: STL Algorithms and Beyond（PDF 第 144～173 页，印刷第 127～156 页）
> 本笔记为中文提炼与再解释，非逐句翻译；代码为教学重写，非原书源码复制。

---

## 1. 本章解决什么问题

- 为什么用 STL 算法而不是手写 for 循环？
- 算法的迭代器约定、输出迭代器、谓词与比较器如何工作？
- 手写循环有哪些隐蔽的崩溃与性能陷阱（`std::rotate` 例子）？
- `std::find` 的隐藏优化（4 路展开）收益有多大？
- `sort`/`partial_sort`/`nth_element` 如何选择？
- 为什么需要 ranges 库？view 如何实现惰性组合？

一句话：**把算法当积木，STL 是首选；手写循环要警惕迭代器失效与复杂度陷阱。**

## 2. 前置知识

- Chapter 5：迭代器、迭代器类别、traits；
- Chapter 2：lambda、谓词；
- Chapter 3：大 O 复杂度。

## 3. PDF 章节结构

| 小节 | PDF 页码 |
|---|---|
| STL algorithm concepts | 144-153 |
| 算法操作迭代器（contains/Grid） | 145-146 |
| 迭代器指向 [first, last) | 147 |
| 算法不改变容器大小（remove/unique） | 147-148 |
| 输出算法需已分配数据 | 148-149 |
| 默认 operator== / operator< | 149-150 |
| 自定义比较器与通用谓词 | 150-151 |
| 算法要求 move 不抛异常 | 151 |
| 算法有复杂度保证 | 152 |
| 与 C 库函数等价（memcpy 等） | 153 |
| STL algorithms versus handcrafted for-loops | 153-159 |
| Example 1 – 异常与性能问题（rotate） | 156-158 |
| Example 2 – STL 的微妙优化（find unroll） | 158-159 |
| Sorting only for the data you need | 160-162 |
| The future of STL and the ranges library | 163-173 |
| Limitations of the iterators / 介绍 ranges | 163-166 |
| Actions, views, and algorithms | 169-173 |

## 4. 核心概念

| 术语 | 含义 |
|---|---|
| **[first, last) 区间** | 迭代器对：first 指向首元素，last 指向末元素之后 |
| **erase-remove 惯用法** | `remove` 把元素移尾 + `erase` 真正删除 |
| **输出迭代器** | 算法写数据的目标：预分配容器/`back_inserter`/`inserter` |
| **谓词（predicate）** | 返回 bool 的可调用对象；`_if` 后缀算法接受谓词 |
| **比较器（comparator）** | 自定义 `<` 语义（如按字符串长度） |
| **`std::rotate`** | 循环移动元素，O(n)，不分配，可处理定长容器 |
| **`nth_element`** | 第 n 个元素就位，前段无序，O(n) |
| **view** | 惰性求值的迭代器代理，不复制数据 |
| **action** | 修改容器的 range 操作 |
| **管道运算符 `\|`** | 从左到右组合 view |

## 5. 工作原理

### 5.1 算法约定

> 来源：PDF 第 144～149 页
> 原始小节：STL algorithm concepts

- 算法只认迭代器，不认识容器（`contains()` 可用于 vector/list/裸指针）；
- 区间是半开 `[first, last)`；
- **算法不改变容器大小**：`std::remove`/`std::unique` 只重排元素并返回
  新 end，必须 `erase` 收尾；
- 输出算法需要已分配空间：预分配 / `back_inserter` / `inserter`
  （直接写空容器 `begin()` 是 UB）。

### 5.2 谓词与比较器

> 来源：PDF 第 149～151 页
> 原始小节：Custom comparator / General-purpose predicates

- 算法默认用 `==`（find/count）与 `<`（sort/max_element）；
- 自定义比较器作为额外参数传入；谓词版算法加 `_if` 后缀；
- 建议建 `preds` 命名空间存放通用谓词（`less_by_size`、
  `equal_case_insensitive` 等）。

### 5.3 手写循环的陷阱：rotate 例子

> 来源：PDF 第 156～158 页
> 原始小节：Example 1 – Unfortunate exceptions and performance problems

把前 n 个元素移到末尾：
- 方案 1（迭代器 + emplace_back）：vector 扩容 → 迭代器失效 → 崩溃；
- 方案 2（索引 + std::next）：安全，但 `std::next(c.begin(), i)` 在 list
  上是 O(i) → 整体 O(n²)；
- 方案 3（`std::rotate`）：O(n)、不分配、适用于定长容器。

**教学点：先浏览 STL 是否有现成算法，再手写。**

### 5.4 find 的隐藏优化

> 来源：PDF 第 158～159 页
> 原始小节：Example 2 – STL has subtle optimizations

libstdc++ 的 `find_if` 把循环 4 路展开（比较次数减 3/4）+ trip_count 与零
比较（x86 用 `test` 而非 `cmp`）。书中实测仅快 0.5%（3420 vs 3402µs）。

本实验实测：`find_fast` 展开版 ~87ns vs 朴素 ~130ns（~1.5 倍，编译器自动
向量化削弱了朴素版的差距）；`std::find` ~86ns 与展开版相当，验证 libstdc++
确实用该技巧。

### 5.5 三种排序

> 来源：PDF 第 160～162 页
> 原始小节：Sorting only for the data you need

- `std::sort` O(n log n)：全排序；
- `std::partial_sort` O(n log m)：前 m 个有序；
- `std::nth_element` O(n)：第 n 个就位。

书中实测（10M，m=1M）：找中位数快 12.4x，部分区间 4.6-8.7x。
本实验实测（2M，m=100k）：partial_sort 快 ~3.1x，nth_element 快 ~8.7x。

### 5.6 ranges 库（现代补充：C++20 std::ranges）

> 来源：PDF 第 163～173 页
> 原始小节：The future of STL and the ranges library

> 现代补充：书中描述的是当时未入标准的 range-v3；C++20 已将其以
> `std::ranges`/`std::views` 落地。

**问题**：STL 算法无法组合。"找出最高等级弓手"需要 copy_if 到新容器再
max_element（浪费复制）。

**解决**：view 是惰性迭代代理，不复制数据；管道 `|` 从左到右组合：

```cpp
auto archer_levels = warriors
  | std::views::filter(is_archer)
  | std::views::transform(level_of);
auto max = *std::ranges::max_element(archer_levels);
```

- **views**：惰性，只迭代一次；**actions**：修改容器；
- 本实验 `cpp20_ranges` 用 C++20 复刻书中所有例子。

## 6. PDF 核心观点

| 观点 | 页码 | 本项目的验证 |
|---|---|---|
| 算法操作迭代器而非容器 | 145 | contains()/Grid 通用 |
| 算法不改变容器大小 | 147 | remove/unique + erase |
| 输出算法需已分配数据 | 148 | 三种输出方式 |
| 用谓词命名空间提高可读性 | 150-151 | comparators 实验 |
| 手写循环有迭代器失效陷阱 | 156 | rotate 三方案对比 |
| 算法有复杂度保证且不分配 | 152 | 复杂度表 |
| find 的 4 路展开优化 | 158-159 | 展开快 ~1.5 倍；std::find 相当 |
| nth_element/partial_sort 避免全排序 | 160-162 | partial 3.1x、nth 8.7x |
| ranges view 惰性组合避免中间容器 | 163-168 | C++20 ranges 复刻 |

## 7. 简单示例

```cpp
// 惰性组合（C++20）
auto odd_squares = numbers | std::views::transform([](int v){return v*v;})
                           | std::views::filter([](int v){return v%2==1;});
for (int s : odd_squares) { /* 1 9 25 ... */ }  // 不创建中间容器
```

## 8. 未优化版本

- 手写 for 循环实现 count/find/transform/copy_if/accumulate
  （`loop_vs_algorithms/baseline.cpp`）；
- 索引循环的 move_n_to_back（O(n²) on list，`rotate_example`）；
- 朴素 `find_slow`（`find_unroll`）。

## 9. 优化版本

- 对应 STL 算法版本（`loop_vs_algorithms/optimized.cpp`）；
- `std::rotate`（`rotate_example`）；
- 4 路展开 `find_fast`（`find_unroll`）；
- `nth_element`/`partial_sort`（`partial_sorting`）；
- C++20 ranges view（`cpp20_ranges`）。

## 10. 为什么可能更快（逐维度分析）

| 维度 | 分析 |
|---|---|
| 时间复杂度 | nth_element O(n) vs sort O(n log n)（实测 8.7x）；rotate O(n) vs 索引 O(n²) |
| 空间复杂度 | view 零额外空间；rotate 零分配 |
| 动态内存分配 | 手写 copy_if 每次 push_back 可能扩容；STL + reserve 避免 |
| Cache Locality | —（与容器布局相关，见 Chapter 4） |
| 函数调用和内联 | 模板算法内联；谓词 lambda 内联 |
| 分支 | find 4 路展开减少循环比较 |
| SIMD / 并行 | STL 算法便于后续并行化（Chapter 11） |

## 11. 该优化什么时候可能无效

- **数据量小**：nth_element 相对 sort 的优势随 n 减小而消失；
- **编译器已优化**：手写循环可能被编译器自动向量化/展开（本实验朴素
  find 差距比书中小正是此因）；
- **可读性**：ranges 管道可读但调试较难；过度组合反而费解；
- **view 悬垂**：view 引用源，源析构后使用是 UB；
- **action vs view 语义混淆**：action 改容器、view 不改；
- **异常安全**：手写索引循环在 list 上 O(n²) 是复杂度问题而非正确性问题。

## 12. 如何验证

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CPP20_EXAMPLES=ON
cmake --build build -j

# 正确性（8 个 tests）
./build/chapter06_algorithms/ch06_*_tests

# Benchmark
./build/chapter06_algorithms/ch06_find_unroll_benchmark
./build/chapter06_algorithms/ch06_partial_sorting_benchmark
./build/chapter06_algorithms/ch06_loop_vs_algos_benchmark

# C++20 ranges
./build/chapter06_algorithms/ch06_cpp20_ranges_example
```

## 13. 汇编观察点

- `std::find` → 4 路展开的 `mov`+`cmp`+`je` 序列；
- `find_slow` → 单元素循环；
- `std::transform` → 向量化（`movdqu`+`pmulld`）；
- ranges view → 内联的管道式迭代器，无运行时开销。

## 14. 常见错误

- 用 `remove`/`unique` 忘了 `erase`（容器没变小）；
- 输出算法写进空容器 `begin()`（UB）；
- 手写循环中 `std::next(c.begin(), i)` 在 list 上造成 O(n²)；
- 迭代器在扩容后继续使用；
- 全排序取 top-m（应该用 nth_element/partial_sort）；
- 在已析构的源上使用 view；
- 把 `std::copy` 用于"反向"区间（正向迭代死循环，Chapter 5 教训）。

## 15. 实践练习

1. **简单**：用 `std::any_of` 改写书中的 conflicting 例子（PDF 155 页）；
2. **简单**：`contains_duplicates` 用 `adjacent_find` + 排序实现
   （PDF 152 页），并测两种复杂度；
3. **中等**：把 `loop_vs_algorithms` 的 copy_if 改为手写 + `reserve`，
   观察与 STL 版差距是否缩小；
4. **中等**：用 `std::ranges` 实现"筛选所有等级≥X 的战士并按名字排序"；
5. **困难**：用 `std::partial_sort` 实现"Top-K 排行榜"（对比 nth_element
   后再排序前 K 个），写 benchmark 对比三种取 top-K 方案。

## 16. 对应代码

| 目录 | 内容 |
|---|---|
| `src/chapter06_algorithms/algorithm_basics/` | 迭代器约定、remove/unique |
| `src/chapter06_algorithms/output_iterators/` | 输出迭代器三种方式 |
| `src/chapter06_algorithms/comparators/` | 谓词与比较器 |
| `src/chapter06_algorithms/rotate_example/` | 手写 vs std::rotate |
| `src/chapter06_algorithms/find_unroll/` | find 隐藏优化 |
| `src/chapter06_algorithms/partial_sorting/` | 三种排序 |
| `src/chapter06_algorithms/loop_vs_algorithms/` | 手写 vs STL 六操作 |
| `src/chapter06_algorithms/cpp20_ranges/` | C++20 ranges（可选） |

## 17. 本章总结

- STL 算法是"积木"：性能（隐藏优化）、安全（边界情况）、可读（意图即名字）、
  未来可并行；
- 手写循环前先找现成算法（`rotate` 例子）；注意迭代器失效与 O(n²) 陷阱；
- `find` 的展开优化等"免费收益"直接用 STL 获得，不要自己手写微优化；
- 只排序需要的数据：`nth_element`/`partial_sort` 明显快于全排序；
- ranges view 解决算法组合问题：惰性、零复制、管道可读；
- 一切以实测为准：本环境数据见 `benchmark_results/chapter06/`。

---

> 来源：PDF 第 144～173 页（Chapter 6: STL Algorithms and Beyond）
> 相关实验：`src/chapter06_algorithms/*`；结果存档：`benchmark_results/chapter06/`
