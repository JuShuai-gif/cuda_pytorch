# 11 Parallel STL 与 GPU 计算

> 对应 PDF Chapter 11: Parallel STL（PDF 第 317～352 页，印刷第 300～335 页）
> 本笔记为中文提炼与再解释，非逐句翻译；代码为教学重写，非原书源码复制。

---

## 1. 本章解决什么问题

- 为什么需要并行？什么样的算法适合并行？
- 手写并行 `std::transform()`：朴素分块版有什么缺陷？分治版如何更好？
- 手写并行 `std::count_if()` 如何累加结果？
- 手写并行 `std::copy_if()` 为什么难？两种方案（原子写位置 / 拆分合并）有何表现？
- C++17 Parallel STL 的执行策略 `seq` / `par` / `par_unseq` 有什么区别？
- `std::reduce` / `std::transform_reduce` 与 `std::accumulate` 有何差异？
- 索引式 for 循环如何并行化（`parallel_for`）？
- 如何用 Boost.Compute + OpenCL 把 STL 算法搬到 GPU 上跑？

一句话：**用执行策略一行并行化 STL 算法；理解手写并行算法的复杂度与权衡；用 Boost.Compute 把算法无缝搬到 GPU。**

## 2. 前置知识

- Chapter 5：`LinearRange`（索引范围迭代器）、迭代器分类；
- Chapter 6：STL 算法（transform / count_if / copy_if / for_each / reduce）；
- Chapter 10：线程、`std::async`、原子、缓存行（伪共享）、争用。

## 3. PDF 章节结构

| 小节 | PDF 页码 |
|---|---|
| Importance of parallelism | 318 |
| 手写并行 std::transform()（朴素实现） | 319-321 |
| 朴素实现缺陷（chunk 成本不均、系统负载） | 321-322 |
| Divide and conquer 实现 | 323-326 |
| 手写并行 std::count_if() | 326-327 |
| 手写并行 std::copy_if()（两种方案） | 327-331 |
| 性能评估（is_odd vs is_prime） | 331-332 |
| Parallel STL / Execution policies（seq/par/par_unseq） | 333-335 |
| std::accumulate vs std::reduce / std::transform_reduce | 335-337 |
| std::for_each 并行化 | 337-338 |
| 索引式 for 循环并行化（parallel_for） | 338-339 |
| 在 GPU 上执行 STL 算法 | 340 |
| Boost Compute / OpenCL 基础 | 341-343 |
| 圆面积 transform-reduce 迁移 | 343-347 |
| 谓词与 Boost Compute | 347-348 |
| 自定义 kernel（box filter） | 348-352 |

## 4. 核心概念

| 术语 | 含义 |
|---|---|
| **并行算法** | 等价算法在单核上更慢，收益来自多核分摊 |
| **加速比度量** | A(串行时间) vs B(并行时间×核数)；相等=完美并行 |
| **朴素分块** | 按 `hardware_concurrency()` 均分；块成本不均时受最慢块限制 |
| **分治（divide and conquer）** | 递归对半拆到 chunk 阈值，一个分支 async 一个分支本线程 |
| **原子写位置** | `copy_if` 用 `atomic fetch_add` 同步目标下标（易伪共享） |
| **拆分合并** | 并行条件复制到稀疏区，再顺序 `std::move` 压实 |
| **执行策略（execution policy）** | 传给算法的并行许可（seq/par/par_unseq） |
| **`seq`** | 串行，无并行 |
| **`par`** | 并行；异常会在主线程重新抛出 |
| **`par_unseq`** | 并行 + 允许 SIMD 向量化；谓词不得抛异常/加锁 |
| **`std::reduce`** | 无序版 accumulate（要求结合律/交换律） |
| **`std::transform_reduce`** | transform 后 reduce 的组合 |
| **`parallel_for`** | `LinearRange` + `for_each(policy, ...)` 包装索引循环 |
| **Boost.Compute** | 基于 OpenCL 的 STL 风格 GPU 库（Device/Context/Queue） |
| **`BOOST_COMPUTE_ADAPT_STRUCT`** | 让自定义 struct 可用于 GPU（成员无 padding） |
| **`BOOST_COMPUTE_FUNCTION`** | 把 C 风格函数体写成宏，在 GPU 编译执行 |
| **OpenCL kernel** | C99 语法 GPU 函数，经 `program` + `kernel` + `enqueue_nd_range` 执行 |

## 5. 工作原理

### 5.1 朴素并行 transform 及其缺陷

> 来源：PDF 第 319～322 页
> 原始小节：Naive implementation / Shortcomings

把范围按 `hardware_concurrency()` 均分成块，每块一个 `std::async`：

- 优点：块数少、每个元素独立变换，理论近乎线性加速（书中 32 元素重函数 7.99x）；
- 缺陷一：块计算成本不均时，整体受**最慢块**限制；
- 缺陷二：系统有其它进程竞争时，部分块无法并行执行；
- 结论：拆成更多更小的块，让调度器动态平衡。

### 5.2 分治并行 transform

> 来源：PDF 第 323～326 页
> 原始小节：Divide and conquer

递归对半拆分直到达到 chunk 阈值：

```cpp
template <typename SrcIt, typename DstIt, typename Func>
void par_transform(SrcIt first, SrcIt last, DstIt dst, Func f, size_t chunk) {
    const auto n = static_cast<size_t>(std::distance(first, last));
    if (n <= chunk) { std::transform(first, last, dst, f); return; }
    const auto src_mid = std::next(first, n / 2);
    auto future = std::async([=] { par_transform(first, src_mid, dst, f, chunk); });
    par_transform(src_mid, last, std::next(dst, n / 2), f, chunk);  // 本线程
    future.wait();
}
```

- chunk 越小任务越多，调度越灵活但开销越大；书中实测 chunk≈10'000 最佳；
- 成本随值变化的函数（`i_max = v/100000`）：过小 chunk 使 10 元素级任务开销反噬（0.55x）。

### 5.3 并行 count_if

> 来源：PDF 第 326～327 页
> 原始小节：Implementing parallel std::count_if

同分治结构，只是把两个分支的计数相加：`return num + future.get();`。

### 5.4 并行 copy_if 的两种方案

> 来源：PDF 第 327～331 页
> 原始小节：Implementing parallel std::copy_if

顺序 `copy_if` 简单，但并行时**多个线程并发写同一目标位置**是未定义行为。两种方案：

1. **原子写位置（sync）**：全局 `std::atomic<size_t> dst_idx`，每命中
   `fetch_add` 取唯一下标。缺点：多线程写相邻缓存行→**伪共享灾难**。
   书中 is_odd（轻谓词）时仅 0.07x（比串行慢 14 倍）；
2. **拆分合并（split）**：第一步各 chunk 并行复制到各自稀疏区间
   （记下每块 `[dst_first, dst_last)`）；第二步顺序 `std::move` 压实。
   无共享写、无伪共享；is_prime（重谓词）时 5.09x。

### 5.5 执行策略

> 来源：PDF 第 333～335 页
> 原始小节：Execution policies

```cpp
// 给 std::find 加一个参数即可并行
*std::find(std::execution::par, v.begin(), v.end(), "loopy");
```

- `seq`：串行（大小低于阈值时用）；
- `par`：并行；谓词可抛异常，异常在调用线程重新抛出、算法中途停止（位置未定义）；
  **注意：GCC 13 libstdc++ 实测 `par` 谓词抛异常会直接 `terminate`（不传播），
  与书中 GCC 7 行为不同；生产代码应避免在并行谓词中抛异常**；
- `par_unseq`：并行 + 允许向量化；谓词**不得抛异常、不得加锁**（同线程交错执行会死锁）；
- 头文件 `<execution>`，命名空间 `std::execution`。

### 5.6 reduce 系列

> 来源：PDF 第 335～337 页
> 原始小节：std::accumulate and std::reduce

- `std::accumulate` 要求按序执行，无法并行；
- `std::reduce` 无序归约，要求操作**交换律 + 结合律**；整数加法/乘法 OK，
  字符串拼接顺序不定；
- `std::transform_reduce`：先 transform 再归约（如每字符串长度求和）。

### 5.7 并行 for_each 与索引循环

> 来源：PDF 第 337～339 页
> 原始小节：std::for_each / Parallelizing an index-based for-loop

- 并行 `for_each` 返回 `void`（原版返回 functor，但并行时调用序不定）；
- 索引式 for 循环无算法等价物；用 `LinearRange`（Chapter 5）生成下标范围，
  再 `for_each(policy, ...)` 并行执行，包装成 `parallel_for(first, last, f)`。

### 5.8 Boost.Compute 与 GPU

> 来源：PDF 第 340～352 页
> 原始小节：Executing STL algorithms on the GPU / Boost Compute

- 基本概念：`device`（GPU）、`context`（设备门）、`queue`（命令队列）；
- 数据必须拷贝进 `bc::vector`（GPU 内存），算完拷回 `std::vector`；
- 自定义 struct 用 `BOOST_COMPUTE_ADAPT_STRUCT(Circle, Circle, (x,y,r))`
  适配（成员需对齐无 padding）；
- 函数用 `BOOST_COMPUTE_FUNCTION(返回, 名, (参数), { 函数体 })` 宏，
  由 OpenCL 驱动在运行时编译（C99 语法）；
- 使用 `bc::transform` / `bc::reduce` / `bc::sort` / `bc::iota` / `bc::fill`
  等与 STL 同名 API；
- 自定义 kernel：OpenCL `program::create_with_source` + `build()` + `kernel`，
  `enqueue_nd_range_kernel(kernel, 2, offset, elems)` 二维并行（box filter）；
- GPU 结果用 `std::equal` + epsilon 与 CPU 结果比对验证；
- 书中强调：GPU 常受**数据往返拷贝**瓶颈，但计算密度高时 30x 不罕见。

## 6. PDF 核心观点

| 观点 | 页码 | 本项目的验证 |
|---|---|---|
| 并行算法单核更慢、多核分摊 | 318 | benchmark 单核基线 |
| 完美并行很罕见（8 核约 4.6-8x） | 303-304 | 本机实测加速比 |
| 朴素分块受最慢块限制 | 321 | 变成本函数 chunk 扫描 |
| 分治 + 小 chunk 更鲁棒 | 323-326 | 变 chunk 大小对比 |
| chunk 过小开销反噬 | 308 | chunk=10 时 <1x |
| copy_if 并行难在并发写 | 327 | 原子版伪共享灾难 |
| split 方案无共享写、赢重谓词 | 331 | is_prime 显著加速 |
| par_unseq 谓词不得抛异常/加锁 | 318 | 文档记录 |
| reduce 无序，要求可交换结合 | 319-320 | 整数求和等价 |
| 执行策略一行并行化 | 333 | transform/find + par |
| GPU 受数据搬运瓶颈 | 335 | GPU/CPU benchmark 实测 |

## 7. 简单示例

```cpp
// 一行把 reduce 换成并行（PDF p.319）
auto sum = std::reduce(std::execution::par, v.begin(), v.end(), 0);
```

## 8. 未优化版本

- 手写分块并行 transform：块数固定、成本不均时慢、系统负载敏感；
- 原子写位置 copy_if：伪共享使其比串行还慢 14 倍（书中实测）；
- 手写索引循环：无法用执行策略并行；
- 手写 `accumulate` 作并行归约：要求按序，无法并行。

## 9. 优化版本

- 分治 `par_transform`：递归拆块、调度器动态平衡；
- 拆分合并 `par_copy_if`：无共享写，重谓词下 5x；
- 执行策略一行并行化：`std::execution::par` / `par_unseq`；
- `std::reduce` / `transform_reduce`：无序归约可并行；
- `parallel_for`：LinearRange + for_each 包装索引循环；
- Boost.Compute：STL 风格 GPU 算法，把 transform/reduce/sort 搬到 GPU。

## 10. 为什么可能更快（逐维度分析）

| 维度 | 分析 |
|---|---|
| 时间复杂度 | 算法复杂度不变；多核分摊墙钟时间 |
| 空间复杂度 | split copy_if 需要完整目标区间（稀疏）+ 返回迭代器 |
| 动态内存分配 | `std::async` 任务对象有开销（chunk 过小即分配过多） |
| 对象复制和移动 | split 合并用 `std::move` 避免拷贝 |
| Cache Locality | 原子写位置→伪共享灾难；split 每块顺序写局部 |
| 函数调用和内联 | 谓词/变换函数可内联；task 调度有开销 |
| 编译期计算 | — |
| SIMD / 并行 | par_unseq 允许 SIMD；多核并行是核心收益 |

## 11. 该优化什么时候可能无效

- **chunk 过小**：任务创建/调度开销超过计算量（书中 0.55x）；
- **轻谓词 + 共享写**：原子写位置 copy_if 比串行慢（伪共享）；
- **GPU 数据量小**：往返拷贝开销占主导，GPU 无优势；
- **操作不满足交换/结合律**：`reduce` 结果不定（字符串拼接）；
- **单核机器**：并行算法单核更慢，无收益；
- **争用/带宽受限**：并行受外部参数限制（书中 8 核常仅 4-5x）。

## 12. 如何验证

```bash
cmake --build build --target ch11_par_transform_example ch11_par_count_if_example \
    ch11_par_copy_if_example ch11_execution_policies_example \
    ch11_parallel_for_example -j

# 正确性（串行/并行结果逐位一致）
./build/chapter11_parallel_stl/ch11_*_tests

# 性能（分治 chunk 扫描、copy_if 双方案、GPU vs CPU）
./build/chapter11_parallel_stl/ch11_par_transform_benchmark
./build/chapter11_parallel_stl/ch11_par_copy_if_benchmark

# GPU（需 ENABLE_BOOST_COMPUTE + ENABLE_OPENCL，本机已验证可用）
./build/chapter11_parallel_stl/ch11_boost_compute_example
```

## 13. 汇编观察点

- 并行算法 `-fopenmp`/TBB 运行时调度；`par_unseq` 下生成 SIMD 指令；
- 原子 `fetch_add` 编译为 `lock xadd`（copy_if sync 版热路径）；
- GPU 代码由 OpenCL 驱动运行时编译，不在此次编译产物中。

## 14. 常见错误

- 手写并行 `copy_if` 并发写同一目标迭代器（未定义行为）；
- `std::reduce` 用于非交换/结合操作（结果不定）；
- `par_unseq` 谓词里抛异常或加锁（UB / 死锁）；`par` 谓词抛异常在 GCC 13
  会 terminate（libstdc++ 实现差异，勿依赖传播语义）；
- `hardware_concurrency()` 可能返回 0（未 clamp）；
- Boost.Compute 自定义 struct 含 padding（成员未对齐）→ GPU 读取错位；
- 忘记把数据拷回 CPU 就看结果（`bc::copy` 回传）；
- GPU 用 `==` 精确比对浮点（应用 epsilon 容差）。

## 15. 实践练习

1. **简单**：给 `par_transform_naive` 加 `num_tasks=1` 时直接串行快路径；
2. **简单**：用 `std::execution::par_unseq` 重跑 reduce 求和并对比 `par`；
3. **中等**：实现 `par_reduce`（分治求和，返回 long long），与 `reduce(par)` 对比；
4. **中等**：给 split `copy_if` 换不同 chunk 大小扫描性能；
5. **困难**：用 Boost.Compute 实现 `par_copy_if_split` 的 GPU 版本并验证。

## 16. 对应代码

| 目录 | 内容 |
|---|---|
| `src/chapter11_parallel_stl/par_transform/` | 朴素 + 分治并行 transform |
| `src/chapter11_parallel_stl/par_count_if/` | 分治并行 count_if |
| `src/chapter11_parallel_stl/par_copy_if/` | 原子写位置 vs 拆分合并 |
| `src/chapter11_parallel_stl/execution_policies/` | seq/par/par_unseq + reduce |
| `src/chapter11_parallel_stl/parallel_for/` | 索引循环并行化 |
| `src/chapter11_parallel_stl/boost_compute/` | GPU 圆面积 + box filter |

## 17. 本章总结

- 并行算法 = 单核更慢 + 多核分摊；加速比受调度/带宽/负载限制；
- 手写并行 transform/count_if 用分治最鲁棒；copy_if 难点在并发写，
  拆分合并方案避免共享写（原子写位置会伪共享灾难）；
- 执行策略是 C++17 的一行并行化入口：seq / par / par_unseq；
- `reduce` 无序归约（要求交换+结合）；`transform_reduce` 组合变换与归约；
- 索引循环用 `LinearRange + for_each(policy)` 包装成 `parallel_for`；
- Boost.Compute 让 STL 算法以近乎相同语法跑在 GPU；注意数据往返与
  自定义 kernel 的编写、GPU 结果需 CPU 验证。

---

> 来源：PDF 第 317～352 页（Chapter 11: Parallel STL）
> 相关实验：`src/chapter11_parallel_stl/*`
