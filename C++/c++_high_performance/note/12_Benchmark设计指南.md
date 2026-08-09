# 12 Benchmark 设计指南

> 综合全书（重点 Chapter 3 与各章 benchmark 实验）的方法论笔记，非单章翻译。
> 本项目的实测基准统一使用 `src/common/benchmark.hpp`。

---

## 1. 为什么需要 benchmark

- 性能优化必须以**测量**为前提：没有数据支撑的优化是臆测（Chapter 3 开篇立场）；
- 编译器、CPU 行为复杂且易变：直觉结论经常与实测相反（本书多章反复印证）；
- 结论必须可重复、可溯源：记录数据规模、编译器、标准库、硬件（本项目每个
  `print_result` 都输出 system 信息）。

## 2. 测量前要回答的问题

1. **测什么**：完整运行时间，还是热点函数？后者用 perf / gprof（paranoid 限制
   时用 gprof 免 root）；
2. **数据规模**：小数据 vs 大数据结论可能相反（Chapter 3 `release_vs_debug`、
   Chapter 11 并行算法对规模敏感）；
3. **I/O bound 还是 CPU bound**：决定优化方向（Chapter 3）；
4. **对比基准**：未优化版 vs 优化版，或与标准库实现对比；
5. **统计口径**：单次运行不可信，需多轮 + 统计量（本项目用 mean/median/min/
   stddev）。

## 3. 本项目 benchmark 基础设施（src/common）

| 组件 | 职责 |
|---|---|
| `benchmark.hpp` | `chp::benchmark(iterations, rounds, warmup, fn)` 统一测量 |
| `compiler_barrier.hpp` | 防止编译器提升/消除被测工作 |
| `statistics.hpp` | mean/median/min/max/stddev |
| `system_info.hpp` | 输出 OS/CPU/编译器，保证可溯源 |
| `test_utils.hpp` | 正确性断言（CHP_CHECK），先正确后测速 |

设计要点：

- **warmup rounds**：预热缓存、分支预测器、频率缩放后才计时；
- **checksum 参数**：回调把结果累进 `checksum`，配合 `compiler_barrier()` 防止
  编译器把整个循环优化掉（关键：基准可能被"优化成什么都不做"）；
- **相同回调签名**：被对比的两个实现用同一签名，簿记开销一致。

## 4. 编译器可能如何"破坏"你的 benchmark

| 问题 | 表现 | 对策 |
|---|---|---|
| 死代码消除 | 结果未使用 → 整个计算被删 | checksum 累加 + compiler_barrier |
| 循环提升 | 不变计算被移出循环 | barrier 在循环内 |
| 常量折叠 | 编译期算出结果，运行期零工作 | 用运行期输入（随机/文件） |
| 内联 | 函数调用消失，测不到调用成本 | 若目标就是测内联则无所谓 |
| 未定义行为优化 | 数据竞争/越界导致"优化到意外" | 先用 Sanitizer 保证正确 |

> 本项目实例：Chapter 10 伪共享 benchmark 若不加 `compiler_barrier()`，编译器把
> `a += 1` 提升到寄存器、最后才写内存，伪共享根本不会发生（1.02x）；加 barrier
> 后实测 9.4x。

## 5. 常见坑

- **测量太短**：单次迭代 < 微秒级，计时噪声占比高 → 增加 iterations/rounds；
- **未预热**：冷缓存第一轮远慢于稳态 → 加 warmup；
- **顺序偏差**：先测的实现热了缓存 → 交替/随机化顺序，或各用独立数据；
- **数据未初始化**：`vector(n)` 默认构造后直接计时，实际包含构造 → 先填数据；
- **只比 mean**：看 min/stddev，高方差说明受系统扰动 → 多次运行；
- **优化级别不同**：Debug vs Release 结果不可比（Chapter 3 有专门实验）；
- **同一程序跑多核**：共享资源（带宽/缓存）使并行 benchmark 失真 → 控制环境。

## 6. 如何设计可靠的对照实验

1. **正确性先行**：两个实现先用 `tests.cpp`（CHP_CHECK）证明结果一致；
2. **独立数据**：两个实现分别测，避免一方改写数据影响另一方；
3. **相同工作量**：checksum 一致（本项目 print 时校验双方 checksum）；
4. **多轮取统计**：`benchmark(iterations, rounds, warmup)` 输出 mean/min/stddev；
5. **记录环境**：打印编译器/CPU/数据规模；
6. **结论标注条件**：如"本机 GCC 13.3 / i7-13700K 实测 5.2x"，注明书中硬件
   对比（i7-7700k 10.7x），避免误导。

## 7. 何时不用 micro-benchmark

- 测**整体程序**：用 perf stat / time（end-to-end）；
- 测**内存/IO 瓶颈**：micro-benchmark 缓存状态难以控制，perf cache-misses 更准；
- 测**多线程可扩展性**：需完整工作负载 + 不同线程数扫描；
- 测**GPU**：数据搬运开销大，micro-benchmark 会严重低估实际收益（Chapter 11）。

## 8. 本项目各章 benchmark 对照表

| 实验 | 对比项 | 本机实测 | 书中（i7-7700k） |
|---|---|---|---|
| ch01 contiguous_vs_pointer | 连续 vs 指针存储 | （见对应 README） | — |
| ch03 release_vs_debug | 优化开关影响 | （见对应 README） | — |
| ch09 concat_proxy | 免临时字符串 | 5.2x | 10.7x |
| ch09 distance_proxy | 免 sqrt | 1.14x | 2x |
| ch10 false_sharing | 缓存行对齐 | 9.4x | 显著 |
| ch11 par_transform | 手写并行 | 17.2x | 5.9x |
| ch11 par_copy_if | split vs sync | split 3.9-9.9x | 5.1x |

> 差异原因：硬件（本机 24 核 vs 书中 8 核）、libstdc++ 版本、TBB 版本、分配器。
> 这正说明：**性能结论必须实测 + 标注环境**，书上的数字只是参考。

## 9. 推荐的工作流

```
1. 用 tests 证明正确性（先对后快）
2. 用 perf/gprof 找热点（避免优化非热点）
3. 对热点写 micro-benchmark（warmup + barrier + 统计）
4. 优化 → 重测 → 对比（记录环境）
5. 若优化无效/变慢，回滚并记录
```

---

> 相关实现：`src/common/benchmark.hpp`、`src/common/statistics.hpp`、
> `src/common/compiler_barrier.hpp`、`src/common/system_info.*`
> 相关实验：各章 `benchmark.cpp`
