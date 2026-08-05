# cpu_vs_memory_bound

CPU-bound vs memory-bound。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 96 页：任务是 CPU-bound（换更快 CPU 会更快）还是 I/O/memory-bound
（换更快内存/磁盘会更快）。识别瓶颈类型决定优化方向。

本实验构造两种负载：
- **CPU loop**：寄存器内 LCG 运算，纯算术，CPU 吞吐主导；
- **Memory loop**：256 MiB 数组读+写一次，DRAM 带宽主导。

## 构建与运行

```bash
cmake --build build --target ch03_cpu_mem_benchmark
./build/chapter03_measurement/ch03_cpu_mem_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX）

- CPU loop（1 亿次 LCG）：~73 ms（纯计算）；
- Memory loop（256 MiB）：~20 ms，有效带宽 ~14 GiB/s。

结论：本环境单次 256MiB 顺序扫描由内存带宽主导（远低于 CPU 能处理的
数据率），属于 memory-bound。CPU loop 则只花时间在算术上。

> 用 `perf stat` 观察 cache-misses：memory-bound 负载的 cache-misses 远高于
> CPU-bound（需要 root 调整 `perf_event_paranoid`，见 `scripts/perf_stat.sh`）。

## 重要实现细节（microbenchmark 陷阱）

早期版本 CPU loop 被编译器**常量折叠**为 0ms（纯函数 + 输入恒定 + 结果未被使用）。
修复：把运行中的 checksum 作为种子传入，并打印最终 checksum 强制结果被消费。
这是 microbenchmark 的头号陷阱，详情见 Chapter 12。
