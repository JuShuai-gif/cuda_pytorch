# Chapter 15 — 性能分析与优化

系统化测量、分析和优化并发代码性能。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_benchmark_basic.cpp` | 基准测试框架 | 微基准、吞吐量测量、扩展性、Amdahl 定律 |
| `02_latency_throughput.cpp` | 延迟vs吞吐量 | P50/P99/P99.9、批处理权衡、尾延迟 |
| `03_contention_analysis.cpp` | 竞争分析 | 竞争级别、锁粒度、shared_mutex vs mutex |
| `04_lock_vs_lockfree_bench.cpp` | 锁vs无锁对比 | mutex/spinlock/CAS/fetch_add/perthread 实测 |
| `05_perf_guide.cpp` | perf工具指南 | perf stat/record/lock、火焰图、cache 分析 |

## 编译运行

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
./ch15_01_benchmark_basic
./ch15_02_latency_throughput
./ch15_03_contention_analysis
./ch15_04_lock_vs_lockfree_bench
./ch15_05_perf_guide
```

## perf 分析

```bash
# 编译时加调试符号
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j$(nproc)

# CPU 热点
perf record -g ./ch15_05_perf_guide && perf report

# 硬件事件统计
perf stat -d ./ch15_04_lock_vs_lockfree_bench

# 锁竞争
perf lock record ./ch15_03_contention_analysis && perf lock report
```

## 学习建议

1. 先理解 latency vs throughput 的本质区别
2. 掌握 benchmark 的正确方法（多次取平均、消除干扰）
3. 学会用 perf 找到真正的瓶颈（而非猜测）
4. 锁 vs 无锁的选择取决于竞争级别，没有绝对最优
