# 10_multithreading —— 多线程

对应笔记：`note/07_多线程优化.md`
对应 PDF：第 111-112 页、第 61-62 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `10_baseline` | 串行求和 |
| `10_optimized` | 数据分解 + 每线程局部累加 + 最后合并；线程数作命令行参数 |
| `10_benchmark` | 串行 vs 局部归约（1/2/4/8/16 线程）vs 原子累加 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 10_baseline 10_optimized 10_benchmark -j

./build/10_multithreading/10_benchmark
./build/10_multithreading/10_optimized 8
./build/10_multithreading/10_baseline
```

## 预期结果与解读

- `local xN` 在 N 较小时有近线性加速；超过逻辑核数后不再提升甚至下降（PDF 第 111 页）。本机实测（64M double，内存受限）：serial 约 33 ms → x8 约 11.6 ms → x16 约 8.4 ms。
- 本实现中 `atomic x4`（mutex 版）每线程只加锁一次，**没有形成真正竞争**，因此与 `local x4` 相当——它演示的是"同步次数少则代价小"，而不是激烈的锁竞争。要观察锁竞争可把锁放进循环（代价会非常大）。
- 本机 i9-14900HX 为 P+E 混合架构，32 逻辑线程；加速比曲线不会完美线性。

## 注意事项

- 本实验 64M double = 512 MiB 数据，**内存带宽受限**，多线程收益会小于纯计算型任务。
- `10_optimized` 中 `partial` 是 `vector<double>`，多个线程写相邻槽位可能触发**伪共享**（见 `src/11_false_sharing`）；这里数据块大、写次数少，影响有限。
