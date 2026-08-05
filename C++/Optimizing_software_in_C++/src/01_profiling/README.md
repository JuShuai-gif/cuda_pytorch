# 01_profiling —— 性能热点分析

对应笔记：`note/03_性能热点分析.md`、`note/14_性能测试与Benchmark.md`
对应 PDF：第 3 章（15-21 页）、第 16 章（167-171 页）

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `01_simple_timer` | `std::chrono` 手动计时；观察首次（冷缓存）与后续（热缓存）读数差异；用 4 累加器打破依赖链 |
| `01_hotspot` | 故意制造明显热点的程序，供 `perf record`/`perf report` 定位热点 |
| `01_benchmark` | 隔离热点、逐个函数测速对比（PDF 第 167 页"isolate the hot spot"） |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 01_simple_timer 01_hotspot 01_benchmark -j

./build/01_profiling/01_simple_timer
./build/01_profiling/01_hotspot
./build/01_profiling/01_benchmark
```

## perf 使用方法

本机 `perf_event_paranoid=4`，普通用户无权限。需要 root 或临时放开：

```bash
# 方案 A：用 sudo 运行
sudo perf record -g ./build/01_profiling/01_hotspot
sudo perf report

# 方案 B：临时放开权限（推荐）
sudo sysctl kernel.perf_event_paranoid=1
perf record -g ./build/01_profiling/01_hotspot
perf report
```

统计硬件计数器：

```bash
perf stat -e cycles,instructions,cache-misses,branch-misses \
    ./build/01_profiling/01_simple_timer
```

## 预期结果与解读

- `01_simple_timer`：第 1 次读数通常高于后续（代码/数据未进缓存，PDF 第 168 页）；`sum_b`（4 累加器）中位数通常低于 `sum_a`。
- `01_hotspot`：`perf report` 里 `heavy_math` 应占绝大部分 CPU 时间。
- `01_benchmark`：`loop_log_sum`（4 累加器）中位数通常低于 `naive_log_sum`。

> 注意：实际数值随 CPU 频率/负载波动。结论看**相对趋势**，不要拿单次读数下结论（PDF 第 168 页）。

## 为什么这样设计

- 防编译器消除：被测函数返回值折叠进 `volatile sink`（`src/common/benchmark.h`）。
- 预热 + 多轮 + 中位数：稳定测量（PDF 第 168 页）。
- `-O3 -g -fno-omit-frame-pointer`：Release 优化同时保留帧指针，`perf report` 才能显示函数名。
