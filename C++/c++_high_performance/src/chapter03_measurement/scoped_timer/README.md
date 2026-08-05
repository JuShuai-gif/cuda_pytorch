# scoped_timer

Instrumentation profiler 风格的计时器。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 99-100 页展示了 `ScopedTimer` 类：在构造时记录 `steady_clock` 时间点，
析构时输出耗时。这是**插桩式（instrumentation）profiler** 的最小形式——
在函数入口插入计时代码。

原书要点：
- 用 `std::chrono::steady_clock`（单调时钟，不会回拨）；
- 可用宏 `MEASURE_FUNCTION()` 包装 `ScopedTimer t{__func__}`；
- 插桩代码本身会影响被测量的程序，且可能阻止编译器优化。

## 构建与运行

```bash
cmake --build build --target ch03_scoped_timer_example
./build/chapter03_measurement/ch03_scoped_timer_example
```

## 输出解读

```
0 ms work_a
1 ms work_b
0 ms work_c
```

三个函数各自报告耗时（ms 粒度）。本例仅用于演示插桩方式；
毫秒粒度对微小函数过粗，需要更高精度的场景用 `benchmark_results` 里的
microbenchmark 工具。

## 局限（书中亦指出）

- 插桩改变被测量行为；
- 编译期优化可能把计时代码优化掉（用 `-O0` 或 `volatile` 防止，但会失真）；
- 不如采样 profiler 准确（见 `hotspot_profiling`）。
