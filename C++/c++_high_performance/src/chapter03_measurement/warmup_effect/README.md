# warmup_effect

Benchmark 热身（warmup）效应。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 96-97 页（性能测试最佳实践）：测量要预热——首次运行时缓存未加载、
频率调节、惰性分配、缺页等导致首轮偏慢。`chp::benchmark` 内置预热轮数。

本实验手写"无预热逐轮计时"循环，展示首轮与稳定轮的差异。

## 构建与运行

```bash
cmake --build build --target ch03_warmup_benchmark
./build/chapter03_measurement/ch03_warmup_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，4M int 求和 ×100 次/轮）

```
round  0: 236.00 ms    <- 冷启动（缺页、缓存未就绪）
round  1: 229.07 ms
...
round  9: 227.93 ms
with warmup (mean): 2.24 ms/iter
```

首轮比稳定轮慢约 3.5%，且后续轮次缓慢下降（缓存/频率稳定过程）。
若只用首轮数据，会高估真实性能。

## 结论

- 丢弃预热轮、取多轮中位数/最小值更接近稳态真实性能；
- 本项目的 `chp::benchmark` 已内置 warmup 参数；
- 大内存缓冲（>cache）预热更关键（缺页成本可达数毫秒）。
