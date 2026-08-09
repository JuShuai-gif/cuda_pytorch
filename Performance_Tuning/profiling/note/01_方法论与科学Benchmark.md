# 性能诊断方法论与科学 Benchmark

## 闭环

```text
Measure → Locate → Explain → Optimize → Verify
```

先把“慢”拆成CPU、GPU、memory、IO、lock、scheduler、network、operator、kernel或pipeline。轻量工具定资源域，采样/Tracing定位，硬件计数器解释，Bad/Good A/B验证。

## 统一实验范式

1. Bad Case和Good Case使用相同输入与有效工作量。
2. checksum、相对误差或`torch.testing.assert_close`先验证正确性。
3. Release构建，记录编译器、`-O0/-O2/-O3`、ISA和依赖版本。
4. 固定线程、affinity、shape、batch、sequence、dtype、功耗模式。
5. CPU/GPU warm-up；冷热cache分别测。
6. 重复运行并保存原始样本。
7. 报告Mean、Median/P50、P90、P95、P99、Min、Max、StdDev、吞吐。
8. CUDA用Event，或wall-clock前后同步。
9. 每次只改变一个机制。
10. 优化后重新运行同一Benchmark和Profiler，检查瓶颈是否转移。

## 防止编译器消除

使用运行时输入、checksum、noinline、volatile sink或类似DoNotOptimize的机制。不要让一个版本因结果未使用而被消除。

## 平均值不是实时保证

50Hz控制周期是20ms。Mean=10ms但P99=35ms仍会Deadline Miss。VLA必须报告stage分布、E2E分布、FPS、Control Hz、frame drop和deadline miss。

## 构建与运行

```bash
cd /home/ghr/code/cuda_pytorch/Performance_Tuning/profiling/src
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./scripts/run_cpu_pathologies.sh
```

完整实验证据见[环境与验证结果](00_环境与验证结果.md)，症状到工具选择见[诊断手册](10_诊断决策树与指标字典.md)。

## 从“慢”到可验证假设

把模糊描述变成可测量问题：模型慢要写明固定shape与batch下的P50/P99；GPU没跑满要测nsys空洞和launch gap；内存高要区分active heap、RSS和reserved；控制不稳要统计20ms deadline的miss与Max。

写出假设和预测。“CPU预处理喂不饱GPU”应预测GPU空洞前存在长preprocess range；并行后空洞减少且吞吐提高。指标不按预测变化就应推翻假设。

## Baseline记录清单

```text
date, commit, hostname
CPU/GPU, driver, CUDA, compiler, PyTorch
build_type, flags
input_size, shape, batch, sequence, dtype
threads, affinity, NUMA policy
warmup, iterations
power_mode, temperature, clock
mean, p50, p90, p95, p99, min, max, stddev
correctness checksum/tolerance
```

没有环境记录的5%提升通常不可复现。Jetson还必须记录功耗模式、风扇、温度和时钟。

## Warm-up与Cold Start

Warm-up覆盖CPU cache、动态库加载、页首次映射、CUDA context、module加载、allocator初始化、GPU频率爬升和框架compile。若关注开机首帧，Cold Start与Steady State必须作为独立Benchmark。

## 延迟分布

P99表示约99%样本不超过该值，不是最慢1%的平均值。100个样本的P99几乎由最后一个样本决定。实时服务应采集数千周期并保存原始时间序列，同时报告Mean、Median、P50/P90/P95/P99、Min/Max、StdDev、deadline miss、温度、频率、RSS和drop。

## 吞吐、排队与Little定律

提高batch可能改善GPU throughput，却增加排队和单请求latency。

```text
系统内平均请求数 L = 到达率 λ × 平均停留时间 W
```

producer持续快于consumer时，queue增长，stage throughput可能正常，但E2E latency恶化。应同时测service time、queue wait和E2E。

## Amdahl定律

```text
Speedup = 1 / ((1-p) + p/s)
```

只占E2E 5%的阶段即使无限加速，总收益也约5.3%。先用timeline确认占比，再投入kernel优化。

## Profiler扰动

- strace会显著放大wall time；
- ncu replay可能多次执行kernel；
- torch.profiler记录stack/shape/memory增加开销；
- 高频逐事件打印比聚合统计扰动大；
- O0会改变执行结构。

Profiler负责定位解释，最终收益由轻量Benchmark证明。

## Before/After验收

| 项目 | Before | After | 要求 |
|---|---:|---:|---|
| Correctness | checksum A | checksum A | 必须一致 |
| Mean |  |  | 辅助 |
| P99/Max |  |  | 实时关键 |
| Throughput |  |  | 明确模式 |
| 根因指标 |  |  | 应按假设变化 |
| CPU/GPU/Memory |  |  | 检查瓶颈转移 |
| Power/Temperature |  |  | 边缘端必须 |

## 练习

1. 对CPU hotspot分别构建Debug、Release并解释差异。
2. 连续运行30次，比较Mean、Median、P99和Max。
3. 给VLA mock增加逐帧CSV，定位P99帧最慢stage。
4. 在改代码前写出假设、预测指标和反证条件。
