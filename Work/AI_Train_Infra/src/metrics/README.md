# Training performance metrics lab

这个目录提供一个刻意很小、可复现的单卡训练 workload，用于建立指标口径和性能证据闭环。`eager` 是 baseline，`compiled` 是完全相同数值语义的 `torch.compile` 候选方案；候选方案可能更慢，结论只能来自当前机器的输出。

## 文件

- `metrics.py`：不依赖 PyTorch 的延迟、FLOP、MFU/HFU、显存、timeline 和 scaling 公式。
- `baseline.py` / `optimized.py`：eager baseline 与 compiled candidate。
- `benchmark.py`：warmup、同步计时、p50/p90/p99、throughput、峰值显存、JSON 和环境元数据。
- `correctness.py`：比较 forward、loss 和一次更新后的参数。
- `profile.py`：PyTorch Profiler trace 或供 Nsight 使用的 NVTX workload。
- `tests/`：纯 Python correctness tests。

从仓库根目录运行，推荐明确指定工程的 Python：

```bash
PYTHONPATH=Work/src /home/guhaoran/miniconda3/envs/flashrt/bin/python \
  -m unittest discover -s Work/src/metrics/tests -v

PYTHONPATH=Work/src /home/guhaoran/miniconda3/envs/flashrt/bin/python \
  -m metrics.correctness --device cuda --candidate compiled

PYTHONPATH=Work/src /home/guhaoran/miniconda3/envs/flashrt/bin/python \
  -m metrics.benchmark --device cuda --variant both \
  --warmup 10 --iterations 100 --output /tmp/metrics_benchmark.json
```

`--peak-tflops` 没有默认值。只有查证当前设备、当前 dtype、dense/sparse Tensor Core 模式与时钟对应的峰值后才可传入；否则 JSON 中 MFU/HFU 必须为 `null`。例如下面的 `VERIFIED_VALUE` 是占位符，不能原样运行：

```bash
python -m metrics.benchmark --device cuda --dtype bfloat16 \
  --peak-tflops VERIFIED_VALUE
```

## Profiling

PyTorch Profiler：

```bash
PYTHONPATH=Work/src python -m metrics.profile --backend torch --device cuda \
  --steps 10 --trace /tmp/metrics_trace.json
```

Nsight Systems（命令参数以本机 `nsys --help` 为准）：

```bash
nsys profile --trace=cuda,nvtx,osrt,cublas \
  -o /tmp/metrics_nsys_UNIQUE_RUN \
  python -m metrics.profile --backend nvtx --device cuda --steps 20
```

在 UI 中筛选 `metrics_measured_region`。应观察：CPU launch 间隔、kernel 数量/时长、前后 step 的空洞、GEMM 与 elementwise kernel，以及是否存在隐式同步。需要严格 capture range 时，优先使用 `../profiling/scripts/run_nsys.sh` 的独立 run 目录并按本机 Nsight 版本配置。

Nsight Compute 用短 workload 并按 NVTX 过滤，避免采集所有迭代：

```bash
ncu --target-processes all --nvtx --nvtx-include 'train_step/' \
  --set basic --kernel-name 'regex:HOTSPOT_FROM_NSYS' --launch-count 1 \
  -o /tmp/metrics_ncu_UNIQUE_RUN \
  python -m metrics.profile --backend nvtx --device cuda --warmup 5 --steps 1
```

先检查 basic 的 launch/occupancy/SpeedOfLight；有明确问题才升级 detailed/source counters/full。重点看实际存在的 Tensor Core 指令/管线利用率、DRAM/L2 吞吐、achieved occupancy、registers/thread、shared memory/block、warp stall reasons、grid size 和访问合并情况。若出现 `ERR_NVGPUCTRPERM`，停止并由管理员按目标机器安全策略开放计数器；不能把未采集指标写成 0。这个 MLP 只是测量载体，不应把单个 kernel 的 NCU 指标直接等同于整个 step 的 MFU。

## Benchmark 纪律

1. 先跑 correctness；不正确的加速没有意义。
2. 固定 shape、dtype、seed、软件版本和功耗/时钟策略。
3. compilation/autotune 在 warmup，测量区间每一步前后同步。
4. 报告原始样本、p50/p90/p99、mean、throughput 与 peak allocated/reserved。
5. 用 Nsight Systems 判断 launch/空洞/同步，再选热点用 Nsight Compute；不要一开始全量 NCU。
6. 优化后以同一口径重跑。保留变慢或无显著差异的结果，不挑选“好看”的一次。
