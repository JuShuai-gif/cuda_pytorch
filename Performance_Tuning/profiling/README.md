# Performance Engineering Lab

本仓库从“程序慢”出发，建立 Problem → Symptom → Metrics → Tool → Root Cause → Optimization → Verification 闭环。

## 入口

- 实验索引与学习路线：[note/11_实验索引与学习路线.md](note/11_实验索引与学习路线.md)
- 方法论与科学Benchmark：[note/01_方法论与科学Benchmark.md](note/01_方法论与科学Benchmark.md)
- 症状诊断：[诊断手册](note/10_诊断决策树与指标字典.md)
- 指标组合：[指标组合推理](note/10_诊断决策树与指标字典.md)
- 代码与构建：[src/README.md](src/README.md)

## 核心实验索引

| 性能问题 | Demo | 第一工具 | 深入工具 | 关键指标 |
|---|---|---|---|---|
| CPU Hotspot | `20_cpu_hotspot_bad_good` | perf record | FlameGraph/VTune | samples/cycles |
| Cache Locality | `21_cache_locality_bad_good` | perf stat | Cachegrind/VTune | miss rate/IPC |
| Branch Miss | `03_branch_miss` | perf stat | annotate/VTune | branch miss rate |
| Bandwidth | `23_stream_bad_good` | GB/s/perf | LIKWID/PCM | BW/AI |
| Allocation | `25_allocation_bad_good` | allocation counter | heaptrack/Massif | count/bytes/peak |
| Lock | `26_lock_contention_bad_good` | perf/strace | VTune/bpftrace | futex/wait/switch |
| False Sharing | `05_false_sharing` | runtime/perf | VTune | coherence/runtime |
| CUDA Kernel | `cuda_13_kernel_hotspot_bad_good` | nsys | ncu | duration/SM/DRAM |
| GPU Bound | `cuda_02/03_*bound` | nsys | ncu/Roofline | AI/SM/DRAM |
| CUDA Pipeline | `cuda_14_pipeline_overlap_bad_good` | nsys | stream/copy分析 | overlap/E2E |
| PyTorch | `07_operator_hotspot_bad_good.py` | torch.profiler | nsys/ncu | Self/Total CPU/CUDA |
| VLA | `09_vla_e2e_bad_good.py` | stage timer/NVTX | nsys/ncu | P50/P90/P99/FPS |
| Realtime | `30_realtime_jitter` | histogram | cyclictest/ftrace | wakeup/P99/miss |

## 构建

```bash
cd /home/ghr/code/cuda_pytorch/Performance_Tuning/profiling/src
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

危险的 leak、UAF、buffer overflow 不进入自动运行脚本。
