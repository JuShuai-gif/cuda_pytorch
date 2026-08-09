# 性能分析实验室

程序有意制造瓶颈或错误；08/09 只用于诊断实验。

## 构建

```bash
cd /home/ghr/code/cuda_pytorch/Performance_Tuning/profiling/src
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

没有 CUDA 时跳过 CUDA；没有 OpenMP 时 14 使用串行回退。

| 文件 | 性能问题 | 推荐工具 |
|---|---|---|
| 01_cpu_hotspot | CPU 热点 | perf / FlameGraph / VTune |
| 02_cache_miss | Cache Miss | perf / Cachegrind |
| 03_branch_miss | 分支预测 | perf stat |
| 04_memory_bandwidth | 内存带宽 | LIKWID / PCM |
| 05_false_sharing | False Sharing | perf / VTune |
| 06_lock_contention | 锁竞争 | VTune / bpftrace |
| 07_allocation_hotspot | 临时分配 | heaptrack |
| 08_memory_leak | 泄漏 | Valgrind / ASan |
| 09_use_after_free | UAF | Memcheck / ASan |
| 10_syscall_overhead | 系统调用 | strace |
| 11_io_bottleneck | IO/flush | strace / iostat |
| 12_context_switch | 切换 | perf / ftrace |
| 13_simd | 向量化 | Advisor / perf |
| 14_openmp_scaling | OpenMP 扩展 | VTune / LIKWID |
| cuda_02_memory_bound | GPU 带宽 | ncu |
| cuda_03_compute_bound | GPU 算力 | ncu |
| cuda_04/05_access | 合并访存 | ncu |
| cuda_06/07/08 | 分歧/Bank/占用率 | ncu |
| cuda_09/10/11 | Launch/传输/同步 | nsys |
| cuda_12_nvtx_pipeline | Pipeline | nsys |

## 实验约定

先 warm-up，再重复采样；报告 P50/P90/P95/P99 与抖动。固定 shape、batch、sequence length、线程数、频率策略、编译器与优化级别。CUDA 用 Event 或同步后的 wall-clock；冷热 cache 分开测。

## 目标机待验证实验

| 文件 | 目标环境 | 用途 |
|---|---|---|
| `33_numa_local_remote` | 多NUMA + libnuma | local/remote memory A/B |
| `34_v4l2_capture_benchmark` | Linux V4L2 camera | FPS、interval、sequence gap |
| `python/10_cuda_inference_target_lab.py` | CUDA版PyTorch | Event、Profiler、NVTX、memory |
| `python/11_media_pipeline_target_lab.py` | OpenCV/GStreamer | NVDEC/MPP/RGA可注入pipeline |
| `integrations/ros2_vla_profiling` | ROS2 | publish→callback→action tracing |
| `scripts/monitor_jetson_long_run.sh` | Jetson | tegrastats长稳态CSV |
| `scripts/collect_ncu_target.sh` | ncu计数器权限 | kernel过滤与report |
| `scripts/run_optional_tools_target.sh` | 可选工具 | Valgrind/heaptrack/fio/cyclictest |

## FFmpeg目标机实验

| 文件 | 功能 |
|---|---|
| `35_ffmpeg_decode_benchmark` | LibAV demux/decode/swscale逐帧统计 |
| `python/12_ffmpeg_target_lab.py` | FFmpeg软件/NVDEC/custom硬件A/B与JSON |
| `scripts/run_ffmpeg_target.sh` | 能力检测并运行CLI和C++实验 |

FFmpeg开发库不存在时只跳过C++目标，CLI实验仍然可用。
