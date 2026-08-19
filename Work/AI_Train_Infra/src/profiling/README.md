# Stage 1 Profiling Lab

这个目录不是一组“点一下 profiler”的命令，而是一个最小性能证据闭环：固定输入与环境，先测 baseline，再从时间线确认瓶颈，实施一个变化，重新测 optimized，最后用 end-to-end 指标证明收益。

## 文件

- `workloads.py`：五类可辨认的合成瓶颈与等价 baseline/optimized。
- `tests/test_correctness.py`：CPU/CUDA 数值等价测试。
- `benchmark.py`：隔离 latency A/B；保存原始样本、mean/std、p50/p90/p95/p99，不覆盖旧结果。
- `profile_target.py`：无 profiler 内部开销的连续异步 workload，带 NVTX。
- `profile_torch.py`：有 wait/warmup/active schedule 的 PyTorch Profiler trace。
- `inspect_ncu_report.py`：枚举真实报告指标；缺失项写 `unavailable/null`，绝不写成 0。
- `scripts/run_nsys.sh`：系统级 CPU/GPU timeline。
- `scripts/run_ncu.sh`：对已由 timeline 证明的热点做 kernel counter 分析。

默认规模刻意较小。它们用于识别 profiler 特征，不代表真实训练性能，也不附带预先编造的 benchmark 数字。

## 环境与快速运行

需要 Python 3、PyTorch；CUDA/Nsight 实验还需要 CUDA build 的 PyTorch、NVIDIA GPU、`nsys` 和 `ncu`。可用 `PROFILING_PYTHON` 指定 Python，不修改系统环境：

```bash
cd /home/guhaoran/code/cuda_pytorch/Work/src/profiling
export PROFILING_PYTHON=/path/to/python-with-torch

$PROFILING_PYTHON -m unittest discover -s tests -v
scripts/run_benchmark.sh --device cpu --case all --iterations 10
scripts/run_benchmark.sh --device cuda --case all --iterations 30
scripts/run_torch_profiler.sh --device cuda --case launch --variant baseline
```

每次运行自动创建带 UTC 时间、PID 和冲突序号的新目录：

```text
artifacts/
├── benchmarks/<unique-run>/results.json
├── torch_profiler/<unique-run>/{metadata.json,key_averages.txt,traces/}
├── nsys/<unique-run>/{command.txt,reports/,analysis/}
└── ncu/<unique-run>/{command.txt,reports/,analysis/}
```

`artifacts/.gitignore` 保留目录但忽略报告。运行脚本不会复用或覆盖旧 run。

## 五种诊断签名

| case | baseline | optimized | timeline 应观察到 |
|---|---|---|---|
| `launch` | 多个依赖的小 elementwise kernel | 一个 kernel | kernel 数和 CUDA launch API 显著减少 |
| `memory` | 两次全 tensor pass 和中间量 | 单次 affine pass | kernel/分配减少，读写字节应下降 |
| `compute` | 重复两次 GEMM | 一次 GEMM 加 scale | GEMM 时间/次数下降；这是教学用 CSE |
| `sync` | 每个 kernel 后 device sync | 最后统一等待 | baseline 的同步 API 和 host/device serialization 消失 |
| `cpu` | NVTX 范围内故意 sleep | 去掉 sleep | baseline GPU submission 间出现 CPU gap |

这些 optimized 只是构造“变化应该怎样出现在证据中”，不是通用优化处方。

## Benchmark 的边界

```bash
scripts/run_benchmark.sh \
  --device cuda --case launch --variant both \
  --warmup 5 --iterations 50 --numel 262144 --repeats 16
```

`benchmark.py` 为每个 invocation 同步，以获得 isolated wall/GPU-event latency。因此它会主动破坏真实的异步提交、跨 step overlap 和 sync timeline。延迟统计用它；真实连续时间线必须用 `profile_target.py`、PyTorch Profiler 或 Nsight Systems。报告中的 FLOPs 是源代码算术估计，不是硬件 counter，也不是 MFU。

## PyTorch Profiler

```bash
scripts/run_torch_profiler.sh \
  --device cuda --case memory --variant baseline \
  --wait 1 --warmup 1 --active 3 --repeat 1 \
  --record-shapes --profile-memory
```

先看 `key_averages.txt` 的 CPU/CUDA self time，再用 Perfetto 或 TensorBoard 打开 `traces/*.pt.trace.json`。检查 `step/N` 下的 op、kernel、memcpy、CPU gap。`record_shapes`、memory 和 stack 会增加开销，只有问题需要时才打开；性能结论回到无 profiler benchmark。

## Nsight Systems：必须先定位热点

```bash
scripts/run_nsys.sh --case launch --variant baseline --device cuda
scripts/run_nsys.sh --case launch --variant optimized --device cuda
```

比较 raw `.nsys-rep` 与 `analysis/stats.txt`：step 边界是否一致、GPU 是否有 bubble、CPU 是否跟不上、是否存在 `cudaDeviceSynchronize`、kernel 是否过短且数量过多、memcpy 与 compute 是否重叠。对 CUDA Graph node 需要自行在同等 workload 下增加 `--cuda-graph-trace=node`，并保持 graph 模式在 A/B 中一致。

## Nsight Compute：basic 再逐级加深

1. 从 Nsight Systems 复制一个真实热点的 demangled kernel 名与代表性 launch。
2. 先收 basic：

```bash
scripts/run_ncu.sh --kernel 'KERNEL_REGEX' --stage basic --tag baseline -- \
  "$PROFILING_PYTHON" profile_target.py --device cuda --case compute --variant baseline
```

3. 只有 basic 提出具体问题后，才收 detailed；需要 source/SASS 归因时用 `source`：

```bash
scripts/run_ncu.sh --kernel 'KERNEL_REGEX' --stage detailed \
  --hotspot-confirmed --tag baseline -- APPLICATION ARGS...

scripts/run_ncu.sh --kernel 'KERNEL_REGEX' --stage source \
  --hotspot-confirmed --tag baseline -- APPLICATION_AROUND_A_LINEINFO_KERNEL...
```

本机 NCU 2025.3 的 `--list-sets` 没有 `source` set；脚本把用户侧 `--stage source` 映射为 `--set detailed --section SourceCounters`。source attribution 仍要求目标以 `-lineinfo` 构建。`full` replay 很贵，只有具体问题不能由 detailed/source 回答时才使用。CUDA Graph 使用 `--graph-node`；NVTX push/pop range 使用 `--nvtx-range profile_steady_state`，脚本会追加 NCU 要求的 `/`。示例 range 名刻意不用 `/`，因为它在 NCU filter 中是 stack/quantifier 语法；自定义名称含特殊字符时须按当前 NCU 文档转义。

解析报告：

```bash
export PYTHONPATH=/installed/nsight-compute/extras/python:$PYTHONPATH
$PROFILING_PYTHON inspect_ncu_report.py baseline.ncu-rep \
  --compare optimized.ncu-rep --dump-all --output comparison.json
```

脚本先枚举 `action.metric_names()`。Thor/sm_110 上不存在的 B200/H100 counter 是 `unavailable`，不是 0；不能拿别的 counter 冒充。当前板卡资源和阈值必须现场查询，不能把另一型号的峰值、SM 数、tile 或 occupancy 经验值套过来。

若 NCU 报 `ERR_NVGPUCTRPERM`，当前用户没有访问硬件计数器的权限。保留错误现场，由机器管理员按该主机的安全策略授权后重采；不要在报告中补 0。脚本也会在 NCU 返回成功但没有生成非空报告时失败，防止 kernel/NVTX filter 未匹配却被误认为采集成功。

## 一次合格 A/B 的交付物

- 相同 commit、shape、dtype、batch、warmup/steps、graph/compile 模式及电源/温度条件。
- correctness 通过；至少多次独立 run，保存 raw 样本与 p50/p90/p99。
- baseline benchmark + timeline + hotspot/counter 证据。
- 一次只改一个主要变量；optimized 重跑完全相同的采集。
- 比较 step time、throughput、GPU active/bubble、kernel count/duration、峰值显存，以及适用时的 MFU、NCCL exposed time、scaling efficiency。
- 没测到的字段明确写“未测/不可用”，不以 0 或估计冒充实测。

完整判读方法见 `../../note/18_profiling_methodology.md`。
