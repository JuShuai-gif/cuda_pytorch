# GPU execution and memory lab

本目录用小而可控的 PyTorch workload，把 launch-bound、memory-bound、同步阻塞、CPU 供给不足和 GEMM 探针放到同一套“正确性 → benchmark → trace”协议中。`optimized` 是待验证候选，不是预先宣称的赢家。

## 运行环境与入口

只依赖 PyTorch。建议从仓库根目录运行：

```bash
export PYTHONPATH="$PWD/Work/src"
python -m unittest discover -s Work/src/gpu_basics/tests -v
python -m gpu_basics.benchmark --device auto --output /tmp/gpu_basics.json
python -m gpu_basics.memory_demo --device auto --output /tmp/gpu_memory.json
python -m gpu_basics.async_timing --device cuda --output /tmp/gpu_async.json
```

默认 shape 最大约为 512 方阵或 262,144 个元素，不会故意吃满显存。正式 GPU 实验应按机器逐步放大，并固定 clocks、dtype、shape、warmup、迭代数与功耗状态。CPU fallback 只能验证代码/正确性/JSON，不能验证 CUDA 异步、Tensor Core 或 GPU 瓶颈。

提供真实峰值后才计算 Roofline 上限；不要混用不同 dtype 的规格：

```bash
python -m gpu_basics.benchmark \
  --device cuda --workload gemm --dtype float32 \
  --matrix-size 2048 --warmup 10 --iterations 30 --repeats 7 \
  --peak-tflops <该精度实测或厂商峰值> \
  --peak-bandwidth-gbs <该卡显存带宽> \
  --output /tmp/gemm.json
```

JSON 同时保存环境元数据、原始重复样本、mean/std/p50/p90/p99、同步 wall time、CUDA Event time、allocator peak 和分析成本下界。GPU active、bubble、kernel 数等 trace-only 指标明确留空，不用 wall time 猜测。每组前后同步适合测独立调用 latency，但切断了自然异步 pipeline；连续训练 step 的 overlap/bubble 必须看 trace。

## workload 与预期 trace

| workload | baseline | optimized candidate | 首要观察 |
|---|---|---|---|
| `launch` | 多次小 pointwise launch | 代数折叠为一次 | kernel 数、kernel 间隙、CPU launch API |
| `memory` | 多次冗余 clone | 保留一次 clone | DRAM/L2 吞吐、allocator、总 bytes |
| `sync` | 每轮 `.item()` | device 上累积后一次 `.item()` | `cudaStreamSynchronize`、GPU 空洞、CPU 阻塞 |
| `cpu_gap` | GPU 提交前 sleep | 数据已准备好 | timeline 中 GPU idle gap |
| `gemm` | `einsum` | `mm` | 是否落到相同 GEMM、Tensor Core、shape 饱和度 |

`sync` 中的同步是被研究的病理；benchmark 自身只在样本边界同步。Profiler 入口不会逐 step 同步，否则会主动改变所研究的 timeline。

## PyTorch Profiler / Nsight

当前环境若默认 `python3` 没有 PyTorch，可显式选择解释器：

```bash
export GPU_BASICS_PYTHON=/home/guhaoran/miniconda3/envs/flashrt/bin/python
```

```bash
$GPU_BASICS_PYTHON -m gpu_basics.profile_workloads \
  --device cuda --workload sync --variant both --profiler torch \
  --steps 8 --record-shapes --profile-memory \
  --trace-dir /tmp/gpu_basics_traces

bash Work/src/gpu_basics/profile_nsys.sh /tmp/launch_nsys \
  --workload launch --variant both --steps 20

bash Work/src/gpu_basics/profile_ncu.sh /tmp/gemm_basic gemm baseline basic \
  --matrix-size 2048 --steps 5

# 只有 basic 已指出内存/调度问题时才显式升级，并使用新报告名
bash Work/src/gpu_basics/profile_ncu.sh /tmp/gemm_detailed gemm baseline detailed \
  --matrix-size 2048 --steps 5
```

先用 Nsight Systems 定位 CPU gap、同步、kernel 名称和临界路径，再用 Nsight Compute 采单个代表 kernel。脚本的 NVTX 名使用下划线而不是 `/`，因为 `/` 在 NCU filter 中是 range-stack 语法。NCU 默认 `--set basic`，再按问题显式升级 `detailed`、`source` 或 `full`；重采必须用新 report 名，脚本会拒绝覆盖。这里 `source` 是脚本的逻辑阶段：已验证的 NCU 2025.3 没有同名 set，因此映射为 `--set detailed --section SourceCounters`；其他机器应先运行 `ncu --list-sets`。若遇到 `ERR_NVGPUCTRPERM`，应由管理员按目标机器策略授权后重采，禁止补 0。NCU replay 显著扰动时间，不能拿其整程序 wall time 当训练吞吐。正式分析应核对报告中实际存在的 Tensor Core、DRAM/L2、occupancy、register/thread、shared memory/block、warp stall 和访存指标；不同 NCU/GPU（尤其 Thor sm_110）缺失的 metric 要记作 unavailable，不能当 0，也不能照搬 B200 阈值。

## 文件职责

- `baseline.py` / `optimized.py`：成对 workload。
- `workloads.py`：固定输入、解析成本下界和递归正确性比较。
- `common.py`：Event/同步 wall 计时、环境元数据和拒绝覆盖的 JSON 证据。
- `benchmark.py`：最小性能证据闭环。
- `async_timing.py`：专门证明 enqueue time 不等于完成时间。
- `memory_demo.py`：参数、梯度、AdamW 状态、saved tensor 与 allocated/reserved。
- `profile_workloads.py`：PyTorch Profiler 或外部 NVTX 入口。
- `tests/`：CPU/CUDA baseline–optimized correctness 与 CUDA Event 测试。

## 解释结果时的边界

- `nvidia-smi` utilization 是采样窗口内“是否忙”，不是 FLOPs/峰值，更不是 MFU。
- Event 测当前 stream 上两个事件之间的设备时间；跨 stream 必须建立 event dependency 后再计时。
- `memory_allocated` 是活跃 tensor，`memory_reserved` 是 PyTorch caching allocator 持有的块，二者都不等于进程在 NVML 中的全部显存。
- 分析 FLOPs/bytes 是模型，不是 counter。缓存、融合、编译、不同算法会改变实际流量。
- 小 workload 易受 clocks、首次初始化和 Python 噪声影响。比较必须保留原始重复样本和 profiler 证据。
