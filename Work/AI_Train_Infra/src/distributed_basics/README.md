# torch.distributed / DDP overlap lab

这个目录把 DDP 从“能启动”变成一个可审计实验：单进程全局 batch 是数值 reference；`baseline` 使用大 bucket，刻意延迟 collective；`optimized` 是小 bucket、`gradient_as_bucket_view=True`、`static_graph=True` 的待验证候选。候选不保证更快，尤其在模型很小、网络延迟高或 bucket 过碎时。

## 文件

- `workload.py`：多层 residual MLP synthetic workload 和显式 FLOP 口径。
- `baseline.py` / `optimized.py`：两套 DDP bucket policy。
- `correctness.py`：验证一次 DDP 更新等价于单进程 global-batch 更新，并验证 rank 间参数一致。
- `benchmark.py`：收集每 rank 原始 step 样本，以每步最慢 rank 作为分布式关键路径。
- `profile.py`：每 rank、每 step 的 forward/backward/optimizer NVTX 范围。
- `analyze_nsys.py`：从 Nsight Systems SQLite 把 backward 内发射的 CUDA kernel 分为 compute/NCCL，做 interval union/intersection。
- `timeline.py` / `tests/`：exposed communication 的纯函数定义与测试。
- `scripts/`：不覆盖历史结果的 correctness、benchmark、PyTorch Profiler、Nsight Systems 入口。

## 1. 先跑 correctness

本机只有一张 GPU，因此用两个 CPU rank + Gloo 验证真实分布式求梯度语义：

```bash
cd /home/guhaoran/code/cuda_pytorch
export DISTRIBUTED_TORCHRUN=/home/guhaoran/miniconda3/envs/flashrt/bin/torchrun
export DISTRIBUTED_PYTHON=/home/guhaoran/miniconda3/envs/flashrt/bin/python

Work/src/distributed_basics/scripts/run_correctness.sh \
  --device cpu --backend gloo --nproc-per-node 2 --variant baseline
```

真实两卡 NCCL：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
Work/src/distributed_basics/scripts/run_correctness.sh \
  --device cuda --backend nccl --nproc-per-node 2 --variant optimized
```

不要在一个可见 GPU 上启动两个 NCCL rank 来伪造双卡实验。

## 2. Benchmark contract

```bash
CUDA_VISIBLE_DEVICES=0,1 \
Work/src/distributed_basics/scripts/run_benchmark.sh \
  --device cuda --backend nccl --nproc-per-node 2 --variant baseline -- \
  --warmup 10 --iterations 100 --hidden-size 1024 --layers 12

CUDA_VISIBLE_DEVICES=0,1 \
Work/src/distributed_basics/scripts/run_benchmark.sh \
  --device cuda --backend nccl --nproc-per-node 2 --variant optimized -- \
  --warmup 10 --iterations 100 --hidden-size 1024 --layers 12
```

固定 commit、shape、dtype、rank placement、电源/温度状态、bucket 配置和迭代数。JSON 保存每 rank raw latency，关键 step time 取同一步最慢 rank。脚本只给 ring all-reduce 的解析通信量；它不是实测 NCCL 算法、带宽或 exposed time。未提供已验证峰值时 MFU 为 `null`。

一次比较同时改变了 bucket、gradient view、static graph 时不能归因到某一个开关。正式调参应使用 `--bucket-cap-mb` 固定其他变量，逐项消融。

## 3. PyTorch Profiler

```bash
CUDA_VISIBLE_DEVICES=0,1 \
Work/src/distributed_basics/scripts/run_torch_profiler.sh \
  --nproc-per-node 2 --variant baseline -- --warmup 3 --steps 3
```

每个 rank 单独生成 trace。先看 `ddp_backward`、collective op、kernel launch 与 memory，再用 Nsight Systems 判断不同 CUDA stream 上的真实重叠。Profiler 的 record-shapes/memory 有开销，性能结论必须回到无 profiler benchmark。

## 4. Nsight Systems 与 exposed communication

```bash
CUDA_VISIBLE_DEVICES=0,1 \
Work/src/distributed_basics/scripts/run_nsys.sh \
  --nproc-per-node 2 --variant baseline -- --warmup 3 --steps 3
```

脚本生成 `.nsys-rep`、SQLite、stats 和 `overlap.json`。分析器做三件事：

1. 找到 `ddp_backward_rank_<rank>_step_<step>` CPU NVTX 范围；
2. 通过 CUDA runtime correlation ID 找到该范围发射的 GPU kernel；
3. 对 compute/NCCL interval 分别求 union，再求 intersection。

```text
NCCL total       = union(NCCL kernel intervals)
overlap          = intersection(union(compute), union(NCCL))
exposed NCCL     = NCCL total - overlap
```

`exposed NCCL` 仍不是自动等于 step penalty：重叠通信可能争用 SM、L2、DRAM，计算也可能不是关键路径。必须同时比较 no-communication/single-rank control 和端到端 step time。

## 5. Nsight Compute

NCU 不用于判断跨 stream overlap。先由 Systems 选择一个实际 backward compute hotspot，再复用 `../profiling/scripts/run_ncu.sh` 从 `basic` 开始。检查 launch、occupancy、Tensor Core、DRAM/L2 和 stall；Thor/sm_110 不套用 B200 阈值。若 `ERR_NVGPUCTRPERM`，保留错误并把 counter 写为 unavailable，不写 0。

## 当前验证边界

- 已验证：双进程 Gloo correctness、CPU 分布式 benchmark、单 GPU/world-size=1 NCCL 控制路径。
- 未验证：两卡/八卡 NCCL overlap、NVLink/PCIe/IB/RoCE、跨机 scaling。
- 当前单卡 Nsight control 应看不到 NCCL 数据传输；这只能证明工具链工作，不能证明 DDP 通信性能。

完整原理、通信量和故障排查见 `../../note/03_torch_distributed_ddp.md`。
