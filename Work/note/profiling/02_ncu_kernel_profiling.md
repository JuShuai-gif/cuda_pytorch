# 02｜Nsight Compute：分析单个 kernel 为什么慢

## 本模块解决的问题

nsys 告诉你"哪个 kernel 最慢、哪里有气泡"，但没告诉你"为什么慢"。Nsight Compute（ncu）回答的是后者，它读取 GPU 硬件性能计数器，把单个 kernel 的执行细节摊开：

```text
它是 memory-bound 还是 compute-bound？
occupancy 到没到位？
register / shared memory 是否限制了并行？
warp 都 stall 在什么地方？
访存是否 coalesced？
```

配套代码：`src/profiling/scripts/run_ncu.sh`（封装采集命令）、`src/profiling/profile_target.py`（可复现的 kernel）。

---

## 1. ncu 回答什么问题

| 问题 | 看哪个 section / metric |
|---|---|
| 是 memory-bound 还是 compute-bound | `SpeedOfLight`：DRAM throughput vs SM/Tensor Core utilization |
| occupancy 是否够 | `Occupancy`：achieved vs theoretical |
| 被什么限制 | register/thread、shared memory/block、block limit |
| warp 为什么 stall | `Warp State Statistics`（stall 原因分布） |
| 访存是否合并 | `Memory Workload Analysis`（gld/gst efficiency、L2 sectors） |
| Tensor Core 有没有用起来 | `SpeedOfLight` 的 Tensor Core 部分 / `Instruction Statistics` |

**核心判断**：打开 `SpeedOfLight`，看"SM 利用率"和"DRAM 吞吐"谁先到顶：

```text
DRAM throughput 接近峰值 + SM 利用率低  → memory-bound
SM/Tensor Core 高 + DRAM 低             → compute-bound
两者都低                                → latency-bound / occupancy 不足 / 访存模式差
```

这比任何"感觉慢"的判断都硬。

---

## 2. 采集命令

```bash
ncu \
  --set basic \
  --launch-count 1 --launch-skip 1 \
  --export /tmp/report \
  --force-overwrite \
  python profile_target.py
```

- `--set basic`：先看基础指标；确有问题再升级 `detailed` / `source` / `full`（先 `ncu --list-sets` 确认本机有哪些 set）。
- `--launch-count 1 --launch-skip 1`：跳过第一个（常含初始化），只采一个代表性 kernel，避免输出爆炸。
- 指定 kernel：`--kernel-name regex:xxx` 可只采目标 kernel。

本仓库封装脚本：

```bash
export PROFILING_PYTHON=/home/guhaoran/miniconda3/envs/flashrt/bin/python
bash Work/src/profiling/scripts/run_ncu.sh \
  --hidden 1024 --layers 4 --batch 1 --steps 3 --set basic --output-root /tmp/ncu
```

查看：`ncu-ui /tmp/ncu/<run>/reports/report.ncu-rep`。

---

## 3. 本机权限限制（重要，Not Validated）

本机（Jetson/Thor）驱动设置了 `RmProfilingAdminOnly=1`，非 root 运行 `ncu` 会直接失败：

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

**处理原则**（与仓库其它模块一致）：

1. **不伪造 ncu 数据**。拿不到硬件计数器时，ncu 相关结论一律标记 `Not Validated`，不补 0、不照搬其它架构（如 B200/A100）的阈值。
2. 授权方式：管理员把 `RmProfilingAdminOnly` 设为 0，或用 `sudo ncu ...` 采集。
3. 在授权之前，用 **nsys（不需要 root）** 完成时间维度的全部定位，ncu 只补"kernel 为什么慢"这最后一步。

`nsys` 已经本机验证可用；`ncu` 的**采集脚本已验证可运行**（能连上进程、正确报权限错误），只是计数器读数未验证。

---

## 4. 关键 metric 的读法

### SpeedOfLight（首选第一屏）

```text
SM [%]            SM 峰值吞吐的利用率（compute 强度）
DRAM [%]          显存带宽利用率（memory 强度）
```

二者之一接近 100% = 它就是瓶颈。都低 = 看 occupancy 和 stall。

### Occupancy

```text
achieved occupancy   实际 active warps / SM
theoretical          上限（受 register/shared/block 限制）
```

achieved 远低于 theoretical → 有其它因素（依赖、同步、小 grid）没吃满。同时看 `registers/thread`、`shared memory/block`、`block limit` 判断是哪个资源卡住了。

### Warp State Statistics

warp 平均 stall 在哪个原因上：

```text
stall long scoreboard   等 global memory 回来（memory-bound 的实锤）
stall wait              等 ALU 结果 / 依赖
stall barrier           等 __syncthreads（block 内同步）
stall not selected      有可运行 warp，但 SM 没选它（通常是访存不足或 grid 太小）
```

### Memory Workload Analysis

```text
gld/gst efficiency      全局访存合并程度（<100% 说明有 stride/非合并访问）
L2 sectors              每次访存实际触发的 L2 sector 数
```

### Instruction Statistics

```text
Tensor Core 指令占比    是否真的在用 Tensor Core 做 GEMM（vs 普通 FMA）
```

---

## 5. 典型诊断决策树

```text
DRAM 接近峰值？
  ├─ 是 → memory-bound → 减少搬运（fusion）、低精度（fp16/bf16 同带宽搬更多）
  └─ 否 → SM/Tensor Core 高？
         ├─ 是 → compute-bound → 换低精度、更好 GEMM 选型
         └─ 否 → occupancy 低？
                ├─ 是 → register/shared 压力大 → 调 BLOCK_SIZE / num_warps / 拆 kernel
                └─ 否 → stall 在 scoreboard？→ 访存延迟未隐藏 → 提 occupancy / 改访存
```

---

## 6. 为什么 ncu 的整段 wall time 不能当延迟

ncu 会**重放（replay）** kernel 多次来读计数器，还会锁定时钟，所以它测出的整段 wall time 会被显著拉长，**不能拿来当推理延迟或吞吐**。ncu 的输出只有"计数器比率/利用率"可信，时间一律回 nsys / benchmark 拿。

---

## 7. 与 Triton / CUDA kernel 调优的衔接

后续 Stage 4（Triton）、Stage 5（CUDA kernel）里，每个算子的性能分析都会走这条路：

```text
写 kernel → 正确性 → benchmark（latency/吞吐）→ nsys 找瓶颈 → ncu 看计数器
       → 调整 BLOCK_SIZE / num_warps / num_stages / 访存布局 → 重测
```

ncu 里的 `occupancy`、`DRAM`、`stall`、`gld_efficiency` 就是判断"为什么这个 BLOCK_SIZE 更快"的直接证据，而不是停留在"运行时间变短了"。

---

## 8. 本模块闭环小结

```text
问题：nsys 找到的那个 kernel 为什么慢
      ↓
工具：ncu（硬件计数器）+ SpeedOfLight 首屏
      ↓
判断：memory-bound / compute-bound / occupancy / stall / 访存
      ↓
约束：本机 ncu 需授权（ERR_NVGPUCTRPERM）→ 未授权则 Not Validated
      ↓
输出：可行动的优化项（fusion / 低精度 / 调 block 配置 / 改访存）
```

---

## 9. Stage 1 收尾：一次完整诊断流程回顾

把 Stage 1 的三篇 inference + 两篇 kernel + 两篇 profiling 串起来，就是处理"模型推理慢"的最小闭环：

```text
1. benchmark_latency.py   → wall vs event，判断 host 开销 vs device 开销
2. benchmark_throughput.py → batch sweep，找吞吐饱和点
3. run_nsys.sh            → timeline 找最慢 kernel / 最大 gap / 同步气泡
4. run_ncu.sh             → 分析该 kernel 是 memory/compute/occupancy 哪种瓶颈
5. 只改一处 → 回到 1 重测 → 用 before/after 数据证明
```

这套闭环就是后面所有优化（Triton kernel、fusion、TensorRT、量化、LLM/VLA 推理）反复复用的地基。
