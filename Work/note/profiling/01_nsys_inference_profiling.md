# 01｜Nsight Systems：从 timeline 找到推理慢在哪一段

## 本模块解决的问题

`ncu` 回答"单个 kernel 为什么慢"，但**推理首先是个系统问题**：CPU 有没有喂饱 GPU？kernel 之间有没有空隙？H2D 有没有和计算重叠？有没有偷偷 `synchronize`？这些都要在**时间轴（timeline）**上看，而 timeline 的工具就是 Nsight Systems（nsys）。

本章建立一条标准 SOP：

```text
nsys → 找到最慢的 kernel / 最大的 CPU gap / 最可疑的同步
       ↓
ncu  → 分析那个 kernel 为什么慢（见 02 篇）
```

配套代码：`src/profiling/profile_target.py`（NVTX 标注的推理目标）、`src/profiling/scripts/run_nsys.sh`、`src/profiling/analyze_nsys.py`。

---

## 1. nsys 回答什么问题

| 问题 | 在 timeline 里看什么 |
|---|---|
| CPU 是否喂得饱 GPU？ | CPU 线程忙 / GPU 空闲的交替 |
| kernel 是否存在空隙？ | GPU 行上 kernel 之间的 gap |
| CUDA API 是否成瓶颈？ | `cudaLaunchKernel` 等 API 调用密度 |
| H2D 是否和 compute overlap？ | memcpy 与 kernel 是否在时间上重叠 |
| 是否存在 `cudaDeviceSynchronize`？ | 同步 API 的长条 + GPU 停摆 |
| 哪一阶段最慢？ | NVTX range 的时长对比 |
| 是否大量 tiny kernel？ | kernel 数量、每个的时长 |
| GPU 是否存在 bubble？ | 大段 GPU idle |

关键区分：nsys 给的是**时间维度的证据**（谁在什么时候忙），不是硬件计数器的效率证据。先看时间，再进 ncu 看效率。

---

## 2. 采集命令

最小可用的采集：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output /tmp/timeline \
  python profile_target.py
```

- `--trace=cuda,nvtx,osrt`：CUDA API + NVTX range + OS runtime（线程）三张关键视图。
- `--sample=none`：不做 CPU 采样（否则会引入额外开销，且我们主要看 GPU）。
- `--cpuctxsw=none`：不采上下文切换。

本仓库封装好的脚本：

```bash
export PROFILING_PYTHON=/home/guhaoran/miniconda3/envs/flashrt/bin/python
bash Work/src/profiling/scripts/run_nsys.sh \
  --hidden 1024 --layers 4 --batch 1 --steps 3 --output-root /tmp/nsys
```

脚本会生成时间戳目录，写入 `command.txt`（可复现），并自动跑 `nsys stats` 生成 kernel/API/NVTX 汇总表到 `analysis/stats.txt`。查看：

```bash
nsys-ui /tmp/nsys/<run>/reports/timeline.nsys-rep
```

---

## 3. 为什么要在代码里打 NVTX

没有 NVTX 时，timeline 里只有一串无上下文的内核名（`at::native::...`），无法对应到"这是第几层、是 preprocess 还是 postprocess"。打 NVTX range 后：

```python
torch.cuda.nvtx.range_push("h2d")
x = x_cpu.to(device, non_blocking=True)
torch.cuda.nvtx.range_pop()
```

timeline 上就会出现有名字的色块，把"这一段 200us 里，preprocess 占多少、block_0 占多少"看清楚。

**命名规范**（重要）：NVTX 名用**下划线**，不要用 `/`，因为 `/` 在 NCU 的 filter 里是 range-stack 语法，混用会导致 ncu 过滤失败。见 `src/profiling/profile_target.py` 的 `h2d` / `block_0` / `postprocess`。

---

## 4. timeline 的四种典型病态

### 病态 A：GPU 空隙 + CPU 满载 → CPU/launch-bound

```text
CPU: ████████████████████████████████（一直忙）
GPU: ███  ███  ███  ███  ███（kernel 之间大量空隙）
```

诊断：CPU 喂得太慢。要么算子太多（launch overhead），要么 Python/dispatcher 太重。对策：fusion、CUDA Graph、C++ runtime。

### 病态 B：GPU 连续满载 → compute/memory-bound

```text
CPU: ██  ██  ██  ██（有空闲，说明 CPU 不慢）
GPU: ████████████████████████████████（连续无空隙）
```

诊断：瓶颈在 GPU 本身。进 ncu 判断 compute vs memory（见 02 篇）。

### 病态 C：同步长条 → synchronization-bound

```text
CPU: ████████[cudaDeviceSynchronize ████████████]（一大段等待）
GPU: ███████████████（早已跑完，空等 CPU）
```

诊断：`.item()` / `.cpu()` / 显式 sync 切断了异步流水。对策：去掉隐式同步、异步化、累积后再同步。

### 病态 D：H2D 与 compute 串行 → I/O-bound

```text
GPU: [H2D ██████][kernel ███][D2H ██████]（copy 和 compute 不重叠）
```

诊断：H2D 没 pin、没 non_blocking，或数据加载本身慢。对策：pinned + async H2D + prefetch，见 `note/kernel/02`。

---

## 5. 从 `nsys stats` 快速定位

不打开 GUI，先看汇总表也能缩小范围：

```text
cuda_gpu_kern_sum  按 kernel 总耗时排序 → 谁最慢
cuda_api_sum       按 CUDA API 总耗时排序 → 同步/分配是不是大头
nvtx_sum           按 NVTX range 排序 → 哪个 stage 最慢
```

`src/profiling/analyze_nsys.py` 能把 `nsys stats` 的文本解析成 JSON。**注意**：`nsys stats` 里的时间受 profiler 干扰，只用于**排序定位**，最终结论要回 timeline 确认，不要拿它当权威 latency 数字。

---

## 6. 本机实测观察（Thor, sm_110）

用 `run_nsys.sh` 跑 4 层小模型，timeline 上应看到：

```text
1. 每个 block 内是若干短小的 at::native kernel（Linear + LayerNorm + GELU）
2. batch=1 时 kernel 都很短（几 us 到几十 us），launch 开销占 wall 的一定比例
3. h2d 只在第一次出现，因为输入非阻塞复制后 GPU 立即开始算
```

具体的 kernel 数量、最慢 kernel 名、gap 比例以 `analysis/stats.txt` 为准，**不在这里贴静态数字冒充结果**——每次跑都会变，且依赖 shape。

---

## 7. nsys 的局限与误区

- **不是硬件计数器工具**：看不到 occupancy、DRAM 带宽、Tensor Core 利用率，那要 ncu。
- **profiler 会扰动**：nsys 会给每次 launch 增加额外开销，wall time 会偏大，所以 nsys 数据用于**相对定位**，不用于对外宣称的绝对延迟。
- **不要拿 nsys 的整段 wall 当吞吐**：同上一级，见 `note/inference/01` 的测量纪律。

---

## 8. 本模块闭环小结

```text
问题：推理时间花在哪一段
      ↓
工具：nsys（时间轴）+ NVTX（分 stage）
      ↓
判断：CPU/launch/memory/compute/sync/I/O bound（四种病态）
      ↓
输出：最慢 kernel / 最大 gap / 可疑同步 → 交给 ncu
```

下一模块：`note/profiling/02_ncu_kernel_profiling.md`，回答"最慢的那个 kernel 为什么慢"。
