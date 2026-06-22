# 13_kernel_profile - GPU Kernel 性能分析与调试模块

## 工业背景：为什么 Kernel Profiling 是必备技能

在 LLM 推理 / 训练的工程实践中，以下场景每天都在发生：

1. **"这个 kernel 为什么慢了 40%？"** —— 没有 profiling 数据，你只能靠猜。可能是 register spilling，可能是 bank conflict，可能是 occupancy 不够，也可能是 memory-bound 根本没发现。
2. **"H2D 拷贝占了 30% 的端到端延迟"** —— nsys timeline 一看便知，优化方向是 pinned memory + async copy + stream overlap。
3. **"CPU 预处理成了瓶颈"** —— GPU 在等数据准备好，`perf stat` 发现 IPC 只有 0.3，cache miss 率 45%，显然需要 SIMD 向量化或数据预取优化。
4. **"Tensor Core 利用率只有 15%"** —— ncu Speed of Light 显示 kernel 完全 memory-bound，需要 tiling 减少显存访问。

本模块提供从 CPU 到 GPU 的**完整性能分析链路**，覆盖工具链、指标解读、实战案例和自动化脚本。

---

## 目录结构

```
13_kernel_profile/
├── README.md                 ← 你在这里
├── __init__.py               ← 空文件，使模块可导入
├── cpu_profiling_guide.md    ← CPU 端性能分析指南（perf / VTune / SIMD）
├── gpu_profiling_guide.md    ← GPU 端性能分析指南（ncu / nsys / 优化决策树）
└── profile_runner.py         ← 自动化 profiling 运行脚本
```

---

## 学习路径

### 第一步：理解瓶颈类型

在动手 profiling 之前，先建立 mental model：

| 瓶颈类型 | 典型特征 | 定位工具 | 典型优化手段 |
|----------|----------|----------|--------------|
| **Memory-bound（显存带宽）** | 计算单元空闲，大量时间等待显存传输 | ncu Speed of Light 显示 memory 利用率 > 80% | 减小显存访问（fusion / tiling / shared memory / vectorized load） |
| **Compute-bound（计算吞吐）** | 显存带宽未用满，但 FMA / Tensor Core 跟不上 | ncu 显示 SM throughput 接近峰值但性能仍差 | 增加并行度（tensor core / warp-level 优化） |
| **Latency-bound（延迟）** | occupancy 低，warp 无法隐藏延迟 | warp occupancy < 30%，大量 warp 在 stall | 提高 occupancy / 多 stream overlap / 减少同步 |
| **CPU overhead-bound** | GPU kernel 很快但 CPU 端调度慢 | nsys timeline 显示 GPU idle 间隙 | 减少 kernel launch 次数 / CUDA Graph / 异步 API |

### 第二步：按照场景选择工具

```
需要分析什么？
├─ 整体端到端延迟 → nsys (Nsight Systems)
│  ├─ kernel 执行时间分布
│  ├─ H2D/D2H 拷贝时间分布
│  └─ CPU-GPU 同步开销
├─ 单个 kernel 为什么慢 → ncu (Nsight Compute)
│  ├─ memory workload（显存带宽、cache 命中率）
│  ├─ compute workload（SM 利用率、tensor core 利用率）
│  ├─ scheduler stats（occupancy、warp stall 原因）
│  └─ Speed of Light（自动汇总瓶颈类型）
├─ CPU 端预处理 / 后处理 → perf
│  ├─ IPC、cache miss、branch miss
│  ├─ SIMD 向量化效率
│  └─ numa 亲和性
└─ 内存瓶颈 → perf mem / numastat
   ├─ 内存访问延迟分布
   └─ numa 跨节点访问
```

### 第三步：掌握具体的诊断方法

- [cpu_profiling_guide.md](./cpu_profiling_guide.md) —— 从 CPU 角度做性能分析
- [gpu_profiling_guide.md](./gpu_profiling_guide.md) —— 从 GPU 角度做性能分析

### 第四步：使用自动化工具

```bash
# 对 flash attention kernel 做 ncu 分析
python profile_runner.py --kernel flash_attention --tool ncu

# 对 rmsnorm 做 nsys 分析
python profile_runner.py --kernel rmsnorm --tool nsys

# 对所有注册的 kernel 做 ncu 分析
python profile_runner.py --all --tool ncu

# 只做 CPU 端 perf 分析
python profile_runner.py --kernel matmul --tool perf

# 导出结果到指定目录
python profile_runner.py --kernel softmax --tool ncu --output-dir ./reports/
```

---

## 前置条件

### 硬件

- NVIDIA GPU（推荐 A100 / H100 / RTX 4090 或更新）
- x86_64 CPU（用于 perf / VTune，ARM 需使用 `perf` 替代部分功能）

### 软件

```bash
# Nsight Systems（用于系统级 profiling）
sudo apt install nsight-systems

# Nsight Compute（用于 kernel 级 profiling）
sudo apt install nsight-compute

# perf（CPU 性能分析，Linux 内核自带）
sudo apt install linux-tools-common linux-tools-$(uname -r)

# 可选：Intel VTune（更深入的 CPU 微架构分析）
# 从 Intel 官网下载：https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html

# 可选：AMD uProf（AMD CPU 微架构分析）
# 从 AMD 官网下载：https://developer.amd.com/amd-uprof/

# Python 依赖
pip install pynvml
```

---

## 快速开始

### 1. 验证工具链

```bash
# 检查 ncu 是否可用
ncu --version

# 检查 nsys 是否可用
nsys --version

# 检查 perf 是否可用
perf --version
```

### 2. 运行第一个 profiling

```bash
# 用 ncu 对所有 kernel 做快速分析（--set basic 开销低）
ncu --set basic --csv python 01_cuda_basics/benchmark_cuda_basics.py

# 用 nsys 看整体 timeline
nsys profile --trace=cuda,nvtx -o timeline_report python 01_cuda_basics/benchmark_cuda_basics.py
```

### 3. 深入分析单个 kernel

```bash
# 只看 flash_attention_fwd kernel
ncu --kernel-name "flash_attention_fwd" --set full python 01_cuda_basics/benchmark_cuda_basics.py
```

---

## 工具对比速查表

| 工具 | 分析层级 | 输出形式 | 开销 | 适用场景 |
|------|----------|----------|------|----------|
| perf stat | CPU 事件计数 | 终端统计 | 极低 | 快速判断 CPU 瓶颈类型 |
| perf record | CPU 采样 | perf.data → report/flamegraph | 低 | 定位 CPU 热点函数 |
| perf mem | 内存访问 | 终端统计 | 中 | 分析内存访问延迟和 NUMA 问题 |
| Intel VTune | CPU 微架构 | GUI / CLI 报告 | 中 | 深入 CPU 微架构瓶颈 |
| Nsight Systems | GPU 系统级 | .nsys-rep（GUI） | 低-中 | 端到端 timeline 分析 |
| Nsight Compute | GPU kernel 级 | 终端输出 / .ncu-rep（GUI） | 高 | 单个 kernel 深入分析 |

---

## 常见问题

### ncu 分析导致 kernel 变慢很多正常吗？

正常。ncu 会注入 profiler 指令来收集 GPU 硬件计数器，有显著开销（特别是 `--set full` 模式下）。Profiling 结果的**相对指标**（如 occupancy、cache hit rate）是准确的，但**绝对时间**不可用作性能基准。

### nsys 和 ncu 应该先用哪个？

先 nsys 后 ncu。先用 nsys 找到哪个 kernel / 哪个阶段是瓶颈，再用 ncu 深入分析那个 kernel 的具体问题。

### perf 在 WSL / 虚拟机上能用吗？

`perf stat` 通常可用。`perf record` 需要内核支持硬件 PMU（Performance Monitoring Unit），虚拟机环境可能受限。建议在裸金属 Linux 上运行完整 profiling。

### root 权限问题

ncu 和 nsys 通常不需要 root。perf 的某些功能（如 `perf record -a` 全系统采样）需要 root 或调整 `perf_event_paranoid`：

```bash
# 临时降低限制（不推荐生产环境）
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

# 或永久设置
echo 'kernel.perf_event_paranoid = 0' | sudo tee -a /etc/sysctl.conf
```
