# Nsight Systems 与 Nsight Compute 性能分析指南

## 概览

NVIDIA Nsight Systems 提供系统级性能分析（CPU、GPU、内存、I/O）。
NVIDIA Nsight Compute 提供详细的 GPU kernel 级性能分析。

经验法则：使用 Nsight Systems 进行端到端性能分析（哪个 kernel 耗时最长？瓶颈在哪里？）。使用 Nsight Compute 进行深层 kernel 分析（这个 kernel 为什么慢？）。

---

## 安装

### Nsight Systems CLI（nsys）

```bash
# 从 NVIDIA Developer 网站下载
# https://developer.nvidia.com/nsight-systems

# Ubuntu/Debian (.deb)
sudo dpkg -i nsight-systems-<version>.deb
sudo apt-get install -f

# 验证安装
nsys --version

# 安装后的常见位置：
# /opt/nvidia/nsight-systems/<version>/target-linux-x64/nsys
```

### Nsight Compute CLI（ncu）

```bash
# 从 NVIDIA Developer 网站下载
# https://developer.nvidia.com/nsight-compute

# Ubuntu/Debian (.deb)
sudo dpkg -i nsight-compute-<version>.deb
sudo apt-get install -f

# 验证安装
ncu --version

# 安装后的常见位置：
# /opt/nvidia/nsight-compute/<version>/ncu
```

### NVIDIA Management Library (NVML) 绑定

```bash
pip install pynvml  # 用于程序化 GPU 监控
```

---

## Nsight Systems CLI 命令

### 基本性能分析

```bash
# 使用 CUDA tracing 分析 Python 脚本
nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --output=profile_report \
    python my_benchmark.py

# 使用 GPU 指标进行性能分析（开销更高）
nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --gpu-metrics-device=0 \
    --output=gpu_metrics_report \
    python my_benchmark.py

# 使用采样进行性能分析（长时间运行开销更低）
nsys profile \
    --trace=cuda,nvtx \
    --sample=cpu \
    --backtrace=dwarf \
    --output=sample_report \
    python my_benchmark.py
```

### 常用标志

| 标志 | 描述 |
|------|-------------|
| `--trace=cuda` | 追踪 CUDA API 调用和 kernel |
| `--trace=nvtx` | 追踪 NVTX 范围（配合 `torch.cuda.nvtx.range_push/pop` 使用） |
| `--trace=osrt` | 追踪 OS 运行时库（线程等） |
| `--trace=cublas` | 追踪 cuBLAS 库调用 |
| `--trace=cudnn` | 追踪 cuDNN 库调用 |
| `--gpu-metrics-device=0` | 在设备 0 上收集 GPU 硬件指标 |
| `--stats=true` | 将统计摘要输出到 stdout |
| `--force-overwrite=true` | 覆盖现有输出文件 |

### 报告生成

```bash
# 生成摘要统计
nsys stats profile_report.nsys-rep

# 导出为 CSV 以供分析
nsys stats --report cuda_gpu_kern_sum profile_report.nsys-rep \
    --format csv --output kernel_stats.csv

# 导出为 SQLite 数据库
nsys stats --report cuda_gpu_kern_sum profile_report.nsys-rep \
    --format sqlite --output kernel_stats.sqlite
```

---

## Nsight Compute CLI 命令

### 基本 Kernel 性能分析

```bash
# 分析脚本启动的所有 kernel
ncu \
    --set full \
    --csv \
    --log-file kernel_profile.csv \
    python my_benchmark.py

# 按名称分析特定 kernel
ncu \
    --kernel-name "my_custom_kernel" \
    --set full \
    --launch-skip 0 \
    --launch-count 1 \
    python my_benchmark.py

# 使用详细内存分析进行性能分析
ncu \
    --set full \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    python my_benchmark.py
```

### 性能指标模块

| 模块 | 它告诉你什么 |
|---------|-------------------|
| `SpeedOfLight` | 高级利用率：计算吞吐量、内存带宽 |
| `MemoryWorkloadAnalysis` | 内存访问模式、合并效率 |
| `MemoryWorkloadAnalysis_Chart` | 内存访问模式的可视化表示 |
| `MemoryWorkloadAnalysis_Tables` | 内存指令统计表 |
| `SchedulerStats` | Warp 调度、停顿原因 |
| `Occupancy` | 理论和实际 occupancy |
| `WarpStateStats` | 每周期 warp 状态分解 |
| `InstructionStats` | 指令组合、吞吐量 |
| `LaunchStatistics` | Grid/block 维度、寄存器/shared memory 使用量 |

### Baseline / Comparison Mode

```bash
# 保存 baseline 性能分析
ncu --set full --save baseline.ncu-rep python baseline.py

# 与 baseline 对比新版本
ncu --set full --import-source yes \
    --compare baseline.ncu-rep \
    python optimized.py
```

### 有用的 ncu 输出指标

Kernel 优化需要检查的关键指标：

```
Memory Throughput:    X GB/s      （与设备峰值带宽比较）
Compute Throughput:   X%          （实际 vs 理论峰值）
Occupancy:            X%          （活跃 warp / 每个 SM 最大 warp 数）
L1 Cache Hit Rate:    X%
L2 Cache Hit Rate:    X%
Registers per Thread: X           （如果太高会限制 occupancy）
Shared Memory:        X KB        （限制每个 block 的 occupancy）
```

---

## 解读结果

### 内存带宽分析

**良好信号：**
- 达到的带宽 > 峰值的 80%
- 高 L2 缓存命中率（> 40%）
- `Memory Throughput` 图表显示合并的全局内存事务（红色部分最少）

**不良信号：**
- 达到的带宽 < 峰值的 30%
- 大量非合并访问（Nsight Compute："Uncoalesced Global Accesses"）
- 低 L1/L2 命中率（< 20%）

**示例**：A100 峰值带宽为 1555 GB/s（HBM2e）。一个经过良好调优的内存受限 kernel（逐元素操作）应达到 1200-1400 GB/s。

### Occupancy 分析

**良好信号：**
- Occupancy > 50%
- Warp 停顿原因主要是 "Memory Throttle"（对内存受限 kernel 来说等待内存是正常的）

**不良信号：**
- Occupancy < 25%
- 因 "Not Selected" 或 "Barrier" 导致的停顿
- 寄存器使用过多（A100 上每个线程 > 128 个寄存器限制为 512 threads/SM）

**公式**：`Occupancy = active_warps / max_warps_per_SM`

### 延迟分析

- **内存受限 kernel**：停顿由内存依赖（Long Scoreboard）主导
- **计算受限 kernel**：停顿由执行依赖（Short Scoreboard）主导
- **指令获取停顿**：kernel 太短，launch 开销占主导

### Roofline 分析方法论

1. **测量 kernel 的操作强度**（FLOPs/byte）：
   - 计算 kernel 中的计算操作（adds、muls、FMA）数量
   - 计算读/写到全局内存的总字节数
   - Flops：总 FLOPs / runtime_seconds
   - Op intensity：总 FLOPs / total_bytes

2. **在 roofline 上绘图**：
   - X 轴：操作强度（FLOPs/byte）- 对数尺度
   - Y 轴：可达到性能（FLOPs/second）- 对数尺度
   - 对角线：内存带宽上限（bandwidth * op_intensity）
   - 水平线：峰值计算上限（如 A100 上 312 TFLOPS）

3. **确定模式**：
   - 如果在对角线以下：内存受限。优化内存访问模式。
   - 如果在水平渐近线以下：计算受限。优化指令组合，使用 tensor core。
   - 如果接近两者：良好优化的 kernel。

### 常见瓶颈及其特征

| 瓶颈 | Nsight Systems 特征 | Nsight Compute 特征 | 修复方法 |
|-----------|-------------------------|-------------------------|-----|
| 非合并内存 | 高 kernel 时间，许多小的 memcpy | 低 L1 命中率，每事务多个扇区 | 对齐数据，使用向量化加载 |
| 低 occupancy | Kernel 时间在时间线上占主导 | Occupancy < 30% | 减少寄存器，减少 shared memory，调整 block size |
| Launch 开销 | 许多微小 kernel 之间有间隙 | 非常短的 kernel 持续时间 | 融合 kernel，批量操作 |
| Bank conflict | 不可见（Systems 中不显示） | 高 "Shared Memory Bank Conflicts" | 向 shared memory 数组添加填充 |
| Warp divergence | Kernel 时间高于预期 | 高 "Divergent Branch" 停顿 | 重构分支使其 warp-uniform |
| 指令缓存未命中 | 不可见 | 高 "I-Cache Misses" | 减少 kernel 代码大小，拆分大型 kernel |
| 同步开销 | Kernel 不重叠 | 不可见 | 使用 streams 进行并发执行 |
| 寄存器溢出 | Kernel 使用比预期更多的寄存器 | "Local Memory" 流量 > 0 | 减少局部变量，拆分为更小的 kernel |

---

## 使用 PyTorch 进行性能分析

### 内置 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        output = model(input)
        prof.step()

# 打印关键平均值
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### NVTX Ranges 用于 Nsight Systems

```python
import torch

# 为 Nsight Systems 标记一个区域
torch.cuda.nvtx.range_push("attention_forward")
output = attention(query, key, value)
torch.cuda.nvtx.range_pop()

# 装饰器模式
def nvtx_range(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            result = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return result
        return wrapper
    return decorator

@nvtx_range("fused_mlp")
def fused_mlp_forward(x, w1, w2):
    ...
```

### 快速内存带宽测试

```python
import torch

def test_bandwidth(size_mb=256):
    """使用拷贝 kernel 快速估算带宽。"""
    n = (size_mb * 1024 * 1024) // 4  # 每个 float32 4 bytes
    a = torch.randn(n, device='cuda')
    b = torch.randn(n, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    b.copy_(a)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    bytes_moved = a.numel() * a.element_size() * 2  # 读 + 写
    bw_gb_s = (bytes_moved / elapsed_ms) / 1e6
    print(f"Bandwidth: {bw_gb_s:.2f} GB/s")
    print(f"Peak theoretical: get from device specs")

test_bandwidth()
```

---

## 快速参考卡片

```
# 系统级：什么慢？
nsys profile --trace=cuda,nvtx -o report python script.py
nsys stats report.nsys-rep

# Kernel 级：为什么慢？
ncu --set full --csv -o profile python script.py

# 特定 kernel 深入分析：
ncu --kernel-name "my_kernel" --set full python script.py

# 对比两个版本：
ncu --set full --save baseline.ncu-rep python baseline.py
ncu --compare baseline.ncu-rep python optimized.py
```
