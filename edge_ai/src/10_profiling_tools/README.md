# 性能分析工具

用于机器人/自动驾驶系统性能分析的实用性能分析脚本。

## 文件概览

| 文件 | 类型 | 用途 |
|------|------|---------|
| `profiling.sh` | Bash 封装脚本 | 封装 `perf stat` 以收集 CPU 硬件计数器事件（周期数、指令数、缓存未命中、分支预测失败、上下文切换）并计算衍生指标（IPC、缓存未命中率、分支预测失败率）。最适合用于快速了解 CPU 微架构行为。 |
| `flamegraph.sh` | Bash 脚本 | 录制 `perf` 采样数据并生成可交互的 SVG 火焰图。支持 on-CPU 分析（热点函数）和 off-CPU 分析（阻塞/等待分析）。若本地未找到 FlameGraph 脚本则自动下载。 |
| `ebpf_trace.py` | Python 脚本 | 模拟 eBPF 风格的函数延迟追踪器。提供两种追踪模式：轻量级计时器（纳秒精度的按函数测量）和 cProfile（Python 内置性能分析器）。同时还提供用于生产环境部署的真实 `bpftrace` 和 BCC 工具集示例。 |

### 何时使用各个工具

1. **`profiling.sh`** - 第一步。运行它以获取快速的硬件级别概览（IPC、缓存行为、上下文切换）。告诉你程序是 CPU 瓶颈、内存瓶颈还是 I/O 瓶颈。
2. **`flamegraph.sh`** - 深入分析 CPU 使用情况。使用 `--cmd` 模式查找程序中的热点函数。使用 `--offcpu` 模式查找程序在哪里阻塞/等待。
3. **`ebpf_trace.py`** - 函数级别的延迟分解。使用 `--simulate` 获取快速延迟直方图。使用 `--bpftrace-examples` 查看用于内核级追踪的生产环境 bpftrace 示例。

---

## profiling.sh

封装 `perf stat` 以收集关键硬件计数器事件并计算衍生指标（IPC、缓存未命中率、分支预测失败率）。

### 前置条件

```bash
sudo apt-get install linux-tools-generic
# 若无需 root 权限运行：
sudo sysctl kernel.perf_event_paranoid=-1
```

### 用法

```bash
# 基本性能分析
./profiling.sh ./my_program arg1 arg2

# 自定义事件
./profiling.sh --events "cycles,instructions,cache-misses" ./my_program

# 重复测量
./profiling.sh --repeat 5 ./my_program

# 保存输出
./profiling.sh --output results.txt ./my_program

# 列出可用事件
./profiling.sh --list
```

### 示例输出

```
============================================================
  性能分析：./my_program
============================================================
  程序：    ./my_program
  重复：    1
============================================================

 Performance counter stats for './my_program':
      1,234,567,890   cycles
        456,789,012   instructions        # 0.37 IPC
         89,012,345   cache-references
         12,345,678   cache-misses        # 13.87% 缓存未命中率
          1,234,567   branch-misses       # 2.34% 分支预测失败率

============================================================
  衍生指标
============================================================
  IPC（每周期指令数）：               0.37
    -> 可能是内存瓶颈（IPC 较低）
  缓存未命中率：                      13.87%
  分支预测失败率：                    2.34%
============================================================
```

---

## flamegraph.sh

录制 perf 采样数据并生成可交互的 SVG 火焰图。

### 前置条件

```bash
sudo apt-get install linux-tools-generic perl
# FlameGraph 脚本如果在 /opt/FlameGraph 中未找到，将自动从 GitHub 下载
```

### 用法

```bash
# 对正在运行的进程采样 30 秒
./flamegraph.sh --pid 12345 --duration 30

# 直接分析一个命令
./flamegraph.sh --cmd ./my_program arg1 arg2

# 生成 off-CPU 火焰图（阻塞/等待分析）
./flamegraph.sh --pid 12345 --offcpu --duration 60

# 自定义频率和输出路径
./flamegraph.sh --pid 12345 --frequency 997 --output /tmp/flamegraphs
```

### 输出

- `flamegraph_output/cpu_flamegraph.svg` - 可交互的 CPU 火焰图
- `flamegraph_output/offcpu_flamegraph.svg` - Off-CPU 火焰图（如果指定了 --offcpu）
- `flamegraph_output/perf.data` - 原始 perf 录制数据
- `flamegraph_output/perf.folded` - 中间折叠栈数据

---

## ebpf_trace.py

模拟 eBPF 风格的函数延迟追踪器。使用轻量级 Python 计时器或 cProfile 来测量函数执行时间并打印延迟直方图。

### 前置条件

- Python 3.7+（模拟模式）
- 模拟模式无需 root 权限
- bpftrace + bcc-tools（可选，用于真实 eBPF 示例）

### 用法

```bash
# 以默认 100 帧运行模拟
python3 ebpf_trace.py --simulate

# 自定义帧数
python3 ebpf_trace.py --simulate --frames 500

# 使用 cProfile 进行更深入的分析
python3 ebpf_trace.py --simulate --cprofile

# 显示真实 bpftrace/BCC 示例
python3 ebpf_trace.py --bpftrace-examples
```

### 示例输出

```
============================================================
  eBPF 风格函数延迟追踪器（模拟）
============================================================
  模式：       轻量级计时器
  帧数：       100

  已处理 20/100 帧... 端到端：15234 us
  ...

==============================================================
  延迟分布：object_detection
==============================================================
  样本数：      100
  平均值：   16432.51 us
  P50：          16234 us
  P95：          23456 us
  P99：          24891 us

  范围           数量   分布
  -------------- --------  ------------------------------
  8K-16K             45  ##############################
  16K-32K            55  ####################################
```

---

## 推荐工作流程

> 本项目是三个独立的可执行工具，彼此之间无代码依赖，无需按顺序阅读源码。以下为推荐使用流程：

1. **运行 `profiling.sh`** 获取快速概览（IPC、缓存行为）
2. 若为 CPU 瓶颈，运行 **`flamegraph.sh --cmd`** 查找热点函数
3. 若存在延迟问题，运行 **`flamegraph.sh --offcpu`** 查找阻塞点
4. 要进行函数级别的延迟分解，运行 **`ebpf_trace.py --simulate`** 或在生产环境中部署真实的 bpftrace uprobe
5. 迭代优化：优化热点路径、重新测量、对比结果

