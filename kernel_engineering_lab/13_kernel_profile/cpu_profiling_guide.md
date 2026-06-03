# CPU 端性能分析指南

## 工业背景

GPU kernel 不是孤岛。在生产环境中，每一次 GPU 计算前后都有大量 CPU 端工作：

- **预处理**：tokenization、KV cache 管理、batch 组装、attention mask 构建
- **数据调度**：H2D 拷贝前的内存布局整理、padding/truncation、RoPE 位置编码计算
- **后处理**：logits 采样（top-k / top-p）、beam search 状态管理、D2H 结果验证
- **框架开销**：Python 解释器 / torch dispatcher / CUDA driver API 调用

如果 CPU 端慢了，GPU 就在空转。AVX512/AVX2 指令使用不足、cache miss 过高、分支预测错误、numa 跨节点访问——每一样都可能成为端到端延迟的瓶颈。

---

## 目录

- [perf 工具详解](#perf-工具详解)
- [Intel VTune / AMD uProf](#intel-vtune--amd-uprof)
- [SIMD 向量化检查](#simd-向量化检查)
- [CPU 内存分析](#cpu-内存分析)
- [numa 感知优化](#numa-感知优化)
- [实战命令速查表](#实战命令速查表)

---

## perf 工具详解

perf 是 Linux 内核自带的性能分析工具，直接读取 CPU PMU（Performance Monitoring Unit）硬件计数器，开销极低（通常 < 1%）。

### perf stat：快速统计关键指标

```bash
# 基础用法：统计程序运行期间的关键事件
perf stat python script.py

# 输出示例解读：
#  Performance counter stats for 'python script.py':
#
#    15,234.56 msec task-clock          # CPU 利用率（16 核中用了多少）
#    45,123,456,789 cycles              # 总 CPU 周期数
#    18,049,382,716 instructions        # 总指令数
#      0.40  insn per cycle             # IPC = 指令数 / 周期数
#     1,234,567,890 cache-misses        # L1/L2/L3 cache miss 总数
#       345,678,901 branch-misses       # 分支预测错误数
#      1.234  GHz  cpu-clock            # 实际 CPU 频率
```

#### 关键指标解读

| 指标 | 含义 | 好 | 需要关注 | 严重 |
|------|------|-----|----------|------|
| **IPC**（Instructions Per Cycle） | 每周期执行的指令数 | > 2.0（向量化密集计算可达 3-5） | 1.0-2.0（中等效率） | < 1.0（大量 stall，CPU 在等待内存或 I/O） |
| **cache-miss rate** | 所有 cache 级别的综合 miss 率 | < 3%（数据局部性好） | 3-10%（可接受） | > 10%（需检查数据布局） |
| **branch-miss rate** | 分支预测错误率 | < 1%（分支规整） | 1-5%（需关注） | > 5%（严重，考虑重构分支逻辑或用无分支算法） |
| **L1-dcache-load-misses** | L1 数据缓存加载 miss | < 3% of L1 loads | 3-5% | > 5%（数据访问模式需要优化） |
| **LLC-load-misses**（Last Level Cache） | L3 cache miss，数据必须从内存读取 | 越低越好 | < 5% of LLC loads | > 10%（内存带宽是瓶颈） |

#### 针对特定事件的统计

```bash
# 只看 IPC
perf stat -e cycles,instructions python script.py

# 关注内存访问
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses python script.py

# 关注分支预测
perf stat -e cycles,instructions,branch-instructions,branch-misses python script.py

# 查看 SIMD 指令使用情况（Intel）
perf stat -e fp_arith_inst_retired.128b_packed_single,\
             fp_arith_inst_retired.256b_packed_single,\
             fp_arith_inst_retired.512b_packed_single \
         python script.py

# 查看 SIMD 指令使用情况（AMD）
perf stat -e ls_dispatch.ld_st_dispatch,\
             deOpCountPackedFP,ex_ret_brn \
         python script.py

# 查看 CPU 前端/后端 stall 分布
perf stat -e cycles,\
             uops_retired.retire_slots,\
             uops_issued.any,\
             resource_stalls.any,\
             cpu_clk_unhalted.thread_any,\
             idq_uops_not_delivered.core \
         python script.py
```

#### 常用 perf stat 参数

```bash
# 重复运行取平均值（减少噪声）
perf stat -r 10 python script.py

# 显示更详细的信息（-d = detailed）
perf stat -d python script.py

# 显示非常详细的信息（-ddd = very detailed）
perf stat -ddd python script.py

# 按 CPU 核心分开统计
perf stat -a -A python script.py  # -a = 全系统, -A = 每核分开

# 指定刷新间隔
perf stat -I 1000 python script.py  # 每秒输出一次
```

---

### perf record：热点函数定位

`perf record` 按固定频率采样调用栈，找到 CPU 时间消耗最多的函数。

```bash
# 基础采样：默认以 4000 Hz 采样
perf record python script.py

# 生成报告
perf report

# 更常用的：采样 + 报告一步到位
perf record -g python script.py && perf report -g

# 提高采样频率（对短程序有用）
perf record -F 999 python script.py  # 999 Hz 采样

# 只采样用户态（排除内核）
perf record -e cycles:u python script.py

# 带调用图（callgraph）的采样
perf record --call-graph dwarf python script.py
# 或使用 fp（frame pointer）模式（开销更低）
perf record --call-graph fp python script.py
```

#### 生成火焰图

```bash
# 安装 FlameGraph 工具
git clone https://github.com/brendangregg/FlameGraph.git /tmp/FlameGraph

# 采集数据
perf record -F 99 -g python script.py

# 生成火焰图
perf script | /tmp/FlameGraph/stackcollapse-perf.pl | /tmp/FlameGraph/flamegraph.pl > flamegraph.svg
```

#### perf report 常用选项

```bash
# 交互模式：按函数查看
perf report

# 终端输出模式
perf report --stdio

# 按调用图排序
perf report --stdio -g graph

# 只看用户态
perf report --stdio --dsos=python

# 只看特定进程
perf report --stdio --pid=$(pgrep -f script.py)
```

---

### perf top：实时观察

```bash
# 实时查看系统热点函数
perf top

# 只观察特定进程
perf top -p $(pgrep -f script.py)

# 增加调用图深度
perf top -g

# 只显示用户态函数
perf top -e cycles:u
```

---

### perf annotate：函数内汇编分析

```bash
# 采样后对热点函数做汇编级分析
perf record python script.py
perf annotate -l <function_name>

# 或者在 perf report 中选择函数后按 'a'
```

---

### IPC < 1 说明什么？

IPC < 1.0 意味着 CPU 在每个周期内执行不到 1 条指令。在现代超标量 CPU（Alder Lake / Zen 4 每个核心每周期可执行 4-6 条指令）上，这是严重低效的信号。

常见原因：
1. **Cache miss**：CPU 在等待数据从内存加载（LLC miss）
2. **分支预测错误**：流水线被清空，浪费已执行的指令
3. **指令依赖链**：后续指令依赖前一条指令的结果，无法并行执行
4. **数据依赖**：pointer chasing 模式，下一条 load 的地址依赖当前 load 的结果
5. **系统调用/上下文切换**：OS 开销

诊断路径：
```bash
# 1. 先看整体
perf stat python script.py

# 2. 如果 IPC < 1，同时 branch-misses > 5%，先优化分支
# 3. 如果 IPC < 1，cache-misses 很高，先优化数据布局
# 4. 如果 IPC < 1 但 cache-misses 和 branch-misses 都不高，可能是 backend bound
#    → 用 perf record 找到热点函数 → perf annotate 看指令依赖链
```

---

### branch miss rate > 5% 严重吗？

**严重。** 现代 CPU 的 branch predictor 准确率通常在 97-99%。如果 branch-miss rate > 5%，说明你的代码有高频的不规则分支，每次预测错误会清空流水线（损失 15-20 个周期）。

常见元凶：
- 在循环内部做 if/else（特别是热路径上的 data-dependent branch）
- 大量 switch-case
- 类虚函数调用（vtable lookup 也是一种间接分支）

优化手段：
- 用 CMOV（conditional move）替代分支
- 用 bitwise 操作替代条件判断
- 用 SIMD 向量化替代循环内的 if
- 数据预排序（如果必须分支，先排序让分支结果统一）

---

## Intel VTune / AMD uProf

perf 适合日常快速分析，但如果需要深入 CPU 微架构（memory bandwidth、frontend/backend bound 比例、内存访问延迟分布等），需要更专业的工具。

### Intel VTune

```bash
# 安装（从 Intel oneAPI 仓库）
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html

# 命令行分析（不需 GUI）
vtune -collect hotspots -- python script.py

# 内存带宽分析
vtune -collect memory-access -- python script.py

# 微架构探索（frontend/backend bound 分析）
vtune -collect uarch-exploration -- python script.py

# 生成报告（不需要 GUI）
vtune -report summary -r <result_dir>
vtune -report hotspots -r <result_dir>
```

#### VTune 关键指标

| 指标 | 含义 | 好 | 差 |
|------|------|-----|-----|
| Front-End Bound | CPU 前端（取指/解码）受限比例 | < 20% | > 40%（I-cache miss 或代码密度问题） |
| Back-End Bound | CPU 后端（执行单元/内存）受限比例 | < 20% | > 40%（数据等待） |
| Bad Speculation | 错误推测（分支预测失败）浪费的比例 | < 10% | > 20% |
| Retiring | 有效执行比例 | > 50% | < 30%（大量 CPU 时间在做无用功） |
| Memory Bound | 内存带宽受限比例 | < 30% | > 60%（需优化内存访问） |
| DRAM Bound | DRAM 带宽受限（LLC miss 后） | < 20% | > 50%（严重的内存瓶颈） |

### AMD uProf

```bash
# 命令行分析
AMDuProfCLI collect --config tbp -- python script.py

# 内存分析
AMDuProfCLI collect --config mem -- python script.py

# 生成报告
AMDuProfCLI report -i <result_dir>
```

---

## SIMD 向量化检查

现代编译器会自动做自动向量化（auto-vectorization），但不保证。以下是检查编译结果是否被正确向量化的方法。

### 编译期检查：循环向量化报告

#### GCC：`-fopt-info-vec`

```bash
# 编译时输出向量化信息
gcc -O3 -march=native -fopt-info-vec -c my_code.c

# 输出示例：
# my_code.c:42:9: optimized: loop vectorized using 16 byte vectors
# my_code.c:67:3: missed: couldn't vectorize loop
# my_code.c:78:5: note: not vectorized: data dependency

# 只看未向量化的循环（最重要）
gcc -O3 -march=native -fopt-info-vec-missed -c my_code.c

# 更详细的优化信息
gcc -O3 -march=native -fopt-info-all -c my_code.c
```

#### Clang：`-Rpass=vectorize`

```bash
# 编译时报告成功向量化的循环
clang -O3 -march=native -Rpass=vectorize -c my_code.c

# 报告向量化失败的循环
clang -O3 -march=native -Rpass-missed=vectorize -c my_code.c

# 报告所有优化
clang -O3 -march=native -Rpass=.* -c my_code.c
```

#### 在 PyTorch C++ 扩展中启用

```python
# setup.py 中添加编译标志
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    ext_modules=[
        CUDAExtension(
            "my_module",
            ["my_kernel.cu", "my_bindings.cpp"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-march=native",
                    "-fopt-info-vec",          # 向量化报告
                    "-fopt-info-vec-missed",
                ],
                "nvcc": [
                    "-O3",
                    "-Xptxas=-v",              # 查看 PTX 汇编器输出
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### 运行时检查：AVX 指令使用率

```bash
# 检查程序是否使用了 256 位 AVX 指令
perf stat -e \
  fp_arith_inst_retired.128b_packed_single,\
  fp_arith_inst_retired.256b_packed_single,\
  fp_arith_inst_retired.512b_packed_single \
  python script.py

# 输出示例：
#   12,345,678  fp_arith_inst_retired.128b_packed_single   # SSE 指令
#   89,012,345  fp_arith_inst_retired.256b_packed_single   # AVX/AVX2 指令
#            0  fp_arith_inst_retired.512b_packed_single   # AVX-512 未使用
```

如果 128 位指令占比过高，说明编译器没有生成高效的向量化代码。

### 常见 SIMD 反模式及修复

#### 反模式 1：循环体内数据依赖

```c
// 反模式：每次迭代依赖前一次结果，编译器无法向量化
float acc = 0;
for (int i = 0; i < n; i++) {
    acc = acc * data[i] + bias[i];  // acc 存在循环依赖
}

// 修复：打破依赖链（如果可以接受近似结果）
// 或者使用 -ffast-math 放松浮点精度约束
```

#### 反模式 2：非对齐访问

```c
// 反模式：指针未对齐到 SIMD 边界
float *data = (float *)malloc(n * sizeof(float));
// 如果 malloc 返回的地址不是 32 字节对齐，SIMD 加载会降级

// 修复：使用对齐分配
float *data = (float *)aligned_alloc(32, n * sizeof(float));
// 或使用 posix_memalign
```

#### 反模式 3：间接索引（gather）

```c
// 反模式：间接索引，编译器必须生成 scatter/gather 指令（比连续访问慢 3-10x）
for (int i = 0; i < n; i++) {
    output[i] = input[index[i]] * scale;  // index 是间接索引
}

// 修复：预先对数据进行重排，使访问变为连续
```

#### 反模式 4：混合数据类型

```c
// 反模式：循环内不同类型混合操作
for (int i = 0; i < n; i++) {
    double result = (double)int_data[i] * float_data[i];  // 类型转换开销
}

// 修复：保持数据类型一致，在循环外完成类型转换
```

---

## CPU 内存分析

### perf mem：内存访问延迟分析

```bash
# 记录内存访问事件
perf mem record python script.py

# 查看内存访问延迟报告
perf mem report

# 输出示例：
#  Samples  Weight  Memory access  Symbol
#  -------  ------  -------------  ------
#  12.3%    45 ns   L1 hit         my_hot_function
#  34.5%    212 ns  L3 hit         my_hot_function
#  53.2%    387 ns  Local RAM      my_hot_function   ← 大部分访问去了主存

# 按延迟级别汇总
perf mem report --sort=mem
```

关键信息：如果超过 50% 的内存访问 latency > 200ns，说明 L1/L2/L3 cache 都未命中，数据局部性极差。

### 页错误（Page Fault）分析

对于 GPU 编程，pinned memory 至关重要。非 pinned memory 的第一次访问可能触发 major page fault（需从磁盘换入），阻塞 GPU 数据传输。

```bash
# 查看页错误统计
perf stat -e page-faults,major-faults,minor-faults python script.py

# 输出示例：
#   123,456 page-faults      # 总页错误
#     2,345 major-faults     # 大页错误（需要磁盘 I/O）→ 严重
#   121,111 minor-faults     # 小页错误（内存中的页表操作）→ 可接受
```

**Golden rule**：与 GPU 交互的内存必须使用 pinned memory（`cudaHostAlloc` 或 `torch.tensor(..., pin_memory=True)`），否则：
1. CUDA 驱动需要先内部拷贝到 pinned staging buffer
2. 拷贝操作变成同步的（不能与 GPU 计算重叠）
3. 可能触发 major page fault

---

## numa 感知优化

在多 socket 服务器上，GPU 通过 PCIe 连接到特定的 CPU socket。如果 CPU 端的内存分配在错误的 NUMA 节点，数据传输将经过跨 socket 的 QPI/UPI 链路，带宽显著下降。

### 诊断 NUMA 拓扑

```bash
# 查看 NUMA 拓扑
numactl --hardware

# 输出示例：
# available: 2 nodes (0-1)
# node 0 cpus: 0-15      ← CPU 0-15 在 NUMA node 0
# node 0 size: 128 GB
# node 1 cpus: 16-31     ← CPU 16-31 在 NUMA node 1
# node 1 size: 128 GB

# 查看 GPU 连接在哪个 NUMA node
nvidia-smi topo -m

# 输出会显示 GPU 与 NUMA node 的亲和关系
```

### 绑定进程到正确的 NUMA 节点

```bash
# 将进程绑定到 NUMA node 0 的 CPU 核心
numactl --cpunodebind=0 --membind=0 python script.py

# 更精确：绑定到特定 CPU 核心
taskset -c 0-15 numactl --membind=0 python script.py
```

### 监控 NUMA 状态

```bash
# 实时查看 NUMA 内存分配
numastat -c python

# 输出示例：
# Per-node process memory usage (in MBs) for PID 12345 (python)
#                 Node 0  Node 1  Total
#                ------- ------- -----
# Huge            0.00    0.00   0.00
# Heap            1.23   45.67  46.90    ← 大部分分配在 Node 1（错误！）
# Stack           0.00    0.12   0.12

# 如果 GPU 连接在 node 0，但 Python 内存分配在 node 1，产生跨 NUMA 访问

# 查看进程的 NUMA 内存分布
cat /proc/$(pgrep -f script.py)/numa_maps | head -20
```

### PyTorch 中指定 Pinned Memory 的 NUMA 位置

```python
import torch

# PyTorch 默认在调用线程的 NUMA 节点分配 pinned memory
# 确保 Python 线程运行在正确的 NUMA 节点上

# 或在 C++ 扩展中使用：
# cudaSetDeviceFlags(cudaDeviceMapHost);  // 启用 NUMA-aware pinned memory
# cudaHostAlloc(&ptr, size, cudaHostAllocPortable);  // 跨 NUMA 可访问
```

---

## 实战命令速查表

### perf 常用命令

```bash
# ─── 快速统计 ───
perf stat python script.py
perf stat -d python script.py                          # 详细统计
perf stat -ddd python script.py                        # 非常详细
perf stat -r 10 python script.py                       # 重复 10 次取平均
perf stat -I 1000 python script.py                     # 每秒输出

# ─── 事件过滤 ───
perf stat -e cycles,instructions,cache-misses python script.py
perf stat -e cycles:u,instructions:u python script.py  # 仅用户态
perf stat -e "syscalls:sys_enter_*" python script.py   # 跟踪系统调用

# ─── 热点函数 ───
perf record -g python script.py && perf report
perf record -F 99 -g python script.py                  # 99 Hz 采样
perf record --call-graph dwarf python script.py        # DWARF 调用图

# ─── 实时监控 ───
perf top
perf top -p $(pgrep -f script.py)
perf top -e cycles:u

# ─── 内存分析 ───
perf mem record python script.py && perf mem report
perf stat -e page-faults,major-faults python script.py

# ─── 调度分析 ───
perf sched record python script.py && perf sched latency

# ─── Tracepoint 追踪 ───
perf list tracepoint                                    # 列出所有追踪点
perf record -e sched:sched_switch python script.py     # 追踪上下文切换

# ─── Scripting（将 perf.data 转为可读文本） ───
perf script > perf_output.txt
perf script -F comm,pid,tid,cpu,time,event,ip,sym,dso,trace
```

### Intel VTune 常用命令

```bash
# ─── 热点分析 ───
vtune -collect hotspots -- python script.py
vtune -report summary -r <result_dir>

# ─── 内存访问分析 ───
vtune -collect memory-access -- python script.py

# ─── 微架构探索 ───
vtune -collect uarch-exploration -- python script.py

# ─── 输入输出分析 ───
vtune -collect io -- python script.py

# ─── HPC 性能分析（含 AVX-512 利用率） ───
vtune -collect hpc-performance -- python script.py

# ─── 线程分析 ───
vtune -collect threading -- python script.py
```

### numactl 常用命令

```bash
# ─── 拓扑信息 ───
numactl --hardware
lscpu | grep NUMA

# ─── 绑定运行 ───
numactl --cpunodebind=0 --membind=0 python script.py
numactl --cpunodebind=0-1 --membind=0-1 python script.py  # 跨两个 NUMA node

# ─── 查看分配 ───
numastat
numastat -c python
cat /proc/$(pgrep -f python)/numa_maps

# ─── 跨 NUMA 延迟测试 ───
numactl --hardware | grep distances
```
