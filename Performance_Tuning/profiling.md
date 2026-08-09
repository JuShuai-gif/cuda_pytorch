你现在是一名资深 Linux / C++ / CUDA / HPC / PyTorch / AI Inference / Jetson / Robot Runtime 性能工程师。

第一阶段的 Profiling 工具说明、基础示例代码和 Markdown 学习笔记已经创建完成。

当前所有 Profiling 相关代码和文档统一位于：

/home/ghr/code/cuda_pytorch/Performance_Tuning/profiling

现在进入第二阶段：

“真实性能瓶颈实验 + 性能诊断方法论 + Bad/Good优化对照”。

这一阶段不要继续简单介绍：

perf是什么
nsys是什么
ncu是什么
Valgrind怎么运行

第一阶段已经解决工具入门问题。

第二阶段必须围绕：

真实性能瓶颈
    ↓
如何构造/复现
    ↓
如何观测
    ↓
应该使用哪个工具
    ↓
看哪些指标
    ↓
如何判断Root Cause
    ↓
如何优化
    ↓
如何重新Benchmark
    ↓
如何证明优化真的有效

展开。

我的主要应用方向：

1. C++ 高性能计算 HPC
2. CUDA Kernel优化
3. PyTorch模型推理优化
4. Transformer / VLM / VLA
5. NVIDIA GPU / Jetson边缘部署
6. 机器人实时推理系统
7. Camera + Decode + Preprocess + CPU/GPU Transfer + Model + Action + ROS + Controller端到端性能分析
8. 长时间运行的机器人/VLA服务稳定性、Latency、Jitter、Power、Memory问题

==================================================
0. 第一原则：先检查现有仓库，不要盲目创建
==================================================

首先完整检查：

/home/ghr/code/cuda_pytorch/Performance_Tuning/profiling

包括：

find /home/ghr/code/cuda_pytorch/Performance_Tuning/profiling -maxdepth 4 -type f | sort

查看：

README
CMakeLists.txt
Makefile
cpp
cuda
python
src
note
docs
scripts

等现有目录。

必须先理解第一阶段已经创建了什么。

要求：

1. 不删除第一阶段已有文件。
2. 不覆盖有价值的已有内容。
3. 如果已有相同实验，优先扩展。
4. 不重复创建功能完全相同的Demo。
5. 不再创建新的独立profiling根目录。
6. 所有第二阶段内容必须继续放在：

   /home/ghr/code/cuda_pytorch/Performance_Tuning/profiling

7. 不修改该目录之外任何文件。
8. 不使用sudo。
9. 如果某工具不存在：
   - 不安装
   - 不导致整个任务失败
   - 仍然补全文档和命令
   - 标注“当前环境未验证”
10. 先制定修改计划，再执行。

==================================================
1. 第二阶段核心目标：性能问题驱动
==================================================

第一阶段是：

工具 → 命令 → 指标

第二阶段必须是：

问题 → 症状 → 指标 → 工具 → Root Cause → Optimization → Verification

例如：

程序运行慢
↓
CPU 100%
↓
perf stat发现IPC很低
↓
LLC Miss很高
↓
Memory Bandwidth接近上限
↓
判断可能Memory Bound
↓
检查数据访问方式
↓
发现随机访存
↓
改成连续访问
↓
重新perf stat
↓
比较：
runtime
IPC
LLC miss
bandwidth

我要最终建立：

“性能诊断能力”

而不仅仅是：

“Profiler工具使用能力”。

==================================================
2. 为所有实验建立统一结构
==================================================

先根据已有目录结构决定最合理方案。

推荐：

profiling/
├── README.md
├── ...
│
├── pathologies/
│   ├── cpu/
│   ├── memory/
│   ├── concurrency/
│   ├── syscall_io/
│   ├── cuda/
│   ├── pytorch/
│   ├── vla/
│   └── realtime/
│
└── note/
    └── pathologies/

但：

如果第一阶段已经有：

cpp/
cuda/
python/
note/

等成熟目录，

不要强制迁移。

可以继续在原目录中扩展。

重点不是目录名字。

重点是：

每一个性能问题必须：

1. 有实验代码
2. 有对应Markdown
3. 有Bad Case
4. 有Good Case
5. 有Profiler命令
6. 有指标解释
7. 有Before / After
8. 能从README快速找到

==================================================
3. 所有核心实验必须遵循统一实验范式
==================================================

每一个重要实验都尽量实现：

Bad Case
↓
Correctness Check
↓
Warmup
↓
Benchmark
↓
Profiler
↓
关键指标
↓
Root Cause
↓
Good Case
↓
Correctness Check
↓
Warmup
↓
Benchmark
↓
Profiler
↓
Before / After

不要只写一个：

slow.cpp

然后告诉我“这个比较慢”。

必须让Profiler能够真实捕获问题。

==================================================
4. CPU Hotspot
==================================================

创建或完善CPU Hotspot实验。

Bad Case：

设置：

hot_function()
medium_function()
cold_function()

让：

hot_function

明显占CPU时间，例如60%~80%。

避免编译器把计算完全优化掉。

可以使用：

volatile sink
返回值累计
输入依赖

等方式保证工作真实存在。

测试工具：

/usr/bin/time
perf stat
perf record
perf report
perf annotate
FlameGraph
VTune Hotspots（如果存在）

重点指标：

elapsed time
cycles
instructions
IPC
CPU utilization
samples
function percentage

必须解释：

perf report：

70% hot_function

真正代表什么。

不要简单理解成：

“函数单次调用需要70%的时间”。

而应该解释采样占比。

Good Case：

优化hot_function。

重新测试。

输出Before/After。

==================================================
5. Cache Miss / Cache Locality
==================================================

至少实现两个实验。

实验A：

连续访问
vs
随机访问

实验B：

二维数组：

row-major遍历
vs
column-major遍历

要求数据规模：

明显超过L1/L2，
尽量能够体现LLC/DRAM行为，

但不要占用过大内存。

工具：

perf stat
Cachegrind
VTune Memory Access

重点：

cache-references
cache-misses
LLC-loads
LLC-load-misses
L1相关事件（如果硬件支持）
cycles
instructions
IPC

必须计算：

Cache Miss Rate

并说明：

Spatial Locality
Temporal Locality

==================================================
6. TLB Miss / Page Fault
==================================================

增加：

大内存随机页访问实验。

解释：

Cache Miss
TLB Miss
Page Fault

三者完全不同。

重点指标：

page-faults
minor-faults
major-faults

如果perf list存在对应TLB event：

再增加TLB分析。

不要硬编码当前CPU不存在的event。

先：

perf list

检测。

联系：

Large Tensor
KV Cache
模型权重
机器人长期运行buffer

==================================================
7. Branch Miss
==================================================

实现：

Predictable Branch
vs
Unpredictable Branch

例如：

规则数据

vs

预生成随机条件。

不要把随机数生成本身混入Benchmark核心区域。

应该提前生成输入。

指标：

branches
branch-misses
branch miss rate
cycles
IPC

说明：

Branch Predictor
Pipeline Flush
Speculative Execution

以及：

CPU Branch Miss

与：

CUDA Warp Divergence

不是一回事。

==================================================
8. Memory Bandwidth
==================================================

实现教学版STREAM：

Copy
Scale
Add
Triad

至少输出：

GB/s

比较：

single-thread
multi-thread

如果有OpenMP则使用。

工具：

perf
VTune
LIKWID
Intel PCM

按环境可用性选择。

结合Roofline：

Arithmetic Intensity = FLOPs / Bytes

解释：

为什么Vector Add通常Memory Bound。

==================================================
9. AoS vs SoA
==================================================

增加：

Array of Structures

vs

Structure of Arrays

例如：

Particle：

x
y
z
velocity

测试：

只处理x/y

时：

AoS
vs
SoA

观察：

cache
bandwidth
vectorization

联系：

PointCloud
Camera Pixel
Tensor
机器人状态
Particle Simulation

==================================================
10. SIMD / Vectorization
==================================================

建立：

Baseline
vs
Compiler Auto Vectorization

使用：

-O0
-O2
-O3

合理比较。

支持：

-march=native

但不要默认所有环境都支持相同ISA。

检查：

GCC vectorization report

例如：

-fopt-info-vec
-fopt-info-vec-missed

以及：

objdump

查看生成代码。

说明：

x86：

SSE
AVX
AVX2
AVX-512

ARM：

NEON
SVE

为什么这对：

HPC
Jetson CPU preprocessing
图像处理

重要。

==================================================
11. Frequent malloc/new
==================================================

Bad：

每个iteration：

malloc/free
new/delete
vector反复扩容

Good：

Pre-allocation
Buffer Reuse
std::vector::reserve
Memory Pool思想

工具：

heaptrack
Valgrind
perf

输出：

runtime
allocation count
allocated bytes
peak heap

重点联系VLA：

每帧不要重新创建：

image buffer
tensor buffer
state buffer
action buffer

==================================================
12. Memory Leak
==================================================

创建独立危险实验。

要求：

不能默认被run-all脚本执行。

必须手动执行。

Bad：

new后不delete

Good：

RAII
unique_ptr
vector

工具：

Valgrind Memcheck
ASan
LSan

解释：

definitely lost
indirectly lost
possibly lost
still reachable

==================================================
13. Use-After-Free / Buffer Overflow
==================================================

增加独立bug实验。

不能进入正常benchmark集合。

分别演示：

heap-use-after-free
heap-buffer-overflow

使用：

ASan
Valgrind

说明：

它们主要是正确性工具，

不是严格意义上的性能Profiler，

但优化高性能代码时非常重要。

==================================================
14. Memory Fragmentation
==================================================

创建：

大量不同大小allocation/free

的实验或者模拟。

解释：

Leak：

资源没有释放

Fragmentation：

资源可能已经释放，
但heap布局碎片化。

说明：

RSS不下降
并不一定意味着Memory Leak。

介绍：

jemalloc profiling
tcmalloc profiling

==================================================
15. Lock Contention
==================================================

这是重点。

Bad：

多个线程竞争：

std::mutex

并且：

Critical Section过大。

Good：

缩小Critical Section
局部计算移出锁
batch update

必要时比较：

mutex
shared_mutex

工具：

perf
VTune Locks and Waits
strace
bpftrace

重点观察：

futex
context switches
CPU utilization
thread waiting
spin/wait

必须解释：

为什么：

CPU利用率只有40%

程序却非常慢，

可能是：

大量线程都在等锁。

==================================================
16. shared_mutex读多写少实验
==================================================

增加一个实际：

Read-mostly

场景。

比较：

std::mutex

vs

std::shared_mutex
+
std::shared_lock
+
std::unique_lock

例如：

100个reader
1个writer

但必须说明：

shared_mutex

不是必然更快。

读写比例不合适、
critical section过短、
线程数少时，

它可能反而更慢。

==================================================
17. False Sharing
==================================================

必须实现清晰实验。

Bad：

struct Counters {
    std::atomic<uint64_t> a;
    std::atomic<uint64_t> b;
};

两个线程分别修改a/b。

Good：

alignas(64)

或者：

std::hardware_destructive_interference_size

如果可用。

解释：

同一个Cache Line
↓
不同CPU Core
↓
MESI / Cache Coherence
↓
Cache Line Ping-Pong

必须强调：

False Sharing != Data Race

==================================================
18. Thread Load Imbalance
==================================================

实现：

多个线程：

thread0处理70%
其他线程处理剩余任务

vs

均衡partition。

观察：

total runtime
per-thread runtime
CPU utilization
timeline

工具：

pidstat
perf
VTune
nsys CPU timeline

==================================================
19. Context Switch / CPU Migration
==================================================

实现多线程压力实验。

观察：

context-switches
cpu-migrations

演示：

taskset

如果NUMA存在：

numactl

但不要修改系统全局配置。

解释：

线程迁移为什么可能破坏：

cache locality

以及：

Realtime Jitter。

==================================================
20. NUMA
==================================================

先检查：

lscpu
numactl --hardware
numastat

如果系统是多NUMA节点：

设计：

Local Memory
vs
Remote Memory

实验。

如果不是：

只补文档。

重点：

CPU pinning
Memory binding
First Touch Policy

HPC必须重点解释。

==================================================
21. Syscall Overhead
==================================================

Bad：

大量：

read
write
open/close

Good：

batch
buffering
减少syscall次数

工具：

strace
strace -c
strace -T
strace -f

指标：

calls
errors
total time
time/call

重点解释：

read
write
openat
futex
poll
epoll_wait

分别可能说明什么。

==================================================
22. Disk IO
==================================================

创建安全的小型实验。

不要写GB级垃圾数据。

限制例如：

64MB~256MB

根据磁盘环境调整。

测试：

Small IO
vs
Large Sequential IO

工具：

fio
iostat
iotop

重点：

IOPS
Bandwidth
Latency
Queue Depth
iowait

==================================================
23. CUDA Kernel Hotspot
==================================================

进入GPU部分。

原则：

nsys：

“哪里慢”

ncu：

“为什么慢”

创建一个包含多个Kernel的程序：

kernel_fast
kernel_medium
kernel_slow

使用NVTX标记：

preprocess
kernel_A
kernel_B
postprocess

使用：

nsys profile

找到热点。

然后：

ncu

只分析最慢Kernel。

==================================================
24. GPU Memory Bound
==================================================

实现低Arithmetic Intensity Kernel：

Vector Add
AXPY
Elementwise

重点：

DRAM Throughput
Memory Throughput
SM Throughput
Bytes
FLOPs

结合Roofline分析。

不要简单地说：

Memory Throughput高 = Memory Bound。

应该结合：

Arithmetic Intensity
Compute utilization
Memory utilization

一起判断。

==================================================
25. GPU Compute Bound
==================================================

实现高Arithmetic Intensity Kernel。

可以选择：

Repeated FMA
小型Matmul
Dense Computation

如果使用Tensor Core：

单独说明。

重点：

SM throughput
instruction throughput
Tensor Core
DRAM throughput

和Memory Bound实验对照。

==================================================
26. Uncoalesced Global Memory Access
==================================================

Bad：

Warp内线程跨stride访问。

Good：

连续线程连续地址。

使用：

ncu

注意：

先通过：

ncu --query-metrics

或当前版本等价方式，

确认metric名称。

不要复制旧版本指标名字导致命令失效。

重点：

Global Memory Load/Store
Memory Transactions
L1/L2
DRAM

==================================================
27. Shared Memory Bank Conflict
==================================================

Bad：

多个线程映射同一个bank模式。

Good：

Padding消除冲突。

解释：

Bank
Warp
Address Mapping
32-bank常见结构

但不要硬编码所有GPU架构细节完全相同。

使用：

ncu

分析shared memory相关metrics和warp stalls。

==================================================
28. Warp Divergence
==================================================

Bad：

同一Warp中：

if(threadIdx.x % 2)

不同路径执行复杂工作。

Good：

调整数据/线程映射
减少Warp内部路径分裂。

使用：

ncu

重点看：

branch
warp execution
scheduler
stall

不要使用不存在的legacy metric。

==================================================
29. Low Occupancy
==================================================

至少制造一种：

高register usage

以及可选：

高shared memory usage。

观察：

Theoretical Occupancy
Achieved Occupancy
Registers Per Thread
Shared Memory Per Block
Active Warps

必须解释：

Occupancy 100%

不代表最快。

Occupancy 50%

也不代表有问题。

真正要问：

Latency Hiding是否不足？

==================================================
30. CUDA Kernel Launch Overhead
==================================================

Bad：

循环中执行大量tiny kernel。

Good：

Kernel Fusion

扩展：

CUDA Graph

使用：

nsys

看：

CUDA API
Kernel Launch
GPU Gap
CPU Launch

联系：

Transformer里大量小elementwise kernel。

==================================================
31. H2D / D2H
==================================================

Bad：

每一步：

Host → Device
Device → Host

Good：

Device Resident Buffer

进一步实现：

Pinned Memory
cudaMemcpyAsync
Streams

使用：

nsys

观察：

HtoD
DtoH
Duration
Overlap

==================================================
32. CPU-GPU Synchronization
==================================================

Bad：

循环中：

cudaDeviceSynchronize()

Good：

减少全局同步。

需要说明：

cudaDeviceSynchronize
cudaStreamSynchronize
cudaEventSynchronize

粒度差异。

使用：

nsys

让Timeline清晰看到：

CPU Wait
GPU Bubble

==================================================
33. CUDA Pipeline Overlap
==================================================

实现：

Serial：

H2D
↓
Kernel
↓
D2H

vs

Pipeline：

Pinned Memory
+
cudaMemcpyAsync
+
Streams
+
Double Buffer

验证：

是否真的发生Overlap。

必须强调：

async API

不等于：

实际并发执行。

必须从nsys timeline证明。

==================================================
34. PyTorch Operator Hotspot
==================================================

创建一个小模型或计算图。

至少包含：

Linear
Matmul
Softmax
LayerNorm/RMSNorm
Attention-like computation

使用：

torch.profiler

重点：

Self CPU Time
CPU Time Total
Self CUDA Time
CUDA Time Total

找到Top Operators。

然后：

torch.profiler
↓
nsys
↓
ncu

建立三级分析流程。

==================================================
35. PyTorch GPU Timing
==================================================

必须单独创建实验：

错误：

start = time.time()
model(x)
end = time.time()

正确：

torch.cuda.synchronize()

以及：

torch.cuda.Event

解释：

CUDA asynchronous execution。

==================================================
36. PyTorch Memory
==================================================

分析：

torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()

如果版本支持：

memory snapshot

实验：

大量temporary tensor
clone
contiguous
中间buffer

Good：

buffer reuse
合理in-place
避免不必要clone

不能破坏Autograd/Correctness语义。

==================================================
37. Transformer典型Kernel分析
==================================================

增加教学文档或PyTorch实验，至少讨论：

GEMM
RMSNorm
Softmax
Attention
SwiGLU
Elementwise
KV Cache

建立：

Kernel
→
常见瓶颈类型

例如：

GEMM：
偏Compute Bound

RMSNorm：
常偏Memory Bandwidth / Reduction

Softmax：
Reduction + Memory

KV Cache：
Memory Bandwidth / Capacity

但必须注明：

具体Bound需要根据：

shape
dtype
hardware
implementation

Profiler判断，

不能死记。

==================================================
38. VLA End-to-End Mock Pipeline
==================================================

这是整个第二阶段最重点实验之一。

创建：

Mock VLA Pipeline

阶段：

Camera Capture
↓
Video Decode
↓
Resize
↓
Normalize
↓
H2D
↓
Vision Encoder
↓
Projector
↓
LLM/VLM Backbone
↓
Action Head
↓
Action Decode
↓
ROS Publish Mock
↓
Robot Controller Mock

要求：

每个stage都能够单独计时。

输出每帧：

capture
decode
preprocess
H2D
vision
projector
language
action
control
E2E

至少统计：

Mean
Median
P50
P90
P95
P99
Min
Max
StdDev
FPS
Control Hz

必须有：

Warmup。

==================================================
39. VLA Pipeline Bubble
==================================================

创建：

Serial Pipeline

Camera
↓
CPU Preprocess
↓
GPU Inference
↓
CPU Action

然后：

Producer / Consumer
Double Buffer
Async GPU
Pipeline Overlap

建立Timeline。

重点理解：

单个stage降低5ms

不一定比：

CPU/GPU并行

收益更大。

==================================================
40. Camera / Decode / Preprocess
==================================================

可以通过Mock实现，

同时写真实工程说明。

覆盖：

V4L2
NVDEC
MPP
RGA
OpenCV CPU Resize
CUDA Preprocess

重点指标：

Capture FPS
Frame Drop
Decode Latency
Preprocess Latency
CPU utilization
GPU utilization

必须说明：

GPU Util低

有时候不是GPU程序差，

可能是：

Camera / Decode / CPU preprocess

喂不饱GPU。

==================================================
41. VLA阶段级NVTX
==================================================

对CUDA/PyTorch Mock VLA增加：

NVTX Range

例如：

capture
decode
preprocess
vision_encoder
projector
llm
action_head
control

让：

nsys

可以直接看到整个Pipeline。

==================================================
42. Robot Realtime Control Jitter
==================================================

创建一个C++固定周期控制循环模拟器。

例如：

50Hz

Period：

20ms。

记录：

scheduled wakeup
actual wakeup
task begin
task end

计算：

Wake-up Latency
Execution Time
Period
Jitter
Deadline Miss

输出：

Mean
P50
P90
P95
P99
Max
Deadline Miss Count
Deadline Miss Ratio

加入可选干扰：

CPU Load
Lock Contention
Memory Pressure

==================================================
43. Linux Scheduler / Realtime
==================================================

介绍并按环境使用：

cyclictest
perf
ftrace
trace-cmd
bpftrace

重点：

scheduler latency
wake-up latency
context switch
IRQ
softirq

解释：

机器人：

Average 10ms

P99 35ms

如果Deadline：

20ms

仍然不合格。

==================================================
44. Power / Thermal / Frequency
==================================================

边缘端必须重点补充。

特别针对Jetson：

tegrastats
jtop

指标：

GR3D
EMC
CPU
RAM
SWAP
Power
Temperature
Clock

解释：

为什么：

刚启动：

50ms

运行20分钟：

70ms

可能来自：

Thermal Throttling
Power Limit
Clock Drop

而不是代码逻辑变化。

==================================================
45. Jetson Memory Bandwidth
==================================================

单独重点解释：

EMC

因为Jetson：

CPU
GPU
Video Codec
ISP
其他模块

共享内存系统。

因此：

GPU kernel
Video Decode
CPU preprocess

可能争抢Memory Bandwidth。

联系：

VLA pipeline。

==================================================
46. ROS1 / ROS2性能
==================================================

覆盖：

ROS1：

rostopic hz
rostopic bw

ROS2：

ros2 topic hz
ros2 topic bw
ros2_tracing
tracetools

重点：

Serialization
Deserialization
Copy
DDS
Shared Memory
Executor
Callback
Queue
Drop

尤其：

sensor_msgs/Image
PointCloud2

大消息。

建立：

Camera Publish
↓
Transport
↓
Callback
↓
Inference
↓
Action Publish
↓
Control Callback

端到端分析思路。

==================================================
47. ROS2 Shared Memory
==================================================

增加概念说明：

普通DDS传输

vs

Shared Memory Transport

重点：

Copy
Serialization
Latency
Large Message

但不要把：

Shared Memory

等同于：

完全Zero-Copy。

必须区分：

Shared Memory Transport
Loaned Message
Zero-Copy

==================================================
48. 性能实验文档
==================================================

在：

/home/ghr/code/cuda_pytorch/Performance_Tuning/profiling

现有note/docs体系下，

创建一个明确的：

性能瓶颈实验

目录。

具体路径根据第一阶段结构决定。

不要再次建立第二个不必要的note根目录。

建议包含：

01_CPU热点.md
02_Cache_Miss.md
03_TLB_PageFault.md
04_Branch_Miss.md
05_Memory_Bandwidth.md
06_AoS_vs_SoA.md
07_SIMD_Vectorization.md
08_Frequent_Allocation.md
09_Memory_Leak.md
10_Memory_Fragmentation.md
11_Lock_Contention.md
12_shared_mutex读多写少.md
13_False_Sharing.md
14_Thread_Imbalance.md
15_Context_Switch_CPU_Migration.md
16_NUMA.md
17_Syscall_Overhead.md
18_Disk_IO.md

20_CUDA_Kernel_Hotspot.md
21_GPU_Memory_Bound.md
22_GPU_Compute_Bound.md
23_Uncoalesced_Access.md
24_Shared_Memory_Bank_Conflict.md
25_Warp_Divergence.md
26_Occupancy.md
27_Kernel_Launch_Overhead.md
28_H2D_D2H.md
29_CPU_GPU_Synchronization.md
30_CUDA_Pipeline_Overlap.md

31_PyTorch_Operator_Hotspot.md
32_PyTorch_GPU_Timing.md
33_PyTorch_Memory.md
34_Transformer_Kernel_Performance.md

40_VLA_End_to_End_Latency.md
41_VLA_Pipeline_Bubble.md
42_Camera_Decode_Preprocess.md
43_VLA_NVTX_Nsys.md

50_Robot_Realtime_Jitter.md
51_Linux_Scheduler_Latency.md
52_Power_Thermal_Throttling.md
53_Jetson_EMC.md
54_ROS_ROS2_Performance.md

==================================================
49. 每篇性能问题Markdown统一模板
==================================================

所有重要文档统一包含：

# 性能问题

## 1. 问题是什么

## 2. 底层原理

## 3. 工程中通常表现为什么症状

## 4. Bad Case

对应代码路径。

## 5. 如何Benchmark

## 6. 第一层观测工具

例如：

top
pidstat
nvidia-smi
tegrastats

## 7. 定位工具

例如：

perf
torch.profiler
nsys

## 8. 深入分析工具

例如：

VTune
ncu
Cachegrind
bpftrace

## 9. 最重要指标

| 指标 | 含义 | 高/低意味着什么 | 能否单独下结论 |

## 10. 实际Profiling命令

命令必须可以复制。

## 11. 如何阅读结果

不要只告诉我：

“IPC低”。

必须建立指标组合推理。

例如：

IPC低
+
LLC Miss高
+
Memory Bandwidth高
+
Front/Backend Memory Stall高

→ Memory Bound概率较高。

## 12. Root Cause

## 13. Optimization

## 14. Good Case

## 15. Correctness验证

## 16. Before vs After

| 指标 | Before | After | 变化 |

## 17. 常见误判

## 18. HPC应用

## 19. CUDA应用

## 20. PyTorch/VLA应用

## 21. Jetson/机器人应用

==================================================
50. 建立“症状 → 指标 → 工具 → 原因”诊断表
==================================================

创建一个核心文档：

性能症状诊断表.md

至少覆盖：

CPU 100%
CPU低但程序慢
IPC低
Cache Miss高
Branch Miss高
Context Switch高
CPU Migration高
RSS增长
Malloc频繁
Futex很多
IOwait高

GPU Util低
GPU Util高但Latency高
GPU Timeline空洞
Kernel时间长
DRAM Throughput高
SM Throughput高
Occupancy低
Warp Stall高
Memcpy占比高
Kernel数量非常多

PyTorch CPU时间高
PyTorch CUDA算子热点
显存峰值高

VLA Mean正常但P99高
Camera FPS不足
Decode慢
Preprocess慢
GPU等待CPU
Control Deadline Miss

每一个症状都给出：

第一工具
↓
第二工具
↓
第三工具

==================================================
51. 建立Profiler决策树
==================================================

创建：

Profiler决策树.md

CPU：

程序慢
↓
/usr/bin/time
↓
top / pidstat
↓
perf stat
↓
perf record
↓
FlameGraph
↓
VTune


Memory：

RSS异常
↓
heaptrack
↓
Valgrind / ASan


Syscall：

sys CPU高
↓
strace
↓
perf


GPU：

GPU程序慢
↓
nvidia-smi / tegrastats
↓
nsys
↓
找到Kernel
↓
ncu
↓
Roofline


PyTorch：

torch.profiler
↓
nsys
↓
ncu


VLA：

Stage Timer
↓
NVTX
↓
nsys
↓
Kernel热点
↓
ncu
↓
P50/P90/P99


Realtime：

Latency Histogram
↓
cyclictest
↓
perf
↓
ftrace
↓
bpftrace

==================================================
52. 建立性能指标字典
==================================================

创建：

性能指标字典.md

至少覆盖：

CPU：

cycles
instructions
IPC
CPI
cache reference
cache miss
L1
L2
LLC
branch
branch miss
TLB
page fault
context switch
CPU migration

Memory：

RSS
VSZ
allocation count
allocated bytes
peak heap
bandwidth
latency

Disk：

IOPS
throughput
latency
queue depth
iowait

GPU：

GPU Utilization
SM Throughput
DRAM Throughput
L1
L2
Registers
Shared Memory
Occupancy
Active Warps
Eligible Warps
Warp Stall
Kernel Duration
Memcpy Duration

PyTorch：

Self CPU
CPU Total
Self CUDA
CUDA Total
Allocated
Reserved
Peak Memory

Latency：

Mean
Median
P50
P90
P95
P99
Max
StdDev
Jitter

Realtime：

Period
Deadline
Deadline Miss
Wake-up Latency
Execution Time

每一个指标必须说明：

定义
如何计算
怎么看
高代表什么
低代表什么
不能单独说明什么

==================================================
53. 建立“指标组合推理”
==================================================

创建：

性能指标组合推理.md

这是重点。

至少包含：

情况1：

IPC低
LLC Miss高
DRAM高

→ Memory Bound候选

情况2：

IPC低
Branch Miss高
DRAM不高

→ Branch / frontend问题候选

情况3：

CPU Util低
futex高
Context Switch高

→ Lock Contention候选

情况4：

GPU Util低
nsys GPU大量空洞

→ CPU Feeding / Sync / Launch / IO候选

情况5：

Memory Throughput高
SM低
Arithmetic Intensity低

→ GPU Memory Bound候选

情况6：

SM高
DRAM低
Arithmetic Intensity高

→ Compute Bound候选

情况7：

Occupancy低
Long Scoreboard高
Eligible Warps不足

→ Latency hiding可能不足

情况8：

Mean很好
P99很差

→ Tail Latency / Scheduler / Contention问题

必须强调：

性能分析通常不是看单一指标，

而是：

多个证据组合。

==================================================
54. Benchmark必须科学
==================================================

所有Benchmark统一考虑：

Warmup
Iteration
Compiler optimization
CPU frequency
CPU affinity
GPU warmup
CUDA asynchronous execution
Cache warm/cold
Thread count
Input size
Tensor shape
Batch
Sequence length
dtype

统计：

Mean
Median
P50
P90
P95
P99
Min
Max
StdDev

不能：

只运行一次。

==================================================
55. 防止编译器优化掉实验
==================================================

性能教学实验必须检查：

Dead Code Elimination
Constant Folding
Loop Elimination

避免：

计算结果从未使用。

可以合理使用：

volatile sink
DoNotOptimize思想
输入运行时生成
输出checksum

确保：

Bad/Good差异来自真实性能问题，

而不是：

一个版本被编译器优化没了。

==================================================
56. Correctness优先
==================================================

任何Bad/Good优化比较：

必须先验证结果一致。

例如：

CPU数组：

checksum

CUDA：

cudaMemcpy结果回来比较

PyTorch：

torch.testing.assert_close

禁止：

通过减少工作量

制造“优化”。

==================================================
57. 自动化脚本
==================================================

根据已有scripts结构，

增加：

run_cpu_pathologies.sh
run_memory_pathologies.sh
run_concurrency_pathologies.sh
run_cuda_pathologies.sh
run_pytorch_pathologies.sh
run_realtime_pathologies.sh

要求：

1. command -v检查工具
2. CUDA不存在时跳过GPU
3. PyTorch不存在时跳过PyTorch
4. 不自动运行危险memory bug程序
5. 不自动运行长时间磁盘测试
6. 不使用sudo

==================================================
58. Build系统
==================================================

检查第一阶段已有：

CMakeLists.txt
Makefile

优先扩展已有Build System。

不要重新建立另一套相互冲突的构建系统。

要求：

CPU-only环境：

仍然可以编译CPU实验。

CUDA不存在：

只跳过CUDA Targets。

OpenMP不存在：

只跳过相关Target。

Build：

Release优先。

建议：

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

根据实际目录结构调整。

==================================================
59. 代码质量
==================================================

C++：

C++17或第一阶段项目已有标准。

CUDA：

独立清晰。

Python：

PEP8不强制，但代码必须易读。

注释：

中文为主。

每一个实验文件顶部说明：

Purpose
Bad Case
Good Case
Recommended Profiler

==================================================
60. 重点联系我的实际VLA部署工作
==================================================

所有重要性能问题的Markdown最后增加：

“在VLA/机器人中的表现”。

例如：

Frequent malloc：

每帧创建Tensor/Buffer
↓
Latency Jitter
↓
Heap Pressure

CPU Preprocess：

GPU空闲等待
↓
GPU Util低

H2D：

CPU/GPU频繁搬数据
↓
Pipeline不能Overlap

Kernel Launch：

大量小Elementwise Kernel
↓
GPU timeline碎片化

Memory Bound：

RMSNorm / Activation / KV Cache
↓
DRAM压力

Lock Contention：

Camera线程
Inference线程
Control线程
共享队列锁竞争

Scheduler Jitter：

50Hz control loop
↓
20ms deadline
↓
P99 > 20ms
↓
控制不稳定

==================================================
61. 性能瓶颈优先级
==================================================

如果任务量太大，

不要通过生成大量低质量文件一次完成。

按照以下优先级逐步完成：

P0：

CPU Hotspot
Cache Miss
Branch Miss
Memory Bandwidth
Frequent Allocation
Lock Contention
False Sharing

CUDA Kernel Hotspot
GPU Memory Bound
GPU Compute Bound
Uncoalesced Access
Warp Divergence
Occupancy
Launch Overhead
H2D/D2H
CPU-GPU Sync
Pipeline Overlap

PyTorch Operator Hotspot

VLA E2E Latency
Realtime Jitter

P1：

TLB
NUMA
SIMD
Memory Fragmentation
Disk IO
shared_mutex
PyTorch Memory
Camera Decode
Power/Thermal
ROS

P2：

其他扩展实验。

优先保证P0实验质量。

==================================================
62. 每完成一个实验都必须验证
==================================================

完成实验后：

1. 编译。
2. 运行。
3. 检查Correctness。
4. 检查Bad/Good性能是否存在合理差异。
5. 如果差异不明显：
   - 调整数据规模
   - iteration
   - workload
6. 防止compiler optimization。
7. 不为了制造差异使用不公平参数。

==================================================
63. Profiler实际验证
==================================================

如果当前环境存在：

perf
strace
valgrind
heaptrack
nsys
ncu

选择安全的小型实验真实运行。

例如：

perf stat ./cache_miss

不要只把命令写进Markdown，

至少对核心工具跑一次真实验证。

如果：

perf权限受限

记录：

perf_event_paranoid

但不要sudo修改。

==================================================
64. README最终索引
==================================================

完善：

/home/ghr/code/cuda_pytorch/Performance_Tuning/profiling

现有README。

建立核心表格：

| 性能问题 | Demo | 第一工具 | 深入工具 | 关键指标 |

例如：

CPU Hotspot
→ CPU hotspot demo
→ perf record
→ FlameGraph / VTune
→ samples / cycles

Cache Miss
→ cache demo
→ perf stat
→ Cachegrind / VTune
→ cache miss / IPC

Lock Contention
→ lock demo
→ perf / strace
→ VTune
→ futex / wait

GPU Kernel
→ CUDA demo
→ nsys
→ ncu
→ kernel duration / SM / DRAM

Memory Bound
→ vector demo
→ ncu
→ Roofline
→ DRAM / AI

PyTorch
→ torch demo
→ torch.profiler
→ nsys/ncu

VLA
→ mock pipeline
→ stage timer/NVTX
→ nsys
→ P50/P90/P99

Realtime
→ control loop
→ histogram
→ cyclictest/ftrace

==================================================
65. 最终学习路线
==================================================

最终给我生成：

学习路线.md

推荐顺序：

Level 1：
CPU Hotspot
perf stat
perf record

Level 2：
Cache
Branch
Memory Bandwidth

Level 3：
Allocation
Valgrind
Sanitizer
strace

Level 4：
Thread
Lock
False Sharing
NUMA

Level 5：
nsys

Level 6：
ncu
Memory Bound
Compute Bound
Warp
Occupancy

Level 7：
torch.profiler

Level 8：
VLA End-to-End

Level 9：
Realtime / Jetson / ROS

==================================================
66. 最终验收
==================================================

全部完成后：

1. 输出最终目录树。
2. 输出新增文件。
3. 输出修改文件。
4. 输出编译成功Target。
5. 输出编译失败Target及原因。
6. 输出实际执行过的Profiler。
7. 输出当前环境不存在的Profiler。
8. 输出哪些实验Bad/Good差异明显。
9. 输出哪些实验还需要在Jetson/GPU机器验证。
10. 输出推荐下一步学习顺序。

==================================================
67. 最终目标
==================================================

最终这个目录：

/home/ghr/code/cuda_pytorch/Performance_Tuning/profiling

必须成为一个：

Performance Engineering Lab

而不是一个：

Profiler命令大全。

我要能够从：

“程序慢”

逐步判断成：

CPU Hotspot？
Cache Miss？
Branch Miss？
Memory Bandwidth？
NUMA？
Malloc？
Lock？
False Sharing？
Syscall？
IO？
Scheduler？
GPU Kernel？
GPU Memory Bound？
GPU Compute Bound？
Warp Divergence？
Occupancy？
Launch Overhead？
H2D/D2H？
CPU-GPU Sync？
PyTorch Operator？
Pipeline Bubble？
Thermal？
Realtime Jitter？

然后知道：

第一步看什么指标，
第二步用什么Profiler，
第三步如何确认Root Cause，
第四步如何优化，
第五步如何证明优化有效。

现在开始：

第一步先扫描：

/home/ghr/code/cuda_pytorch/Performance_Tuning/profiling

已有文件和目录，

总结第一阶段现状，

制定第二阶段修改计划，

然后再开始创建和修改代码。