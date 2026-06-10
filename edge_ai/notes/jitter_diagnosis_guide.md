# Jitter 诊断指南

## 快速开始

```bash
# 编译（纯 C++，无 CUDA 依赖）
g++ -std=c++17 -O2 -pthread -o test_jitter test_jitter.cpp

# 运行所有测试
./test_jitter

# 自定义参数
./test_jitter --noise 16 --contention 8 --duration 10 --iters 500

# 查看帮助
./test_jitter --help
```

---

## 6 类 Jitter 来源的判断方法

### 1. OS 调度 Jitter（CFS 时间片抢占）

#### test_jitter 表现

```
[OS Scheduling] p99-p50 gap: 3,944,330 ns  ⚠ SEVERE tail latency (p99/mean=3.2x)
```

如果 p99 远大于 p50（> 3x），说明存在严重的调度抖动。

#### 系统工具确认

```bash
# 查看测试期间的上下文切换次数
perf stat -e context-switches ./test_jitter --duration 5

# 查看调度延迟分布
perf sched record ./test_jitter --duration 5
perf sched latency   # 列出最大调度延迟的任务

# 查看线程被抢占的频率
perf stat -e sched:sched_switch ./test_jitter --duration 5

# 查看 CPU 上的中断分布
cat /proc/interrupts | head -20
```

#### 判断标准

| 现象 | 诊断结论 |
|------|----------|
| `perf sched latency` 某个任务最大延迟 > 1ms | CFS 调度器抢占导致 |
| `context-switches` > 10k/s | 线程过多，调度开销大 |
| p50 正常但 p99 突增 3x+ | 定时器中断（tick）或 IO 中断抢占 |

#### 解决方案

```bash
# 1. 绑核：将关键线程绑定到隔离 CPU
taskset -c 2,3 ./your_app

# 2. 实时调度策略
sudo chrt --fifo 80 ./your_app

# 3. 关闭 CPU 变频
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 4. 隔离 CPU（从内核调度器中移除）
# 在 grub 启动参数中添加：isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
```

---

### 2. 缓存/TLB 未命中 Jitter

#### test_jitter 表现

```
[Cache] Sequential DRAM-size (cache miss)
  per-access mean: X ns (取决于是否命中 L3)
[TLB]   Stride=512 (TLB miss every access)
  per-access mean: Y ns (Y >> X 表示 TLB miss)
```

较大的 stride 导致每次访问跳过一个 TLB 页（4KB），触发 TLB miss。

#### 系统工具确认

```bash
# 统计各级缓存未命中率
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./test_jitter

# 统计 TLB 未命中
perf stat -e dTLB-loads,dTLB-load-misses ./test_jitter

# 查看具体哪行代码导致缓存未命中
perf record -e cache-misses -g ./your_app
perf report
```

#### 判断标准

| 现象 | 诊断结论 |
|------|----------|
| `L1-dcache-load-misses` > 10% | 数据布局差，L1 未命中过多 |
| `LLC-load-misses` > 50% | 数据超出 L3 容量，频繁访 DRAM |
| `dTLB-load-misses` > 1% | 访问模式跳跃导致 TLB miss |
| per-access 延迟 > 50ns | 访问大概率到了 DRAM |

#### 解决方案

```
1. 数据重排：Struct of Arrays → Array of Structs，减少 padding
2. 预取：__builtin_prefetch(addr, 0, 3) 提前加载
3. 使用大页（2MB/1GB huge pages），减少 TLB miss
   echo 128 | sudo tee /proc/sys/vm/nr_hugepages
4. 数据对齐到 cache line 边界（64 字节）
   alignas(64) struct { ... };
```

---

### 3. 内存带宽竞争 Jitter

#### test_jitter 表现

```
[Memory Bandwidth] 16 contender threads
  slow-down factor:   6.69x
  contended stddev:   65988 ns (increased = jitter from contention)
```

如果 contended stddev 明显大于 solo stddev，说明多线程竞争 DRAM 带宽。

#### 系统工具确认

```bash
# 查看内存带宽使用
perf stat -e uncore_imc/data_reads/,uncore_imc/data_writes/ ./your_app

# Intel PCM 查看详细内存带宽（需要安装）
pcm-memory 1  # 每秒刷新一次

# 查看 DRAM 延迟分布（Intel MLC）
mlc --latency_matrix

# 简单方法：用 stress-ng 模拟竞争
stress-ng --vm 4 --vm-bytes 2G &
./test_jitter --duration 10
kill %1
```

#### 判断标准

| 现象 | 诊断结论 |
|------|----------|
| `slow-down` > 2x with N threads | 内存带宽瓶颈 |
| contended stddev > 5x solo stddev | 内存访问时序被其他核心打乱 |
| pcm 显示 > 80% 带宽利用率 | 内存带宽饱和 |

#### 解决方案

```
1. 减少同时访 DRAM 的线程数（错峰）
2. 使用 NUMA 感知分配，数据尽量在 local node
   numactl --membind=0 --cpunodebind=0 ./your_app
3. 批量处理 + 预取，提高每次内存访问的效率
4. 压缩数据：FP32 → FP16 推理，减少一半带宽压力
```

---

### 4. 中断（IRQ）Jitter

#### test_jitter 表现

```
[IRQ Detection] Monitoring for 3s
  spikes > 10us: 16 (5.3 per million iterations)
  max spike: 94.2 us
```

如果每条百万次迭代有多次 > 10us 的尖刺，说明有中断抢占。

#### 系统工具确认

```bash
# 查看各 CPU 的中断计数
watch -n1 cat /proc/interrupts

# 查看各中断类型的频率
cat /proc/irq/*/smp_affinity

# 查看 timer 中断频率
grep CONFIG_HZ /boot/config-$(uname -r)

# 实时追踪中断处理延迟
perf top -e irq:irq_handler_entry

# 使用 ftrace 追踪 > 10us 的中断
echo 10 > /sys/kernel/debug/tracing/tracing_thresh
echo 1 > /sys/kernel/debug/tracing/events/irq/enable
cat /sys/kernel/debug/tracing/trace_pipe
```

#### 判断标准

| 现象 | 诊断结论 |
|------|----------|
| spike > 10us 频率 > 1000/百万次 | 中断风暴（NIC 或磁盘 IO 过多） |
| spike 集中在特定 CPU | 该 CPU 未做 IRQ affinity 隔离 |
| spike 呈周期性（如 250Hz / 1000Hz） | 内核时钟中断（tick）抢占 |

#### 解决方案

```bash
# 1. 将中断导向非关键 CPU
echo 2 > /proc/irq/NN/smp_affinity  # bitmask: CPU1

# 2. 查看哪些中断最频繁
cat /proc/interrupts | sort -rnk2 | head -10

# 3. 禁用无关中断（驱动/硬件）
# 例如关闭蓝牙、WiFi 等不用的设备

# 4. 使用低延迟内核
sudo apt install linux-lowlatency  # Ubuntu

# 5. 减少内核 tick 频率（noHZ）
# grub: nohz_full=2,3 rcu_nocbs=2,3
```

---

### 5. GPU 内核启动 Jitter

#### 测试方法

GPU 部分的测试需要 CUDA，单独编译：

```bash
nvcc -std=c++17 -arch=sm_75 -o test_gpu_jitter test_gpu_jitter.cu
```

#### test_jitter 表现

```
[GPU Launch] 100 iterations
  mean:  8.4 us
  stddev: 2.3 us
  p99:   18.2 us
  max:   35.1 us
```

CUDA kernel launch 本身有 ~5-20us 开销，p99 跟 mean 差距大说明 GPU 驱动内部有锁等待。

#### 系统工具确认

```bash
# Nsight Systems 时间线（最直观）
nsys profile ./your_cuda_app

# 查看 CUDA API 调用延迟
nsys nvprof --print-api-trace ./your_cuda_app

# 统计 launch latency
perf stat -e nvidia/cuda-kernel-launch/ ./your_cuda_app
```

#### 判断标准

| 现象 | 诊断结论 |
|------|----------|
| p99 launch > 50us | CUDA 驱动内部队列锁竞争 |
| launch 延迟随时间递增 | GPU 显存碎片导致分配变慢 |
| launch 延迟每隔 N 次突增 | 显存 defrag / GC 操作 |

#### 解决方案

```cpp
// 1. 使用 CUDA Graph（一次 capture，多次 replay）—— 消除 launch overhead
cudaGraphLaunch(graph_exec, stream);

// 2. 使用 cudaStream 批量提交，减少 driver 交互频率

// 3. 预分配 CUDA memory pool，避免运行时 cudaMalloc 进入驱动
cudaMemPool_t pool;
cudaDeviceGetMemPool(&pool, device);
cudaMallocAsync(&ptr, size, stream);  // 从 pool 分配

// 4. 关闭 ECC（仅测试环境）
nvidia-smi -e 0
```

---

### 6. CPU 频率调节 / 热降频 Jitter

#### test_jitter 表现

```
[CPU Frequency] cycle variation: 22.9% ⚠ DVFS / thermal throttling may be active
```

RDTSC 周期数波动 > 5% 说明 CPU 频率不稳定。

#### 系统工具确认

```bash
# 实时监控 CPU 频率
watch -n1 "cat /proc/cpuinfo | grep 'MHz'"

# 或使用 turbostat
sudo turbostat --quiet --show PkgWatt,GFXWatt,Bzy_MHz --interval 1

# 查看是否触发了 thermal throttling
cat /sys/class/thermal/thermal_zone*/temp

# 查看 CPU governor 是否用了 powersave
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### 判断标准

| 现象 | 诊断结论 |
|------|----------|
| governor = `powersave` 或 `ondemand` | 允许动态变频 → 延迟不稳 |
| Bzy_MHz 波动 > 200MHz | 频率在频繁切换 |
| 温度 > 90°C 时频率骤降 | 热降频（thermal throttling） |

#### 解决方案

```bash
# 1. 固定为最高频率
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 2. 锁定最高 P-state（Intel）
sudo cpupower frequency-set --max $(cpupower frequency-info -l | tail -1 | awk '{print $2}')

# 3. 禁用 Intel Turbo Boost（减少频率波动）
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# 4. 加强散热（硬件层面）
nvidia-smi -pm 1         # GPU 持久模式
nvidia-smi -ac 877,1530  # 锁定 GPU 频率
```

---

## 诊断流程总览

```
1. 运行 test_jitter → 输出报告
        ↓
2. 看哪个测试的 stddev 最大 / p99-p50 差距最大
        ↓
3. 对应该测试，用系统工具确认根因（perf / proc / tracing）
        ↓
4. 应用对应解决方案
        ↓
5. 重新运行 test_jitter → 对比改善幅度
```

### 快速判断口诀

| test_jitter 信号 | 先看什么 |
|------------------|----------|
| OS Scheduling p99/mean > 3x | `perf sched latency` + `cat /proc/interrupts` |
| Cache 某级访问异常慢 | `perf stat -e LLC-load-misses` |
| Memory Bandwidth slow-down > 2x | `pcm-memory` 或 `perf stat -e uncore_imc/` |
| IRQ spikes > 10us 频繁 | `watch cat /proc/interrupts` |
| CPU Frequency variation > 5% | `turbostat` + `cpufreq governor` |
| malloc p99/mean > 5x | `perf record -e page-faults` |

---

## 预期输出参考（干净系统上）

一个优化好的实时系统应该接近：

```
[OS Scheduling]       p99/mean < 1.5x, p50 ≈ 0
[Cache DRAM]          per-access < 50ns (DRAM latency)
[Memory Bandwidth]    slow-down < 1.5x with 1 contender
[IRQ Detection]       spikes > 10us: 0 per 3 seconds
[CPU Frequency]       cycle variation < 2%
[malloc 4KB]          p99 < 100ns
```
