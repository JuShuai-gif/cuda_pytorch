# 计算机体系结构与性能优化

## 1. CPU 缓存层次结构

### 1.1 缓存层次概览

```
寄存器 (0 cycle)   ← 编译器管理，最快
    ↓
L1 缓存 (4-5 cycles)  ← 32KB I-Cache + 32KB D-Cache，每核私有
    ↓
L2 缓存 (12-14 cycles) ← 256KB-1MB，每核私有
    ↓
L3 缓存 (40-50 cycles) ← 8MB-64MB，所有核共享
    ↓
主存 RAM (100-200 cycles) ← DDR4/DDR5
    ↓
磁盘/SSD (数百万 cycles)
```

**关键数值（现代 x86 CPU，如 Intel Core i7 / AMD Zen4）**：

| 层级 | 大小 | 延迟（cycles） | 延迟（ns @ 4GHz） | 带宽 |
|------|------|--------------|-------------------|------|
| L1 | 32KB I + 32KB D | 4-5 | ~1ns | ~2TB/s |
| L2 | 256KB-1MB | 12-14 | ~3ns | ~500GB/s |
| L3 | 8-64MB | 40-50 | ~10ns | ~200GB/s |
| DRAM | GB 级别 | 100-200 | ~50ns | ~50GB/s |

**对于机器人系统**：当数据访问延迟从 L1 变为 DRAM 时，速度下降 50-200 倍。这在 1kHz 控制循环中至关重要。

### 1.2 缓存行（Cache Line）

缓存以固定大小的块（缓存行）加载数据。x86 上为 64 字节。

```cpp
// 缓存行 = 64 字节
// sizeof(int) = 4 字节，所以一个缓存行可装 16 个 int
// 读取 array[0] 时，array[0]...array[15] 这 64 字节都被加载到缓存

int array[1000];
array[0] = 1;  // 加载缓存行 [0..15]
array[2] = 3;  // 缓存命中（同一缓存行）
array[16] = 5; // 加载新缓存行 [16..31]
```

**缓存行的对齐影响**：

```cpp
struct BadLayout {
    int a;       // 4B
    double b;    // 8B → 需要 8 字节对齐
    int c;       // 4B
}; // 总共 24B（填充后），但 c 可能跨缓存行

struct GoodLayout {
    double b;    // 8B
    int a;       // 4B
    int c;       // 4B
}; // 总共 16B，在一个缓存行内
```

### 1.3 缓存关联度（Associativity）

缓存不是全相联的——每个内存地址只能映射到有限的缓存槽。

```
直接映射（1-way）：
│ Addr 0x0   → Set 0     │ ← 冲突多
│ Addr 0x40  → Set 1     │
│ Addr 0x80  → Set 2     │
│ Addr 0x100 → Set 0 ← 冲突！│

8-way 组相联：
│ Addr 0x0   → Set 0, Way 0-7 │ ← 8 个槽位可选
│ Addr 0x100 → Set 0, Way 0-7 │ ← 同时存在

全相联：
│ 任意地址 → 任意位置     │ ← 硬件复杂度高
```

**典型关联度**：

| 缓存 | 关联度 |
|------|--------|
| L1 D-Cache | 8-way |
| L1 I-Cache | 8-way |
| L2 | 4-16 way |
| L3 | 12-16 way |

**对性能的影响**：步长为缓存路数的地址访问会导致路径冲突（cache thrashing）。

```cpp
// 示例：4096-byte stride on 32KB 8-way L1
// N = 8192, stride = 512 → 8192/512 = 16 个地址映射到同一 set
// 8-way 只能容纳 8 个 → 持续 eviction → "thrashing"
#define N 8192
#define STRIDE 512
int data[N];
for (int i = 0; i < N; i += STRIDE / sizeof(int)) {
    sum += data[i];  // 每次访问都是 cache miss
}
```

### 1.4 缓存行对机器人应用的影响

```cpp
// 机器人感知管线中的 EKF 协方差矩阵
// float cov[6][6] // 6-DOF 状态
// 6x6x4 = 144 bytes → 3 个缓存行
// 针对局部性进行排序，使访问模式为行优先（row-major）

// 错误：跨步访问
for (int j = 0; j < 6; j++)
    for (int i = 0; i < 6; i++)
        cov[i][j] *= alpha;  // 列优先访问，每次跨 24 字节

// 正确：顺序访问
for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++)
        cov[i][j] *= alpha;  // 行优先访问，缓存利用率高
```

## 2. 缓存一致性协议

### 2.1 MESI 协议

**通俗理解：几个室友共用一本笔记**

多核 CPU 的每个核心都有自己独立的 L1/L2 缓存，但它们共享同一块主内存。
问题来了：Core 0 在自己的缓存里改了数据，Core 1 怎么知道？MESI 就是解决这个
"各自手上的副本是否还有效"的同步协议。

把缓存行想象成室友们手抄的笔记副本，四个状态就是：

| 状态 | 含义 | 通俗解释 | 本核 | 其他核 |
|------|------|---------|------|--------|
| **M**odified | 独占且已修改 | "我改了笔记，只有我手里的是对的，其他人的都作废了。" | 有最新数据 | 无效 |
| **E**xclusive | 独占且未修改 | "只有我有这份笔记，还没改过，跟原版一模一样。" | 有数据 | 无效 |
| **S**hared | 共享 | "好几个人都有这份笔记，大家看的都一样，谁也别改。" | 有数据 | 可能有 |
| **I**nvalid | 无效 | "我手里的已经过时了，要看必须找别人借。" | 无数据 | 可能有 |

**完整交互流程示例**：

```
1. Core 0 读变量 X → 没人有 → 从内存加载 → 状态 E（"只有我有"）
2. Core 1 也读 X → Core 0 说"我也有份" → 两人都变 S（"一起看，谁都别改"）
3. Core 0 要写 X → 广播"你们手里的都作废！" → Core 1 变 I，Core 0 变 M
   （"只有我的是最新版"）
4. Core 1 再读 X → 发现自己手里是 I → 找 Core 0 要最新数据 →
   Core 0 把数据给 Core 1，自己也退回 S
```

**为什么原子变量的竞争访问特别慢**：

之前 `test_os_scheduling` 里所有噪声线程都在疯狂写同一个 `dummy` 原子变量：

```
Core 0 写 dummy → 缓存行变 M → Core 1/2/3 的全部失效（变 I）
Core 1 想写 dummy → 必须先找 Core 0 拿最新数据 → Core 0 的变 I → Core 1 变 M
Core 2 想写 → 又得找 Core 1 要 → ...
```

就像一群室友同时抢同一本笔记改来改去，每次只允许一个人持有最新版。
这被称为 **invalidation 风暴**—— 同一缓存行在多个核心间 ping-pong，
Cache Coherence 总线流量爆炸，性能比各写各的缓存行慢 20-100 倍。 |

**状态转换（简化）**：

```
PrRd (本地读)：
  I → E/S (发 BusRd，其他核响应则 S，否则 E)
  M/E → M/E (无变化)
  S → S (无变化)

PrWr (本地写)：
  I → M (发 BusRdX = Read Exclusive)
  E → M (静默升级)
  S → M (发 BusUpgr = 使无效)
  M → M (无变化)

BusRd (远程读窥探)：
  M → S (写回并共享)
  E → S (共享)
  S → S (保持)

BusRdX (远程写窥探)：
  M → I (写回并失效)
  E → I (失效)
  S → I (失效)
```

### 2.2 MOESI 协议（AMD 使用）

在 MESI 基础上增加 **O**wned 状态：

- Modified 且被共享 → Owned
- Owned 缓存行被修改过，但其他核有只读副本
- 驱逐时 Owned 必须写回，Modified 也必须写回

**为什么需要 O 状态**：

```
MESI：Modified → 收到 BusRd → 写回内存 → 变为 Shared
MOESI：Modified → 收到 BusRd → 变为 Owned → 共享数据不写回内存
→ 减少内存带宽消耗
```

### 2.3 伪共享（False Sharing）

两个不同核心各自修改不同变量，但它们恰好在同一个缓存行中。

```cpp
struct Counters {
    alignas(64) int64_t counter_a = 0;  // 独自占用一个缓存行
    char padding_a[64 - sizeof(int64_t)];
    alignas(64) int64_t counter_b = 0;  // 独自占用另一个缓存行
    char padding_b[64 - sizeof(int64_t)];
};

// 无填充时：counter_a 和 counter_b 在同一缓存行
// → Core 0 写 counter_a → 缓存行变为 M → Core 1 的副本失效
// → Core 1 写 counter_b → 缓存行变为 M → Core 0 的副本失效
// → 反复 invalidation 抖动！即使访问的是不同变量
```

**性能损失**：1000 万次迭代中，伪共享比无伪共享慢 **20-100 倍**，取决于一致性协议流量。

**机器人系统中的常见场景**：多线程更新各自负责的关节状态数组。

## 3. NUMA 架构

### 3.1 NUMA 拓扑

非一致性内存访问（Non-Uniform Memory Access）：每个 CPU 有本地内存和远程内存。

```
┌─────────────────┐     ┌─────────────────┐
│  CPU Socket 0   │     │  CPU Socket 1   │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Core 0..7 │  │     │  │ Core 8..15│  │
│  └───────────┘  │     │  └───────────┘  │
│       │ QPI/UPI │───────│       │        │
│  ┌───┴───┐      │     │  ┌───┴───┐      │
│  │ NUMA  │      │     │  │ NUMA  │      │
│  │ Node 0│      │     │  │ Node 1│      │
│  └───────┘      │     │  └───────┘      │
│  Local 64GB    │     │  Local 64GB     │
└─────────────────┘     └─────────────────┘
```

- 本地节点访问：~100ns
- 远程节点访问：~150-300ns（1.5x-3x 开销）

```bash
# 查看 NUMA 拓扑
numactl --hardware
# available: 2 nodes (0-1)
# node 0 cpus: 0 1 2 3 4 5 6 7
# node 0 size: 65456 MB
# node 1 cpus: 8 9 10 11 12 13 14 15
# node 1 size: 65536 MB
# node distances:
# node   0   1
#   0:  10  21   ← 本地:10, 远程:21（乘以 10 得到纳秒的近似值）

lstopo --of console  # 更详细的拓扑
```

### 3.2 NUMA 感知编程

**libnuma API**：

```cpp
#include <numa.h>

// 在 node 0 上分配内存
void *buf = numa_alloc_onnode(1024 * 1024, 0);

// 在本地节点分配（跟随调用线程的 CPU）
void *buf = numa_alloc_local(1024 * 1024);

// 将线程绑定到指定节点
struct bitmask *mask = numa_allocate_nodemask();
numa_bitmask_setbit(mask, 0);
numa_bind(mask);

// 设置内存策略
numa_set_preferred(0);  // 优先从 node 0 分配
numa_set_interleave_mask(mask);  // 交替分配

numa_free(buf, 1024 * 1024);
```

**NUMA 绑定方法**：

```bash
# 在 node 0 上运行程序，并优先使用 node 0 的内存
numactl --cpunodebind=0 --membind=0 ./my_program

# 让内存分配在 node 0 和 1 之间交替
numactl --interleave=0,1 ./my_program

# 显示 NUMA 统计
numastat -p <pid>
```

**机器人系统中的应用**：

- 将传感器处理固定到离 PCIe 最近（IOH 所在）的 NUMA 节点
- 控制循环与决策规划分离到不同 NUMA 节点
- 共享数据使用 interleave 策略避免单节点带宽瓶颈

## 4. CPU 流水线与分支预测

### 4.1 经典五级流水线

```
IF → ID → EX → MEM → WB
指令取指 → 译码 → 执行 → 访存 → 写回

现代 CPU：10-20 级流水线
```
- **流水线停顿（Stall）**：数据依赖、缓存未命中、分支预测失败
- **数据转发（Bypass/Forwarding）**：EX 结果直接传给下一个 EX，无需等待 WB

### 4.2 分支预测

**静态预测**：向后分支（循环）→ 跳转；向前分支（if-else）→ 不跳转

**动态预测**：

- **1-bit 预测器**：记住上次分支方向
- **2-bit 饱和计数器**：00, 01, 10, 11 → 需要连续两次预测错误才改变
- **全局历史（GShare）**：分支历史 异或 分支地址 → 查表
- **锦标赛（Tournament）**：局部+全局，选择历史较好的预测器

```cpp
// 可预测（排序后）
std::sort(data.begin(), data.end());
// 分支高度可预测 → 流水线满载

// 不可预测
//（随机数据 → 分支预测失败 ~50% → 持续刷流水线）
```

**分支预测失败的代价**：

- 短流水线（ARM Cortex-M）：2-3 cycles
- 长流水线（x86）：15-20 cycles
- 误预测率 5% → 整体性能损失 10-15%

### 4.3 推测执行

CPU 在分支结果未确定时，基于预测结果提前执行指令。如预测正确，结果直接提交；如预测错误，丢弃推测结果（ROB 刷新）。

**Spectre/Meltdown 的根源**：推测执行期间访问了不该访问的内存，微架构侧信道泄漏了数据。

### 4.4 对机器人代码的影响

```cpp
// 不好：循环内分支
for (int i = 0; i < N; i++) {
    if (data[i] > threshold) {
        process_a(data[i]);
    } else {
        process_b(data[i]);
    }
}
// → 如果 data 无序，预测失败率 ~50%

// 好：分离数据流
std::vector<float> above, below;
for (int i = 0; i < N; i++) {
    (data[i] > threshold ? above : below).push_back(data[i]);
}
for (auto v : above) process_a(v);  // 无分支
for (auto v : below) process_b(v);  // 无分支
```

## 5. SIMD 基础

### 5.1 SIMD 指令集

| 指令集 | 寄存器宽度 | 首次引入 | 处理器 |
|--------|----------|---------|--------|
| SSE | 128-bit | 1999 (Pentium III) | x86通用 |
| SSE2/3/4 | 128-bit | 2001-2006 | x86通用 |
| AVX | 256-bit | 2011 (Sandy Bridge) | x86主流 |
| AVX2 | 256-bit | 2013 (Haswell) | x86主流 |
| AVX-512 | 512-bit | 2017 (Skylake-X) | 服务器级 |
| NEON | 128-bit | 2011 | ARMv7/v8 |
| SVE | 可变 128-2048 | 2020 | ARMv9 |

### 5.2 何时使用 SIMD

**适合**：

- 统一的数据类型（float, int16, int8）
- 连续内存访问
- 重复的算术/逻辑运算
- 无分支的循环

**不适合**：

- 控制流复杂（大量 if/switch）
- 不连续的内存访问
- 单次操作的延迟敏感代码
- 64-bit 整数除法等无 SIMD 指令的操作

### 5.3 使用示例

```cpp
#include <immintrin.h>

// 128 floats 的点乘，使用 AVX（每次操作 8 个 float）
void dot_product_avx(const float *a, const float *b,
                     float *result, size_t n) {
    __m256 sum = _mm256_setzero_ps();  // 初始化为 0
    for (size_t i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);  // 加载 8 个 float
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);   // FMA: sum += va * vb
    }
    // 水平求和（将 8 个 float 缩减为 1 个）
    alignas(32) float temp[8];
    _mm256_store_ps(temp, sum);
    *result = temp[0] + temp[1] + temp[2] + temp[3]
            + temp[4] + temp[5] + temp[6] + temp[7];
}
```

**编译器自动向量化**：

```bash
# 查看哪些循环被向量化
g++ -O3 -march=native -fopt-info-vec-optimized main.cpp

# 查看哪些循环未能向量化及原因
g++ -O3 -march=native -fopt-info-vec-missed main.cpp
```

### 5.4 机器人感知中的 SIMD 用例

- 图像预处理（归一化、颜色空间转换）
- 点云变换（旋转矩阵乘 3D 点）
- Kalman 滤波预测步骤（矩阵向量乘法）
- 激光雷达扫描点的距离计算

## 6. 线程调度深入

### 6.1 CFS（完全公平调度器）

Linux 默认调度器：

- 每个 CPU 维护红黑树，按 vruntime 排序
- vruntime 小的线程优先运行
- 每次选择最左边的节点（最小 vruntime）

```
vruntime = actual_runtime × (1024 / weight)
```

- nice=0 的权重为 1024（vruntime = 实际运行时间）
- nice=-20（最高优先级用户线程），vruntime 增长缓慢
- 高优先级线程 "看起来" 运行时间更短 → 获得更多 CPU

**CFS 参数**：

```bash
# 调度周期的目标延迟（默认 6ms）
/proc/sys/kernel/sched_latency_ns

# 调度周期内的最小时间片（默认 0.75ms）
/proc/sys/kernel/sched_min_granularity_ns

# 唤醒时的抢占延迟
/proc/sys/kernel/sched_wakeup_granularity_ns
```

### 6.2 实时调度器

**SCHED_FIFO 行为**：

- 运行到完成 / 主动让出 / 阻塞
- 被更高优先级的实时线程抢占
- 没有时间片

**SCHED_RR 行为**：

- 与 FIFO 相同，但同一优先级有时间片轮转
- 时间片大小：`/proc/sys/kernel/sched_rr_timeslice_ms`（默认 100ms）

**实时节流（RT Throttling）**：

```bash
# 实时线程最多占 95% 的 CPU 时间（默认）
/proc/sys/kernel/sched_rt_runtime_us  # = 950000
/proc/sys/kernel/sched_rt_period_us   # = 1000000

# 禁用实时节流（不推荐）
echo -1 > /proc/sys/kernel/sched_rt_runtime_us
```

### 6.3 CPU 隔离（Isolcpus）

从调度器管理范围中移除 CPU，使它们仅用于手动固定线程：

```bash
# 内核启动参数
isolcpus=4,5,6,7 nohz_full=4,5,6,7 rcu_nocbs=4,5,6,7

# 然后将实时线程固定到隔离的 CPU
taskset -c 4 ./control_loop
```

**完整的实时内核配置清单**：

```
CONFIG_PREEMPT_RT=y      # 实时抢占补丁
CONFIG_HZ=1000           # 定时器频率 1kHz
CONFIG_NO_HZ_FULL=y      # 无滴答内核（减少中断）
isolcpus=...             # CPU 隔离
irqaffinity=...          # IRQ 亲和性管理
```

## 7. 性能分析工具

### 7.1 perf

```bash
# 统计缓存未命中
perf stat -e cache-references,cache-misses \
          -e L1-dcache-loads,L1-dcache-load-misses \
          ./benchmark

# 记录调用图
perf record -g --call-graph dwarf ./benchmark
perf report

# 查看具体函数的缓存行为
perf annotate -s cache-misses function_name

# 查看 NUMA 内存访问
perf stat -e node-loads,node-load-misses ./numa_bench
```

### 7.2 Cachegrind（Valgrind）

```bash
valgrind --tool=cachegrind \
         --LL=8388608,16,64 \  # L3: 8MB, 16-way, 64B 行
         ./benchmark
cg_annotate cachegrind.out.xxxx
```

### 7.3 其他工具

```bash
# 查看 CPU 拓扑
lscpu | grep -E "Model name|L1|L2|L3|NUMA|Thread"

# 查看缓存信息
getconf -a | grep CACHE
getconf LEVEL1_DCACHE_SIZE     # L1D 大小
getconf LEVEL1_DCACHE_ASSOC    # L1D 关联度
getconf LEVEL1_DCACHE_LINESIZE # 缓存行大小

# 查看页面大小
getconf PAGE_SIZE  # 4096

# 分析内存访问模式
likwid-perfctr -g MEM ./benchmark  # 需要 likwid 工具
```

## 8. 对机器人性能的影响

### 8.1 代表性场景分析

| 场景 | 关键瓶颈 | 优化方向 |
|------|---------|---------|
| SLAM 后端优化 | 稀疏矩阵计算 | 缓存友好存储（CSR）、L3 缓存拟合 |
| 点云处理 | 大量的点，访问不连续 | SIMD 向量化、预取、流式处理 |
| 控制循环(1kHz) | 延迟敏感、内存锁定 | NUMA 固定、缓存预热、预取 |
| 深度学习推理 | 权重矩阵带宽 | 矩阵分块、数据布局优化 |
| 传感器融合 | 多线程数据同步 | 无锁队列、缓冲区布局 |

### 8.2 缓存编程法则

1. **数据组织**：结构数组（SoA）而非数组结构（AoS）
2. **空间局部性**：顺序访问 > 跨步访问 > 随机访问
3. **时间局部性**：循环分块（tiling）复用数据
4. **缓存行利用**：一次性消费整个缓存行的数据
5. **避免伪共享**：每个线程有自己独立缓存行变量
6. **预取**：硬件自动，但对于不规则模式，软件预取有帮助

### 8.3 实际检查清单

在机器人软件部署前验证：

```
[ ] 使用 perf stat 确认 L1 缓存命中率 > 95%
[ ] 确认控制循环线程未被交换到磁盘（mlockall）
[ ] 确认 CPU 固定正确（taskset/cpuset）
[ ] 确认 NUMA 节点绑定符合硬件拓扑
[ ] 确认无伪共享（使用 perf c2c）
[ ] 确认 SIMD 自动向量化通过 -fopt-info-vec 验证
[ ] 确认分支预测失败率 < 2%（perf stat -e branches,branch-misses）
```
