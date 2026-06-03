# x86 SIMD 性能调优与基准测试

```
-------------------------------------------------------------------------------
参考资料：Modern X86 Assembly Language Programming, 2nd Edition
           Agner Fog's Optimization Manuals
           Intel 64 and IA-32 Architectures Optimization Reference Manual
目标读者：需要测量、理解和提升 SIMD 内核性能的工程师
-------------------------------------------------------------------------------
```

---

## 1. 测量思维

在优化之前，必须先进行测量。性能工程的黄金法则：

1. **永远不要猜测。** 始终使用硬件计数器进行测量。
2. **先看墙上时间**，然后再用 CPU 计数器深入分析。
3. **与标量基线对比** 以确定加速比。
4. **使用真实数据规模进行测试** — 小规模的基准测试会撒谎。
5. **将内核与 I/O、内存分配和操作系统噪声隔离开来**。

### 1.1 测量什么

| 指标 | 工具 | 原因 |
|--------|------|-----|
| 墙上时间 | `clock_gettime()`, `std::chrono` | 唯一重要的真相 |
| CPU 周期 | `RDTSC` / `__rdtsc()` | 时钟周期级精度 |
| 已退役指令数 | `perf stat -e instructions` | SIMD 是否减少了指令数？ |
| CPI（每指令周期数） | `perf stat -e cycles,instructions` | 是否发生了停顿？ |
| L1/L2/L3 缓存未命中 | `perf stat -e cache-misses` | 内存层次结构问题 |
| 分支预测失败 | `perf stat -e branch-misses` | 热循环中存在不可预测的 if/else？ |
| FLOPS | `perf stat -e fp_arith_inst_retired.*` | 距离峰值有多近？ |
| SIMD 端口利用率 | `perf stat -e uops_dispatched_port.*` | 端口 5 瓶颈？ |

### 1.2 基准测试框架模式

```c
#include <time.h>
#include <stdio.h>
#include <stdint.h>

// 使用 RDTSC 的高精度计时器
static inline uint64_t read_tsc(void) {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// 或使用 clock_gettime 获取墙上时间
static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 基准测试封装：运行 fn() `iters` 次，返回平均时间
typedef void (*kernel_fn)(void);

typedef struct {
    const char* name;
    double      time_sec;
    double      cycles_per_element;
    size_t      elements_per_call;
    size_t      bytes_per_call;
    uint64_t    iters;
} benchmark_result_t;

static benchmark_result_t benchmark_kernel(
    kernel_fn fn, size_t n_elements, size_t n_bytes,
    const char* name, uint64_t target_iters)
{
    // 预热：3 次迭代以稳定缓存和频率
    for (int w = 0; w < 3; w++) fn();

    // 时间测量
    uint64_t start_tsc = read_tsc();
    double   start_sec = get_time_sec();

    uint64_t iters = 0;
    double elapsed;
    do {
        fn();
        iters++;
        elapsed = get_time_sec() - start_sec;
    } while (elapsed < 0.5 && iters < target_iters);

    uint64_t end_tsc = read_tsc();
    double   end_sec = get_time_sec();

    // 填充结果
    benchmark_result_t res;
    res.name              = name;
    res.iters             = iters;
    res.elements_per_call = n_elements;
    res.bytes_per_call    = n_bytes;
    res.time_sec          = (end_sec - start_sec) / (double)iters;
    res.cycles_per_element = (double)(end_tsc - start_tsc) / (double)(iters * n_elements);
    return res;
}

// 打印基准测试报告
static void bench_report(const benchmark_result_t* results, int count) {
    printf("%-40s %10s %12s %10s\n", "内核", "时间(ns)", "周期/元素", "GB/s");
    printf("----------------------------------------"
           "------------------------------------------\n");
    for (int i = 0; i < count; i++) {
        double ns = results[i].time_sec * 1e9;
        double cpe = results[i].cycles_per_element;
        double bw = (double)results[i].bytes_per_call / results[i].time_sec / 1e9;
        printf("%-40s %9.2f %11.2f %9.2f\n",
               results[i].name, ns, cpe, bw);
    }
}
```

---

## 2. CPU 频率与睿频加速

### 2.1 问题：不稳定的频率

现代 CPU 会因以下原因持续改变频率：
- **睿频加速（Turbo Boost）**：在负载下提高频率
- **热降频（Thermal throttling）**：温度过高时降低频率
- **AVX 偏移（AVX offset）**：Intel CPU 在执行 256/512 位 SIMD 时会降低频率
- **功耗限制（PL1/PL2）**：长期与短期功耗预算

**这使得基于 RDTSC 的计时在衡量绝对性能时不可靠。** 在冷系统中以 4.0 GHz 运行的内核，经过 30 秒持续负载后可能会降至 3.2 GHz。

### 2.2 为基准测试稳定频率

```bash
# Linux：将 CPU 频率锁定为固定值
sudo cpupower frequency-set -g performance    # 设置调速器为 performance
sudo cpupower frequency-set -d 2.5GHz -u 2.5GHz  # 锁定到指定频率

# 禁用睿频加速
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo   # Intel
# 或
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost            # AMD

# 检查当前频率
watch -n1 "cat /proc/cpuinfo | grep 'cpu MHz'"
```

### 2.3 在基准测试期间测量实际频率

```c
// 读取 APERF/MPERF MSR 以计算基准测试期间的平均频率
// 这需要内核模块或 `sudo` 权限
// 更简单的方式：使用 `perf stat`，它会报告 GHz：
//   perf stat -e cycles,instructions,task-clock ./my_benchmark
//   GHz = cycles / (task-clock * 1e9)
```

---

## 3. x86 的屋顶线模型

屋顶线模型是理解 SIMD 内核性能最重要的分析工具。

### 3.1 理论

```
可达到的性能 (GFLOPS/s) = min(
    峰值计算力 (GFLOPS/s),
    峰值内存带宽 (GB/s) × 算术强度 (FLOPs/Byte)
)
```

**算术强度 (AI)** = 总 FLOPs / 总传输字节数

### 3.2 示例分析

| 内核 | 每元素 FLOPs | 每元素字节数 | AI (FLOP/Byte) | 瓶颈在于 |
|--------|------------------|-------------------|----------------|----------|
| `c[i] = a[i] + b[i]` | 1 | 12（读 a,b + 写 c） | 0.083 | 内存 |
| `c[i] = a[i] * b[i] + c[i]` | 2 | 12 | 0.167 | 内存 |
| ReLU: `c[i] = max(a[i], 0)` | 1 | 8（读 a + 写 c） | 0.125 | 内存 |
| Softmax（小 N） | ~20 | 8 | 2.5 | 过渡区 |
| GEMM (N=1024) | 2×N³/N² ≈ 2N | ~12 | ~170 | **计算** |
| Conv2D 3×3 | 每输出 18 个 | ~12 | 1.5 | 过渡区 |

### 3.3 示例屋顶线图（Sapphire Rapids, AVX-512）

```
GFLOPS
  ^
  |                         ┌──── 峰值 AVX-512 FMA @ 2.0 GHz ────
200+                        │  = 2(FMA/周期) × 16(f32/FMA) × 2.0 GHz = 64 GFLOPS/核
  |                    ┌────┤
  |              GEMM━━┘    │
150+                  /     │
  |                 /       │
  |               /         │
100+             /           │
  |            /             │
  |          /                │
 50+  ┌──Softmax              │
  |  /   ┌──Conv2D             │
  | /  /                       │
  |/ /  ┌──ReLU/FMA            │
 0+─────┴──────────────────────┴──────────────> AI (FLOP/Byte)
  0     1     2     4     8    16    32
  内存受限区    |   过渡区   |   计算受限区
```

### 3.4 如何构建你自己的屋顶线

```bash
# 1. 测量峰值内存带宽（STREAM 基准测试）
git clone https://github.com/jeffhammond/STREAM.git
cd STREAM
gcc -O3 -mavx2 -fopenmp stream.c -o stream
./stream

# 2. 测量峰值 FLOPS（OpenBLAS 的 DGEMM 或自定义微基准测试）
# 快速估算：CPU 基础频率 × 核心数 × SIMD 宽度 × FMA/周期
# Sapphire Rapids: 2.0 GHz × 56 核 × 32(fp64 FMA) × 2(FMA/周期) = 7.2 TFLOPS

# 3. 将你的内核性能绘制到屋顶线上
# 你的内核 = (测得的 GFLOPS, AI) 点
# 如果点远低于屋顶线 → 存在优化空间
# 如果点接近内存带宽屋顶 → 需要重构数据访问模式
```

---

## 4. Perf：通用性能分析器

### 4.1 基本命令

```bash
# 基本事件计数
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses \
    ./my_benchmark

# SIMD 特定事件（Intel）
perf stat -e cycles,instructions,\
    fp_arith_inst_retired.128b_packed_single,\
    fp_arith_inst_retired.256b_packed_single,\
    fp_arith_inst_retired.512b_packed_single \
    ./my_benchmark

# 检查你的代码是否真正使用了 AVX2/AVX-512 指令！
perf stat -e avx_insts.all ./my_benchmark

# 采样分析（火焰图）
perf record -g ./my_benchmark
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

# 缓存未命中分析
perf stat -e L1-dcache-load-misses,L1-dcache-loads,\
    l2_rqsts.miss,LLC-load-misses,LLC-loads ./my_benchmark
```

### 4.2 关键性能计数器

| 计数器 | 描述 | 好/坏信号 |
|---------|-------------|---------------|
| `instructions` | 总退役指令数 | 越低越好（SIMD） |
| `cycles` | 总 CPU 周期数 | 越低越好 |
| `instructions/cycle` (IPC) | > 2 为佳，< 0.5 为差 | 越高越好 |
| `cache-misses` | L1/L2/L3 缓存未命中 | < 1% 的加载操作是理想状态 |
| `branch-misses` | 预测失败的分支 | < 1% 的分支是良好的 |
| `stalled-cycles-frontend` | 等待指令（前端停顿） | > 10% = 前端瓶颈 |
| `stalled-cycles-backend` | 等待执行资源（后端停顿） | > 10% = 后端瓶颈 |
| `uops_retired.all` | 总退役微操作数 | 与指令数对比 |

### 4.3 诊断常见瓶颈

```bash
# 前端瓶颈（指令获取/解码）
perf stat -e idq_uops_not_delivered.core ./my_benchmark

# 后端瓶颈（执行端口饱和）
perf stat -e uops_executed.thread,\
    uops_dispatched_port.port_0,\
    uops_dispatched_port.port_1,\
    uops_dispatched_port.port_5,\
    uops_dispatched_port.port_6 \
    ./my_benchmark

# 内存带宽瓶颈
perf stat -e offcore_requests.demand_data_rd,\
    offcore_requests_outstanding.demand_data_rd \
    ./my_benchmark

# 伪共享检测
perf c2c record ./my_benchmark
perf c2c report
```

---

## 5. SIMD 编译器优化标志

### 5.1 GCC/Clang 标志速查表

```bash
# 基线：AVX2 + FMA（最佳覆盖率，2013+）
gcc -O3 -mavx2 -mfma -march=haswell

# AVX-512 及其所有常用子集
gcc -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma

# 自动向量化报告（查看编译器向量化了什么）
gcc -O3 -mavx2 -fopt-info-vec -fopt-info-vec-missed

# 禁用自动向量化（用于对比手写 intrinsic 与编译器效果）
gcc -O3 -fno-tree-vectorize

# 链接时优化（LTO），用于跨翻译单元内联
gcc -O3 -flto

# 配置文件引导优化（PGO）
gcc -O3 -fprofile-generate  # 第一遍：插桩
./program                   # 使用训练数据运行
gcc -O3 -fprofile-use       # 第二遍：使用配置文件
```

### 5.2 向量化报告

```c
// 强制编译器报告为何无法向量化某个循环
#pragma GCC ivdep              // 忽略向量依赖
#pragma GCC optimize("O3")     // 按函数指定优化级别
__attribute__((optimize("O3"))) void my_func() { ... }

// 循环专用 pragma：
#pragma omp simd                // OpenMP SIMD 提示
#pragma clang loop vectorize(enable)  // Clang 专用
```

---

## 6. LLVM-MCA：静态性能分析

LLVM 机器码分析器（llvm-mca）通过 CPU 流水线模型模拟指令执行，预测吞吐量并识别瓶颈。

### 6.1 基本用法

```bash
# 分析一段汇编代码
llvm-mca -mcpu=skylake -iterations=1000 -timeline my_kernel.s

# 关键输出：
#   - Iterations：模拟的循环迭代次数
#   - Instructions：代码段中的总指令数
#   - Total Cycles：每次迭代的预测周期数
#   - Total uOps：解码出的微操作数
#   - Dispatch Width：每周期分派的指令数
#   - uOps Per Cycle：实际吞吐量
#   - IPC：每周期指令数
#   - Block RThroughput：倒数吞吐量（每周期迭代数的倒数）
#
# 资源压力：
#   显示哪些执行端口已饱和
#   [0]  - FMA、乘法、加法、洗牌
#   [1]  - FMA、乘法、加法、洗牌
#   [5]  - 洗牌、置换、混合
#   [2,3] - 加载
#   [4,7] - 存储
```

### 6.2 示例：分析 AVX2 点积

```asm
# 将内层循环提取到 .s 文件并添加 LLVM-MCA 标记：
    .intel_syntax noprefix
    .text
    .globl  dot_product_inner

dot_product_inner:
    vxorps    ymm0, ymm0, ymm0
    # LLVM-MCA-BEGIN dot_loop
.Lloop:
    vmovups   ymm1, [rdi]
    vmovups   ymm2, [rsi]
    vfmadd231ps ymm0, ymm1, ymm2
    add       rdi, 32
    add       rsi, 32
    sub       edx, 8
    jg        .Lloop
    # LLVM-MCA-END dot_loop
    ret
```

```bash
llvm-mca -mcpu=skylake -iterations=100 -timeline dot_product.s
# 检查：端口 5 是否饱和？瓶颈在加载还是 FMA？
```

### 6.3 理解输出

```
Timeline view（时间线视图）：
Index     0123456789
[0,0]     DR   .    .    .    .    .    .    .    .   vmovups	ymm1, [rdi]
[0,1]     .DR  .    .    .    .    .    .    .    .   vmovups	ymm2, [rsi]
[0,2]     . DeER.    .    .    .    .    .    .    .   vfmadd231ps	ymm0, ymm1, ymm2
[0,3]     . D==eER    .    .    .    .    .    .    .   add	rdi, 32
[0,4]     . D====eER  .    .    .    .    .    .    .   add	rsi, 32
[0,5]     . D======eER.    .    .    .    .    .    .   sub	edx, 8
[0,6]     . D========eER   .    .    .    .    .    .   jg	.Lloop

D = 分派（Dispatch），e = 执行开始（Execute start），E = 执行结束（Execute end），R = 退役（Retire）
```

---

## 7. BmThreadTimer 模式（源自原书）

原书使用一个简单但有效的计时器类进行基准测试：

```c
// BmThreadTimer.h — 受原书启发的简化版本
#include <time.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    double   elapsed_sec;
    uint64_t start_tsc;
} bm_timer_t;

static inline bm_timer_t bm_timer_start(void) {
    bm_timer_t t;
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    t.start_tsc = ((uint64_t)hi << 32) | lo;
    return t;
}

static inline double bm_timer_stop_ms(bm_timer_t* t) {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    uint64_t end = ((uint64_t)hi << 32) | lo;
    uint64_t diff = end - t->start_tsc;
    // 假设标称频率以进行粗略的毫秒转换
    // 如需精确结果，请改用 clock_gettime()
    return (double)diff / 2.5e6;  // 2.5 GHz 标称频率
}

// 用法：
// bm_timer_t t = bm_timer_start();
// run_kernel();
// double ms = bm_timer_stop_ms(&t);
```

---

## 8. 缓存行为测试

### 8.1 通过指针追逐测量延迟

```c
// 通过指针追逐测量 L1/L2/L3/内存延迟
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static double measure_cache_latency(int size_kb) {
    int num_elements = (size_kb * 1024) / sizeof(void*);
    void** array = (void**)malloc(num_elements * sizeof(void*));

    // 创建循环链表（指针追逐）
    for (int i = 0; i < num_elements - 1; i++)
        array[i] = &array[i + 1];
    array[num_elements - 1] = &array[0];

    // 预热
    void* p = array[0];
    for (int i = 0; i < 1000; i++) p = *(void**)p;

    // 测量
    uint64_t start = __rdtsc();
    for (int i = 0; i < 10000; i++) p = *(void**)p;
    uint64_t end = __rdtsc();

    free(array);
    return (double)(end - start) / 10000.0;  // 每次访问的周期数
}
```

### 8.2 STREAM 风格带宽测试

```c
// 带宽测试：我们的读写能有多快？
#include <immintrin.h>
#include <malloc.h>

__attribute__((noinline))
double bench_read_bw(const float* a, int n_bytes) {
    int n = n_bytes / (int)sizeof(float);
    __m256 sum = _mm256_setzero_ps();  // 防止死代码消除

    uint64_t start = __rdtsc();
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        sum = _mm256_add_ps(sum, v);
    }
    uint64_t end = __rdtsc();

    // 将结果存入 volatile 变量以防止被消除
    volatile float sink;
    _mm256_storeu_ps((float*)&sink, sum);
    (void)sink;

    return (double)n_bytes / ((double)(end - start) / 2.5e9);  // GB/s @ 2.5 GHz
}
```

---

## 9. 生产环境基准测试检查清单

```
[ ] 锁定 CPU 频率（cpupower frequency-set）
[ ] 测量期间禁用睿频加速
[ ] 将线程绑定到核心（taskset -c N）
[ ] 预热：测量前运行内核 3-5 次
[ ] 使用足够大的 N 以摊薄开销（>10,000 个元素）
[ ] 报告 5 次以上运行的中位数，而不仅仅是最佳值
[ ] 报告标准差
[ ] 使用 noinline 屏障防止编译器作弊
[ ] 与标量参考实现验证正确性
[ ] 使用 volatile 或 asm(""::"r"(ptr)) 防止死代码消除
[ ] 报告硬件信息：CPU 型号、频率、缓存大小
[ ] 同时测试 L1 驻留和 DRAM 驻留的数据规模
[ ] 包含冷缓存测量（首次运行前刷新缓存）
[ ] 在基准测试期间检查频率降频（perf stat -e power/energy-cores/）
```

---

## 10. 常见性能陷阱

### 10.1 死代码消除

```c
// 错误：编译器可能会消除整个循环！
for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
// ... 从未使用 c ...

// 正确：强制结果可被观测
volatile float sink;
for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
sink = c[0];  // 读取操作强制计算发生
```

### 10.2 因频率缩放导致的不稳定基准测试

```bash
# 检查基准测试期间频率是否发生变化
perf stat -e cycles,instructions,power/energy-cores/ ./my_benchmark
# 如果 cycles/instructions 在各次运行间波动很大 → 频率缩放在起作用
```

### 10.3 首次运行时的缺页异常

```bash
# 首次运行总是较慢，因为有缺页异常
# 解决方案：始终用至少一次完整迭代进行预热
# 基准测试前，先触碰所有页面：
for (int i = 0; i < n; i += 4096 / sizeof(float))
    array[i] = 0.0f;  // 强制操作系统分配物理页面
```

### 10.4 NUMA 效应

```bash
# 绑定到特定的 NUMA 节点
numactl --membind=0 --cpunodebind=0 ./my_benchmark

# 查看 NUMA 拓扑
numactl --hardware
```

### 10.5 超线程争用

```bash
# 禁用超线程以获得干净的测量结果
echo 0 | sudo tee /sys/devices/system/cpu/cpu*/online  # 禁用奇数编号的 CPU

# 或使用 taskset 只使用物理核心
taskset -c 0,2,4,6,8,10,12,14 ./my_benchmark  # 在许多系统上，偶数是物理核心
```

---

## 11. 解读结果：决策树

```
Is IPC < 1.0?
  ├── Yes → 检查前端/后端停顿计数器
  │         ├── 前端停顿占主导 → I-cache 未命中、分支预测失败
  │         │   → 修复：展开循环、减少热路径中的代码量
  │         └── 后端停顿占主导 → 执行端口饱和
  │             ├── 端口 5 饱和 → 过多的洗牌/置换操作
  │             │   → 修复：使用 vaddps 代替 vhaddps、减少洗牌操作
  │             └── 端口 2/3/4 饱和 → 内存瓶颈
  │                 → 修复：重构以获得更好的缓存局部性、使用预取
  └── No → IPC > 1.0，是否接近峰值？
            ├── Yes → 做得不错，可以继续前进
            └── No → 检查 SIMD 宽度利用率
                  ├── 是否使用了所有通道？→ 检查性能计数器的 SIMD 指令组合
                  └── 热路径中存在标量代码？→ 向量化或使用 intrinsic
```

---

## 12. 快速参考：Perf 一行命令

```bash
# 自顶向下分析（仅 Intel CPU）
perf stat --topdown -a -- ./my_benchmark

# 查看有多少时间花在 SIMD 上
perf stat -e fp_arith_inst_retired.128b_packed_single,\
    fp_arith_inst_retired.256b_packed_single,\
    fp_arith_inst_retired.512b_packed_single,\
    fp_arith_inst_retired.scalar_single \
    ./my_benchmark

# 检查是否将寄存器溢出到栈上
perf stat -e cpu/event=0xD1,umask=0x10,name=MEM_LOAD_UOPS_RETIRED.L3_HIT/ \
    ./my_benchmark

# 统计 AVX-512 重型与轻型指令
perf stat -e cpu/event=0xC6,umask=0x10,name=AVX_INSTS.ALL/ \
    ./my_benchmark

# 获取内存层次结构行为的详细视图
perf stat -e cycles,instructions,\
    L1-dcache-loads,L1-dcache-load-misses,\
    LLC-loads,LLC-load-misses,\
    l2_rqsts.all_demand_data_rd,\
    l2_rqsts.demand_data_rd_miss \
    ./my_benchmark
```

（文件结束 - 共 403 行）
