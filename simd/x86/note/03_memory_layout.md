# x86 SIMD 内存布局与数据组织

## 1. 对齐：SIMD 内存访问的基石

### 1.1 各级对齐要求

| 指令集 | 向量宽度 | 对齐字节 | 对齐加载 intrinsic |
|--------|---------|---------|-------------------|
| SSE | 128 bit (16B) | 16 | `_mm_load_ps` |
| AVX/AVX2 | 256 bit (32B) | 32 | `_mm256_load_ps` |
| AVX-512 | 512 bit (64B) | 64 | `_mm512_load_ps` |

**关键认知**：对齐要求不是 SIMD 指令本身的要求，而是**对齐加载指令**的要求。`loadu`（非对齐）版本接受任意对齐（合法地址即可）。

### 1.2 _mm_malloc：对齐内存分配

```c
#include <malloc.h>  // _mm_malloc, _mm_free

// 分配 64 字节对齐的内存（适用于 AVX-512）
float* data = (float*)_mm_malloc(n * sizeof(float), 64);
if (!data) { /* OOM 处理 */ }

// 必须用 _mm_free 释放（不能用标准 free！）
_mm_free(data);
```

**为什么不用 `aligned_alloc` 或 `posix_memalign`**：
- `_mm_malloc` 保证在所有支持相应 SIMD 的平台上都能正常工作
- 部分平台（尤其是 Windows）的 `aligned_alloc` 行为有细微差异
- 一致使用 `_mm_malloc` / `_mm_free` 配对是跨平台最安全的方案

### 1.3 编译器对齐声明

```c
// 栈上对齐变量
__attribute__((aligned(64))) float local_array[1024];

// 结构体对齐
struct __attribute__((aligned(64))) AlignedStruct {
    float data[16];
};

// C11 方式
_Alignas(64) float array[1024];

// C++11 方式
alignas(64) float array[1024];
```

**注意**：局部变量对齐可能增加栈帧大小（编译器会插入 padding），但这对 SIMD 性能至关重要。不要吝啬。

### 1.4 对齐的性能影响

**现代 Intel CPU（Haswell+）的对齐/非对齐延迟**：

| 访问模式 | 延迟（周期） | 说明 |
|---------|------------|------|
| 对齐加载（不跨 cache line） | 4-5 | 基线 |
| 非对齐加载（不跨 cache line） | 4-5 | 与对齐相同！ |
| 非对齐加载（跨 64B cache line） | 5-6 | 轻微增加 |
| 非对齐加载（跨 4KB 页面） | ~100+ | 触发页面遍历 |

**实验数据**（Skylake，L1 内）：
```
对齐加载       : 4.0 周期
非对齐加载     : 4.0 周期  (不跨 cache line)
跨 cache line  : 5.2 周期
跨 4KB 页      : ~150 周期
```

**结论**：在现代 x86 CPU 上，对齐本身不再像过去那样关键。但对于生产代码：
1. 无论如何对齐数组（16/32/64 字节），然后**使用 `loadu`/`storeu`** 以获得最大的灵活性
2. 如果代码中有确定性的对齐保证（例如已验证地址是 64 字节对齐的），使用 `load`/`store` 可以略微提高可读性（作为文档说明对齐要求）
3. 最需要避免的是**跨 4KB 页面的访问**，这比对齐/非对齐的差异大得多

### 1.5 跨页边界的具体影响

4KB 页面边界之所以开销大，是因为它触发了**两次 TLB（Translation Lookaside Buffer）查找**。加载地址的低 12 位属于第一页，但字节位移跨越到下一页，需要两个不同的物理页映射：

```
内存布局（页面边界在 0x...FFF→0x...000）：
字节:   0x..FFC 0x..FFD 0x..FFE 0x..FFF | 0x..000 0x..001 0x..002 0x..003
                                          ^^^^^^^^^^^^^跨页^^^^^^^^^^^^^^
                                         AVX 加载跨越此处：两次 TLB 查表
```

**如何避免**：
- 分配 64 字节对齐的内存，且数组起始地址在 64 字节边界上
- 大部分时间这自然解决了问题：一个 64 字节的对齐块永远不会跨越页面（因为 64 能整除 4096）
- 但动态偏移（例如 `ptr + offset`）时需注意

## 2. 非对齐加载/存储（loadu/storeu）的现代真相

### 2.1 硬件演进

在 Pentium 4 / Core 2 时代，非对齐加载确实很慢（2-3x 延迟）。但从 Nehalem（2008）开始，Intel 在加载/存储单元中增加了非对齐处理逻辑。到 Haswell（2013），非对齐加载在不跨 cache line 时与对齐加载完全等价。

### 2.2 什么情况下 loadu 仍然慢

```c
// 场景 1：不跨 cache line——完全等价
float* ptr = (float*)(aligned_base + 4);  // 16 字节偏移
__m256 v = _mm256_loadu_ps(ptr);  // 尽管非对齐，但 32 字节范围在同一个 64B cache line 内

// 场景 2：跨 cache line——慢约 1-2 周期
float* ptr = (float*)(aligned_base + 10);  // 40 字节偏移
__m256 v = _mm256_loadu_ps(ptr);  // 跨越 64 字节边界（40+32-64=8 字节在下一行）

// 场景 3：跨页面——极慢（~150 周期）
float* ptr = (float*)(page_start + 4080);  // 接近 4KB 页面尾部
__m256 v = _mm256_loadu_ps(ptr);  // 32 字节中有 16 字节在下一页
```

### 2.3 非对齐存储（storeu）

非对齐存储在 Skylake 及以后的 CPU 上同样是单端口执行。主要开销体现在跨 cache line 时，但比加载影响略小（存储可以在 store buffer 中合并）。

## 3. 非时间存储（Non-temporal Store / Streaming Store）

### 3.1 基本原理

标准存储（`store`/`storeu`）在写入前会将数据加载到 cache。非时间存储（`stream`）**绕过 cache**，数据直接从 CPU 写入内存控制器：

```c
// 常规存储：Cache → 写回（write-back）→ 内存。污染 cache
_mm256_storeu_ps(dst + i, data);  // 4-5 周期延迟，数据在 L1

// 非时间存储：CPU → 内存控制器。不经过 cache
_mm256_stream_ps(dst + i, data);  // 写合并（write-combining），延迟更高但不污染 cache
```

### 3.2 何时使用

**流存储的适用条件**（全部满足）：
1. 数据不会被立即重新读取（如果重新读取，就不应该 bypass cache）
2. 写入的数据量 > L1 数据 cache 的 50%（对于 32KB L1d，即 > 16KB）
3. 写入模式是顺序的（streaming store 针对顺序写入优化）

**反面示例**：
```c
// 错误：小数组，不满足条件 2
float small_arr[32];
_mm256_stream_ps(small_arr, data);  // 过度优化！32*4=128B < 16KB，cache 更适合
_mm_sfence();

// 错误：写入后立即读取，违反条件 1
_mm256_stream_ps(dst, data);
float x = dst[0];  // 需要从内存重新加载，抵消了所有优势
```

**正确使用**：
```c
// 大型数据复制/初始化——典型的 stream 使用场景
void large_memcpy_stream(float* __restrict dst, const float* __restrict src, int n) {
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_stream_ps(dst + i, v);
    }
    _mm_sfence();  // 确保所有流存储对后续可见
    // _mm_sfence 是必要的——流存储是弱序的
}
```

### 3.3 写合并缓冲区（Write-Combining Buffer）

x86 CPU 有 4-6 个写合并缓冲区，每个 64 字节（一个 cache line）。流存储先填满一个缓冲区，然后整行写入内存：

```
_mm256_stream_ps(dst + 0, a0);   → WC buffer 0 [0:31]
_mm256_stream_ps(dst + 8, a8);   → WC buffer 0 [32:63]  ← 合并！
// 当 64 字节满后，一次性写入内存
```

**要点**：
- 流存储应写入完整 cache line（64 字节），避免部分行刷新
- 交错使用流存储和常规存储时，写合并缓冲区可能被部分刷新，降低效率

## 4. Gather 和 Scatter：便利的代价

### 4.1 AVX2 Gather 性能分析

```c
// _mm256_i32gather_ps: 从 8 个不同地址加载 8 个 f32
// 微架构实现（Skylake）：~20 个 µop，延迟 ~20 周期
__m256i indices = _mm256_setr_epi32(0, 7, 3, 11, 2, 5, 1, 9);
__m256 gathered = _mm256_i32gather_ps(base, indices, 4);
```

**内部实现分解**（概念上）：
```
for each lane i:
    addr = base + indices[i] * scale
    load temp_i = *addr
merge temp_0..temp_7 → result
```

在硬件中，这被实现为多个独立的加载端口操作（每个周期一个），所以 8 个元素的 gather 最少需要 8 个周期（加上合并开销）。

### 4.2 AVX-512 Gather 性能分析

```c
// _mm512_i32gather_ps 同样昂贵，但相比等效循环仍然有优势
// Skylake-X：16 个元素 gather 延迟约 24 周期
__m512i indices = _mm512_setr_epi32(0,7,3,11,2,5,1,9,4,13,6,8,10,12,14,15);
__m512 gathered = _mm512_i32gather_ps(base, indices, 4);
```

### 4.3 Gather vs 连续加载性能对比

| 操作 | 延迟（周期） | 吞吐（每 N 周期一个） | 元素数 |
|------|------------|---------------------|--------|
| `_mm256_loadu_ps`（连续对齐） | 4 | 0.5 | 8 |
| `_mm256_i32gather_ps`（完全随机） | 20 | ~5 | 8 |
| `_mm512_loadu_ps`（连续对齐） | 4 | 0.5 | 16 |
| `_mm512_i32gather_ps`（完全随机） | 24 | ~6 | 16 |

### 4.4 何时必须使用 Gather

有些场景无法避免 gather：

```c
// 嵌入查找：user_emb 和 item_emb 都是稠密向量
// item_indices 是稀疏的 item ID 列表
__m512 query = _mm512_set1_ps(user_emb[0]);  // 广播查询的第一个分量
__m512i item_ids = _mm512_loadu_si512((__m512i*)batch_indices);
// Gather：从不同的 item embedding 中收集对应分量
__m512 items = _mm512_i32gather_ps((int*)item_embedding_table, item_ids, dim);
__m512 dot = _mm512_mul_ps(query, items);
// 这是无法避免的——indices 是动态的
```

## 5. AoS vs SoA vs AoSoA

这是单指令多数据（SIMD）编程中最关键的数据布局决策。

### 5.1 Array of Structures（AoS）

```c
// AoS：结构体数组——自然的数据表示，但对 SIMD 不友好
struct Particle {
    float x, y, z;      // 12 字节
    float vx, vy, vz;   // 12 字节
    float mass;          // 4 字节
};
// 总共 28 字节，对齐到 32

Particle particles[1024];  // AoS
```

**SIMD 访问 AoS 的问题**：
```c
// 想要做 x += vx * dt (所有粒子的 x 坐标更新)
// AoS 布局：x[0] 在 offset 0, x[1] 在 offset 28 (跨 cache line!)
for (int i = 0; i < 1024; i += 8) {
    // 需要从不同位置 gather 8 个 x 值——极其低效！
    // particles[i].x, particles[i+1].x, ... particles[i+7].x
    // 它们的间隔是 sizeof(Particle) = 28 字节，无法连续加载
}
```

### 5.2 Structure of Arrays（SoA）

```c
// SoA：每个字段独立成数组——SIMD 理想布局
struct ParticleSystem {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
};

// 访问粒子 i 的 x 坐标：x[i]（简单！）
float* xs = ps.x;
float* vxs = ps.vx;
for (int i = 0; i + 8 <= 1024; i += 8) {
    __m256 xv = _mm256_loadu_ps(xs + i);     // 连续加载，完美
    __m256 vxv = _mm256_loadu_ps(vxs + i);   // 连续加载，完美
    xv = _mm256_fmadd_ps(vxv, dt_vec, xv);   // FMA，一条指令
    _mm256_storeu_ps(xs + i, xv);
}
```

**SoA 优势总结**：
- 连续内存访问（> 矢量加载带宽）
- 无需 shuffle/gather
- 预取可预测（线性地址模式）
- 对 cache 更友好（没有"空洞"）

### 5.3 AoSoA 混合布局

对于有多个字段的结构，纯 SoA 的一个问题是：访问单个粒子的所有字段时，需要从多个分散的数组中加载。如果同时需要粒子的 x, y, z，SoA 要求三次独立的流式加载，可能对 cache 和 TLB 不友好。

**AoSoA**（Array of Struct of Arrays）将数据分成 SIMD 宽度的小块：

```c
// AoSoA：每 8 个元素为一个块，每个块内是 AoS 布局
// 但字段用 SoA 组织
#define WIDTH 8  // SIMD 向量宽度

struct ParticleBlock {
    float x[WIDTH];      // 8 个粒子的 x 坐标（连续）
    float y[WIDTH];
    float z[WIDTH];
    float vx[WIDTH];
    float vy[WIDTH];
    float vz[WIDTH];
};

ParticleBlock blocks[NUM_PARTICLES / WIDTH];

// 访问所有 8 个粒子的 x：一次 32 字节加载
// 访问某个粒子的所有字段：都在同一 cache line 附近
```

**AoSoA 的性能特性**：
- 每个字段的 8 个值连续（可以 SIMD 加载）
- 一个粒子的所有字段在空间上接近（cache 友好）
- 块大小等于一个 cache line 时最优（64 / 4 / 8 = 正好 2 个字段可在同一行）

## 6. Cache 层次结构

### 6.1 典型 x86 Cache 参数（Intel Sapphire Rapids 为例）

```
层级    大小     关联度    延迟    带宽          线大小
L1d     48KB     12-way   4-5c    ~2TB/s(读)    64B
L2      2MB      16-way   14c     ~1TB/s        64B
L3      105MB    15-way   50-55c  ~200GB/s      64B
DRAM    -        -        ~100ns  ~100GB/s      64B (行大小，非线)
```

**注意**：这些数字因 CPU 型号、内存配置、负载类型而异。L1 带宽可达每周期 2×512bit（2×64 字节）加载 + 1×512bit 存储。

### 6.2 Cache Line 行为

x86 cache line 为 **64 字节**（自 Pentium 4 以来未变）。这意味着：

```c
// 访问地址 A 会加载包含 A 的整个 64 字节行到 cache
// 地址的低 6 位（0-63）确定行内偏移，高位确定 cache set

float data[16];  // 64 字节 = 一个 cache line，恰好
// 访问 data[0] 会预取 data[0..15] 到 L1

// 结构体大小应为 64 字节的倍数以避免浪费带宽
struct __attribute__((aligned(64))) OptBlock {
    float values[16];   // 64 字节
    // 如果再加 4 字节，整个结构变成 68 字节→对齐到 128→浪费 60 字节
};
```

### 6.3 False Sharing

当两个线程访问不同变量，但这些变量位于同一个 cache line 时：

```c
// 错误设计：两个线程各自递增不同的计数器
struct Counters {
    int counter_a;  // 线程 A
    int counter_b;  // 线程 B
    // counter_a 和 counter_b 在同一 cache line (64 字节内)！
};
struct Counters cnt;  // 全局

// 线程 A: cnt.counter_a++
// 线程 B: cnt.counter_b++
// 即使操作不同的变量，每次写入会无效化对方的 cache line
// → 频繁的 cache coherence 流量，性能退化 10-100x！
```

**解决方案**：填充到不同 cache line：

```c
struct Counters {
    alignas(64) int counter_a;
    alignas(64) int counter_b;  // 强制在不同 cache line
};
// 或直接的填充：
struct Counters {
    int counter_a;
    char _pad[60];  // 填充到 64 字节
    int counter_b;
    char _pad2[60];
};
```

## 7. Prefetch（预取）

### 7.1 软件预取指令

```c
#include <xmmintrin.h>  // _mm_prefetch

// 基本调用
_mm_prefetch((const char*)ptr, _MM_HINT_T0);   // 预取到 L1
_mm_prefetch((const char*)ptr, _MM_HINT_T1);   // 预取到 L2
_mm_prefetch((const char*)ptr, _MM_HINT_T2);   // 预取到 L3
_mm_prefetch((const char*)ptr, _MM_HINT_NTA);  // 非时间预取（最小污染）
```

**各级别用途**：

| 提示 | 目标 | 使用场景 | 说明 |
|------|------|---------|------|
| `_MM_HINT_T0` | L1 | 即将使用（下一个循环迭代） | 最激进，污染 L1 |
| `_MM_HINT_T1` | L2 | 下一个 kernel 使用 | 中度激进 |
| `_MM_HINT_T2` | L3 | 未来的 kernel 使用 | 轻量，不污染 L1/L2 |
| `_MM_HINT_NTA` | L3（非时间） | 流式访问（使用后丢弃） | 最小 cache 污染 |

### 7.2 预取距离

预取需要在数据被使用前足够早地发出，但不要太早（否则在需要前被逐出）：

```c
// 简单的顺序访问循环
void process_array(float* data, int n) {
    const int PREFETCH_DISTANCE = 256;  // 预取距离：256×8×4 = 8KB 超前，约 500 周期
    
    for (int i = 0; i + 8 <= n; i += 8) {
        // 预取远处的数据（如果还在数组内）
        if (i + PREFETCH_DISTANCE * 8 < n) {
            _mm_prefetch((const char*)(data + i + PREFETCH_DISTANCE * 8), 
                         _MM_HINT_T0);
        }
        
        __m256 v = _mm256_loadu_ps(data + i);
        v = _mm256_mul_ps(v, v);  // 模拟计算
        _mm256_storeu_ps(data + i, v);
    }
}
```

**预取距离经验公式**：
```
prefetch_distance_cycles = 目标延迟（周期）
  L1: ~4 周期 → 非常短的预取距离（一般硬件预取器自动处理）
  L2: ~14 周期 → 可以不用软件预取
  L3: ~55 周期 → prefetch_distance = 55 周期 / (循环每迭代周期)
  DRAM: ~200-300 周期 → 通常由硬件跨步预取器处理
```

**一般不建议**手动软件预取，除非：
1. 你使用非顺序访问模式（硬件预取器无法预测）
2. 你确认硬件预取器就是瓶颈（通过 `perf stat -e l2_rqsts.pf_miss` 等计数器验证）
3. gather/scatter 或间接索引访问

### 7.3 硬件预取器

现代 x86 CPU 有强大的硬件预取器，通常不需要手动 `_mm_prefetch`：

```
L1 预取器：
  - DCU（Data Cache Unit）预取器：自动将下一条 cache line 加载到 L1
  - IP（Instruction Pointer）跨步预取器：检测恒定跨步并预取

L2 预取器：
  - 流预取器（Stream Prefetcher）：检测顺序访问模式
  - 空间预取器（Spatial Prefetcher）：检测相邻 cache line 访问模式
```

**测试硬件预取器是否工作**：
```bash
# 对比有/无预取时的 L1 和 L2 未命中率
perf stat -e L1-dcache-load-misses,l2_rqsts.miss ./prog
```

## 8. 内存带宽层次

### 8.1 理论峰值带宽

```
                                                   理论峰值带宽（每核）
L1 数据 Cache：   2×512-bit load + 1×512-bit store  → ~2TB/s
L2 Cache：        1×512-bit load (64B/cycle)         → ~1TB/s  
L3 Cache：        取决于环形总线/网格                  → ~200GB/s
DDR5-5600 内存：  2 channels × 5600 MT/s            → ~90GB/s（总）
```

### 8.2 STREAM 基准测试风格的内存访问

```c
// STREAM Copy: c[i] = a[i]
void stream_copy(float* c, const float* a, int n) {
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(c + i, v);
    }
    // 内存带宽：2 × 32 字节 / 迭代 → 理论约 150 GB/s (DDR5, 2 channel)
    // 实际受限于内存带宽（~90 GB/s）
}
```

### 8.3 Roofline Model on x86

Roofline 模型将内核性能分为两类限制因素：

```
性能 = min(峰值计算能力, 操作强度 × 峰值内存带宽)

操作强度 = FLOPS / 加载+存储的字节数 = (浮点运算数) / (内存流量)
```

**Roofline 分析示例表**（Intel Sapphire Rapids @ 2.5 GHz）：

| 操作强度 (FLOP/Byte) | 限制因素 | 可实现性能 | 示例操作 |
|---------------------|---------|-----------|---------|
| < 16 | 内存带宽 | ~90 GB/s×OI | memcpy, STREAM Copy |
| 16-32 | 过渡区 | 两者 | ReLU, 向量加法 |
| > 32 | 计算 | ~80 GFLOPS | GEMM, 矩阵乘法 |

```
操作强度 = FLOP / Byte

对于 ReLU: 1 FLOP (max) / (4 字节读 + 4 字节写) = 1/8 FLOP/Byte
   → 内存带宽限制，绑定在 ~90 GB/s 上
   → 性能 = 90 × 1/8 = 11.25 GFLOPS

对于 GEMM (N×N): N^3 FLOP / (N^2 + N^2 × 2)Byte ≈ N/3 FLOP/Byte
   → 当 N 足够大时，操作强度趋向无穷
   → 计算限制，绑定在 80 GFLOPS（接近峰值）
```

## 9. 实际的内存布局决策矩阵

```
┌──────────────────────────────────────────────────────────────┐
│                     我应该在哪种布局？                          │
├──────────────────────────────────────────────────────────────┤
│ 是否每个字段都需要独立 SIMD 处理？                              │
│   ├─ 是 → SoA（完美的连续访问，最佳的 SIMD 效率）               │
│   └─ 否 → 所有字段总是一起访问？                                │
│           ├─ 是 → AoS（cache 友好，但不是 SIMD 友好）           │
│           └─ 混合 → AoSoA（两者兼顾，最佳实践）                  │
│                                                              │
│ 数据量 > L2 Cache？ → 使用流存储和非时间预取                      │
│ 数据量 > L3 Cache？ → 确保顺序访问模式被硬件预取器识别             │
│ 多线程？ → 警惕 false sharing，对齐到 64 字节                    │
│ 间接索引（gather）无法避免？ → 先排序索引再访问                    │
└──────────────────────────────────────────────────────────────┘
```

## 10. 完整的对齐与布局示例

```c
#include <immintrin.h>
#include <malloc.h>
#include <string.h>

// SoA 粒子系统（SIMD 友好）
typedef struct {
    float* x;    // 连续数组
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    int    count;
} ParticleSystemSoA;

// 分配并初始化 SoA 粒子系统
ParticleSystemSoA* create_particles_soa(int n) {
    ParticleSystemSoA* ps = (ParticleSystemSoA*)malloc(sizeof(*ps));
    int padded_n = ((n + 7) / 8) * 8;  // 向上取整到 8 的倍数
    
    // 所有数组 64 字节对齐
    ps->x  = (float*)_mm_malloc(padded_n * sizeof(float), 64);
    ps->y  = (float*)_mm_malloc(padded_n * sizeof(float), 64);
    ps->z  = (float*)_mm_malloc(padded_n * sizeof(float), 64);
    ps->vx = (float*)_mm_malloc(padded_n * sizeof(float), 64);
    ps->vy = (float*)_mm_malloc(padded_n * sizeof(float), 64);
    ps->vz = (float*)_mm_malloc(padded_n * sizeof(float), 64);
    ps->count = n;
    
    // 初始化为零（包括 padding）
    memset(ps->x, 0, padded_n * sizeof(float));
    memset(ps->y, 0, padded_n * sizeof(float));
    memset(ps->z, 0, padded_n * sizeof(float));
    memset(ps->vx, 0, padded_n * sizeof(float));
    memset(ps->vy, 0, padded_n * sizeof(float));
    memset(ps->vz, 0, padded_n * sizeof(float));
    
    return ps;
}

// SoA 粒子位置更新：x += vx * dt（所有粒子）
// 连续的 SIMD 循环
void update_positions_soa(ParticleSystemSoA* ps, float dt) {
    __m256 dt_vec = _mm256_set1_ps(dt);
    for (int i = 0; i + 8 <= ps->count; i += 8) {
        __m256 x_vec  = _mm256_loadu_ps(ps->x + i);
        __m256 vx_vec = _mm256_loadu_ps(ps->vx + i);
        x_vec = _mm256_fmadd_ps(vx_vec, dt_vec, x_vec);  // x += vx * dt
        _mm256_stream_ps(ps->x + i, x_vec);  // 非时间存储：不复用 x
        
        __m256 y_vec  = _mm256_loadu_ps(ps->y + i);
        __m256 vy_vec = _mm256_loadu_ps(ps->vy + i);
        y_vec = _mm256_fmadd_ps(vy_vec, dt_vec, y_vec);
        _mm256_stream_ps(ps->y + i, y_vec);
        
        __m256 z_vec  = _mm256_loadu_ps(ps->z + i);
        __m256 vz_vec = _mm256_loadu_ps(ps->vz + i);
        z_vec = _mm256_fmadd_ps(vz_vec, dt_vec, z_vec);
        _mm256_stream_ps(ps->z + i, z_vec);
    }
    _mm_sfence();  // 确保流存储完成
}

void free_particles_soa(ParticleSystemSoA* ps) {
    _mm_free(ps->x);  _mm_free(ps->y);  _mm_free(ps->z);
    _mm_free(ps->vx); _mm_free(ps->vy); _mm_free(ps->vz);
    free(ps);
}
```

编译运行：
```bash
gcc -mavx2 -mfma -O2 -o particles particles.c
# 用 perf 验证 L1 命中率和内存带宽
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,llc-misses ./particles
```
