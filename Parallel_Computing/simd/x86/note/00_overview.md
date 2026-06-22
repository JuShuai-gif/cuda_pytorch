# x86 SIMD 概述：从 MMX 到 AVX-512

## 1. x86 SIMD 指令集演进史

### 1.1 MMX（1997，Pentium MMX）

MMX（MultiMedia eXtensions）是 x86 架构第一个 SIMD 指令集：

- **位宽**：64 位（复用 x87 FPU 的 8 个 80 位寄存器低 64 位，命名为 mm0-mm7）
- **数据类型**：仅整数 —— 8x8bit、4x16bit、2x32bit
- **关键限制**：与 x87 FPU 共享寄存器，浮点和 MMX 代码不能交错执行，需要 `emms` 指令切换状态
- **无新寄存器**：这导致了严重的上下文切换开销

```c
// MMX 示例：饱和加（用于像素混合）
__m64 a = _mm_set_pi16(100, 200, 50, 10);
__m64 b = _mm_set_pi16(50, 100, 200, 20);
__m64 c = _mm_adds_pi16(a, b);  // 150, 255(saturated), 250, 30
_mm_empty();  // 必须！清除 MMX 状态，恢复 FPU
```

MMX 标志着 x86 正式进入 SIMD 时代，但因为寄存器复用问题，实际工程使用繁琐。

### 1.2 SSE（1999，Pentium III）

SSE（Streaming SIMD Extensions）解决了 MMX 的核心痛点：

- **位宽**：128 位
- **寄存器**：全新独立寄存器组 **xmm0-xmm7**（x86-64 扩展到 xmm0-xmm15）
- **数据类型**：4x32bit 单精度浮点（首次支持浮点 SIMD）
- **关键指令**：对齐/非对齐加载存储、算术运算、shuffle、cache 控制

```c
#include <xmmintrin.h>  // SSE 头文件

__m128 a = _mm_load_ps(ptr);         // 16 字节对齐加载
__m128 b = _mm_loadu_ps(ptr2);       // 非对齐加载（代价更高）
__m128 c = _mm_add_ps(a, b);
__m128 d = _mm_mul_ps(a, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0,0,0,0)));
_mm_store_ps(dst, d);
```

SSE 的革命性在于：独立的寄存器文件 + 浮点支持。这直接让 3D 游戏和科学计算的向量运算性能翻倍。

### 1.3 SSE2（2000，Pentium 4）

SSE2 将 SSE 扩展到整数和双精度浮点，是 SSE 系列中最重要的升级：

- **双精度浮点**：`__m128d`（2x f64），`_mm_add_pd` 等
- **整数运算**：`__m128i`，支持 16x8bit、8x16bit、4x32bit、2x64bit
- **新增**：`_mm_cvtsi32_si128`、64 位整数操作、cache 行 flush

```c
// SSE2 双精度示例
__m128d a = _mm_load_pd(ptr);
__m128d b = _mm_set1_pd(3.14159);
__m128d c = _mm_mul_pd(a, b);  // 2 个 double 并行乘法
```

SSE2 是 x86-64 ABI 的基线：所有 x86-64 处理器保证支持 SSE2。浮点参数通过 xmm 寄存器传递。

### 1.4 SSE3/SSSE3/SSE4（2004-2008）

**SSE3**（Prescott，2004）：
- 水平加减：`_mm_hadd_ps`（hadd 虽然有，但在关键路径上延迟高）  
- `_mm_lddqu_si128`（专为可能跨 cache line 的非对齐加载优化）
- `_mm_mwait`/`_mm_monitor`（线程同步）

**SSSE3**（Core 2，2006）：
- `_mm_shuffle_epi8`（PSHUFB）：**SIMD 精华指令**，以字节为单位的任意重排，实现查表、字节反转等功能
- `_mm_alignr_epi8`：双寄存器字节级拼接对齐
- 水平加减的整数版本：`_mm_hadd_epi16` 等

```c
// PSHUFB 的魅力：字节级重排/查表
__m128i table = _mm_set_epi8(/* 16 字节的查找表 */);
__m128i indices = _mm_loadu_si128((__m128i*)input);
__m128i result = _mm_shuffle_epi8(table, indices);
// 当 index 最高位为 1 时，输出 0 —— 实现条件置零！
```

**SSE4.1 / SSE4.2**（Penryn/Nehalem，2008）：
- `_mm_dp_ps`：4 元素点积（虽是一条指令，但延迟很高，不如手动 mul+add）
- `_mm_blendv_ps`：按掩码逐元素选择
- `_mm_minpos_epu16`：找 8 个 u16 中最小值及其位置
- `_mm_crc32_u8/u32/u64`：硬件 CRC32C（存储/网络校验）
- `_mm_cmpestri`：字符串比较，带范围控制（XML/JSON 解析利器）

### 1.5 AVX（2011，Sandy Bridge）

AVX（Advanced Vector Extensions）将向量宽度翻倍至 256 位：

- **寄存器**：**ymm0-ymm15**（256 位），xmm 为其低 128 位
- **VEX 编码**：3 操作数非破坏性指令（`c = a + b` 而非 `a = a + b`），代码密度和性能双赢
- **浮点**：8x f32 或 4x f64
- **关键限制**：仅浮点运算，整数仍在 128 位（由 SSE 处理）

```c
#include <immintrin.h>  // AVX 及以后的总头文件

// AVX: 8 个 float 并行
__m256 a = _mm256_load_ps(p);
__m256 b = _mm256_broadcast_ss(&scalar);
__m256 c = _mm256_add_ps(a, b);
// VEX 编码：c = a + b，不破坏 a
```

**AVX 的新增关键特性**：
- `_mm256_broadcast_ss`：标量广播到 256 位的所有 8 个 lane
- `_mm256_permute_ps` / `_mm256_permutevar8x32_ps`：lane 内重排
- `_mm256_extractf128_ps` / `_mm256_insertf128_ps`：跨 128 位边界操作
- 非对齐加载/存储的开销在现代 CPU 上大幅降低

**AVX 的致命弱点**：没有 256 位整数算术。整数运算必须降级到 128 位 SSE，效率大打折扣。

### 1.6 AVX2 + FMA3（2013，Haswell）

AVX2 补齐了 AVX 最关键的短板，是当前 x86 SIMD 工程的**黄金基线**：

**AVX2 新增内容**：

| 类别 | 关键指令 | 说明 |
|------|---------|------|
| 整数运算 | `_mm256_add_epi32`, `_mm256_mullo_epi32` | 256 位整数算术终于来了 |
| Gather | `_mm256_i32gather_ps` | 从非连续地址加载（微码实现，很慢） |
| 跨 lane 重排 | `_mm256_permutevar8x32_ps` | 任意 cross-lane 重排 |
| 广播 | `_mm256_broadcastss_ps` | 从内存广播（1 条指令） |
| 移位 | `_mm256_sllv_epi32` | 逐元素可变移位 |

**FMA3**（Fused Multiply-Add）：

```c
// FMA: a*b + c 在一个指令内完成，单次舍入
__m256 result = _mm256_fmadd_ps(a, b, c);  // = a*b + c
__m256 result2 = _mm256_fmsub_ps(a, b, c); // = a*b - c
__m256 result3 = _mm256_fnmadd_ps(a, b, c); // = -(a*b) + c
```

FMA 的两个关键优势：
1. 一次舍入代替两次（mul 再 add），精度更高
2. 一个 µop 完成两个运算，吞吐翻倍（每个周期 2 条 FMA 指令 → 16 flops/cycle f32）

**AVX2 的实际地位**：绝大多数 x86 服务器（2015+）和桌面（2013+）都支持 AVX2+FMA。深度学习推理框架（ONNX Runtime、OpenVINO）的 x86 后端以 AVX2 为基线。如果你只能选择一个目标指令集，**选 AVX2**。

### 1.7 AVX-512（2017+，多种子集）

AVX-512 不是一个单一指令集，而是一系列扩展的集合。理解这些子集对可移植性至关重要：

#### AVX-512 Foundation（AVX-512F）—— 核心

- **寄存器**：**zmm0-zmm31**（512 位，32 个！），ymm 为其低 256 位，xmm 为低 128 位
- **最多 32 个** 512 位寄存器（AVX2 只有 16 个 256 位），寄存器压力大幅降低
- **掩码寄存器 k0-k7**（64 位），基于掩码的条件执行
- 新的 EVEX 编码格式
- 基本算术、比较、转换等

#### 关键子集

| 子集 | 全称 | 关键特性 | 典型 CPU |
|------|------|---------|---------|
| AVX-512F | Foundation | 512 位基础运算、掩码 | 所有 AVX-512 CPU |
| AVX-512CD | Conflict Detection | `_mm512_conflict_epi32`，检测 scatter 冲突 | Skylake-X+ |
| AVX-512BW | Byte/Word | 8/16 位整数掩码操作，`kmov` 系列 | Skylake-X+ |
| AVX-512DQ | Dword/Qword | 32/64 位掩码和整数操作 | Skylake-X+ |
| AVX-512VL | Vector Length | 在 128/256 位向量上使用 AVX-512 特性（掩码等） | Skylake-X+ |
| AVX-512VNNI | Vector Neural Network Instructions | `_mm512_dpbusd_epi32`，u8×s8→i32 点积 | Cascade Lake+ |
| AVX-512BF16 | BF16 | bf16×bf16→fp32 点积 | Cooper Lake+, Sapphire Rapids |
| AVX-512VBMI | Vector Byte Manipulation | 字节级 permute，`vpermb` | Cannon Lake, Ice Lake+ |
| AVX-512FP16 | FP16 | 原生 fp16 算术（非仅转换） | Sapphire Rapids |

#### 掩码寄存器（k-register）

这是 AVX-512 最革命性的改进之一：

```c
// 无需 blend 指令！掩码直接控制哪些 lane 参与运算
__m512 a = _mm512_load_ps(src);
__m512 b = _mm512_set1_ps(0.0f);
__mmask16 mask = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);  // a > 0 的 lane 生成掩码

// 仅对 a > 0 的位置执行加法，其他位置保留 a 的原值
__m512 c = _mm512_mask_add_ps(a, mask, a, b);

// 仅对 a > 0 的位置执行加法，其他位置置零
__m512 d = _mm512_maskz_add_ps(mask, a, b);
```

在 AVX2 中这需要 `cmpps` → `andps` → `andnps` → `orps`（至少 4 条指令），AVX-512 一条 `vaddps {k1}` 完成。

#### 嵌入式舍入与 SAE

```c
// SAE = Suppress All Exceptions，同时指定舍入模式
__m512 result = _mm512_add_round_ps(a, b, 
    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
// 可用于：
// _MM_FROUND_TO_NEAREST_INT    - 最近偶数舍入
// _MM_FROUND_TO_NEG_INF        - 向负无穷
// _MM_FROUND_TO_POS_INF        - 向正无穷
// _MM_FROUND_TO_ZERO           - 向零
```

这对于实现高精度数值算法（如 Kahan summation）非常有用，不需要修改全局 MXCSR。

### 1.8 Intel AMX（2023，Sapphire Rapids）

AMX（Advanced Matrix Extensions）是超越 SIMD 的矩阵加速器：

- **tile 寄存器**：8 个 1KB tile 寄存器（tmm0-tmm7），每个可配置为 16x16 bf16 等形状
- **TMUL**：tile 矩阵乘法，一个指令完成 16x16×16x16→16x16 i32 累加
- **适用场景**：大规模 GEMM，超越传统 SIMD GEMM 微内核 4-8x

AMX 已超出本教程范围（需要内核级上下文保存/恢复支持），但它是理解 x86 AI 加速演进的关键一环。

## 2. 寄存器演进全景

```
                 SSE               AVX/AVX2           AVX-512
              ┌─────────┐       ┌─────────┐       ┌─────────────┐
  128-bit    │ xmm0-15 │       │ xmm0-15 │       │ xmm0-31     │
              └─────────┘       └─────────┘       └─────────────┘
                                               
                               ┌─────────┐       ┌─────────────┐
  256-bit                     │ ymm0-15 │       │ ymm0-31     │
                               └─────────┘       └─────────────┘
                                               
                                                 ┌─────────────┐
  512-bit                                        │ zmm0-31     │
                                                 └─────────────┘
                                               
                                                 ┌─────────────┐
  Mask (64-bit)                                  │ k0-k7       │
                                                 └─────────────┘
```

**关于 xmm16-xmm31 和 ymm16-ymm31**：这些寄存器只有 AVX-512 才存在。在 AVX/AVX2 时代，即使 x86-64 也只有 16 个 128/256 位寄存器。

**k0 的特殊用途**：
- k0 是"无掩码"的硬件编码（硬连线全 1），用于不需要掩码的指令
- k1-k7 是真正的可编程掩码寄存器

## 3. 为什么 AVX-512 重要

### 3.1 数量级优势

| 指标 | AVX2 | AVX-512 | 倍率 |
|------|------|---------|------|
| 浮点寄存器数 | 16 | **32** | 2x |
| f32 元素/寄存器 | 8 | **16** | 2x |
| 总 f32 寄存器容量 | 128 | **512** | 4x |
| 掩码机制 | 需要 3-5 条指令模拟 | **原生 1 条** | - |
| 跨 lane 重排 | 有限 | **丰富** | - |

### 3.2 实践中的降频问题

长期以来，AVX-512 被诟病于"一跑就降频"。这个问题的真相：

**Intel 历史**：
- Skylake-X（2017）：AVX-512 确实会导致显著降频（几百 MHz），因为 512 位执行单元热密度大
- Ice Lake（2019+）：大幅改善，降频通常 < 100 MHz
- Sapphire Rapids（2023）：基本不再是一个问题

**AMD Zen4（2022+）**：
- Zen4 用两个 256 位执行单元拼接实现 AVX-512，功耗控制优秀
- 基本没有额外降频，因为物理上就是 256 位宽度

**工程建议**：在现代硬件（2020+）上，AVX-512 的性能优势（2x 宽度 + 掩码）远大于可能的轻微降频。不要因为过去的刻板印象而放弃 AVX-512。

## 4. 工业应用全景

### 4.1 云上机器学习推理

x86 服务器（Intel Xeon、AMD EPYC）是云端 ML 推理的主力：

- **ONNX Runtime**：AVX2/VNNI 后端，int8 量化推理
- **OpenVINO**：Intel 深度优化的推理引擎，充分利用 VNNI/AMX
- **TensorFlow Lite XNNPACK**：针对移动/边缘 x86 优化
- **llama.cpp**：社区 LLM 推理框架，广泛使用 AVX2/FMA

典型场景：推荐系统（双塔模型内积）、BERT 推理、Whisper 语音识别。这些工作负载中，AVX-512 VNNI 相比 AVX2 可以获得 2-4x 加速。

### 4.2 高频交易（HFT）

金融行业是 AVX-512 的早期采用者：

- 订单簿匹配引擎：SIMD 扫描价格/数量数组
- 风险计算：蒙特卡洛模拟中的向量化路径生成
- 时间序列处理：移动平均、波动率计算

快 1 微秒就是竞争优势，金融公司愿意为 AVX-512 硬件付出溢价。

### 4.3 视频编解码

x264/x265 等编解码器广泛使用手写汇编的 SIMD 路径：

- 运动估计：SAD（Sum of Absolute Differences）用 `psadbw` 加速
- DCT/IDCT：蝶形变换的 SIMD 实现
- 去块滤波：边界条件判断 + 逐像素操作
- 色彩空间转换：YUV↔RGB，大量使用 shuffle 和 FMA

x265 中超过 60% 的计算来自 SIMD 优化代码。

### 4.4 数据库引擎

现代分析型数据库极度依赖 SIMD：

**ClickHouse**：
- `memcpy` 用 SIMD 实现（32 字节/周期以上的带宽）
- 聚合函数（sum/count/avg）使用 AVX2/AVX-512
- 字符串搜索：`memchr` 的 AVX2 实现可达到 32+ GB/s

**DuckDB**：
- 列式存储的向量化执行引擎
- Hash join 中的 SIMD 哈希表查找
- 字符串操作：大小写转换、trim 等的 SIMD 加速

### 4.5 游戏引擎

Unreal Engine 5 的 SIMD 数学库：

- SSE/AVX 向量/矩阵运算（`FVector`、`FMatrix`）
- 骨骼动画变换：每帧数千次矩阵乘法
- 粒子系统物理：AABB 碰撞检测用 SIMD 比较
- 解压纹理数据（BCn 格式）

### 4.6 密码学

OpenSSL / BoringSSL 对 SIMD 的重度使用：

- AES：Intel AES-NI 专有指令（`_mm_aesenc_si128`）
- SHA：SHA-NI 扩展
- ChaCha20-Poly1305：AVX2 并行处理多个数据块
- 大整数算术（RSA/ECC）：AVX-512 IFMA（整数 FMA，52 位精度）

## 5. 编译器标志

### 5.1 GCC/Clang 目标指令集

```bash
# AVX2 + FMA 基线（推荐的最低要求）
gcc -mavx2 -mfma -o prog prog.c

# 完整的常用 AVX-512 子集
gcc -mavx512f -mavx512bw -mavx512vl -mavx512dq -o prog prog.c

# 启用 VNNI（int8 推理加速）
gcc -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -o prog prog.c

# 构建机器的本地指令集（不可分发，仅用于本地测试）
gcc -march=native -o prog prog.c

# 对特定 CPU 优化（需查阅文档确认支持哪些子集）
gcc -march=icelake-server -o prog prog.c
gcc -march=znver4 -o prog prog.c          # AMD Zen4
```

### 5.2 目标选择的影响

使用 `-march=native` 时，编译器**可能**自动生成 AVX-512 指令（如果本地 CPU 支持），即使你写的是纯 C。这对性能测试有利，但构建出的二进制文件无法在旧 CPU 上运行。

**生产建议**：
- 构建分发版本时，明确指定最低目标指令集
- 使用运行时检测（如下节所述）在支持高级指令集时使用优化路径

## 6. CPU 特性检测

### 6.1 编译时检测

```c
// GCC/Clang 预定义宏
#if defined(__AVX512F__)
    // AVX-512F 已启用
#endif
#if defined(__AVX512BW__)
    // AVX-512BW 已启用
#endif
#if defined(__FMA__)
    // FMA 已启用
#endif
```

### 6.2 运行时检测

```c
#ifdef __GNUC__
#include <cpuid.h>
#endif

int supports_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;  // AVX2 = CPUID.07H.EBX[5]
    }
    return 0;
}

int supports_avx512f(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;  // AVX-512F = CPUID.07H.EBX[16]
    }
    return 0;
}

// 或使用 GCC 内置函数（更简单）
int supports_avx2_simple(void) {
    return __builtin_cpu_supports("avx2");
}
int supports_avx512f_simple(void) {
    return __builtin_cpu_supports("avx512f");
}
```

### 6.3 函数多版本（Function Multi-Versioning）

GCC/Clang 支持为同一函数提供多个 SIMD 版本，运行时自动选择：

```c
// 基础版本（无 SIMD 要求）
__attribute__((target("default")))
float sum_array(const float* arr, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += arr[i];
    return s;
}

// AVX2 优化版本
__attribute__((target("avx2,fma")))
float sum_array(const float* arr, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i + 8 <= n; i += 8)
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(arr + i));
    float result[8];
    _mm256_storeu_ps(result, sum);
    float s = 0;
    for (int i = 0; i < 8; i++) s += result[i];
    for (int i = n - n % 8; i < n; i++) s += arr[i];
    return s;
}

// AVX-512 优化版本
__attribute__((target("avx512f")))
float sum_array(const float* arr, int n) {
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i + 16 <= n; i += 16)
        sum = _mm512_add_ps(sum, _mm512_loadu_ps(arr + i));
    return _mm512_reduce_add_ps(sum);
}
```

编译时添加 `-mavx2 -mfma -mavx512f` 标志。运行时，动态链接器根据 CPU 能力选择版本。

## 7. AMD vs Intel AVX-512

### 7.1 AMD Zen4 的实现方式

Zen4 采用"双泵"（double-pumping）策略：

- 物理执行单元宽度为 256 位
- 一个 AVX-512 指令被拆分为两个 256 位 µop，在两个连续周期内执行
- 结果：**吞吐量减半**（相对于"真"512 位），但**延迟也相应增加**

```
Intel 原生 512 位：     [=== 512-bit ALU ===]  → 1 条指令/周期
AMD Zen4 双泵：         [== 256-bit ALU ==][== 256-bit ALU ==]  → 0.5 条 512 位指令/周期
```

### 7.2 实际影响

- 对于**内存带宽受限**的工作负载（如 streaming sum、memcpy），Zen4 和 Intel 性能相近，因为瓶颈在内存
- 对于**计算密集**负载（如 GEMM、FFT），Zen4 的 AVX-512 吞吐约为 Intel 的 50-70%（考虑更高的频率弥补）
- **关键优势**：即使用 256 位双泵，有 32 个寄存器 + 掩码 + 更强 permute 仍然比 AVX2 有显著提升

### 7.3 可移植代码建议

编写同时适合 Intel 和 AMD 的 AVX-512 代码：

```c
// 好的：充分使用掩码，减少 blend
__mmask16 mask = _mm512_cmp_ps_mask(a, zero, _CMP_GT_OQ);
a = _mm512_mask_mul_ps(a, mask, a, factor);

// 避免：指望单周期 512 位吞吐来做极窄的延迟关键循环
// 更好的：增加指令级并行，隐藏延迟
```

## 8. 学习路线

```
SSE2 基础（128 位、加载/存储/算术/转换）
    ↓
AVX2 核心（256 位、FMA、整数、gather、跨 lane shuffle）
    ↓
AVX-512 进阶（512 位、掩码、compress/expand、VNNI）
    ↓
领域专项（ML 推理内核、视频编解码、密码学、数据库）
```

**建议的学习项目顺序**：
1. 实现 memcpy（理解加载/存储和带宽）
2. 实现 ReLU / Softmax（理解条件执行和归约）
3. 实现矩阵乘法微内核（理解寄存器分块和 FMA）
4. 实现 LayerNorm（理解两遍算法和 Welford 方差）
5. 实现某种关键内核（根据你的领域选择）

## 9. 参考资料

| 资源 | 说明 |
|------|------|
| [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) | 在线查询每条 intrinsic 的指令、延迟、吞吐 |
| [uops.info](https://uops.info/) | 每条指令在不同微架构上的精确 µop 分解和延迟/吞吐 |
| Agner Fog's Optimization Manuals | x86 微架构深度分析，必读 |
| [simdjson](https://github.com/simdjson/simdjson) | 工业级 SIMD JSON 解析器，代码即文档 |
| [xsimd](https://github.com/xtensor-stack/xsimd) | C++ SIMD 包装库，学习 intrinsic 模式 |
| [highway](https://github.com/google/highway) | Google 的可移植 SIMD 库，支持多平台 |
