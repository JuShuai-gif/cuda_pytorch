# CPUID 与操作系统 SIMD 状态检测

```
-------------------------------------------------------------------------------
Reference:    Modern X86 Assembly Language Programming, 2nd Ed, Chapter 16
              Intel SDM Vol.2A, Chapter 3 (CPUID instruction)
              AMD APM Vol.3, Appendix E
Audience:     Engineers implementing runtime SIMD dispatch
Key goal:     Reliably detect CPU features AND OS support before using SIMD
-------------------------------------------------------------------------------
```

---

## 1. 为何 CPUID 检测很重要

在不支持 AVX/AVX-512 指令的 CPU 上使用这些指令 → **SIGILL**
（非法指令，立即崩溃）。但即使 CPU 支持 AVX-512，**操作系统也必须
通过 XSAVE 状态管理来支持它**。

### 三层检测

| 层级 | 需检查项 | 方法 |
|------|---------|------|
| 1. CPU 硬件 | CPU 是否拥有该指令？ | CPUID 叶 1/7 |
| 2. 操作系统支持 | 操作系统是否保存/恢复 SIMD 状态？ | XGETBV (XCR0) |
| 3. BIOS | BIOS 中是否启用了该特性？ | 同层级 1/2 |

---

## 2. CPUID 指令基础

### 2.1 调用约定

```c
// Input:  EAX = leaf (function ID), ECX = subleaf (0 for most leaves)
// Output: EAX, EBX, ECX, EDX = leaf-specific results

// Inline assembly version:
static inline void cpuid_raw(uint32_t leaf, uint32_t subleaf,
                              uint32_t* eax, uint32_t* ebx,
                              uint32_t* ecx, uint32_t* edx) {
    __asm__ __volatile__(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf)
        : "memory"
    );
}

// GCC builtin (simpler):
#include <cpuid.h>
unsigned int eax, ebx, ecx, edx;
__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);

// Or the even simpler:
__builtin_cpu_supports("avx2");
__builtin_cpu_supports("avx512f");
```

### 2.2 SIMD 检测的关键 CPUID 叶

| 叶 | 子叶 | 寄存器 | 位 | 特性 |
|------|---------|----------|-----|---------|
| 01H | 0 | ECX[0] | SSE3 | Prescott 新指令 |
| 01H | 0 | ECX[9] | SSSE3 | 补充 SSE3 |
| 01H | 0 | ECX[19] | SSE4.1 | Penryn |
| 01H | 0 | ECX[20] | SSE4.2 | Nehalem |
| 01H | 0 | ECX[23] | POPCNT | 人口计数 |
| 01H | 0 | ECX[28] | AVX | Sandy Bridge |
| 01H | 0 | ECX[25] | AESNI | AES 指令 |
| 01H | 0 | ECX[26] | XSAVE | XSAVE/XRSTOR（AVX 必需） |
| 01H | 0 | EDX[25] | SSE | 流式 SIMD 扩展 |
| 01H | 0 | EDX[26] | SSE2 | SSE2（x86-64 保证支持） |
| 07H | 0 | EBX[3] | BMI1 | 位操作指令 |
| 07H | 0 | EBX[5] | AVX2 | Haswell |
| 07H | 0 | EBX[8] | BMI2 | 位操作指令 2 |
| 07H | 0 | EBX[16] | AVX-512F | 基础 |
| 07H | 0 | EBX[17] | AVX-512DQ | 双/四字 |
| 07H | 0 | EBX[21] | AVX-512IFMA | 整数 FMA |
| 07H | 0 | EBX[26] | AVX-512PF | 预取 |
| 07H | 0 | EBX[27] | AVX-512ER | 指数与倒数 |
| 07H | 0 | EBX[28] | AVX-512CD | 冲突检测 |
| 07H | 0 | EBX[30] | AVX-512BW | 字节/字 |
| 07H | 0 | EBX[31] | AVX-512VL | 向量长度 |
| 07H | 0 | ECX[1] | AVX-512VBMI | 向量字节操作 |
| 07H | 0 | ECX[11] | AVX-512VNNI | 向量神经网络 |
| 07H | 0 | ECX[14] | AVX-512VPOPCNTDQ | 向量人口计数 |
| 07H | 1 | EAX[5] | AVX-512BF16 | BFloat16 |
| 0DH | 0 | EAX[0] | XSAVEOPT | 优化版 XSAVE |

### 2.3 厂商字符串

```c
void get_vendor_string(char vendor[13]) {
    unsigned int eax, ebx, ecx, edx;
    cpuid_raw(0, 0, &eax, &ebx, &ecx, &edx);

    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
    // Returns: "GenuineIntel", "AuthenticAMD", "CentaurHauls", etc.
}
```

### 2.4 处理器品牌字符串

```c
// Leaf 80000002H-80000004H: 48-byte brand string
void get_brand_string(char brand[49]) {
    unsigned int eax, ebx, ecx, edx;
    for (unsigned int leaf = 0x80000002; leaf <= 0x80000004; leaf++) {
        cpuid_raw(leaf, 0, &eax, &ebx, &ecx, &edx);
        int offset = (int)(leaf - 0x80000002) * 16;
        memcpy(brand + offset + 0,  &eax, 4);
        memcpy(brand + offset + 4,  &ebx, 4);
        memcpy(brand + offset + 8,  &ecx, 4);
        memcpy(brand + offset + 12, &edx, 4);
    }
    brand[48] = '\0';
    // Returns: "Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz"
}
```

---

## 3. XGETBV：操作系统 SIMD 状态支持

### 3.1 问题所在

AVX/AVX-512 引入了新的架构状态（YMM 高 128 位、ZMM 高 256 位、
opmask 寄存器 k1-k7）。操作系统必须在上下文切换时保存/恢复这些状态。
如果操作系统不支持，状态将在**上下文切换时丢失**。

### 3.2 通过 XGETBV 检查

```c
// XGETBV reads the extended control register (XCR)
// ECX = 0: XCR0 (extended feature enable mask)

static inline uint64_t xgetbv(uint32_t ecx) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(ecx) : "memory");
    return ((uint64_t)edx << 32) | eax;
}

// Check OS support for AVX:
//   XCR0 bit 1 = X87/SIMD state (always set)
//   XCR0 bit 2 = AVX state (YMM_Hi128)
uint64_t xcr0 = xgetbv(0);
int os_avx = (xcr0 & 6) == 6;  // bits 1 and 2 must be set

// Check OS support for AVX-512:
//   XCR0 bit 5 = Opmask state (k0-k7)
//   XCR0 bit 6 = ZMM_Hi256 state
//   XCR0 bit 7 = Hi16_ZMM state (ZMM16-ZMM31)
int os_avx512 = (xcr0 & 0xE6) == 0xE6;  // bits 1,2,5,6,7

// Complete detection sequence for AVX-512:
int has_avx512f = 0;
if (/* CPUID.07H.EBX[16] = AVX-512F */) {
    if (os_avx512) {
        has_avx512f = 1;  // Both CPU and OS support it!
    }
}
```

### 3.3 完整的安全检测函数

```c
int safe_has_avx512f(void) {
    unsigned int eax, ebx, ecx, edx;

    // Step 1: Check CPUID leaf 1 for XSAVE
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    if (!(ecx & (1u << 27))) return 0;  // OSXSAVE not available
    if (!(ecx & (1u << 26))) return 0;  // XSAVE not available

    // Step 2: Check XCR0 for OS-managed AVX-512 state
    uint32_t xcr_eax, xcr_edx;
    __asm__("xgetbv" : "=a"(xcr_eax), "=d"(xcr_edx) : "c"(0));
    if ((xcr_eax & 0xE6) != 0xE6) return 0;  // OS doesn't manage AVX-512 state

    // Step 3: Check CPUID leaf 7 for AVX-512F instruction support
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    if (!(ebx & (1u << 16))) return 0;  // AVX-512F not present

    return 1;  // ✅ AVX-512 fully available
}
```

---

## 4. 缓存拓扑检测（CPUID 叶 04H）

### 4.1 确定性缓存参数

```c
// CPUID leaf 04H provides cache topology in a standardized format.
// Repeated calls with ECX = 0,1,2,... enumerate each cache level.

typedef struct {
    int level;         // 1=L1, 2=L2, 3=L3
    int type;          // 1=Data, 2=Instruction, 3=Unified
    int size_kb;       // Total cache size in KB
    int ways;          // Associativity
    int line_size;     // Cache line size in bytes
    int sets;          // Number of sets
    int partitions;    // Physical line partitions
    int shared_by;     // Number of cores sharing this cache
} CacheInfo;

CacheInfo detect_cache(int ecx_index) {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid_count(4, ecx_index, &eax, &ebx, &ecx, &edx);

    CacheInfo ci = {0};
    ci.type       = (eax >> 5) & 0x07;
    if (ci.type == 0) return ci;  // no more cache levels

    ci.level      = (eax >> 5) & 0x07;
    ci.line_size  = ((ebx & 0xFFF) + 1);          // B = EBX[11:0] + 1
    ci.partitions = ((ebx >> 12) & 0x3FF) + 1;    // P = EBX[21:12] + 1
    ci.ways       = ((ebx >> 22) & 0x3FF) + 1;    // W = EBX[31:22] + 1
    ci.sets       = ecx + 1;                      // S = ECX + 1
    ci.size_kb    = (ci.ways * ci.partitions * ci.line_size * ci.sets) / 1024;
    ci.shared_by  = ((eax >> 14) & 0xFFF) + 1;
    return ci;
}

void print_cache_info(void) {
    printf("--- Cache Topology ---\n");
    for (int i = 0; ; i++) {
        CacheInfo ci = detect_cache(i);
        if (ci.type == 0) break;

        const char* type_str = "Unknown";
        if (ci.type == 1) type_str = "Data";
        else if (ci.type == 2) type_str = "Instruction";
        else if (ci.type == 3) type_str = "Unified";

        printf("  L%d %-11s: %4d KB, %2d-way, %3d B/line, "
               "shared by %d cores\n",
               ci.level, type_str, ci.size_kb, ci.ways,
               ci.line_size, ci.shared_by);
    }
}
```

示例输出：
```
--- Cache Topology ---
  L1 Data       :   32 KB,  8-way,  64 B/line, shared by 1 cores
  L1 Instruction:   32 KB,  8-way,  64 B/line, shared by 1 cores
  L2 Unified     :  256 KB,  4-way,  64 B/line, shared by 1 cores
  L3 Unified     : 8192 KB, 16-way,  64 B/line, shared by 4 cores
```

### 4.2 TLB 信息（CPUID 叶 02H / 18H）

要获取详细的 TLB 信息，可使用 CPUID 叶 18H（新版）或叶 02H（旧版）：

```c
// Leaf 18H subleaf 0: TLB parameters
unsigned int eax, ebx, ecx, edx;
__get_cpuid_count(0x18, 0, &eax, &ebx, &ecx, &edx);
int max_pa_bits = eax & 0xFF;  // Maximum physical address bits
```

---

## 5. 频率与拓扑

### 5.1 标称频率（叶 16H）

```c
unsigned int eax, ebx, ecx, edx;
if (__get_cpuid(0x16, &eax, &ebx, &ecx, &edx)) {
    int base_freq_mhz = eax & 0xFFFF;    // Processor Base Frequency (MHz)
    int max_freq_mhz  = ebx & 0xFFFF;    // Maximum Frequency (MHz)
    int bus_freq_mhz  = ecx & 0xFFFF;    // Bus (Reference) Frequency (MHz)
    printf("CPU Frequency: %d MHz base, %d MHz max (bus: %d MHz)\n",
           base_freq_mhz, max_freq_mhz, bus_freq_mhz);
}
```

### 5.2 核心与线程拓扑（叶 0BH / 1FH）

```c
// Modern: use leaf 0x1F (V2 Extended Topology Enumeration)
// Legacy: use leaf 0x0B

// Determine SMT (Hyper-Threading) support:
__get_cpuid(1, &eax, &ebx, &ecx, &edx);
int logical_cores = (ebx >> 16) & 0xFF;  // logical processors per package

// Existing /proc/cpuinfo or sysfs are usually easier for topology
```

---

## 6. 特性位完整枚举

### 6.1 单表列出所有相关特性位

```c
typedef struct {
    int sse3, ssse3, sse41, sse42;
    int avx, avx2;
    int fma;
    int avx512f, avx512dq, avx512cd, avx512bw, avx512vl;
    int avx512_ifma, avx512_vbmi, avx512_vnni, avx512_bf16;
    int avx512_vbmi2, avx512_vpopcntdq;
    int avx512_fp16;
    int aesni, pclmulqdq, rdrand, rdseed;
    int bmi1, bmi2, adx;
    int sha, sgx;
    int os_avx, os_avx512;
} CpuFeatures;

CpuFeatures detect_all_features(void) {
    CpuFeatures f = {0};
    unsigned int eax, ebx, ecx, edx;

    // Leaf 1
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    f.sse3  = (ecx >> 0)  & 1;
    f.ssse3 = (ecx >> 9)  & 1;
    f.sse41 = (ecx >> 19) & 1;
    f.sse42 = (ecx >> 20) & 1;
    f.avx   = (ecx >> 28) & 1;
    f.fma   = (ecx >> 12) & 1;
    f.aesni = (ecx >> 25) & 1;
    f.pclmulqdq = (ecx >> 1) & 1;
    int osxsave = (ecx >> 27) & 1;

    // Leaf 7.0
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    f.avx2       = (ebx >> 5)  & 1;
    f.bmi1       = (ebx >> 3)  & 1;
    f.bmi2       = (ebx >> 8)  & 1;
    f.avx512f    = (ebx >> 16) & 1;
    f.avx512dq   = (ebx >> 17) & 1;
    f.avx512cd   = (ebx >> 28) & 1;
    f.avx512bw   = (ebx >> 30) & 1;
    f.avx512vl   = (ebx >> 31) & 1;
    f.avx512_ifma = (ebx >> 21) & 1;
    f.rdseed     = (ebx >> 18) & 1;
    f.adx        = (ebx >> 19) & 1;
    f.sha        = (ebx >> 29) & 1;
    f.avx512_vnni      = (ecx >> 11) & 1;
    f.avx512_vpopcntdq = (ecx >> 14) & 1;

    // Leaf 7.1
    __get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx);
    f.avx512_bf16  = (eax >> 5)  & 1;
    f.avx512_vbmi2 = (ecx >> 6)  & 1;
    f.avx512_fp16  = (edx >> 23) & 1;

    // OS XSAVE check
    if (osxsave) {
        uint32_t xcr_eax, xcr_edx;
        __asm__("xgetbv" : "=a"(xcr_eax), "=d"(xcr_edx) : "c"(0));
        f.os_avx    = (xcr_eax & 0x6) == 0x6;
        f.os_avx512 = (xcr_eax & 0xE6) == 0xE6;
    }

    return f;
}
```

---

## 7. 生产级分派模式

### 7.1 CPUID + XGETBV 完整分派

```c
// Step 1: Detect at program start
static int g_isa_level = 0;  // 0=scalar, 1=SSE, 2=AVX, 3=AVX2, 4=AVX-512

void init_isa_dispatch(void) {
    CpuFeatures f = detect_all_features();

    if (f.avx512f && f.avx512bw && f.avx512vl && f.os_avx512) {
        g_isa_level = 4;  // AVX-512
    } else if (f.avx2 && f.fma && f.os_avx) {
        g_isa_level = 3;  // AVX2+FMA
    } else if (f.avx && f.os_avx) {
        g_isa_level = 2;  // AVX
    } else if (f.sse41) {
        g_isa_level = 1;  // SSE4.1
    } else {
        g_isa_level = 0;  // Scalar (always works)
    }
}

// Step 2: Dispatch to best kernel
void compute(float* c, const float* a, const float* b, int n) {
    switch (g_isa_level) {
        case 4: avx512_kernel(c, a, b, n); break;
        case 3: avx2_kernel(c, a, b, n);   break;
        case 2: avx_kernel(c, a, b, n);    break;
        case 1: sse_kernel(c, a, b, n);    break;
        default: scalar_kernel(c, a, b, n); break;
    }
}
```

---

## 8. 常见陷阱

### 8.1 CPUID 与 RDX 损坏

**重要**：在某些 CPU 微架构上，`cpuid` 指令可能会将 RDX 的高 32 位清零。
从汇编中调用 CPUID 时，始终保存 RDX（或在 System V 调用约定中通过
RDX 传递的任何指针参数）：

```asm
; Safe CPUID wrapper that preserves all other registers
cpuid_safe:
    push   rbx
    push   rdx              # save rdx (cpuid may corrupt it!)
    mov    eax, edi
    mov    ecx, esi
    cpuid
    pop    rcx              # restore saved rdx into rcx
    mov    [rcx],     eax
    mov    [rcx + 4], ebx
    mov    [rcx + 8], ecx
    mov    [rcx + 12], edx
    pop    rbx
    ret
```

### 8.2 OSXSAVE 位在 ECX 而非 EDX

`OSXSAVE` 特性标志在 CPUID.01H.**ECX**[27]，而非 EDX。
这是一个常见的混淆点，因为大多数其他基本特性都在 EDX 中。

### 8.3 Intel 与 AMD AVX-512 检测

- Intel：检查 AVX-512F + OSXSAVE + XCR0 位 5-7
- AMD Zen4：相同流程，但 Zen4 使用双泵 256 位单元实现 AVX-512，
  因此性能特征不同（相同 ISA，512 位指令的吞吐量减半）

### 8.4 在信号处理函数中检查

切勿在信号处理函数中调用 CPUID 或 XGETBV。某些操作系统内核
会根据进程上下文修改 CPUID 输出。始终在 `main()` 中或库初始化时
检测特性。

### 8.5 `__builtin_cpu_supports` 的局限性

GCC 的 `__builtin_cpu_supports` 仅检查 CPU 硬件，**不**检查操作系统支持。
对于 AVX/AVX-512，你仍然需要进行 XGETBV 检查：

```c
// WRONG: May return true even if OS doesn't manage AVX-512 state!
if (__builtin_cpu_supports("avx512f")) { ... }

// CORRECT: Check both hardware AND OS
int safe_avx512(void) {
    return __builtin_cpu_supports("avx512f") && os_supports_avx512();
}
```

---

## 9. 快速参考

```c
// ---- GCC built-in (hardware only) ----
__builtin_cpu_supports("avx2");
__builtin_cpu_supports("avx512f");

// ---- Linux getauxval ----
#include <sys/auxv.h>
unsigned long hwcap  = getauxval(AT_HWCAP);
unsigned long hwcap2 = getauxval(AT_HWCAP2);

// ---- Manual CPUID (full control) ----
#include <cpuid.h>
unsigned int eax, ebx, ecx, edx;
__get_cpuid(1, &eax, &ebx, &ecx, &edx);
__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);

// ---- XGETBV (OS support) ----
uint32_t eax, edx;
__asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
int os_avx512 = (eax & 0xE6) == 0xE6;

// ---- Complete safe check ----
int has_avx2 = __builtin_cpu_supports("avx2") && ((xgetbv() & 6) == 6);
int has_avx512 = __builtin_cpu_supports("avx512f") && ((xgetbv() & 0xE6) == 0xE6);
```

（文件结束 - 共 501 行）
