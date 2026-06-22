/**
 * cpuid_full_demo.cpp -- 完整的 CPUID 枚举演示
 *
 * 演示基于 CPUID 的完整 CPU 检测：
 *   1. 厂商字符串（GenuineIntel / AuthenticAMD）
 *   2. 品牌字符串（完整的处理器名称）
 *   3. 缓存拓扑（L1/L2/L3 大小、类型、关联度）
 *   4. 全部 SIMD 特性位（SSE → AVX-512 及子特性）
 *   5. XGETBV 操作系统 XSAVE 状态检查（AVX/AVX-512）
 *   6. CPU 频率信息
 *
 * 参考: Modern X86 Assembly Language Programming 2nd Ed, Chapter 16
 */

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

#include <cpuid.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * 底层 CPUID 和 XGETBV 包装函数
 * ================================================================ */

static inline void cpuid_raw(uint32_t leaf, uint32_t subleaf,
                              uint32_t* eax, uint32_t* ebx,
                              uint32_t* ecx, uint32_t* edx) {
    __get_cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
}

static inline int cpuid_has_leaf(uint32_t leaf) {
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    return leaf <= eax ? 1 : 0;
}

static inline uint64_t xgetbv_u64(uint32_t ecx) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(ecx) : "memory");
    return ((uint64_t)edx << 32) | eax;
}

/* ================================================================
 * 1. 厂商字符串（叶 0）
 * ================================================================ */

static void print_vendor_string(void) {
    uint32_t eax, ebx, ecx, edx;
    cpuid_raw(0, 0, &eax, &ebx, &ecx, &edx);

    char vendor[13];
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    printf("  Vendor:       %s\n", vendor);
    printf("  Max basic leaf: 0x%08X (%u)\n", eax, eax);
}

/* ================================================================
 * 2. 品牌字符串（叶 80000002H-80000004H）
 * ================================================================ */

static void print_brand_string(void) {
    uint32_t eax, ebx, ecx, edx;
    __get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx);
    if (eax < 0x80000004) {
        printf("  Brand:        (not available)\n");
        return;
    }

    char brand[49];
    for (uint32_t leaf = 0x80000002; leaf <= 0x80000004; leaf++) {
        cpuid_raw(leaf, 0, &eax, &ebx, &ecx, &edx);
        int off = (int)(leaf - 0x80000002) * 16;
        memcpy(brand + off + 0,  &eax, 4);
        memcpy(brand + off + 4,  &ebx, 4);
        memcpy(brand + off + 8,  &ecx, 4);
        memcpy(brand + off + 12, &edx, 4);
    }
    brand[48] = '\0';

    /* 去除首尾空格 */
    char* p = brand;
    while (*p == ' ') p++;
    char* end = p + strlen(p) - 1;
    while (end > p && *end == ' ') *end-- = '\0';

    printf("  Brand:        %s\n", p);
}

/* ================================================================
 * 3. 缓存拓扑（叶 04H）
 * ================================================================ */

typedef struct {
    int level, type, size_kb, ways, line_size, sets, partitions, shared_by;
} CacheEntry;

static CacheEntry detect_cache_entry(int index) {
    CacheEntry c;
    memset(&c, 0, sizeof(c));
    uint32_t eax, ebx, ecx, edx;
    cpuid_raw(4, index, &eax, &ebx, &ecx, &edx);

    c.type = (eax >> 5) & 0x07;
    if (c.type == 0) return c;

    c.level      = (int)((eax >> 5) & 0x07);
    c.line_size  = (int)((ebx & 0xFFF) + 1);
    c.partitions = (int)(((ebx >> 12) & 0x3FF) + 1);
    c.ways       = (int)(((ebx >> 22) & 0x3FF) + 1);
    c.sets       = (int)(ecx + 1);
    c.shared_by  = (int)(((eax >> 14) & 0xFFF) + 1);

    int total_bytes = c.ways * c.partitions * c.line_size * c.sets;
    c.size_kb = total_bytes / 1024;
    return c;
}

static void print_cache_topology(void) {
    printf("  Cache Topology:\n");
    int found = 0;
    for (int i = 0; ; i++) {
        CacheEntry c = detect_cache_entry(i);
        if (c.type == 0) break;
        found = 1;

        const char* tname = "?";
        if (c.type == 1) tname = "Data";
        else if (c.type == 2) tname = "Instruction";
        else if (c.type == 3) tname = "Unified";

        printf("    L%d %-11s %5d KB  %2d-way  %3d B/line  "
               "shared x%d\n",
               c.level, tname, c.size_kb, c.ways,
               c.line_size, c.shared_by);
    }
    if (!found) printf("    (not available via CPUID leaf 4)\n");
}

/* ================================================================
 * 4. 全部 SIMD 特性位
 * ================================================================ */

typedef struct {
    const char* name;
    unsigned int value;
} FeatureEntry;

static void print_all_features(void) {
    uint32_t eax, ebx, ecx, edx;

    /* 叶 1 特性 */
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    int osxsave = (ecx >> 27) & 1;

    FeatureEntry base[] = {
        {"SSE3",           (ecx >> 0)  & 1},
        {"PCLMULQDQ",      (ecx >> 1)  & 1},
        {"MONITOR",        (ecx >> 3)  & 1},
        {"SSSE3",          (ecx >> 9)  & 1},
        {"FMA",            (ecx >> 12) & 1},
        {"CMPXCHG16B",     (ecx >> 13) & 1},
        {"SSE4.1",         (ecx >> 19) & 1},
        {"SSE4.2",         (ecx >> 20) & 1},
        {"MOVBE",          (ecx >> 22) & 1},
        {"POPCNT",         (ecx >> 23) & 1},
        {"AESNI",          (ecx >> 25) & 1},
        {"XSAVE",          (ecx >> 26) & 1},
        {"OSXSAVE",        (ecx >> 27) & 1},
        {"AVX",            (ecx >> 28) & 1},
        {"F16C",           (ecx >> 29) & 1},
        {"RDRAND",         (ecx >> 30) & 1},

        {"SSE",            (edx >> 25) & 1},
        {"SSE2",           (edx >> 26) & 1},
        {"HTT",            (edx >> 28) & 1},
        {NULL, 0}
    };

    printf("  Base Features (CPUID.01H):\n");
    for (int i = 0; base[i].name; i++)
        printf("    %-16s %s\n", base[i].name, base[i].value ? "YES" : "NO");

    /* 叶 7 特性 */
    if (!cpuid_has_leaf(7)) return;

    cpuid_raw(7, 0, &eax, &ebx, &ecx, &edx);

    FeatureEntry ext[] = {
        {"BMI1",           (ebx >> 3)  & 1},
        {"AVX2",           (ebx >> 5)  & 1},
        {"BMI2",           (ebx >> 8)  & 1},
        {"ERMS",           (ebx >> 9)  & 1},
        {"INVPCID",        (ebx >> 10) & 1},
        {"RTM",            (ebx >> 11) & 1},
        {"FPU_CSDS",       (ebx >> 13) & 1},  /* 已弃用的 CS/DS */
        {"MPX",            (ebx >> 14) & 1},
        {"AVX-512F",       (ebx >> 16) & 1},
        {"AVX-512DQ",      (ebx >> 17) & 1},
        {"RDSEED",         (ebx >> 18) & 1},
        {"ADX",            (ebx >> 19) & 1},
        {"SMAP",           (ebx >> 20) & 1},
        {"AVX-512IFMA",    (ebx >> 21) & 1},
        {"CLFLUSHOPT",     (ebx >> 23) & 1},
        {"CLWB",           (ebx >> 24) & 1},
        {"AVX-512CD",      (ebx >> 28) & 1},
        {"SHA",            (ebx >> 29) & 1},
        {"AVX-512BW",      (ebx >> 30) & 1},
        {"AVX-512VL",      (ebx >> 31) & 1},

        {"PREFETCHWT1",    (ecx >> 0)  & 1},
        {"AVX-512VBMI",    (ecx >> 1)  & 1},
        {"UMIP",           (ecx >> 2)  & 1},
        {"PKU",            (ecx >> 3)  & 1},
        {"OSPKE",          (ecx >> 4)  & 1},
        {"AVX-512_VBMI2",  (ecx >> 6)  & 1},
        {"GFNI",           (ecx >> 8)  & 1},
        {"VAES",           (ecx >> 9)  & 1},
        {"VPCLMULQDQ",     (ecx >> 10) & 1},
        {"AVX-512VNNI",    (ecx >> 11) & 1},
        {"AVX-512BITALG",  (ecx >> 12) & 1},
        {"AVX-512VPOPCNTDQ",(ecx>> 14) & 1},
        {"RDPID",          (ecx >> 22) & 1},
        {NULL, 0}
    };

    printf("\n  Extended Features (CPUID.07H.0):\n");
    for (int i = 0; ext[i].name; i++)
        printf("    %-20s %s\n", ext[i].name, ext[i].value ? "YES" : "NO");

    /* 叶 7.1 特性（Ice Lake+） */
    if (cpuid_has_leaf(7)) {
        cpuid_raw(7, 1, &eax, &ebx, &ecx, &edx);
        if (eax || ebx || ecx || edx) {
            FeatureEntry ext1[] = {
                {"AVX-512BF16",    (eax >> 5)  & 1},
                {"AVX_VNNI",       (eax >> 4)  & 1},
                {"AVX-512FP16",    (edx >> 23) & 1},
                {"AMX-BF16",       (edx >> 22) & 1},
                {"AMX-TILE",       (edx >> 24) & 1},
                {"AMX-INT8",       (edx >> 25) & 1},
                {NULL, 0}
            };
            printf("\n  Advanced Features (CPUID.07H.1):\n");
            for (int i = 0; ext1[i].name; i++)
                printf("    %-20s %s\n", ext1[i].name, ext1[i].value ? "YES" : "NO");
        }
    }
}

/* ================================================================
 * 5. XGETBV 操作系统 XSAVE 状态检查
 * ================================================================ */

static void print_xsave_state(void) {
    uint32_t eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    int has_xsave = (ecx >> 26) & 1;
    int has_osxsave = (ecx >> 27) & 1;

    printf("\n  OS XSAVE Support:\n");
    printf("    XSAVE inst:  %s\n", has_xsave   ? "YES" : "NO");
    printf("    OSXSAVE:     %s\n", has_osxsave ? "YES" : "NO");

    if (!has_osxsave) {
        printf("    --> AVX/AVX-512 NOT available (OS doesn't support XSAVE)\n");
        return;
    }

    uint64_t xcr0 = xgetbv_u64(0);
    printf("    XCR0 = 0x%016llX\n", (unsigned long long)xcr0);
    printf("      Bit 0 (x87):     %s\n", (xcr0 & 1)    ? "YES" : "NO");
    printf("      Bit 1 (SSE):     %s\n", (xcr0 & 2)    ? "YES" : "NO");
    printf("      Bit 2 (AVX):     %s\n", (xcr0 & 4)    ? "YES" : "NO");

    int avx_ok  = (xcr0 & 0x6) == 0x6;
    printf("      --> OS AVX support: %s\n",
           avx_ok ? "YES (YMM state saved/restored)" : "NO");

    printf("      Bit 5 (Opmask):  %s\n", (xcr0 & 32)   ? "YES" : "NO");
    printf("      Bit 6 (ZMM_Hi256):%s\n", (xcr0 & 64)   ? "YES" : "NO");
    printf("      Bit 7 (Hi16_ZMM): %s\n", (xcr0 & 128)  ? "YES" : "NO");

    int avx512_ok = (xcr0 & 0xE6) == 0xE6;
    printf("      --> OS AVX-512 support: %s\n",
           avx512_ok ? "YES (ZMM+Opmask state saved/restored)" : "NO");
}

/* ================================================================
 * 6. 频率信息（叶 16H）
 * ================================================================ */

static void print_frequency_info(void) {
    uint32_t eax, ebx, ecx, edx;
    if (!cpuid_has_leaf(0x16)) {
        printf("  Frequency:    (not available via CPUID)\n");
        return;
    }
    cpuid_raw(0x16, 0, &eax, &ebx, &ecx, &edx);
    unsigned int base_mhz = eax & 0xFFFF;
    unsigned int max_mhz  = ebx & 0xFFFF;
    unsigned int bus_mhz  = ecx & 0xFFFF;
    if (base_mhz > 0) {
        printf("  Frequency:    %u MHz base, %u MHz max, %u MHz bus\n",
               base_mhz, max_mhz, bus_mhz);
    }
}

/* ================================================================
 * 7. 逻辑核心数
 * ================================================================ */

static void print_core_info(void) {
    uint32_t eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    int log_cores = (ebx >> 16) & 0xFF;

    /* 尝试用叶 0x1F 获取更详细的拓扑（如果可用） */
    if (cpuid_has_leaf(0x1F)) {
        cpuid_raw(0x1F, 0, &eax, &ebx, &ecx, &edx);
        int smt_threads = eax & 0xFFFF;
        if (log_cores > 1) {
            printf("  Cores:        %d logical, SMT threads per core: %d\n",
                   log_cores, smt_threads);
        }
    } else {
        printf("  Cores:        %d logical processors\n", log_cores);
    }
}

/* ================================================================
 * 主函数
 * ================================================================ */

int main() {
    printf("=== Complete CPUID Enumeration Demo ===\n");
    printf("Reference: Modern X86 Assembly Language Programming 2e, Ch.16\n\n");

    printf("--- CPU Identification ---\n");
    print_vendor_string();
    print_brand_string();
    print_core_info();
    print_frequency_info();

    printf("\n--- SIMD Feature Bits ---\n");
    print_all_features();

    printf("\n--- XSAVE / OS SIMD State ---\n");
    print_xsave_state();

    printf("\n--- Cache Topology ---\n");
    print_cache_topology();

    /* 与 GCC 内置函数的快速对比 */
    printf("\n--- GCC __builtin_cpu_supports ---\n");
    printf("  AVX2:        %s\n", __builtin_cpu_supports("avx2")     ? "YES":"NO");
    printf("  AVX-512F:    %s\n", __builtin_cpu_supports("avx512f")  ? "YES":"NO");
    printf("  AVX-512BW:   %s\n", __builtin_cpu_supports("avx512bw") ? "YES":"NO");
    printf("  AVX-512VL:   %s\n", __builtin_cpu_supports("avx512vl") ? "YES":"NO");
    printf("  AVX-512DQ:   %s\n", __builtin_cpu_supports("avx512dq") ? "YES":"NO");
    printf("  AVX-512CD:   %s\n", __builtin_cpu_supports("avx512cd") ? "YES":"NO");
    printf("  FMA:         %s\n", __builtin_cpu_supports("fma")      ? "YES":"NO");
    printf("  AES:         %s\n", __builtin_cpu_supports("aes")      ? "YES":"NO");
    printf("  SSE4.1:      %s\n", __builtin_cpu_supports("sse4.1")   ? "YES":"NO");
    printf("  SSE4.2:      %s\n", __builtin_cpu_supports("sse4.2")   ? "YES":"NO");
    printf("\n  NOTE: __builtin_cpu_supports checks hardware ONLY.\n");
    printf("  OS XSAVE support must be verified separately via XGETBV.\n");

    return 0;
}