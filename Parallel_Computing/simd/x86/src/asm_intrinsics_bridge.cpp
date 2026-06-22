/**
 * asm_intrinsics_bridge.cpp -- C/C++ 与 x86-64 汇编桥接演示
 *
 * 演示将 C/C++ 与汇编混合使用的三种模式：
 *   1. extern "C" 调用汇编函数（单独的 .s 文件）
 *   2. GCC 内联汇编（扩展 asm）用于单指令包装器
 *   3. 通过内联汇编调用 CPUID（经典用例）
 *
 * 配套的汇编文件（x86_64_calling_convention.s）实现了：
 *   - asm_array_sum:    整数数组求和（System V 调用约定）
 *   - asm_simd_add:     AVX2 向量加法（System V 调用约定）
 *   - asm_cpuid:        CPUID 和 XGETBV 指令的包装器
 *
 * 构建需要 .cpp 和 .s 两个文件：
 *   g++ -mavx2 -O2 asm_intrinsics_bridge.cpp x86_64_calling_convention.s
 *
 * 参考资料：Modern X86 Assembly Language Programming, 第二版, 第 2-3 章
 */

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * 外部汇编函数声明（定义在 .s 文件中）
 * ================================================================ */

extern "C" {

/**
 * 整数数组求和。
 * System V：RDI = int 数组指针，ESI = 元素个数
 * 返回值在 EAX 中。
 */
int asm_array_sum(const int* arr, int n);

/**
 * AVX2 浮点向量加法：c[i] = a[i] + b[i]
 * System V：RDI = a，RSI = b，RDX = c，ECX = n
 */
void asm_simd_add_f32(const float* a, const float* b, float* c, int n);

/**
 * CPUID 指令包装器。
 * EAX = leaf，ECX = subleaf
 * 输出写入 CpuidRegs 结构体。
 */
struct CpuidRegs {
    uint32_t eax, ebx, ecx, edx;
};
void asm_cpuid_raw(uint32_t leaf, uint32_t subleaf, CpuidRegs* out);

/**
 * XGETBV 指令包装器：读取扩展控制寄存器。
 * ECX = 寄存器索引（0 表示 XCR0），输出 EAX:EDX。
 */
uint64_t asm_xgetbv(uint32_t ecx);

} /* extern "C" */

/* ================================================================
 * 内联汇编示例
 * ================================================================ */

/* 模式 1：使用 GCC 扩展 asm 的单指令包装器 */
static inline __m128 inline_mm_add_ps(__m128 a, __m128 b) {
    __m128 result;
    __asm__ __volatile__(
        "addps %1, %0"
        : "=x"(result)          /* 输出：x = XMM 寄存器 */
        : "x"(a), "0"(b)        /* 输入：b 与输出共用同一寄存器 */
        :                        /* 除输出外无额外 clobber */
    );
    return result;
}

/* 模式 2：RDTSC —— 读取时间戳计数器 */
static inline uint64_t inline_rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__(
        "rdtsc"
        : "=a"(lo), "=d"(hi)    /* 输出：EAX=lo，EDX=hi */
        :                        /* 无输入 */
        : "memory"               /* clobber：阻止重排序 */
    );
    return ((uint64_t)hi << 32) | lo;
}

/* 模式 3：通过内联汇编调用 CPUID */
static inline void inline_cpuid(uint32_t leaf, uint32_t subleaf,
                                 uint32_t* eax, uint32_t* ebx,
                                 uint32_t* ecx, uint32_t* edx) {
    __asm__ __volatile__(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf)
        : "memory"
    );
}

/* 模式 4：SIMD 清零惯用法（xorps xmm, xmm） */
static inline __m128 inline_mm_zero_ps(void) {
    __m128 result;
    __asm__ __volatile__(
        "xorps %0, %0"
        : "=x"(result)
    );
    return result;
}

/* ================================================================
 * C++ 参考实现（用于正确性校验）
 * ================================================================ */

static int cpp_array_sum(const int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];
    return sum;
}

static void cpp_simd_add_f32(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

/* ================================================================
 * 使用汇编包装器进行 CPUID 特性检测
 * ================================================================ */

static void detect_features_via_asm(void) {
    printf("\n--- 通过汇编包装器调用 CPUID ---\n");

    /* Leaf 0：厂商 ID */
    uint32_t eax, ebx, ecx, edx;
    inline_cpuid(0, 0, &eax, &ebx, &ecx, &edx);

    char vendor[13] = {0};
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    printf("  厂商字符串: %s（最大基础 leaf = %u）\n", vendor, eax);

    /* Leaf 7：扩展特性（AVX2、AVX-512F 等） */
    inline_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    printf("  AVX2:        %s（CPUID.07H.EBX[5]）\n",
           (ebx & (1u << 5))  ? "是" : "否");
    printf("  AVX-512F:    %s（CPUID.07H.EBX[16]）\n",
           (ebx & (1u << 16)) ? "是" : "否");
    printf("  AVX-512BW:   %s（CPUID.07H.EBX[30]）\n",
           (ebx & (1u << 30)) ? "是" : "否");
    printf("  AVX-512VL:   %s（CPUID.07H.EBX[31]）\n",
           (ebx & (1u << 31)) ? "是" : "否");

    /* Leaf 1：SSE/SSE2 等 */
    inline_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    printf("  SSE3:        %s（CPUID.01H.ECX[0]）\n",
           (ecx & (1u << 0))  ? "是" : "否");
    printf("  SSSE3:       %s（CPUID.01H.ECX[9]）\n",
           (ecx & (1u << 9))  ? "是" : "否");
    printf("  SSE4.1:      %s（CPUID.01H.ECX[19]）\n",
           (ecx & (1u << 19)) ? "是" : "否");
    printf("  SSE4.2:      %s（CPUID.01H.ECX[20]）\n",
           (ecx & (1u << 20)) ? "是" : "否");

    /* 同时测试外部 .s 文件版本的 CPUID */
    CpuidRegs regs;
    asm_cpuid_raw(7, 0, &regs);
    printf("\n  （通过外部 .s 文件）CPUID.07H: "
           "EAX=0x%08X EBX=0x%08X ECX=0x%08X EDX=0x%08X\n",
           regs.eax, regs.ebx, regs.ecx, regs.edx);

    /* XGETBV：检查 XCR0 以确认操作系统是否支持 AVX-512 */
    uint64_t xcr0 = asm_xgetbv(0);
    printf("  XCR0 = 0x%016llX\n", (unsigned long long)xcr0);
    printf("  操作系统 XSAVE 已启用: %s\n", (xcr0 & 1) ? "是" : "否");
    printf("  操作系统 AVX 状态保存: %s（XCR0[2]）\n",
           (xcr0 & (1u << 2)) ? "是" : "否");
    printf("  操作系统 AVX-512 状态（ZMM_Hi256 + Opmask）: %s（XCR0[5..7]）\n",
           ((xcr0 >> 5) & 7) == 7 ? "是" : "否");
}

/* ================================================================
 * 主函数
 * ================================================================ */

int main() {
    cpu_print_features();

    printf("\n=== C/C++ ↔ x86-64 汇编桥接演示 ===\n");
    printf("展示内容：extern \"C\" 汇编调用、内联汇编、CPUID\n\n");

    /* ---- 测试 1：外部汇编函数（数组求和） ---- */
    printf("--- 测试 1：extern \"C\" asm_array_sum ---\n");
    {
        int arr[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        int n = (int)(sizeof(arr) / sizeof(arr[0]));

        int sum_cpp = cpp_array_sum(arr, n);
        int sum_asm = asm_array_sum(arr, n);

        printf("  C++ 求和 = %d，汇编求和 = %d\n", sum_cpp, sum_asm);
        CHECK_EQ(sum_asm, sum_cpp, "外部汇编数组求和与 C++ 一致");
    }

    /* ---- 测试 2：外部 AVX2 SIMD 加法 ---- */
    printf("\n--- 测试 2：extern \"C\" asm_simd_add_f32 ---\n");
    {
        const int n = 1003;
        float* a = ALIGNED_ALLOC(float, n, 32);
        float* b = ALIGNED_ALLOC(float, n, 32);
        float* c_cpp = ALIGNED_ALLOC(float, n, 32);
        float* c_asm = ALIGNED_ALLOC(float, n, 32);

        rand_xorshift64_seed(42);
        fill_random_f32(a, n);
        rand_xorshift64_seed(99);
        fill_random_f32(b, n);

        cpp_simd_add_f32(a, b, c_cpp, n);
        asm_simd_add_f32(a, b, c_asm, n);

        CHECK_NEAR_ARRAY(c_asm, c_cpp, n, 1e-6f,
            "外部汇编 AVX2 向量加法与 C++ 一致");

        ALIGNED_FREE(a); ALIGNED_FREE(b);
        ALIGNED_FREE(c_cpp); ALIGNED_FREE(c_asm);
    }

    /* ---- 测试 3：内联汇编（单指令包装器） ---- */
    printf("\n--- 测试 3：GCC 扩展内联汇编 ---\n");
    {
        /* 对比测试 inline_add_ps 与 intrinsic */
        __m128 a = _mm_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);
        __m128 b = _mm_setr_ps(5.0f, 6.0f, 7.0f, 8.0f);

        __m128 r_intrin = _mm_add_ps(a, b);
        __m128 r_inline = inline_mm_add_ps(a, b);

        float r1[4], r2[4];
        _mm_storeu_ps(r1, r_intrin);
        _mm_storeu_ps(r2, r_inline);

        int ok = 1;
        for (int i = 0; i < 4; i++)
            if (fabsf(r1[i] - r2[i]) > 1e-6f) { ok = 0; break; }
        printf("  [%s] 内联 addps 与 _mm_add_ps intrinsic 结果一致\n",
               ok ? "通过" : "失败");
        if (!ok) exit(1);
    }

    /* 测试清零惯用法 */
    {
        __m128 z = inline_mm_zero_ps();
        __m128 ref = _mm_setzero_ps();
        float fz[4], fr[4];
        _mm_storeu_ps(fz, z);
        _mm_storeu_ps(fr, ref);
        int ok = 1;
        for (int i = 0; i < 4; i++)
            if (fabsf(fz[i] - fr[i]) > 1e-6f) { ok = 0; break; }
        printf("  [%s] 内联 xorps 清零与 _mm_setzero_ps 结果一致\n",
               ok ? "通过" : "失败");
        if (!ok) exit(1);
    }

    /* ---- 测试 4：RDTSC 计时 ---- */
    printf("\n--- 测试 4：RDTSC 计时开销 ---\n");
    {
        /* 测量 RDTSC 配对开销 */
        const int trials = 100;
        uint64_t min_overhead = UINT64_MAX;
        for (int i = 0; i < trials; i++) {
            uint64_t t1 = inline_rdtsc();
            uint64_t t2 = inline_rdtsc();
            uint64_t diff = t2 - t1;
            if (diff < min_overhead) min_overhead = diff;
        }
        printf("  RDTSC 配对开销: ~%llu 个周期（%d 次试验中的最小值）\n",
               (unsigned long long)min_overhead, trials);
    }

    /* ---- 测试 5：通过汇编调用 CPUID ---- */
    detect_features_via_asm();

    /* ---- 总结 ---- */
    printf("\n--- 汇编桥接关键要点 ---\n");
    printf("1. extern \"C\"：从 C++ 调用汇编的最简洁方式。\n");
    printf("2. System V x86-64 ABI：参数在 RDI,RSI,RDX,RCX,R8,R9；返回值在 RAX\n");
    printf("3. 内联汇编最适用场景：CPUID、RDTSC、MSR 访问、短指令序列\n");
    printf("4. 外部 .s 文件最适用场景：完整循环、复杂 SIMD、寄存器分配\n");
    printf("5. 本书使用 MASM + MS x64 ABI；此处移植为 GAS + System V ABI\n");
    printf("6. 被调用者保存寄存器：RBX,RBP,R12-R15（通用），XMM8-XMM15（SIMD，SysV）\n");

    return 0;
}
