/**
 * avx2_string_search.cpp -- SIMD 字符串搜索 (simdjson / memchr 风格)
 *
 * 演示 AVX2 加速的字符串操作:
 *   1. memchr: 查找字节的首次出现位置 (每次处理 32 字节)
 *   2. find_structural_chars: 定位 JSON 语法字符
 *   3. strlen: 计算字符串长度 (搜索空字节)
 *
 * 所有实现都使用 _mm256_cmpeq_epi8 + _mm256_movemask_epi8
 * 每次迭代处理 32 字节。
 *
 * 参考: Modern X86 Assembly Language Programming 2nd Ed, 第 13 章
 *       simdjson (Lemire 等人)
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
 * 1. AVX2 memchr: 查找字节值的首次出现位置
 *    (标量 memchr 通常由 glibc 优化过, 但这里展示的是
 *     高性能库中使用的 SIMD 模式)
 * ================================================================ */

__attribute__((noinline))
static const uint8_t* memchr_avx2(const uint8_t* data, uint8_t target,
                                   size_t len) {
    const __m256i vtarget = _mm256_set1_epi8((char)target);
    size_t i = 0;

    /* 对齐到 32 字节边界，走对齐加载快速路径 */
    while (((uintptr_t)(data + i) & 31) && i < len) {
        if (data[i] == target) return data + i;
        i++;
    }

    /* 快速 AVX2 循环: 每次迭代 32 字节 */
    for (; i + 32 <= len; i += 32) {
        __m256i v = _mm256_load_si256((const __m256i*)(data + i));
        __m256i cmp = _mm256_cmpeq_epi8(v, vtarget);
        uint32_t mask = _mm256_movemask_epi8(cmp);
        if (mask) {
            return data + i + (unsigned)__builtin_ctz(mask);
        }
    }

    /* 标量尾部处理 */
    for (; i < len; i++) {
        if (data[i] == target) return data + i;
    }
    return NULL;
}

__attribute__((noinline))
static const uint8_t* memchr_scalar(const uint8_t* data, uint8_t target,
                                     size_t len) {
    for (size_t i = 0; i < len; i++)
        if (data[i] == target) return data + i;
    return NULL;
}

/* ================================================================
 * 2. JSON 语法字符扫描器
 *
 * 定位: " \ { } [ ] : ,
 * 返回 64 位位图, 其中第 i 位 = 1 表示字节 i 是语法字符。
 * ================================================================ */

__attribute__((noinline))
static void find_structural_chars_avx2(const uint8_t* data, size_t len,
                                        uint64_t* bitmap) {
    /* 预广播全部 8 个语法字符 */
    static const uint8_t structural_chars[8] = {
        '"', '\\', '{', '}', '[', ']', ':', ','
    };
    __m256i targets[8];
    for (int j = 0; j < 8; j++)
        targets[j] = _mm256_set1_epi8((char)structural_chars[j]);

    /* 将位图清零 */
    size_t words = (len + 63) / 64;
    for (size_t w = 0; w < words; w++) bitmap[w] = 0;

    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));

        /* 与全部 8 个字符比较, 对结果做 OR 运算 */
        __m256i any = _mm256_setzero_si256();
        for (int j = 0; j < 8; j++) {
            __m256i m = _mm256_cmpeq_epi8(v, targets[j]);
            any = _mm256_or_si256(any, m);
        }

        uint32_t mask = _mm256_movemask_epi8(any);
        /* 存储到位图: 每个 uint64 字存放 N 位 */
        size_t word = i / 64;
        size_t bit_off = i % 64;
        bitmap[word] |= ((uint64_t)mask) << bit_off;
        if (bit_off + 32 > 64 && word + 1 < words) {
            bitmap[word + 1] |= ((uint64_t)mask) >> (64 - bit_off);
        }
    }

    /* 标量尾部处理 */
    for (; i < len; i++) {
        uint8_t c = data[i];
        int is_structural = (c == '"' || c == '\\' || c == '{' || c == '}' ||
                             c == '[' || c == ']' || c == ':' || c == ',');
        if (is_structural) {
            size_t word = i / 64;
            bitmap[word] |= ((uint64_t)1) << (i % 64);
        }
    }
}

__attribute__((noinline))
static void find_structural_chars_scalar(const uint8_t* data, size_t len,
                                          uint64_t* bitmap) {
    size_t words = (len + 63) / 64;
    for (size_t w = 0; w < words; w++) bitmap[w] = 0;
    for (size_t i = 0; i < len; i++) {
        uint8_t c = data[i];
        if (c == '"' || c == '\\' || c == '{' || c == '}' ||
            c == '[' || c == ']' || c == ':' || c == ',') {
            bitmap[i / 64] |= ((uint64_t)1) << (i % 64);
        }
    }
}

/* ================================================================
 * 3. AVX2 strlen: 查找空终止符
 * ================================================================ */

__attribute__((noinline))
static size_t strlen_avx2(const char* str) {
    const __m256i zero = _mm256_setzero_si256();
    const char* p = str;

    /* 对齐到 32 字节 */
    while (((uintptr_t)p & 31) && *p != '\0') p++;
    if (*p == '\0') return (size_t)(p - str);

    /* 每次处理 32 字节 */
    for (;; p += 32) {
        __m256i v = _mm256_load_si256((const __m256i*)p);
        __m256i cmp = _mm256_cmpeq_epi8(v, zero);
        uint32_t mask = _mm256_movemask_epi8(cmp);
        if (mask) {
            return (size_t)(p - str) + (unsigned)__builtin_ctz(mask);
        }
    }
}

/* ================================================================
 * 性能测试基础设施
 * ================================================================ */

static const size_t N = 200000;
static uint8_t* g_data = NULL;
static uint64_t* g_bitmap = NULL;
static const uint8_t* g_found = NULL;
static size_t g_strlen_result = 0;

__attribute__((noinline)) static void bn_memchr_scalar() {
    g_found = memchr_scalar(g_data, '\n', N);
}
__attribute__((noinline)) static void bn_memchr_avx2() {
    g_found = memchr_avx2(g_data, '\n', N);
}
__attribute__((noinline)) static void bn_struct_scalar() {
    find_structural_chars_scalar(g_data, N, g_bitmap);
}
__attribute__((noinline)) static void bn_struct_avx2() {
    find_structural_chars_avx2(g_data, N, g_bitmap);
}

/* ================================================================
 * 主函数
 * ================================================================ */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported. Exiting.\n");
        return 0;
    }

    printf("\n=== AVX2 String Search Demo ===\n");
    printf("SIMD width: 256-bit (32 bytes per iteration)\n");
    printf("Key instruction: _mm256_cmpeq_epi8 + _mm256_movemask_epi8\n\n");

    /* 分配数据 */
    g_data   = ALIGNED_ALLOC(uint8_t, N, 32);
    size_t words = (N + 63) / 64;
    g_bitmap = ALIGNED_ALLOC(uint64_t, words, 32);

    /* 填充文本类数据 (可打印 ASCII, 夹杂少量特殊字符) */
    rand_xorshift64_seed(42);
    for (size_t i = 0; i < N; i++) {
        uint8_t c = (uint8_t)(rand_xorshift64_next() % 128);
        g_data[i] = (c < 32) ? (uint8_t)32 : c;  /* 可打印 ASCII */
    }

    /* 在已知位置插入换行符 */
    g_data[N / 3] = '\n';

    /* ---- 测试 1: memchr ---- */
    printf("--- memchr (查找 0x0A 换行符) ---\n");
    {
        const uint8_t* p_scalar = memchr_scalar(g_data, '\n', N);
        const uint8_t* p_avx2   = memchr_avx2(g_data, '\n', N);

        printf("  插入位置偏移:       %zu\n", N / 3);
        printf("  标量找到位置:       %td\n", p_scalar ? p_scalar - g_data : -1);
        printf("  AVX2 找到位置:      %td\n", p_avx2   ? p_avx2 - g_data : -1);

        if (p_scalar == p_avx2) {
            printf("  [通过] memchr 结果一致\n");
        } else {
            printf("  [失败] memchr 结果不一致\n");
        }
    }

    /* ---- 测试 2: 语法字符扫描 ---- */
    printf("\n--- 语法字符扫描器 ---\n");
    {
        /* 插入一些语法字符 */
        g_data[100] = '{';  g_data[101] = '"';
        g_data[500] = ':';  g_data[501] = '[';
        g_data[900] = '}';  g_data[901] = ',';

        uint64_t* bitmap_ref = ALIGNED_ALLOC(uint64_t, words, 32);
        memset(bitmap_ref, 0, words * sizeof(uint64_t));
        memset(g_bitmap,    0, words * sizeof(uint64_t));

        find_structural_chars_scalar(g_data, N, bitmap_ref);
        find_structural_chars_avx2(g_data, N, g_bitmap);

        int ok = 1;
        for (size_t w = 0; w < words; w++) {
            if (g_bitmap[w] != bitmap_ref[w]) { ok = 0; break; }
        }
        printf("  [%s] 语法字符位图与参考值一致 (%zu 个字)\n",
               ok ? "通过" : "失败", words);

        /* 打印找到的位置 */
        int found_count = 0;
        for (size_t i = 0; i < N && found_count < 12; i++) {
            if (g_bitmap[i / 64] & ((uint64_t)1 << (i % 64))) {
                printf("    位置 %5zu: 0x%02X '%c'\n",
                       i, g_data[i],
                       (g_data[i] >= 32 && g_data[i] < 127) ? (char)g_data[i] : '?');
                found_count++;
            }
        }

        ALIGNED_FREE(bitmap_ref);
    }

    /* ---- 测试 3: strlen ---- */
    printf("\n--- AVX2 strlen ---\n");
    {
        char test_str[64] = "Hello, SIMD World!";
        size_t s1 = strlen(test_str);
        size_t s2 = strlen_avx2(test_str);
        printf("  字符串: \"%s\"\n", test_str);
        printf("  glibc strlen: %zu\n", s1);
        printf("  AVX2  strlen: %zu\n", s2);
        printf("  [%s] strlen 结果一致\n", s1 == s2 ? "通过" : "失败");

        /* 空字符串 */
        test_str[0] = '\0';
        s1 = strlen(test_str);
        s2 = strlen_avx2(test_str);
        printf("  空字符串: glibc=%zu, AVX2=%zu [%s]\n",
               s1, s2, s1 == s2 ? "通过" : "失败");
    }

    /* ---- 性能测试 ---- */
    printf("\n--- 性能测试 (N = %zu) ---\n", N);
    {
        benchmark_result_t results[4];
        memset(results, 0, sizeof(results));

        BENCH_COMPUTE(bn_memchr_scalar(), N, N, 100, results[0]);
        results[0].name = "memchr scalar";

        BENCH_COMPUTE(bn_memchr_avx2(), N, N, 100, results[1]);
        results[1].name = "memchr AVX2 (32B/iter)";

        BENCH_COMPUTE(bn_struct_scalar(), N, N, 30, results[2]);
        results[2].name = "struct scan scalar";

        BENCH_COMPUTE(bn_struct_avx2(), N, N, 30, results[3]);
        results[3].name = "struct scan AVX2 (8 cmp + OR)";

        bench_report(results, 4);
    }

    printf("--- 字符串搜索要点 ---\n");
    printf("  _mm256_cmpeq_epi8: 并行比较 32 字节 (1 个微操作)\n");
    printf("  _mm256_movemask_epi8: 提取 32 个符号位到通用寄存器 (1 个微操作)\n");
    printf("  __builtin_ctz: 找到第一个置位位 (1 个周期)\n");
    printf("  吞吐量: memchr 约 3-5 GB/s (受内存带宽限制)\n");
    printf("  应用场景: JSON 解析 (simdjson), 日志扫描,\n");
    printf("    协议解析, 数据库列扫描\n");

    ALIGNED_FREE(g_data);
    ALIGNED_FREE(g_bitmap);
    return 0;
}
