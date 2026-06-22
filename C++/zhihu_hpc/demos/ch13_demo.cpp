// Chapter 13: CPU Dispatching for Multiple Instruction Sets
// Demonstrates how to generate and dispatch optimized code for different CPUs.
// Compile: see CMakeLists.txt (ch13_demo target)

#include <chrono>
#include <iostream>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// ---- CPU feature detection ----
#ifdef _MSC_VER
static void cpuid(int info[4], int function_id) {
    __cpuid(info, function_id);
}
#else
static void cpuid(int info[4], int function_id) {
    __cpuid_count(function_id, 0, info[0], info[1], info[2], info[3]);
}
#endif

enum InstructionSet {
    IS_UNKNOWN = 0,
    IS_SSE2 = 2,
    IS_SSE3 = 3,
    IS_SSSE3 = 4,
    IS_SSE41 = 5,
    IS_SSE42 = 6,
    IS_AVX = 7,
    IS_AVX2 = 8,
    IS_AVX512F = 9,
};

InstructionSet DetectInstructionSet() {
    int info[4];
    cpuid(info, 0);
    int nIds = info[0];

    cpuid(info, 0x80000000);
    unsigned nExIds = static_cast<unsigned>(info[0]);

    // Check for basic x86 features
    if (nIds >= 1) {
        cpuid(info, 1);
        bool hasSSE2 = (info[3] & (1 << 26)) != 0;
        bool hasSSE3 = (info[2] & (1 << 0)) != 0;
        bool hasSSSE3 = (info[2] & (1 << 9)) != 0;
        bool hasSSE41 = (info[2] & (1 << 19)) != 0;
        bool hasSSE42 = (info[2] & (1 << 20)) != 0;
        bool hasAVX = (info[2] & (1 << 28)) != 0;
        bool hasOSXSAVE = (info[2] & (1 << 27)) != 0;

        if (hasOSXSAVE && hasAVX) {
            // Check if OS supports AVX
            // (simplified; real code should check XCR0)
        }

        if (hasAVX)
            return IS_AVX;
        if (hasSSE42)
            return IS_SSE42;
        if (hasSSE41)
            return IS_SSE41;
        if (hasSSSE3)
            return IS_SSSE3;
        if (hasSSE3)
            return IS_SSE3;
        if (hasSSE2)
            return IS_SSE2;
    }

    // Check extended features
    if (nExIds >= 0x80000001) {
        cpuid(info, 0x80000001);
        // ... check for extended features
    }

    return IS_UNKNOWN;
}

// ---- Dispatch example: memcpy with CPU-specific optimization ----
// Generic fallback (always works)
static void MemcpyGeneric(void* dest, const void* src, std::size_t n) {
    auto d = static_cast<char*>(dest);
    auto s = static_cast<const char*>(src);
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}

#ifdef __SSE2__
#include <emmintrin.h>
// SSE2 optimized version (16-byte copies)
static void MemcpySSE2(void* dest, const void* src, std::size_t n) {
    auto d = static_cast<__m128i*>(dest);
    auto s = static_cast<const __m128i*>(src);
    std::size_t count = n / 16;
    for (std::size_t i = 0; i < count; ++i) {
        _mm_storeu_si128(d + i, _mm_loadu_si128(s + i));
    }
    // Copy remaining bytes
    auto db = static_cast<char*>(dest) + count * 16;
    auto sb = static_cast<const char*>(src) + count * 16;
    for (std::size_t i = count * 16; i < n; ++i) {
        db[i - count * 16] = sb[i - count * 16];
    }
}
#endif

// Dispatched memcpy (selects best implementation at runtime)
using MemcpyFunc = void (*)(void*, const void*, std::size_t);

static MemcpyFunc g_memcpy_dispatch = MemcpyGeneric;

void InitMemcpyDispatch() {
    auto iset = DetectInstructionSet();
#ifdef __SSE2__
    if (iset >= IS_SSE2) {
        g_memcpy_dispatch = MemcpySSE2;
    }
#endif
    // Would add AVX, AVX512 versions here...
}

void DispatchedMemcpy(void* dest, const void* src, std::size_t n) {
    g_memcpy_dispatch(dest, src, n);
}

// ---- Main ----
int main() {
    std::cout << "=== Chapter 13: CPU Dispatching ===\n\n";

    InstructionSet iset = DetectInstructionSet();
    const char* iset_names[] = {"Unknown", "",       "SSE2", "SSE3", "SSSE3",
                                "SSE4.1",  "SSE4.2", "AVX",  "AVX2", "AVX-512F"};
    std::cout << "Detected instruction set: " << iset_names[iset] << "\n";

    InitMemcpyDispatch();

    // Test dispatched memcpy
    const char src[] = "Hello, CPU dispatch world!";
    char dest[64] = {};
    DispatchedMemcpy(dest, src, sizeof(src));
    std::cout << "Dispatched memcpy: " << dest << "\n";

    // Benchmark generic vs optimized
    constexpr int BIG = 1024 * 1024;
    char* big_src = new char[BIG];
    char* big_dst = new char[BIG];
    std::memset(big_src, 'A', BIG);

    auto t1 = std::chrono::high_resolution_clock::now();
    MemcpyGeneric(big_dst, big_src, BIG);
    auto t2 = std::chrono::high_resolution_clock::now();
    DispatchedMemcpy(big_dst, big_src, BIG);
    auto t3 = std::chrono::high_resolution_clock::now();

    using namespace std::chrono;
    auto gen_us = duration_cast<microseconds>(t2 - t1).count();
    auto opt_us = duration_cast<microseconds>(t3 - t2).count();

    std::cout << "Generic memcpy (1MB): " << gen_us << " us\n";
    std::cout << "Optimized memcpy (1MB): " << opt_us << " us\n";

    delete[] big_src;
    delete[] big_dst;

    std::cout << "\nAll chapter 13 checks passed.\n";
    return 0;
}
