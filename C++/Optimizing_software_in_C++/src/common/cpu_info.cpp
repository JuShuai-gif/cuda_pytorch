#include "cpu_info.h"

#include <cstdio>

int cpu_instruction_set_level() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return 10;
    if (__builtin_cpu_supports("avx2")) return 8;
    if (__builtin_cpu_supports("avx")) return 7;
    if (__builtin_cpu_supports("sse4.2")) return 6;
    if (__builtin_cpu_supports("sse4.1")) return 5;
    if (__builtin_cpu_supports("ssse3")) return 4;
    if (__builtin_cpu_supports("sse3")) return 3;
    if (__builtin_cpu_supports("sse2")) return 2;
    if (__builtin_cpu_supports("sse")) return 1;
    return 0;
}

bool cpu_has_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}

bool cpu_has_avx512() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f");
}

void cpu_print_info() {
    int level = cpu_instruction_set_level();
    std::printf("CPU instruction-set level: %d", level);
    if (level >= 10) std::printf(" (AVX-512)");
    else if (level >= 8) std::printf(" (AVX2)");
    else if (level >= 7) std::printf(" (AVX)");
    else if (level >= 2) std::printf(" (SSE2 or higher)");
    else std::printf(" (SSE or generic)");
    std::printf("\n");
}
