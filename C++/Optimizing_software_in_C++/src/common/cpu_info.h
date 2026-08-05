// CPU feature detection (CPUID via GCC/Clang builtins).
//
// __builtin_cpu_supports() already performs an OS-level check (XGETBV),
// so it returns true only if both the CPU and the OS support the feature.
// This is the same information the CPU dispatcher needs (PDF ch.13, p135).
#pragma once

// Highest supported instruction-set level:
//   0 = generic x86-64, 1 = SSE, 2 = SSE2, 3 = SSE3, 4 = SSSE3,
//   5 = SSE4.1, 6 = SSE4.2, 7 = AVX, 8 = AVX2, 10 = AVX512F
int cpu_instruction_set_level();

bool cpu_has_avx2();
bool cpu_has_avx512();

void cpu_print_info();
