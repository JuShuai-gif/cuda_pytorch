// 13_intrinsics: AVX-512 conditional example -- masked select.
//
// PDF 12.4d (p123-124): AVX-512 mask registers enable compute+select in one
// instruction. This file is compiled with -mavx512f; it is GUARDED so that on
// a CPU without AVX-512 (like this machine) it prints a message and returns
// instead of crashing (PDF p124: "The program will crash if it contains an
// instruction that the CPU does not support").
#include <cstdio>
#include <immintrin.h>
#include <vector>

#include "common/cpu_info.h"

// SelectAddMul using AVX-512 mask arithmetic (PDF Example 12.4d).
static void avx512_select_add_mul(short int aa[], const short int bb[],
                                  const short int cc[]) {
    __m512i zero = _mm512_setzero_si512();
    __m512i two  = _mm512_set1_epi16(2);
    for (int i = 0; i < 256; i += 32) {
        __m512i b  = _mm512_loadu_si512(bb + i);
        __m512i c  = _mm512_loadu_si512(cc + i);
        __m512i bc = _mm512_mullo_epi16(b, c);
        __mmask32 mask = _mm512_cmp_epi16_mask(b, zero, 6);  // b > 0
        __m512i r = _mm512_mask_add_epi16(bc, mask, c, two); // c+2 where b>0
        _mm512_storeu_epi16(aa + i, r);
    }
}

int main() {
    if (!cpu_has_avx512()) {
        std::printf(
            "AVX-512 NOT supported by this CPU; skipping masked example.\n"
            "This demonstrates why AVX-512 code needs runtime CPU dispatch\n"
            "(PDF p124, ch.13). Current machine: i9-14900HX (no AVX-512).\n");
        return 0;
    }

    short aa[256] = {0}, bb[256], cc[256];
    for (int i = 0; i < 256; ++i) { bb[i] = (short)(i - 128); cc[i] = 3; }
    avx512_select_add_mul(aa, bb, cc);
    long long sum = 0;
    for (int i = 0; i < 256; ++i) sum += aa[i];
    std::printf("AVX-512 ran; checksum = %lld\n", sum);
    return 0;
}
