// 08_memory_cache: matrix transpose -- power-of-two cache contention.
//
// PDF 9.10 (p106-108, Example 9.9a, Table 9.1): when the row distance is a
// multiple of the critical stride, cache sets collide and the transpose
// slows dramatically (on the PDF's machine 512x512 was ~6x slower than 513).
// The effect depends on the machine's cache geometry, so we just time both.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/benchmark.h"

template <int SIZE>
void transpose(double a[SIZE][SIZE]) {
    for (int r = 1; r < SIZE; ++r)
        for (int c = 0; c < r; ++c) {
            double t = a[r][c];
            a[r][c] = a[c][r];
            a[c][r] = t;
        }
}

template <int SIZE>
void fill(double a[SIZE][SIZE]) {
    for (int r = 0; r < SIZE; ++r)
        for (int c = 0; c < SIZE; ++c) a[r][c] = (double)(r * 31 + c);
}

template <int SIZE>
double checksum(const double a[SIZE][SIZE]) {
    double s = 0.0;
    for (int r = 0; r < SIZE; ++r)
        for (int c = 0; c < SIZE; ++c) s += a[r][c];
    return s;
}

int main() {
    std::printf("== power-of-two vs non-power-of-two transpose ==\n");

    {
        static double m[64][64];  fill(m);
        bench("transpose 64",  [&] { transpose<64>(m); return checksum<64>(m); });
        static double n[65][65];  fill(n);
        bench("transpose 65",  [&] { transpose<65>(n); return checksum<65>(n); });
    }
    {
        static double m[512][512]; fill(m);
        bench("transpose 512", [&] { transpose<512>(m); return checksum<512>(m); });
        static double n[513][513]; fill(n);
        bench("transpose 513", [&] { transpose<513>(n); return checksum<513>(n); });
    }
    return 0;
}
