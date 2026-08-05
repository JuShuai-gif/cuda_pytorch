// Experiment 17: Non-temporal (streaming) store.
//
// Fills a large array (>> LLC) with:
//   - normal stores
//   - non-temporal stores (_mm_stream_si128) if SSE2 is available
// Then measures "write then read back soon" to show when NT is a bad idea.
// Runtime CPUID check; falls back gracefully if unsupported.
//
// Reference: PDF 6.1 (Bypassing the Cache, Table 6.1).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark.h"

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <emmintrin.h>
#endif

static constexpr size_t N = 1 << 26;   // 64M ints = 256 MB
static constexpr int kRounds = 3;

static bool has_sse2() {
#if defined(__x86_64__)
    return true;  // x86-64 always has SSE2
#elif defined(__i386__)
    unsigned a, b, c, d;
    if (__get_cpuid(1, &a, &b, &c, &d)) return (d & (1u << 26)) != 0;
    return false;
#else
    return false;
#endif
}

int main() {
    std::printf("Experiment 17: non-temporal store (array %zu MB)\n",
                N * sizeof(int) / (1024 * 1024));
    bool sse2 = has_sse2();
    std::printf("SSE2 available: %s\n", sse2 ? "yes" : "no");

    std::vector<int> data(N, 0);

    auto normal_fill = [&] {
        for (size_t i = 0; i < N; ++i) data[i] = 1;
        bm::compiler_barrier();
    };

    auto stream_fill = [&] {
#if defined(__x86_64__) || defined(__i386__)
        char* p = reinterpret_cast<char*>(data.data());
        size_t bytes = N * sizeof(int);
        __m128i v = _mm_set1_epi32(1);
        size_t i = 0;
        for (; i + 16 <= bytes; i += 16)
            _mm_stream_si128(reinterpret_cast<__m128i*>(p + i), v);
        for (; i < bytes; ++i) p[i] = 1;
        _mm_sfence();
#else
        normal_fill();
#endif
        bm::compiler_barrier();
    };

    auto normal_readback = [&] {
        for (size_t i = 0; i < N; ++i) data[i] = i & 1;
        long long s = 0;
        for (size_t i = 0; i < N; ++i) s += data[i];
        bm::do_not_optimize(s);
    };

    normal_fill();
    stream_fill();
    normal_readback();

    auto r_norm = bm::time_rounds(kRounds, normal_fill);
    auto r_stream = bm::time_rounds(kRounds, stream_fill);
    auto r_rb = bm::time_rounds(kRounds, normal_readback);

    std::printf("normal_fill : mean=%.3f ms\n", r_norm.mean_ms);
    if (sse2)
        std::printf("stream_fill : mean=%.3f ms\n", r_stream.mean_ms);
    std::printf("normal_rb   : mean=%.3f ms (write-then-read)\n", r_rb.mean_ms);
    return 0;
}
