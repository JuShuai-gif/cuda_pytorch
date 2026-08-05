// Pitfall P5: assuming heap objects are cache-line / SIMD aligned.
//
// malloc/new only guarantee alignment suitable for max_align_t (16 B on
// x86-64), NOT 64 B. For SIMD (AVX2=32 B, AVX-512=64 B) and for keeping a
// hot object within one cache line, that is not enough. This benchmark
// allocates many objects and shows that `new` does not reliably give 64-byte
// alignment, while posix_memalign does.
//
// Related PDF: 6.2.1 (alignment), Figure 6.4 (unaligned cost).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int kSamples = 16;

int main() {
    std::printf("Pitfall P5: heap alignment (want 64 B for SIMD/cache line)\n");
    std::printf("max_align_t on this platform: %zu bytes\n",
                alignof(std::max_align_t));

    std::vector<void*> news;
    std::vector<void*> aligned;

    std::printf("\nnew  (16-byte guarantee only):\n");
    for (int i = 0; i < kSamples; ++i) {
        void* p = ::operator new(128);
        news.push_back(p);
        std::printf("  %p  mod64=%ld\n", p, (long)((uintptr_t)p % 64));
    }

    std::printf("\nposix_memalign(64):\n");
    for (int i = 0; i < kSamples; ++i) {
        void* p = nullptr;
        if (posix_memalign(&p, 64, 128) != 0) p = nullptr;
        aligned.push_back(p);
        if (p) std::printf("  %p  mod64=%ld\n", p, (long)((uintptr_t)p % 64));
    }

    int unaligned = 0;
    for (void* p : news)
        if (((uintptr_t)p % 64) != 0) ++unaligned;
    std::printf("\n%d/%d `new` allocations were NOT 64-byte aligned.\n",
                unaligned, kSamples);
    std::printf("Lesson: for SIMD (16/32/64 B) and cache-line placement, do\n"
                "not assume `new`; use alignas / aligned_alloc / posix_memalign.\n");

    for (void* p : news) ::operator delete(p);
    for (void* p : aligned) if (p) free(p);
    return 0;
}
