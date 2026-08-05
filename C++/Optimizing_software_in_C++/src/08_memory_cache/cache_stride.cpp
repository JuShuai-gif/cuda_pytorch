// 08_memory_cache: sweep array size and stride to reveal cache levels.
//
// The classic "cache timing" experiment. For each array size we touch every
// stride-th element; when the working set no longer fits a cache level the
// average access time jumps (PDF 3.13 p21: cached 2-4 cycles vs uncached
// hundreds).
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
    // Sizes from 4 KiB to 128 MiB, doubling each step.
    const size_t max_bytes = 128u << 20;
    std::vector<char> buf(max_bytes);

    std::printf("%12s  %10s  %8s\n", "bytes", "stride", "ns/access");
    for (size_t bytes = 4u << 10; bytes <= max_bytes; bytes <<= 1) {
        // pick a stride that walks this working set
        size_t stride = 64;               // one cache line
        size_t touches = bytes / stride;

        // warm-up
        volatile char sink = 0;
        for (int w = 0; w < 3; ++w)
            for (size_t i = 0; i < touches; ++i) sink += buf[i * stride];

        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 20; ++rep)
            for (size_t i = 0; i < touches; ++i) sink += buf[i * stride];
        auto t1 = std::chrono::steady_clock::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double per = ns / (20.0 * touches);
        std::printf("%12zu  %10zu  %8.2f\n", bytes, stride, per);
        (void)sink;
    }
    return 0;
}
