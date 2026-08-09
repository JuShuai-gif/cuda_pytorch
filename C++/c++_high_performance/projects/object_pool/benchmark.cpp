// Benchmark: pooled allocation vs system new/delete for small objects.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "object_pool.hpp"

namespace {

struct Particle {
    float x = 0.0F, y = 0.0F, vx = 0.0F, vy = 0.0F;
    int alive = 0;
};

// Compiler barrier: forces the address to materialize and blocks memory
// reordering/elimination around the pool operations.
inline void keep_alive(void* p) {
    __asm__ volatile("" : : "r"(p) : "memory");
}

// Barriers so the compiler cannot elide the measured allocations.
template <typename Fn>
double measure(std::size_t reps, Fn fn) {
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < reps; ++i) {
        fn(acc);
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("  checksum=%llu\n",
                static_cast<unsigned long long>(acc));
    return std::chrono::duration<double>(t1 - t0).count() * 1e9 /
           static_cast<double>(reps);
}

constexpr std::size_t kCount = 200'000;

}  // namespace

int main() {
    std::printf("== object_pool benchmark ==\n");

    // System new/delete: write through a pointer so the allocation is used.
    const double sys_ns = measure(kCount, [](std::uint64_t& acc) {
        auto* p = new Particle{};
        p->x = 1.0F;
        p->y = 2.0F;
        acc += static_cast<std::uint64_t>(p->x + p->y);
        keep_alive(p);  // barrier: allocation must survive
        delete p;
    });

    // Pool allocate/deallocate (raw).
    chp::ObjectPool pool(sizeof(Particle), kCount);
    const double pool_raw_ns = measure(kCount, [&](std::uint64_t& acc) {
        auto* p = static_cast<Particle*>(pool.allocate());
        p->x = 1.0F;
        p->y = 2.0F;
        acc += static_cast<std::uint64_t>(p->x + p->y);
        keep_alive(p);
        pool.deallocate(p);
    });

    // Pool with RAII wrapper.
    const double pool_raii_ns = measure(kCount, [&](std::uint64_t& acc) {
        chp::Pooled<Particle> p(pool);
        p->x = 1.0F;
        p->y = 2.0F;
        acc += static_cast<std::uint64_t>(p->x + p->y);
        keep_alive(p.get());
    });
    std::printf("new/delete:        %8.1f ns/op\n", sys_ns);
    std::printf("pool raw:          %8.1f ns/op\n", pool_raw_ns);
    std::printf("pool RAII:         %8.1f ns/op\n", pool_raii_ns);
    std::printf("new/pool-raw ratio: %.1fx\n", sys_ns / pool_raw_ns);

    return 0;
}
