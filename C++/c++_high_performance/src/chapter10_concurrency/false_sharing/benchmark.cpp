// False sharing benchmark: padded vs unpadded counters under contention.
//
// Two threads hammer their own counters simultaneously. With false sharing
// they invalidate each other's cache line every iteration; padding each
// counter to a cache line removes that traffic.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <thread>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kRounds = 5'000'000;

struct PlainCounters {
    long a = 0;
    long b = 0;
};

struct alignas(std::hardware_destructive_interference_size) PaddedCounter {
    long value = 0;
};

// Runs two threads each incrementing its own counter; used by the benchmark
// callback which accumulates the (deterministic) final sum into `acc`.
// The compiler barrier forces each increment to hit memory, otherwise the
// compiler hoists the loop body into a register and no false sharing occurs.
template <typename Counters, typename FieldFn>
void run_two(Counters& c, FieldFn field_a, FieldFn field_b,
             std::uint64_t& acc) {
    std::thread t1{[&] {
        for (std::size_t i = 0; i < kRounds; ++i) {
            (c.*field_a) += 1;
            chp::compiler_barrier();
        }
    }};
    std::thread t2{[&] {
        for (std::size_t i = 0; i < kRounds; ++i) {
            (c.*field_b) += 1;
            chp::compiler_barrier();
        }
    }};
    t1.join();
    t2.join();
    acc += static_cast<std::uint64_t>(c.a + c.b);
}

constexpr std::size_t kIterations = 5;
constexpr std::size_t kRounds2 = 5;
constexpr std::size_t kWarmup = 1;

}  // namespace

int main() {
    std::printf("== false_sharing benchmark ==\n");
    std::printf("cache line size: %zu bytes\n",
                std::hardware_destructive_interference_size);

    const auto r_plain = chp::benchmark(kIterations, kRounds2, kWarmup,
        [](std::uint64_t& acc) {
            PlainCounters c;
            run_two(c, &PlainCounters::a, &PlainCounters::b, acc);
        });

    const auto r_padded = chp::benchmark(kIterations, kRounds2, kWarmup,
        [](std::uint64_t& acc) {
            PaddedCounter a, b;
            // Two separate objects: each is cache-line aligned, so the two
            // counters are guaranteed to be on different cache lines.
            std::thread t1{[&] {
                for (std::size_t i = 0; i < kRounds; ++i) {
                    a.value += 1;
                    chp::compiler_barrier();
                }
            }};
            std::thread t2{[&] {
                for (std::size_t i = 0; i < kRounds; ++i) {
                    b.value += 1;
                    chp::compiler_barrier();
                }
            }};
            t1.join();
            t2.join();
            acc += static_cast<std::uint64_t>(a.value + b.value);
        });

    std::printf("Data: 2 threads, %zu increments each, per full pass\n\n",
                kRounds);

    chp::print_result("false sharing (same cache line)", r_plain);
    chp::print_result("padded (own cache line)", r_padded);

    const double ratio = r_plain.mean_ns / r_padded.mean_ns;
    std::printf("unpadded/padded time ratio: %.2fx\n", ratio);

    return 0;
}
