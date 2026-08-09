// False sharing: threads invalidate each other's cache lines (PDF p.315).
//
// Two threads each increment their own counter. If the counters sit in the
// same cache line, every write invalidates the line on the other core and
// both threads stall. Padding each counter to its own cache line (via
// alignas) eliminates the effect.

#include <cstddef>
#include <cstdio>
#include <new>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kRounds = 100'000;

// Adjacent elements share cache lines -> false sharing.
struct PlainCounters {
    long a = 0;
    long b = 0;
};

// Each counter is padded to a full cache line -> no false sharing.
struct alignas(std::hardware_destructive_interference_size) PaddedCounter {
    long value = 0;
};

}  // namespace

int main() {
    std::printf("== false_sharing ==\n");
    std::printf("cache line size (interference): %zu bytes\n",
                std::hardware_destructive_interference_size);

    // False sharing: a and b live in one cache line.
    PlainCounters plain;
    std::thread t1{[&] {
        for (std::size_t i = 0; i < kRounds; ++i) {
            plain.a += 1;
        }
    }};
    std::thread t2{[&] {
        for (std::size_t i = 0; i < kRounds; ++i) {
            plain.b += 1;
        }
    }};
    t1.join();
    t2.join();
    std::printf("plain:  a=%ld b=%ld\n", plain.a, plain.b);

    // Padded: each counter is on its own cache line.
    PaddedCounter pad_a;
    PaddedCounter pad_b;
    std::thread t3{[&] {
        for (std::size_t i = 0; i < kRounds; ++i) {
            pad_a.value += 1;
        }
    }};
    std::thread t4{[&] {
        for (std::size_t i = 0; i < kRounds; ++i) {
            pad_b.value += 1;
        }
    }};
    t3.join();
    t4.join();
    std::printf("padded: a=%ld b=%ld\n", pad_a.value, pad_b.value);

    return 0;
}
