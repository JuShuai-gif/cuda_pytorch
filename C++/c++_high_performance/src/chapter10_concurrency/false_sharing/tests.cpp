// Correctness checks: both implementations produce identical results.

#include <cstddef>
#include <cstdio>
#include <new>
#include <thread>

#include "test_utils.hpp"

namespace {

constexpr std::size_t kRounds = 50'000;

struct PlainCounters {
    long a = 0;
    long b = 0;
};

struct alignas(std::hardware_destructive_interference_size) PaddedCounter {
    long value = 0;
};

}  // namespace

int main() {
    PlainCounters plain;
    {
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
    }

    PaddedCounter pad_a;
    PaddedCounter pad_b;
    {
        std::thread t1{[&] {
            for (std::size_t i = 0; i < kRounds; ++i) {
                pad_a.value += 1;
            }
        }};
        std::thread t2{[&] {
            for (std::size_t i = 0; i < kRounds; ++i) {
                pad_b.value += 1;
            }
        }};
        t1.join();
        t2.join();
    }

    CHP_CHECK(plain.a == static_cast<long>(kRounds));
    CHP_CHECK(plain.b == static_cast<long>(kRounds));
    CHP_CHECK(pad_a.value == static_cast<long>(kRounds));
    CHP_CHECK(pad_b.value == static_cast<long>(kRounds));
    CHP_CHECK(plain.a + plain.b == pad_a.value + pad_b.value);

    return chp::test_summary("false_sharing");
}
