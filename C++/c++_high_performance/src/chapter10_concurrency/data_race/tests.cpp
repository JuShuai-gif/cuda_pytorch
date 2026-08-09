// Correctness checks for the mutex/atomic fixes (and the data race).
//
// The plain version is a data race (undefined behavior); we still run it and
// observe it usually deviates from the expected value, but we do NOT assert
// on the wrong value -- that would depend on scheduling. TSan builds catch it
// deterministically. The protected versions must always be exact.

#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#include "test_utils.hpp"

namespace {

constexpr int kNTimes = 50'000;
constexpr int kThreads = 4;

int g_plain = 0;

int g_mutex = 0;
std::mutex g_mutex_m;

std::atomic<int> g_atomic{0};

void increment_plain(int n) {
    for (int i = 0; i < n; ++i) {
        ++g_plain;
    }
}

void increment_mutex(int n) {
    for (int i = 0; i < n; ++i) {
        const std::lock_guard<std::mutex> lock{g_mutex_m};
        ++g_mutex;
    }
}

void increment_atomic(int n) {
    for (int i = 0; i < n; ++i) {
        ++g_atomic;
    }
}

}  // namespace

int main() {
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; ++t) {
            ts.emplace_back(increment_mutex, kNTimes);
        }
        for (auto& t : ts) {
            t.join();
        }
        CHP_CHECK(g_mutex == kNTimes * kThreads);
    }

    {
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; ++t) {
            ts.emplace_back(increment_atomic, kNTimes);
        }
        for (auto& t : ts) {
            t.join();
        }
        CHP_CHECK(g_atomic.load() == kNTimes * kThreads);
    }

    // The plain version is UB; verify it is not reliably correct by running
    // it and reporting whether it happened to be right this time.
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; ++t) {
            ts.emplace_back(increment_plain, kNTimes);
        }
        for (auto& t : ts) {
            t.join();
        }
        std::printf("[info] plain counter ended at %d (expected %d); "
                    "this run %s\n", g_plain, kNTimes * kThreads,
                    g_plain == kNTimes * kThreads ? "matched" : "deviated");
    }

    return chp::test_summary("data_race");
}
