// Data race and its two fixes: mutex and atomic (PDF p.284-285, 293-294, 300-301).
//
// Two threads increment a shared counter 100k times each. Without protection
// the read-modify-write of ++counter interleaves and the final value is
// wrong; with a mutex or an atomic it is exactly 2 * n_times.

#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

namespace {

// --- 1. No protection: data race, undefined behavior ---
int g_plain = 0;
void increment_plain(int n) {
    for (int i = 0; i < n; ++i) {
        ++g_plain;
    }
}

// --- 2. Mutex protects the critical section ---
int g_mutex = 0;
std::mutex g_mutex_m;
void increment_mutex(int n) {
    for (int i = 0; i < n; ++i) {
        const std::lock_guard<std::mutex> lock{g_mutex_m};
        ++g_mutex;
    }
}

// --- 3. Atomic makes the increment itself indivisible ---
std::atomic<int> g_atomic{0};
void increment_atomic(int n) {
    for (int i = 0; i < n; ++i) {
        ++g_atomic;  // == fetch_add(1)
    }
}

constexpr int kNTimes = 100'000;

}  // namespace

int main() {
    std::printf("== data_race ==\n");
    std::printf("expected counter after %d increments x2 threads: %d\n",
                kNTimes, kNTimes * 2);

    {
        std::thread t1{increment_plain, kNTimes};
        std::thread t2{increment_plain, kNTimes};
        t1.join();
        t2.join();
        std::printf("plain (data race):   %d\n", g_plain);
    }

    {
        std::thread t1{increment_mutex, kNTimes};
        std::thread t2{increment_mutex, kNTimes};
        t1.join();
        t2.join();
        std::printf("mutex:               %d\n", g_mutex);
    }

    {
        std::thread t1{increment_atomic, kNTimes};
        std::thread t2{increment_atomic, kNTimes};
        t1.join();
        t2.join();
        std::printf("atomic:              %d\n", g_atomic.load());
    }

    return 0;
}
