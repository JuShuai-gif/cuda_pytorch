// Correctness checks: single-writer/single-reader stress over the queue.

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <thread>
#include <vector>

#include "test_utils.hpp"

#include "lock_free_queue.hpp"

namespace {

constexpr std::size_t kCapacity = 64;
constexpr std::size_t kItems = 100'000;

}  // namespace

int main() {
    // Producer pushes 0..kItems-1; consumer sums and counts them.
    chp10::LockFreeQueue<int, kCapacity> q;
    std::atomic<bool> stop{false};

    std::thread consumer{[&] {
        long sum = 0;
        std::size_t count = 0;
        while (true) {
            if (q.size() > 0) {
                sum += q.front();
                q.pop();
                ++count;
            } else if (stop.load()) {
                break;
            }
        }
        CHP_CHECK(count == kItems);
        CHP_CHECK(sum == static_cast<long>(kItems) * (kItems - 1) / 2);
    }};

    std::thread producer{[&] {
        for (std::size_t i = 0; i < kItems; ++i) {
            // Back-pressure: wait for space instead of overflowing the
            // fixed-capacity queue (a real-time producer would instead
            // control its rate).
            while (q.size() >= kCapacity) {
                std::this_thread::yield();
            }
            q.push(static_cast<int>(i));
        }
        stop.store(true);
    }};

    producer.join();
    consumer.join();

    return chp::test_summary("lock_free_queue");
}
