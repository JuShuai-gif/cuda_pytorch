// Correctness checks: producer-consumer delivers every item exactly once.

#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "test_utils.hpp"

namespace {

class Channel {
public:
    void push(int v) {
        {
            std::lock_guard<std::mutex> lock{mtx_};
            q_.push(v);
        }
        cv_.notify_one();
    }

    // Blocks until an item is available; returns false on the sentinel.
    bool pop(int& out) {
        std::unique_lock<std::mutex> lock{mtx_};
        while (q_.empty()) {
            cv_.wait(lock);
        }
        out = q_.front();
        q_.pop();
        return out != kDone;
    }

private:
    std::condition_variable cv_;
    std::mutex mtx_;
    std::queue<int> q_;
    static constexpr int kDone = -1;
};

constexpr int kItems = 1'000;

}  // namespace

int main() {
    Channel ch;
    std::vector<int> received;
    received.reserve(static_cast<std::size_t>(kItems));

    std::thread producer{[&ch] {
        for (int i = 0; i < kItems; ++i) {
            ch.push(i);
        }
        ch.push(-1);  // sentinel
    }};
    std::thread consumer{[&ch, &received] {
        int v = 0;
        while (ch.pop(v)) {
            received.push_back(v);
        }
    }};
    producer.join();
    consumer.join();

    CHP_CHECK(received.size() == static_cast<std::size_t>(kItems));
    int sum = 0;
    for (const int v : received) {
        sum += v;
    }
    // sum(0..kItems-1) == n*(n-1)/2
    CHP_CHECK(sum == kItems * (kItems - 1) / 2);

    return chp::test_summary("producer_consumer");
}
