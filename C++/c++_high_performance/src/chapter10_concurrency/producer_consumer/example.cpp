// Producer-consumer with a condition variable (PDF p.296-298).
//
// The producer pushes one integer per second and notifies the consumer,
// which sleeps in cv.wait() instead of busy-polling. A sentinel (-1) tells
// the consumer to stop.

#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <queue>
#include <thread>

namespace {

std::condition_variable cv;
std::queue<int> q;
std::mutex mtx;
constexpr int kDone = -1;

void generate_ints() {
    for (const int v : {1, 2, 3, kDone}) {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        {
            const std::lock_guard<std::mutex> lock{mtx};
            q.push(v);
        }
        cv.notify_one();  // notification does not require holding the lock
    }
}

void print_ints() {
    int i = 0;
    while (i != kDone) {
        {
            std::unique_lock<std::mutex> lock{mtx};
            while (q.empty()) {
                cv.wait(lock);  // releases the lock while sleeping
            }
            i = q.front();
            q.pop();
        }
        if (i != kDone) {
            std::printf("got: %d\n", i);
        }
    }
}

}  // namespace

int main() {
    std::printf("== producer_consumer ==\n");

    std::thread producer{generate_ints};
    std::thread consumer{print_ints};
    producer.join();
    consumer.join();

    std::printf("done\n");
    return 0;
}
