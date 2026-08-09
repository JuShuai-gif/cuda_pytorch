// std::thread lifecycle and identification (PDF p.289-292).
//
// Demonstrates thread creation, join/detach semantics, joinable() state,
// thread ids, and hardware_concurrency(). Thread arguments are forwarded
// to the callable exactly like std::bind.

#include <chrono>
#include <cstdio>
#include <thread>

namespace {

void worker_with_arg(int id, int times) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    for (int i = 0; i < times; ++i) {
        std::printf("thread %d running\n", id);
    }
}

}  // namespace

int main() {
    std::printf("== threads_basics ==\n");

    std::printf("hardware concurrency: %u\n",
                std::thread::hardware_concurrency());
    std::printf("main thread id: %llu\n",
                static_cast<unsigned long long>(
                    std::hash<std::thread::id>{}(std::this_thread::get_id())));

    // join(): main waits until the worker finishes.
    {
        std::thread t1{worker_with_arg, 1, 2};
        std::printf("t1 joinable after construction: %d\n", t1.joinable());
        t1.join();
        std::printf("t1 joinable after join: %d\n", t1.joinable());
    }

    // detach(): worker keeps running in the background; thread is not
    // joinable anymore. Use sparingly (book warns it is rarely needed).
    {
        std::thread t2{[] {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
            std::printf("detached worker finished\n");
        }};
        t2.detach();
        std::printf("t2 joinable after detach: %d\n", t2.joinable());
    }

    // Default-constructed threads are not joinable.
    std::thread t3;
    std::printf("default thread joinable: %d\n", t3.joinable());

    // Give the detached worker a chance to print before main exits.
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    return 0;
}
