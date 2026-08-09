// Correctness checks for std::thread lifecycle.

#include <cstdio>
#include <thread>

#include "test_utils.hpp"

namespace {

int g_ran = 0;

void record() { ++g_ran; }

}  // namespace

int main() {
    std::thread t;
    CHP_CHECK(!t.joinable());  // default constructed

    t = std::thread{record};
    CHP_CHECK(t.joinable());
    t.join();
    CHP_CHECK(!t.joinable());  // already joined
    CHP_CHECK(g_ran == 1);

    // Moved-from threads are not joinable; the running thread is transferred.
    std::thread t2{record};
    std::thread t3 = std::move(t2);
    CHP_CHECK(!t2.joinable());
    CHP_CHECK(t3.joinable());
    t3.join();
    CHP_CHECK(g_ran == 2);

    // Arguments are forwarded to the callable.
    int sum = 0;
    std::thread t4{[&sum](int a, int b) { sum = a + b; }, 2, 3};
    t4.join();
    CHP_CHECK(sum == 5);

    return chp::test_summary("threads_basics");
}
