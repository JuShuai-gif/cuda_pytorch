// Correctness checks for packaged_task and std::async.

#include <cstdio>
#include <exception>
#include <future>
#include <stdexcept>

#include "test_utils.hpp"

namespace {

int divide(int a, int b) {
    if (b == 0) {
        throw std::runtime_error{"Divide by zero exception"};
    }
    return a / b;
}

}  // namespace

int main() {
    // async success + error.
    {
        auto f = std::async(divide, 100, 4);
        CHP_CHECK(f.get() == 25);
    }
    {
        auto f = std::async(divide, 1, 0);
        bool caught = false;
        try {
            (void)f.get();
        } catch (const std::runtime_error&) {
            caught = true;
        }
        CHP_CHECK(caught);
    }

    // async with a lambda, deferred policy runs on get().
    {
        auto f = std::async(std::launch::deferred, [] { return 7 * 6; });
        CHP_CHECK(f.get() == 42);
    }

    // packaged_task value transport.
    {
        std::packaged_task<int(int, int)> task{divide};
        auto f = task.get_future();
        std::thread{std::move(task), 81, 9}.detach();
        CHP_CHECK(f.get() == 9);
    }

    // packaged_task exception transport.
    {
        std::packaged_task<int(int, int)> task{divide};
        auto f = task.get_future();
        std::thread{std::move(task), 1, 0}.detach();
        bool caught = false;
        try {
            (void)f.get();
        } catch (const std::runtime_error&) {
            caught = true;
        }
        CHP_CHECK(caught);
    }

    return chp::test_summary("async_tasks");
}
