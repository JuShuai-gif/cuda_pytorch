#include <cstdio>
#include <memory>
#include <type_traits>

#include "test_utils.hpp"

int main() {
    // unique_ptr is move-only and zero-overhead.
    static_assert(std::is_move_constructible<std::unique_ptr<int>>::value,
                  "unique_ptr is movable");
    static_assert(!std::is_copy_constructible<std::unique_ptr<int>>::value,
                  "unique_ptr is NOT copyable");
    static_assert(sizeof(std::unique_ptr<int>) == sizeof(int*),
                  "unique_ptr == one pointer");

    // Ownership transfer.
    {
        auto a = std::make_unique<int>(7);
        auto b = std::move(a);
        CHP_CHECK(a == nullptr);
        CHP_CHECK(*b == 7);
    }

    // shared_ptr ref counting.
    {
        auto a = std::make_shared<int>(5);
        CHP_CHECK(a.use_count() == 1);
        auto b = a;
        CHP_CHECK(a.use_count() == 2);
        CHP_CHECK(*b == 5);
        b.reset();
        CHP_CHECK(a.use_count() == 1);
    }

    // weak_ptr does not keep alive.
    {
        std::weak_ptr<int> weak;
        {
            auto a = std::make_shared<int>(9);
            weak = a;
            CHP_CHECK(!weak.expired());
            auto locked = weak.lock();
            CHP_CHECK(locked != nullptr);
            CHP_CHECK(*locked == 9);
        }
        CHP_CHECK(weak.expired());
        CHP_CHECK(weak.lock() == nullptr);
    }

    // make_shared vs new: both produce valid shared_ptr.
    {
        auto a = std::make_shared<double>(3.5);
        std::shared_ptr<double> b(new double{3.5});
        CHP_CHECK(*a == *b);
    }

    return chp::test_summary("smart_pointers");
}
