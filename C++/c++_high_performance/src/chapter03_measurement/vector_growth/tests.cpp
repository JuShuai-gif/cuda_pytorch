#include <cstddef>
#include <cstdio>
#include <vector>

#include "test_utils.hpp"

int main() {
    // Reserve avoids reallocation: capacity stays stable.
    {
        std::vector<int> v;
        v.reserve(100);
        CHP_CHECK(v.capacity() >= 100);
        const std::size_t cap = v.capacity();
        for (std::size_t i = 0; i < 100; ++i) {
            v.push_back(static_cast<int>(i));
        }
        CHP_CHECK(v.capacity() == cap);
        CHP_CHECK(v.size() == 100);
    }

    // Without reserve, capacity grows geometrically (doubling).
    {
        std::vector<int> v;
        std::size_t prev_capacity = v.capacity();
        bool grew = false;
        for (std::size_t i = 0; i < 1000; ++i) {
            v.push_back(static_cast<int>(i));
            if (v.capacity() > prev_capacity) {
                grew = true;
                // A reallocation happened; capacity should have grown, not
                // shrunk, and typically multiplies.
                CHP_CHECK(v.capacity() > prev_capacity);
                prev_capacity = v.capacity();
            }
        }
        CHP_CHECK(grew);
        CHP_CHECK(v.size() == 1000);
    }

    // Capacity is >= size always.
    {
        std::vector<int> v;
        for (std::size_t i = 0; i < 500; ++i) {
            v.push_back(static_cast<int>(i));
            CHP_CHECK(v.capacity() >= v.size());
        }
    }

    return chp::test_summary("vector_growth");
}
