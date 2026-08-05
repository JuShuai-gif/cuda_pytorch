#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "test_utils.hpp"

namespace {

int g_constructed = 0;
int g_destroyed = 0;

struct Tracked {
    explicit Tracked(int v = 0) : value(v) { ++g_constructed; }
    Tracked(const Tracked& other) : value(other.value) { ++g_constructed; }
    Tracked& operator=(const Tracked&) = default;
    ~Tracked() { ++g_destroyed; }
    int value;
};

}  // namespace

int main() {
    // Placement new constructs without allocating; destructor must be called
    // manually; free() releases the memory (PDF p.183).
    {
        g_constructed = 0;
        g_destroyed = 0;
        void* mem = std::malloc(sizeof(Tracked));
        CHP_CHECK(mem != nullptr);
        auto* t = new (mem) Tracked(7);
        CHP_CHECK(t->value == 7);
        CHP_CHECK(g_constructed == 1);
        t->~Tracked();
        CHP_CHECK(g_destroyed == 1);
        std::free(mem);
    }

    // uninitialized_fill_n + destroy_at (PDF p.184).
    {
        g_constructed = 0;
        g_destroyed = 0;
        void* mem = std::malloc(sizeof(Tracked));
        CHP_CHECK(mem != nullptr);
        auto* t = static_cast<Tracked*>(mem);
        std::uninitialized_fill_n(t, 1, Tracked{3});
        CHP_CHECK(t->value == 3);
        std::destroy_at(t);
        CHP_CHECK(g_constructed == 2);  // temp + placed object
        CHP_CHECK(g_destroyed == 2);    // temp + destroyed object
        std::free(mem);
    }

    // new[]/delete[] must pair (PDF p.184-185).
    {
        g_constructed = 0;
        g_destroyed = 0;
        auto* arr = new Tracked[3];
        for (int i = 0; i < 3; ++i) {
            arr[i] = Tracked(i);
        }
        CHP_CHECK(g_constructed >= 3);
        delete[] arr;
        CHP_CHECK(g_destroyed == g_constructed);
    }

    return chp::test_summary("new_delete_placement");
}
