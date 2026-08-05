// Smart pointers and the make_shared allocation difference.
//
// The book (PDF p.193-195): std::unique_ptr has zero overhead; std::shared_ptr
// uses an atomic reference count; std::make_shared() performs ONE allocation
// (object + control block together), while `shared_ptr(new T)` performs TWO.

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

long g_allocations = 0;
long g_frees = 0;
bool g_tracking = false;

struct TrackingGuard {
    long a0 = g_allocations;
    long f0 = g_frees;
    TrackingGuard() { g_tracking = true; }
    ~TrackingGuard() { g_tracking = false; }
    long allocs() const { return g_allocations - a0; }
    long frees() const { return g_frees - f0; }
};

}  // namespace

void* operator new(std::size_t size) {
    if (g_tracking) {
        ++g_allocations;
    }
    if (void* p = std::malloc(size)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* p) noexcept {
    if (g_tracking) {
        ++g_frees;
    }
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete[](void* p) noexcept { ::operator delete(p); }

void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

int main() {
    std::printf("== smart_pointers ==\n");

    // unique_ptr: no overhead, ownership transfer via move (PDF p.193).
    {
        auto owner = std::make_unique<int>(42);
        std::unique_ptr<int> new_owner = std::move(owner);
        std::printf("unique_ptr: owner=%s new_owner=%d\n",
                    owner ? "set" : "null", *new_owner);
        static_assert(sizeof(std::unique_ptr<int>) == sizeof(int*),
                      "unique_ptr is a single pointer");
    }

    // make_shared: one allocation (PDF p.194).
    {
        TrackingGuard g;
        auto i = std::make_shared<double>(42.0);
        (void)i;
        std::printf("make_shared: allocs=%ld frees=%ld\n", g.allocs(), g.frees());
    }

    // shared_ptr(new T): two allocations (PDF p.195).
    {
        TrackingGuard g;
        std::shared_ptr<double> i(new double{42.0});
        (void)i;
        std::printf("shared_ptr(new T): allocs=%ld frees=%ld\n", g.allocs(),
                    g.frees());
    }

    // weak_ptr: does not keep the object alive (PDF p.195-196).
    {
        std::weak_ptr<int> weak;
        {
            auto i = std::make_shared<int>(10);
            weak = i;
            if (auto shared = weak.lock()) {
                std::printf("weak_ptr while alive: %d\n", *shared);
            }
        }
        // i destroyed: weak expired.
        if (auto shared = weak.lock()) {
            std::printf("weak_ptr STILL alive (bug!)\n");
        } else {
            std::printf("weak_ptr expired after owner destroyed\n");
        }
    }

    return 0;
}
