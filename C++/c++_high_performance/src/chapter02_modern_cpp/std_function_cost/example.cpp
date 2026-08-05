// Detects whether std::function heap-allocates its captured state, and how
// the Small Buffer Optimization (SBO) affects that.
//
// The book (PDF p.59) claims: "An std::function heap allocates and captures
// variables ... note that some implementations of std::function do not
// heap-allocate if the size of the captured variable is less than a specific
// threshold." We verify that claim on libstdc++ by counting global operator
// new calls.
//
// Also inspect the assembly to confirm that a std::function call is an
// indirect call (cannot be inlined), unlike a plain lambda:
//   g++ -std=c++17 -O3 -S example.cpp

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>

namespace {

long g_allocations = 0;
long g_frees = 0;
bool g_tracking = false;

// Tracking state, guarded by a plain bool (single-threaded example).
struct TrackingGuard {
    long allocs_before;
    long frees_before;
    TrackingGuard() : allocs_before(g_allocations), frees_before(g_frees) {
        g_tracking = true;
    }
    ~TrackingGuard() { g_tracking = false; }
    long alloc_delta() const { return g_allocations - allocs_before; }
    long free_delta() const { return g_frees - frees_before; }
};

}  // namespace

// Global allocation counting. malloc is used instead of ::new to avoid
// recursion; malloc returns memory aligned to alignof(max_align_t), which is
// sufficient for all over-aligned objects used in this example.
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

namespace {

struct SmallCapture {
    int value;
};

struct LargeCapture {
    // Larger than std::function's internal buffer (SBO) on libstdc++.
    char data[64];
};

}  // namespace

int main() {
    std::printf("sizeof(std::function<void()>) = %zu bytes\n",
                sizeof(std::function<void()>));
    std::printf("sizeof(SmallCapture) = %zu, sizeof(LargeCapture) = %zu\n\n",
                sizeof(SmallCapture), sizeof(LargeCapture));

    // --- Lambda without capture: no allocation at all ---
    {
        TrackingGuard g;
        std::function<void()> f = []() {};
        f();
        std::printf("no-capture lambda -> new: %+ld, delete: %+ld\n",
                    g.alloc_delta(), g.free_delta());
    }

    // --- Small capture: fits SBO, no heap allocation ---
    {
        TrackingGuard g;
        int value = 7;
        std::function<int()> f = [value]() { return value; };
        const int r = f();
        std::printf("small capture (%zu bytes) -> new: %+ld, delete: %+ld, result %d\n",
                    sizeof(SmallCapture), g.alloc_delta(), g.free_delta(), r);
    }

    // --- Large capture: exceeds SBO, heap allocation observed ---
    {
        TrackingGuard g;
        LargeCapture big{};
        std::function<const char*()> f = [big]() { return big.data; };
        const char* r = f();
        std::printf("large capture (%zu bytes) -> new: %+ld, delete: %+ld, ptr %p\n",
                    sizeof(LargeCapture), g.alloc_delta(), g.free_delta(),
                    static_cast<const void*>(r));
    }

    std::printf("\nTo inspect assembly (indirect call, not inlined):\n");
    std::printf("  g++ -std=c++17 -O3 -S src/chapter02_modern_cpp/"
                "std_function_cost/example.cpp\n");
    return 0;
}
