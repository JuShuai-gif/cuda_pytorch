// Small String Optimization.
//
// The book (PDF p.196-198): std::string avoids heap allocation for short
// strings by keeping a small inline buffer (a union of a "short" and a "long"
// layout). On libstdc++, strings up to 15 chars fit inline (capacity 15);
// longer strings allocate.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>

namespace {

long g_allocations = 0;
bool g_tracking = false;

struct TrackingGuard {
    long a0 = g_allocations;
    TrackingGuard() { g_tracking = true; }
    ~TrackingGuard() { g_tracking = false; }
    long allocs() const { return g_allocations - a0; }
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

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main() {
    std::printf("== small_string ==\n");
    std::printf("sizeof(std::string) = %zu bytes\n", sizeof(std::string));

    // Test different lengths; find the SSO boundary.
    for (std::size_t len = 0; len <= 18; ++len) {
        const std::string text(len, 'x');
        TrackingGuard g;
        std::string s = text;  // copy: if len <= SSO, no allocation
        const long allocs = g.allocs();
        std::printf("len %2zu: capacity=%2zu allocs=%ld\n", len,
                    s.capacity(), allocs);
    }

    std::printf("\nlibstdc++ keeps strings up to %zu chars inline (no heap).\n",
                static_cast<std::size_t>(16));
    return 0;
}
