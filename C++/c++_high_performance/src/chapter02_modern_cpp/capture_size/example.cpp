#include <cstdio>
#include <cstddef>

// Demonstrates that the size of a lambda object equals the size of its
// captured members (book PDF p.51-52: the capture block is the class's
// member variables). Combined with std::function this determines whether
// the SBO can avoid a heap allocation.

namespace {

struct SmallCapture {
    int value;
};

struct LargeCapture {
    char data[64];
};

}  // namespace

int main() {
    std::printf("== capture_size ==\n");

    int a = 1;
    long b = 2;
    LargeCapture big{};

    auto no_capture = []() { return 0; };
    auto small = [a]() { return a; };
    auto two_small = [a, b]() { return static_cast<long>(a) + b; };
    auto large = [big]() { return big.data; };
    auto by_ref = [&big]() { return big.data; };

    std::printf("sizeof(int)          = %2zu\n", sizeof(int));
    std::printf("sizeof(SmallCapture) = %2zu\n", sizeof(SmallCapture));
    std::printf("sizeof(LargeCapture) = %2zu\n", sizeof(LargeCapture));
    std::printf("no_capture lambda    = %2zu bytes\n", sizeof(no_capture));
    std::printf("small capture lambda = %2zu bytes\n", sizeof(small));
    std::printf("two small captures   = %2zu bytes\n", sizeof(two_small));
    std::printf("large capture lambda = %2zu bytes\n", sizeof(large));
    std::printf("by-reference capture = %2zu bytes (a pointer)\n", sizeof(by_ref));

    std::printf("\nA lambda's size is the sum of its captures. If that size\n");
    std::printf("exceeds std::function's SBO buffer (16 bytes on libstdc++),\n");
    std::printf("storing it in std::function heap-allocates (see std_function_cost).\n");
    return 0;
}
