// 16_division_optimization: baseline -- division by a runtime variable.
//
// PDF 14.5 (p150): integer division costs 27-80 cycles; a variable divisor
// cannot be strength-reduced by the compiler.
#include <cstdio>
#include <vector>

int sum_div(const std::vector<int>& v, int d) {
    int s = 0;
    for (int x : v) s += x / d;   // runtime divisor -> slow idiv
    return s;
}

int main() {
    std::vector<int> v(8'000'000, 1000);
    volatile int r = sum_div(v, 7);
    std::printf("checksum = %d\n", r);
    return 0;
}
