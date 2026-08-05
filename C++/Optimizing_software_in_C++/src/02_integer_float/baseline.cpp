// 02_integer_float: baseline -- mixed precision + runtime divisor.
//
// PDF p32: do not mix float and double (conversion instructions).
// PDF p150: division by a variable is slow (27-80 cycles).
#include <cstdint>
#include <cstdio>
#include <vector>

float work_mixed(const std::vector<float>& x, double divisor) {
    float s = 0.0f;
    for (float v : x) {
        // v is float, divisor is double: every iteration converts float->double
        // and back (PDF p32, p153).
        s += static_cast<float>(static_cast<double>(v) / divisor);
    }
    return s;
}

int main() {
    std::vector<float> x(4'000'000, 1.0f);
    float r = work_mixed(x, 3.0);
    std::printf("checksum = %.6f\n", r);
    return 0;
}
