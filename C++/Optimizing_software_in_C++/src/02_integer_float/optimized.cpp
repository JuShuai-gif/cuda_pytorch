// 02_integer_float: optimized -- single precision + multiply by reciprocal.
//
// PDF p32: keep one precision everywhere.
// PDF p152: float division by a constant -> multiply by the reciprocal.
#include <cstdint>
#include <cstdio>
#include <vector>

float work_single(const std::vector<float>& x, float inv_divisor) {
    float s = 0.0f;
    for (float v : x) {
        s += v * inv_divisor;   // all float, no conversions, no division
    }
    return s;
}

int main() {
    std::vector<float> x(4'000'000, 1.0f);
    float r = work_single(x, 1.0f / 3.0f);   // reciprocal hoisted (PDF p152)
    std::printf("checksum = %.6f\n", r);
    return 0;
}
