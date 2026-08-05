// 13_intrinsics: baseline -- scalar dot product, reduction, min/max.
#include <cfloat>
#include <cstdio>
#include <vector>

float scalar_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

float scalar_reduce(const float* a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i];
    return s;
}

void scalar_minmax(const float* a, int n, float& mn, float& mx) {
    mn = FLT_MAX;
    mx = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        mn = a[i] < mn ? a[i] : mn;
        mx = a[i] > mx ? a[i] : mx;
    }
}

int main() {
    const int n = 8'000'000;
    std::vector<float> a(n), b(n, 1.0f);
    for (int i = 0; i < n; ++i) a[i] = (float)(i % 1024);

    float d = scalar_dot(a.data(), b.data(), n);
    float s = scalar_reduce(a.data(), n);
    float mn, mx;
    scalar_minmax(a.data(), n, mn, mx);
    std::printf("dot=%.1f sum=%.1f min=%.0f max=%.0f\n", d, s, mn, mx);
    return 0;
}
