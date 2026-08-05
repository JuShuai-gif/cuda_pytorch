// 04_loop: loop optimizations -- accumulators, unrolling, invariant hoisting.
//
// PDF 7.13 (p45-48) and 11 (p113-114). Note: with -O3 -ffast-math the
// compiler may already do these transforms; run WITHOUT fast-math first to
// see the manual effect, then with it to see what the compiler does.
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

// 1 accumulator (serial chain)
double sum1(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x;
    return s;
}

// 2 accumulators
double sum2(const std::vector<double>& a) {
    double s0 = 0.0, s1 = 0.0;
    size_t i = 0;
    for (; i + 1 < a.size(); i += 2) { s0 += a[i]; s1 += a[i + 1]; }
    for (; i < a.size(); ++i) s0 += a[i];
    return s0 + s1;
}

// 4 accumulators
double sum4(const std::vector<double>& a) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    for (; i + 3 < a.size(); i += 4) {
        s0 += a[i]; s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3];
    }
    for (; i < a.size(); ++i) s0 += a[i];
    return (s0 + s1) + (s2 + s3);
}

// 8 accumulators
double sum8(const std::vector<double>& a) {
    double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    size_t i = 0;
    for (; i + 7 < a.size(); i += 8) {
        s0+=a[i]; s1+=a[i+1]; s2+=a[i+2]; s3+=a[i+3];
        s4+=a[i+4]; s5+=a[i+5]; s6+=a[i+6]; s7+=a[i+7];
    }
    for (; i < a.size(); ++i) s0 += a[i];
    return (s0+s1)+(s2+s3)+(s4+s5)+(s6+s7);
}

// Loop invariant code motion: division hoisted out of the loop (PDF p72).
double div_in_loop(const std::vector<double>& a, double d) {
    double s = 0.0;
    for (double x : a) s += x / d;      // division recomputed every step
    return s;
}

double div_hoisted(const std::vector<double>& a, double d) {
    double s = 0.0;
    double inv = 1.0 / d;               // hoisted (PDF p152)
    for (double x : a) s += x * inv;
    return s;
}

int main() {
    std::vector<double> a(16'000'000, 1.0);
    bench("sum1", [&] { return sum1(a); });
    bench("sum2", [&] { return sum2(a); });
    bench("sum4", [&] { return sum4(a); });
    bench("sum8", [&] { return sum8(a); });
    bench("div_in_loop",  [&] { return div_in_loop(a, 3.0); });
    bench("div_hoisted",  [&] { return div_hoisted(a, 3.0); });

    std::printf("\nresults equal: %.0f %.0f %.0f %.0f %.6f %.6f\n",
                sum1(a), sum2(a), sum4(a), sum8(a),
                div_in_loop(a, 3.0), div_hoisted(a, 3.0));
    return 0;
}
