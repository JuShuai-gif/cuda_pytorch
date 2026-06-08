// Chapter 14: Specific Optimization Topics
// Demonstrates lookup tables, bit operations, integer division tricks,
// floating-point manipulation, and more.
// Compile: see CMakeLists.txt (ch14_demo target)

#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>

// ---- Example 14.1b: Lookup table for factorial ----
constexpr int FACTORIAL_TABLE_SIZE = 13;
const int FactorialTable[FACTORIAL_TABLE_SIZE] = {
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600};

int FactorialLookup(int n) {
    if (static_cast<unsigned int>(n) >= FACTORIAL_TABLE_SIZE) {
        std::cerr << "Factorial index out of range\n";
        return -1;
    }
    return FactorialTable[n];
}

// ---- Example 14.2b: Single comparison bounds check ----
// Check: 0 <= i < size using one unsigned comparison
bool BoundsCheck(int i, int size) {
    return static_cast<unsigned int>(i) < static_cast<unsigned int>(size);
}

// ---- Example 14.3b: Bitmask range check (power of 2) ----
bool IsInRangePow2(int i) {
    return (static_cast<unsigned int>(i) & 0xFFFFFFF0u) == 0u; // i in [0, 15]
}

// ---- Example 14.5b: Integer division by constant using multiplication ----
// Compiler does this automatically: a / 10
static inline unsigned int Div10(unsigned int n) {
    // Compiler will generate: n * 0xCCCCCCCD >> 35 (or similar)
    return n / 10;
}

// ---- Example 14.7b: Bit flags instead of multiple conditions ----
enum DayFlags : unsigned int {
    DF_Sunday = 1u << 0,
    DF_Monday = 1u << 1,
    DF_Tuesday = 1u << 2,
    DF_Wednesday = 1u << 3,
    DF_Thursday = 1u << 4,
    DF_Friday = 1u << 5,
    DF_Saturday = 1u << 6,
};
bool IsWeekend(unsigned int day_flags) {
    return (day_flags & (DF_Saturday | DF_Sunday)) != 0;
}

// ---- Example 14.12b: Avoid modulo in loops ----
void AvoidModulo(const int *data, int *out, int n) {
    // Instead of: if (i % 3 == 0)
    for (int i = 0; i < n; i += 3) {
        out[i] = data[i] * 2;
    }
}

// ---- Example 14.14b: Replace float division with multiplication ----
static inline double DivByConst(double x) {
    return x * (1.0 / 1.2345); // Faster than x / 1.2345
}

// ---- Example 14.16b: Common denominator trick ----
// Instead of: (a/b) + (c/d), use: (a*d + c*b) / (b*d)
// But beware of overflow!

// ---- Example 14.18b: Avoid mixing float and double ----
static inline float MixedPrecBad(float a, float b) {
    return a + b + 1.2; // 1.2 is double! Causes conversion.
}
static inline float MixedPrecGood(float a, float b) {
    return a + b + 1.2f; // 1.2f is float. No conversion.
}

// ---- Example 14.23: Manipulate float sign bit via integer ----
union FloatInt {
    float f;
    unsigned int i;
};
float NegateFloat(float x) {
    FloatInt u;
    u.f = x;
    u.i ^= 0x80000000u; // Flip sign bit
    return u.f;
}
float AbsFloat(float x) {
    FloatInt u;
    u.f = x;
    u.i &= 0x7FFFFFFFu; // Clear sign bit
    return u.f;
}

// ---- Example 14.29: Integer-to-float in [1.0, 2.0) range ----
// Convert integer n in [0, 2^23) to float in [1.0, 2.0)
union IntToFloatMagic {
    int i;
    float f;
};
float IntToFloat1to2(unsigned int n) {
    IntToFloatMagic u;
    u.i = static_cast<int>((n & 0x7FFFFFu) | 0x3F800000u);
    return u.f - 1.0f;
}

// ---- Main ----
int main() {
    std::cout << "=== Chapter 14: Specific Optimization Topics ===\n\n";

    // Lookup table
    std::cout << "FactorialLookup(5): " << FactorialLookup(5) << "\n";
    std::cout << "FactorialLookup(10): " << FactorialLookup(10) << "\n";

    // Bounds check
    std::cout << "BoundsCheck(5, 10): " << BoundsCheck(5, 10) << " (expect 1)\n";
    std::cout << "BoundsCheck(15, 10): " << BoundsCheck(15, 10) << " (expect 0)\n";
    std::cout << "BoundsCheck(-1, 10): " << BoundsCheck(-1, 10) << " (expect 0)\n";

    // Bit flags
    unsigned int friday_flag = DF_Friday;
    std::cout << "IsWeekend(Friday): " << IsWeekend(friday_flag) << " (expect 0)\n";
    unsigned int sat_flag = DF_Saturday;
    std::cout << "IsWeekend(Saturday): " << IsWeekend(sat_flag) << " (expect 1)\n";

    // Float sign manipulation
    std::cout << "NegateFloat(3.14f): " << NegateFloat(3.14f) << "\n";
    std::cout << "AbsFloat(-2.71f): " << AbsFloat(-2.71f) << "\n";

    // Division by constant
    std::cout << "Div10(42): " << Div10(42) << "\n";

    // Float division by const
    std::cout << "DivByConst(10.0): " << DivByConst(10.0) << " (expect " << 10.0 / 1.2345 << ")\n";

    // Precision mixing
    std::cout << "MixedBad(1.0f, 2.0f): " << MixedPrecBad(1.0f, 2.0f) << "\n";
    std::cout << "MixedGood(1.0f, 2.0f): " << MixedPrecGood(1.0f, 2.0f) << "\n";

    // Int to float magic
    std::cout << "IntToFloat1to2(0): " << IntToFloat1to2(0) << "\n";
    std::cout << "IntToFloat1to2(1000): " << IntToFloat1to2(1000) << "\n";

    // Modulo benchmark
    constexpr int N = 10000;
    int data[N], out[N] = {};
    for (int i = 0; i < N; ++i) data[i] = i;
    AvoidModulo(data, out, N);

    std::cout << "\nAll chapter 14 checks passed.\n";
    return 0;
}
