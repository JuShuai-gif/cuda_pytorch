// Chapter 14: "具体的优化主题" (Specific Optimization Topics)
// Consolidated runnable examples
// Compile: g++ -std=c++11 -O2 -msse2 ch14_optimization.cpp -o ch14_optimization

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <emmintrin.h>

using std::cout;
using std::endl;

// =============================================================================
// Section 14.1: Look-up tables - factorial computation vs table
// =============================================================================

// Example 14.1a: Compute factorial by loop
int factorial_loop(int n) {
    int i, f = 1;
    for (i = 2; i <= n; i++)
        f *= i;
    return f;
}

// Example 14.1b: Factorial via precomputed lookup table with bounds check
int factorial_table(int n) {
    const int FactorialTable[13] = {1,    1,     2,      6,       24,       120,      720,
                                    5040, 40320, 362880, 3628800, 39916800, 479001600};
    if ((unsigned int)n < 13)
        return FactorialTable[n];
    else
        return 0;
}

// Global factorial table (Example 14.1c: table defined outside critical loop)
const int g_FactorialTable[13] = {1,    1,     2,      6,       24,       120,      720,
                                  5040, 40320, 362880, 3628800, 39916800, 479001600};

void demo_14_1() {
    cout << "=== Section 14.1: Look-up tables (Factorial) ===" << endl;

    // Verify correctness
    bool ok = true;
    for (int n = 0; n < 13; n++) {
        if (factorial_loop(n) != factorial_table(n)) {
            cout << "Mismatch at n=" << n << endl;
            ok = false;
        }
    }
    if (ok)
        cout << "  factorial_loop and factorial_table produce same results" << endl;

    // Example 14.1c: Use global table in a critical inner loop
    volatile int sink = 0;  // prevent optimization
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000000; iter++) {
        for (int b = 0; b < 12; b++) {
            sink += g_FactorialTable[b];
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    cout << "  14.1c (global table loop, 1M iters): " << ns << " ns" << endl;
}

// =============================================================================
// Section 14.2: Ternary operator vs lookup table
// =============================================================================

float ternary_select(int b) {
    return (b == 0) ? 1.0f : 2.5f;
}

float table_select(int b) {
    const float OneOrTwo5[2] = {1.0f, 2.5f};
    return OneOrTwo5[b & 1];
}

void demo_14_2() {
    cout << "=== Section 14.2: Ternary operator vs lookup table ===" << endl;
    cout << "  ternary_select(0) = " << ternary_select(0) << endl;
    cout << "  ternary_select(1) = " << ternary_select(1) << endl;
    cout << "  table_select(0)   = " << table_select(0) << endl;
    cout << "  table_select(1)   = " << table_select(1) << endl;
}

// =============================================================================
// Section 14.3: switch vs pointer array lookup
// =============================================================================

const char* switch_greek(int n) {
    switch (n) {
        case 0:
            return "Alpha";
        case 1:
            return "Beta";
        case 2:
            return "Gamma";
        case 3:
            return "Delta";
        default:
            return "Unknown";
    }
}

const char* const Greek[4] = {"Alpha", "Beta", "Gamma", "Delta"};

const char* array_greek(int n) {
    if ((unsigned int)n < 4)
        return Greek[n];
    return "Unknown";
}

void demo_14_3() {
    cout << "=== Section 14.3: switch vs pointer array lookup ===" << endl;
    cout << "  switch_greek(1) = " << switch_greek(1) << endl;
    cout << "  array_greek(1)  = " << array_greek(1) << endl;
}

// =============================================================================
// Section 14.4: Bounds checking optimization (unsigned int trick)
// =============================================================================

const int size_14_4 = 16;
float list_14_4[size_14_4];

// 14.4a: Bounds check with two comparisons
bool bounds_check_two_cmps(int i) {
    if (i < 0 || i >= size_14_4)
        return false;
    list_14_4[i] += 1.0f;
    return true;
}

// 14.4b: Bounds check with single unsigned comparison
bool bounds_check_unsigned(int i) {
    if ((unsigned int)i >= (unsigned int)size_14_4)
        return false;
    list_14_4[i] += 1.0f;
    return true;
}

void demo_14_4() {
    cout << "=== Section 14.4: Bounds checking optimization ===" << endl;
    for (int k = 0; k < size_14_4; k++)
        list_14_4[k] = (float)k;

    bool ok = true;
    for (int i = -5; i < 20; i++) {
        if (bounds_check_two_cmps(i) != bounds_check_unsigned(i)) {
            ok = false;
        }
    }
    cout << "  both bounds-check methods agree: " << (ok ? "yes" : "NO") << endl;
}

// =============================================================================
// Section 14.5: Range checking optimization
// =============================================================================

const int min_14_5 = 100, max_14_5 = 110;

// 14.5a: Range check with two comparisons
bool range_check_two_cmps(int i) {
    return (i >= min_14_5 && i <= max_14_5);
}

// 14.5b: Range check with unsigned subtraction trick
bool range_check_unsigned(int i) {
    return (unsigned int)(i - min_14_5) <= (unsigned int)(max_14_5 - min_14_5);
}

void demo_14_5() {
    cout << "=== Section 14.5: Range checking optimization ===" << endl;
    bool ok = true;
    for (int i = 90; i < 120; i++) {
        if (range_check_two_cmps(i) != range_check_unsigned(i)) {
            cout << "  Mismatch at i=" << i << endl;
            ok = false;
        }
    }
    if (ok)
        cout << "  both range-check methods agree: yes" << endl;
}

// =============================================================================
// Section 14.6: Modulo by power of 2 using bitwise AND
// =============================================================================

float list_14_6[16];

void demo_14_6() {
    cout << "=== Section 14.6: Modulo by power of 2 using bitwise AND ===" << endl;
    for (int k = 0; k < 16; k++)
        list_14_6[k] = (float)k;

    // i & 15 is equivalent to i % 16 for non-negative i
    volatile float sink = 0;
    for (int i = 0; i < 100; i++) {
        sink += list_14_6[i & 15];
    }
    cout << "  list[i & 15] used in loop (bitwise AND replaces modulo 16)" << endl;
}

// =============================================================================
// Section 14.7: Multiple flag testing using bitwise OR
// =============================================================================

// 14.7a: Weekdays with sequential values, testing with ||
enum Weekdays_old {
    Sunday_old,
    Monday_old,
    Tuesday_old,
    Wednesday_old,
    Thursday_old,
    Friday_old,
    Saturday_old
};

bool is_target_day_old(Weekdays_old d) {
    return (d == Tuesday_old || d == Wednesday_old || d == Friday_old);
}

// 14.7b: Weekdays with bit-flag values, testing with &
enum Weekdays {
    Sunday = 1,
    Monday = 2,
    Tuesday = 4,
    Wednesday = 8,
    Thursday = 0x10,
    Friday = 0x20,
    Saturday = 0x40
};

const int WeekdayMask = (Tuesday | Wednesday | Friday);

bool is_target_day(Weekdays d) {
    return (d & WeekdayMask) != 0;
}

const char* weekday_name(Weekdays d) {
    switch (d) {
        case Sunday:
            return "Sunday";
        case Monday:
            return "Monday";
        case Tuesday:
            return "Tuesday";
        case Wednesday:
            return "Wednesday";
        case Thursday:
            return "Thursday";
        case Friday:
            return "Friday";
        case Saturday:
            return "Saturday";
        default:
            return "Unknown";
    }
}

void demo_14_7() {
    cout << "=== Section 14.7: Multiple flag testing using bitwise OR ===" << endl;
    Weekdays week[] = {Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday};
    cout << "  Target days (Tue, Wed, Fri): ";
    for (int i = 0; i < 7; i++) {
        if (is_target_day(week[i]))
            cout << weekday_name(week[i]) << " ";
    }
    cout << endl;
}

// =============================================================================
// Section 14.8: Multidimensional array access
// =============================================================================

const int rows_14_8 = 10, columns_14_8 = 8;
float matrix_14_8[rows_14_8][columns_14_8];

int order_14_8(int x) {
    // Example ordering function
    return (x * 3 + 1) % rows_14_8;
}

void demo_14_8() {
    cout << "=== Section 14.8: Multidimensional array access ===" << endl;
    for (int i = 0; i < rows_14_8; i++) {
        int j = order_14_8(i);
        matrix_14_8[j][0] = (float)i;
    }
    cout << "  matrix[j][0] = i for i=0.." << rows_14_8 - 1 << " (indirect row access)" << endl;
}

// =============================================================================
// Section 14.9: Struct size aligned to power of 2
// =============================================================================

// Struct with 3 ints would be 12 bytes (not power of 2).
// Adding a filler makes it 16 bytes, aligning to a power of 2
// to avoid cache bank conflicts.
struct S1 {
    int a;
    int b;
    int c;
    int UnusedFiller;  // pad to 16 bytes (power of 2)
};

const int size_14_9 = 100;
S1 list_14_9[size_14_9];

int order_14_9(int x) {
    return (x * 5 + 7) % size_14_9;
}

void demo_14_9() {
    cout << "=== Section 14.9: Struct size aligned to power of 2 ===" << endl;
    cout << "  sizeof(S1) = " << sizeof(S1) << " bytes (padded to power of 2)" << endl;

    // Initialize
    for (int i = 0; i < size_14_9; i++) {
        list_14_9[i].a = i;
        list_14_9[i].b = i * 2;
        list_14_9[i].c = i * 3;
    }

    // Indirect access pattern
    volatile int sink = 0;
    for (int i = 0; i < size_14_9; i++) {
        int j = order_14_9(i);
        list_14_9[j].a = list_14_9[j].b + list_14_9[j].c;
        sink += list_14_9[j].a;
    }
}

// =============================================================================
// Section 14.10: Integer division optimization
// =============================================================================

void demo_14_10() {
    cout << "=== Section 14.10: Integer division optimization ===" << endl;

    volatile int a_sink = 0;
    int b = 123456, c = 7;

    // Slow: division by variable
    a_sink += b / c;

    // Faster: division by constant (compiler can optimize)
    a_sink += b / 10;

    // Still faster: unsigned division by constant
    a_sink += (unsigned int)b / 10;

    // Fast: division by power of 2 (becomes shift)
    a_sink += b / 16;

    // Still faster: unsigned division by power of 2
    a_sink += (unsigned int)b / 16;

    cout << "  Various integer division forms evaluated" << endl;
}

// =============================================================================
// Section 14.11: Modulo optimization
// =============================================================================

void demo_14_11() {
    cout << "=== Section 14.11: Modulo optimization ===" << endl;

    volatile int a_sink = 0;
    int b = 123456, c = 7;

    // Slow
    a_sink += b % c;
    // Faster: modulo by constant
    a_sink += b % 10;
    // Still faster: unsigned
    a_sink += (unsigned int)b % 10;
    // Fast: modulo by power of 2
    a_sink += b % 16;
    // Still faster: unsigned
    a_sink += (unsigned int)b % 16;

    cout << "  Various modulo forms evaluated" << endl;
}

// =============================================================================
// Section 14.12: Eliminating slow division in loops (induction variable)
// =============================================================================

// 14.12a: Division inside loop
void fill_division_14_12a(int* list, int n) {
    for (int i = 0; i < n; i++) {
        list[i] += i / 3;
    }
}

// 14.12b: Eliminate division using induction variable
void fill_induction_14_12b(int* list, int n) {
    int i, i_div_3;
    // Process in groups of 3; n should be divisible by 3 for correctness
    for (i = i_div_3 = 0; i < n; i += 3, i_div_3++) {
        list[i] += i_div_3;
        list[i + 1] += i_div_3;
        list[i + 2] += i_div_3;
    }
}

void demo_14_12() {
    cout << "=== Section 14.12: Eliminating slow division in loops ===" << endl;

    const int N = 300;
    int list1[N] = {};
    int list2[N] = {};

    fill_division_14_12a(list1, N);
    fill_induction_14_12b(list2, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (list1[i] != list2[i]) {
            ok = false;
            break;
        }
    }
    cout << "  Results match: " << (ok ? "yes" : "NO") << endl;

    // Timing comparison
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000000; iter++) {
        fill_division_14_12a(list1, N);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ns_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000000; iter++) {
        fill_induction_14_12b(list2, N);
    }
    t2 = std::chrono::high_resolution_clock::now();
    auto ns_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  14.12a (division in loop, 1M iters): " << ns_a << " ms" << endl;
    cout << "  14.12b (induction variable, 1M iters): " << ns_b << " ms" << endl;
}

// =============================================================================
// Section 14.13: Eliminating slow modulo in loops
// =============================================================================

// 14.13a: Modulo inside loop
void fill_modulo_14_13a(int* list, int n) {
    for (int i = 0; i < n; i++) {
        list[i] = i % 3;
    }
}

// 14.13b: Unrolled to eliminate modulo (assumes n divisible by 3)
void fill_unrolled_14_13b(int* list, int n) {
    for (int i = 0; i < n; i += 3) {
        list[i] = 0;
        list[i + 1] = 1;
        list[i + 2] = 2;
    }
}

// 14.13c: Handle remainder when n not divisible by 3
void fill_unrolled_14_13c(int* list, int n) {
    int i;
    for (i = 0; i < n - 2; i += 3) {
        list[i] = 0;
        list[i + 1] = 1;
        list[i + 2] = 2;
    }
    if (i < n)
        list[i] = 0;
    if (i + 1 < n)
        list[i + 1] = 1;
}

// Alternate version matching the book exactly (n=301)
void fill_unrolled_14_13c_book(int* list) {
    for (int i = 0; i < 301; i += 3) {
        list[i] = 0;
        list[i + 1] = 1;
        list[i + 2] = 2;
    }
    list[300] = 0;
}

void demo_14_13() {
    cout << "=== Section 14.13: Eliminating slow modulo in loops ===" << endl;

    const int N = 300;
    int list1[N] = {};
    int list2[N] = {};

    fill_modulo_14_13a(list1, N);
    fill_unrolled_14_13b(list2, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (list1[i] != list2[i]) {
            ok = false;
            break;
        }
    }
    cout << "  Results match (300): " << (ok ? "yes" : "NO") << endl;

    // Test 14.13c with odd size
    const int N_odd = 301;
    int list3[N_odd] = {};
    int list4[N_odd] = {};

    fill_modulo_14_13a(list3, N_odd);
    fill_unrolled_14_13c(list4, N_odd);

    ok = true;
    for (int i = 0; i < N_odd; i++) {
        if (list3[i] != list4[i]) {
            ok = false;
            break;
        }
    }
    cout << "  Results match (301): " << (ok ? "yes" : "NO") << endl;

    // Timing
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000000; iter++) {
        fill_modulo_14_13a(list1, N);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000000; iter++) {
        fill_unrolled_14_13b(list2, N);
    }
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  14.13a (modulo in loop, 1M iters): " << ms_a << " ms" << endl;
    cout << "  14.13b (unrolled, 1M iters):       " << ms_b << " ms" << endl;
}

// =============================================================================
// Section 14.14: Float division to multiplication (1/c)
// =============================================================================

// 14.14a: Division by constant
double div_14_14a(double b) {
    return b / 1.2345;
}

// 14.14b: Multiply by reciprocal (compiler usually does this anyway at -O2)
double mul_14_14b(double b) {
    return b * (1.0 / 1.2345);
}

void demo_14_14() {
    cout << "=== Section 14.14: Float division to multiplication ===" << endl;
    double val = 100.0;
    cout << "  div_14_14a(100) = " << div_14_14a(val) << endl;
    cout << "  mul_14_14b(100) = " << mul_14_14b(val) << endl;

    // Timing
    const int N = 100000000;
    volatile double sink = 0;
    double b = 3.14159;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink += div_14_14a(b);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink += mul_14_14b(b);
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  14.14a (division, 100M iters): " << ms_a << " ms" << endl;
    cout << "  14.14b (multiply, 100M iters): " << ms_b << " ms" << endl;
}

// =============================================================================
// Section 14.15: Moving division across inequality
// =============================================================================

// 14.15a: Division on RHS of comparison
bool cmp_14_15a(double a, double b, double c) {
    return a > b / c;
}

// 14.15b: Multiply both sides (avoiding division)
// NOTE: only valid when c > 0
bool cmp_14_15b(double a, double b, double c) {
    return a * c > b;
}

void demo_14_15() {
    cout << "=== Section 14.15: Moving division across inequality ===" << endl;
    bool ok = true;
    double test_vals[] = {0.5, 1.0, 2.0, 10.0};
    for (double a : test_vals) {
        for (double b : test_vals) {
            for (double c : test_vals) {
                if (c <= 0)
                    continue;
                if (cmp_14_15a(a, b, c) != cmp_14_15b(a, b, c)) {
                    ok = false;
                }
            }
        }
    }
    cout << "  Both methods agree (c > 0): " << (ok ? "yes" : "NO") << endl;
}

// =============================================================================
// Section 14.16: Common denominator to reduce divisions
// =============================================================================

// 14.16a: Two separate divisions
double two_divs_14_16a(double a1, double b1, double a2, double b2) {
    return a1 / b1 + a2 / b2;
}

// 14.16b: Combine into one division
double one_div_14_16b(double a1, double b1, double a2, double b2) {
    return (a1 * b2 + a2 * b1) / (b1 * b2);
}

void demo_14_16() {
    cout << "=== Section 14.16: Common denominator to reduce divisions ===" << endl;
    double a1 = 1.0, b1 = 2.0, a2 = 3.0, b2 = 4.0;
    cout << "  two_divs: " << two_divs_14_16a(a1, b1, a2, b2) << endl;
    cout << "  one_div:  " << one_div_14_16b(a1, b1, a2, b2) << endl;

    // Timing
    const int N = 100000000;
    volatile double sink = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink += two_divs_14_16a(a1, b1, a2, b2);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink += one_div_14_16b(a1, b1, a2, b2);
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  14.16a (two divisions, 100M iters): " << ms_a << " ms" << endl;
    cout << "  14.16b (one division, 100M iters):  " << ms_b << " ms" << endl;
}

// =============================================================================
// Section 14.17: Sharing reciprocal across multiple divisions
// =============================================================================

// 14.17a: Two separate divisions
void two_divs_14_17a(double a1, double b1, double a2, double b2, double& y1, double& y2) {
    y1 = a1 / b1;
    y2 = a2 / b2;
}

// 14.17b: Share reciprocal: compute 1/(b1*b2) and reuse
void shared_recip_14_17b(double a1, double b1, double a2, double b2, double& y1, double& y2) {
    double reciprocal_divisor = 1.0 / (b1 * b2);
    y1 = a1 * b2 * reciprocal_divisor;
    y2 = a2 * b1 * reciprocal_divisor;
}

void demo_14_17() {
    cout << "=== Section 14.17: Sharing reciprocal across multiple divisions ===" << endl;
    double y1a, y2a, y1b, y2b;
    double a1 = 3.0, b1 = 4.0, a2 = 5.0, b2 = 6.0;

    two_divs_14_17a(a1, b1, a2, b2, y1a, y2a);
    shared_recip_14_17b(a1, b1, a2, b2, y1b, y2b);

    cout << "  14.17a: y1=" << y1a << " y2=" << y2a << endl;
    cout << "  14.17b: y1=" << y1b << " y2=" << y2b << endl;

    // Timing
    const int N = 100000000;
    volatile double sink = 0;
    double y1 = 0, y2 = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        two_divs_14_17a(a1, b1, a2, b2, y1, y2);
        sink += y1 + y2;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        shared_recip_14_17b(a1, b1, a2, b2, y1, y2);
        sink += y1 + y2;
    }
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  14.17a (two divs, 100M iters): " << ms_a << " ms" << endl;
    cout << "  14.17b (shared recip, 100M):    " << ms_b << " ms" << endl;
}

// =============================================================================
// Section 14.18: Mixed precision (float/double) avoiding implicit conversions
// =============================================================================

// 14.18a: Bad - mixing float and double (1.2 is double literal)
float mixed_precision_bad(float b) {
    return b * 1.2;  // b is promoted to double, then result truncated to float
}

// 14.18b: Good - everything float (1.2f is float literal)
float mixed_precision_float(float b) {
    return b * 1.2f;
}

// 14.18c (in 14.18b file): Good - everything double
double mixed_precision_double(double b) {
    return b * 1.2;
}

void demo_14_18() {
    cout << "=== Section 14.18: Mixed precision avoiding implicit conversions ===" << endl;
    float bf = 100.0f;
    double bd = 100.0;
    cout << "  bad (float*double):    " << mixed_precision_bad(bf) << endl;
    cout << "  good_float (float*float):  " << mixed_precision_float(bf) << endl;
    cout << "  good_double (double*double): " << mixed_precision_double(bd) << endl;

    // Timing
    const int N = 200000000;
    volatile float sink_f = 0;
    volatile double sink_d = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink_f += mixed_precision_bad(bf);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink_f += mixed_precision_float(bf);
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink_d += mixed_precision_double(bd);
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_c = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  14.18a (mixed float/double, 200M iters): " << ms_a << " ms" << endl;
    cout << "  14.18b (float only, 200M iters):         " << ms_b << " ms" << endl;
    cout << "  14.18c (double only, 200M iters):        " << ms_c << " ms" << endl;
}

// =============================================================================
// Section 14.19: Inline assembly for lrint (double -> int rounding)
// =============================================================================

// GCC inline assembly version of lrint
// Uses x87 fldl/fistpl to round double to int
static inline int lrint_asm(double const x) {
    int n;
    __asm__ __volatile__(
        "fldl %1  \n\t"
        "fistpl %0 \n\t"
        : "=m"(n)
        : "m"(x)
        : "memory");
    return n;
}

void demo_14_19() {
    cout << "=== Section 14.19: Inline assembly for lrint ===" << endl;
    double test_vals[] = {0.0, 1.3, 2.6, -1.3, -2.6, 3.5, -3.5};
    for (double x : test_vals) {
        int result = lrint_asm(x);
        cout << "  lrint_asm(" << x << ") = " << result << "  (std::lrint = " << std::lrint(x)
             << ")" << endl;
    }
}

// =============================================================================
// Section 14.21: SSE2 fast lrint/lrintf
// =============================================================================

// Use SSE2 intrinsics for fast rounding conversion
static inline int lrintf_sse(float const x) {
    return _mm_cvtss_si32(_mm_load_ss(&x));
}

static inline int lrint_sse(double const x) {
    return _mm_cvtsd_si32(_mm_load_sd(&x));
}

void demo_14_21() {
    cout << "=== Section 14.21: SSE2 fast lrint/lrintf ===" << endl;

    float test_f[] = {0.0f, 1.3f, 2.6f, -1.3f, -2.6f, 3.5f, -3.5f};
    for (float x : test_f) {
        cout << "  lrintf_sse(" << x << ") = " << lrintf_sse(x)
             << "  (std::lrintf = " << std::lrint(x) << ")" << endl;
    }

    double test_d[] = {0.0, 1.3, 2.6, -1.3, -2.6, 3.5, -3.5};
    for (double x : test_d) {
        cout << "  lrint_sse(" << x << ") = " << lrint_sse(x) << "  (std::lrint = " << std::lrint(x)
             << ")" << endl;
    }

    // Timing comparison for float lrint
    const int N = 100000000;
    volatile int sink = 0;
    float xf = 3.14159f;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink = lrintf_sse(xf);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_sse = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink = (int)std::lrint(xf);
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_std = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  SSE lrintf (100M iters): " << ms_sse << " ms" << endl;
    cout << "  std lrint   (100M iters): " << ms_std << " ms" << endl;
}

// =============================================================================
// Section 14.22: unsigned int -> double conversion optimization
// =============================================================================

// 14.22a: Direct unsigned int to double (slower on some architectures)
double uitod_direct(unsigned int u) {
    double d;
    d = u;
    return d;
}

// 14.22b: Cast to signed int first, then to double (faster on some CPUs)
double uitod_via_signed(unsigned int u) {
    double d;
    d = (double)(signed int)u;  // Only correct when u <= INT_MAX
    return d;
}

void demo_14_22() {
    cout << "=== Section 14.22: unsigned int -> double conversion ===" << endl;

    unsigned int test_vals[] = {0, 1, 100, 1000000, 2147483647u};
    for (unsigned int u : test_vals) {
        cout << "  u=" << u << " -> direct=" << uitod_direct(u)
             << " via_signed=" << uitod_via_signed(u) << endl;
    }

    // Timing
    const int N = 100000000;
    volatile double sink = 0;
    unsigned int u = 123456789;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink += uitod_direct(u);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink += uitod_via_signed(u);
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  14.22a (direct uint->double, 100M iters): " << ms_a << " ms" << endl;
    cout << "  14.22b (via signed, 100M iters):          " << ms_b << " ms" << endl;
}

// =============================================================================
// Section 14.23: Union for float sign bit manipulation
// =============================================================================

// 14.23: Flip sign bit of float using union
float flip_float_sign(float x) {
    union {
        float f;
        int i;
    } u;
    u.f = x;
    u.i ^= 0x80000000;  // flip sign bit
    return u.f;
}

// 14.23b: Test sign bit of double using union
bool is_double_negative(double x) {
    union {
        double d;
        int i[2];
    } u;
    u.d = x;
    // On little-endian, i[1] is the high 32 bits containing the sign bit
    return u.i[1] < 0;
}

void demo_14_23() {
    cout << "=== Section 14.23: Union for float sign bit manipulation ===" << endl;

    float fv = 3.14f;
    cout << "  flip_float_sign(" << fv << ") = " << flip_float_sign(fv) << endl;
    cout << "  flip_float_sign(" << -fv << ") = " << flip_float_sign(-fv) << endl;

    double dv1 = -3.14, dv2 = 3.14;
    cout << "  is_double_negative(" << dv1 << ") = " << (is_double_negative(dv1) ? "true" : "false")
         << endl;
    cout << "  is_double_negative(" << dv2 << ") = " << (is_double_negative(dv2) ? "true" : "false")
         << endl;
}

// =============================================================================
// Section 14.24: Float absolute value (clear sign bit)
// =============================================================================

float fast_abs_float(float x) {
    union {
        float f;
        int i;
    } u;
    u.f = x;
    u.i &= 0x7FFFFFFF;  // clear sign bit
    return u.f;
}

void demo_14_24() {
    cout << "=== Section 14.24: Float absolute value (clear sign bit) ===" << endl;
    float test_vals[] = {3.14f, -3.14f, 0.0f, -0.0f, -100.5f};
    for (float x : test_vals) {
        cout << "  fast_abs(" << x << ") = " << fast_abs_float(x) << "  (fabsf = " << fabsf(x)
             << ")" << endl;
    }

    // Timing
    const int N = 200000000;
    volatile float sink = 0;
    float x = -3.14159f;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink += fast_abs_float(x);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sink += fabsf(x);
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  fast_abs (bitwise, 200M iters): " << ms_a << " ms" << endl;
    cout << "  fabsf   (library, 200M iters):  " << ms_b << " ms" << endl;
}

// =============================================================================
// Section 14.25: Float zero detection (check bits)
// =============================================================================

bool fast_is_zero_float(float x) {
    union {
        float f;
        int i;
    } u;
    u.f = x;
    // Check bits 0-30 (exponent + mantissa); if all zero, value is +/-0
    return (u.i & 0x7FFFFFFF) == 0;
}

bool fast_is_nonzero_float(float x) {
    union {
        float f;
        int i;
    } u;
    u.f = x;
    return (u.i & 0x7FFFFFFF) != 0;
}

void demo_14_25() {
    cout << "=== Section 14.25: Float zero detection (check bits) ===" << endl;
    float test_vals[] = {0.0f, -0.0f, 0.001f, -0.001f, 3.14f, -3.14f};
    for (float x : test_vals) {
        cout << "  fast_is_zero(" << x << ") = " << (fast_is_zero_float(x) ? "true" : "false")
             << "  (x==0.0f = " << (x == 0.0f ? "true" : "false") << ")" << endl;
    }
}

// =============================================================================
// Section 14.26: Float multiply by power of 2 (manipulate exponent)
// =============================================================================

// Multiply float by 2^n by adding n to the exponent field
float fast_mul_pow2(float x, int n) {
    union {
        float f;
        int i;
    } u;
    u.f = x;
    // Only works if x is nonzero (normal or subnormal)
    if (u.i & 0x7FFFFFFF) {
        u.i += n << 23;  // add n to exponent bits
    }
    return u.f;
}

void demo_14_26() {
    cout << "=== Section 14.26: Float multiply by power of 2 (manipulate exponent) ===" << endl;
    float base = 1.5f;
    for (int n = -3; n <= 3; n++) {
        float fast = fast_mul_pow2(base, n);
        float ref = base * powf(2.0f, (float)n);
        cout << "  " << base << " * 2^" << n << " = " << fast << "  (ref = " << ref << ")" << endl;
    }
}

// =============================================================================
// Section 14.27: Compare positive floats as integers
// =============================================================================

// Compare two positive floats by comparing their bit patterns as integers
bool pos_float_greater(float a, float b) {
    union {
        float f;
        int i;
    } u, v;
    u.f = a;
    v.f = b;
    // Only valid for positive floats (both >= 0)
    return u.i > v.i;
}

void demo_14_27() {
    cout << "=== Section 14.27: Compare positive floats as integers ===" << endl;
    float a = 3.14f, b = 2.71f, c = 3.14f;
    cout << "  pos_float_greater(3.14, 2.71) = " << (pos_float_greater(a, b) ? "true" : "false")
         << endl;
    cout << "  pos_float_greater(2.71, 3.14) = " << (pos_float_greater(b, a) ? "true" : "false")
         << endl;
    cout << "  pos_float_greater(3.14, 3.14) = " << (pos_float_greater(a, c) ? "true" : "false")
         << endl;
}

// =============================================================================
// Section 14.28: Float absolute value comparison
// =============================================================================

// Compare absolute values of two floats
// Shift left by 1 (i*2) effectively shifts out the sign bit
bool abs_float_greater(float a, float b) {
    union {
        float f;
        unsigned int i;
    } u, v;
    u.f = a;
    v.f = b;
    // Multiply by 2 to shift out sign bit; compare unsigned
    return u.i * 2 > v.i * 2;
}

void demo_14_28() {
    cout << "=== Section 14.28: Float absolute value comparison ===" << endl;

    struct {
        float a;
        float b;
    } tests[] = {{3.5f, 2.5f}, {-3.5f, 2.5f}, {1.0f, -5.0f}, {-1.0f, -1.0f}};

    for (auto t : tests) {
        cout << "  abs(" << t.a << ") > abs(" << t.b
             << ") = " << (abs_float_greater(t.a, t.b) ? "true" : "false")
             << "  (ref = " << (fabsf(t.a) > fabsf(t.b) ? "true" : "false") << ")" << endl;
    }
}

// =============================================================================
// Section 14.29: Construct float from integer bit pattern
// =============================================================================

// Construct a float in range [1.0, 2.0) from lower 23 bits of n
float make_float_1_to_2(int n) {
    union {
        float f;
        int i;
    } u;
    // Lower 23 bits go into mantissa; exponent = 127 -> bias gives 2^0 = 1
    u.i = (n & 0x7FFFFF) | 0x3F800000;
    return u.f;
}

void demo_14_29() {
    cout << "=== Section 14.29: Construct float from integer bit pattern ===" << endl;
    for (int n = 0; n <= 5; n++) {
        float f = make_float_1_to_2(n);
        cout << "  n=" << n << " -> f=" << f << endl;
    }

    // Verify range: should be in [1.0, 2.0)
    float min_f = make_float_1_to_2(0);
    float max_f = make_float_1_to_2(0x7FFFFF);
    cout << "  range: [" << min_f << ", " << max_f << ")" << endl;
}

// =============================================================================
// Section 14.30: Fast max absolute value in double array
// =============================================================================

// Union type for double/int aliasing (used in Section 14.30)
union DoubleAlias {
    double d;
    unsigned int u[2];
};

// Find the element with largest absolute value in a double array
// Uses integer comparison on the upper 32 bits for speed
int find_largest_abs_double(const double* array, int size) {
    const DoubleAlias* a = reinterpret_cast<const DoubleAlias*>(array);

    unsigned int absvalue, largest_abs = 0;
    int largest_index = 0;

    for (int i = 0; i < size; i++) {
        // Get upper 32 bits and shift out sign bit (multiply by 2)
        absvalue = a[i].u[1] * 2;
        if (absvalue > largest_abs) {
            largest_abs = absvalue;
            largest_index = i;
        }
    }
    return largest_index;
}

// Reference: find max abs using standard library
int find_largest_abs_double_ref(const double* array, int size) {
    int idx = 0;
    double max_abs = fabs(array[0]);
    for (int i = 1; i < size; i++) {
        double abs_val = fabs(array[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
            idx = i;
        }
    }
    return idx;
}

void demo_14_30() {
    cout << "=== Section 14.30: Fast max absolute value in double array ===" << endl;

    const int size = 100;
    double arr[size];
    for (int i = 0; i < size; i++) {
        arr[i] = sin((double)i * 0.5) * 100.0;
    }
    // Make one element clearly the largest in abs value
    arr[37] = -999.9;

    int idx_fast = find_largest_abs_double(arr, size);
    int idx_ref = find_largest_abs_double_ref(arr, size);

    cout << "  Fast method: index=" << idx_fast << " value=" << arr[idx_fast] << endl;
    cout << "  Reference:   index=" << idx_ref << " value=" << arr[idx_ref] << endl;
    cout << "  Match: " << (idx_fast == idx_ref ? "yes" : "NO") << endl;

    // Timing
    const int N = 1000000;
    volatile int sink = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < N; iter++) {
        sink += find_largest_abs_double(arr, size);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < N; iter++) {
        sink += find_largest_abs_double_ref(arr, size);
    }
    t2 = std::chrono::high_resolution_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    cout << "  Fast method (1M iters): " << ms_a << " ms" << endl;
    cout << "  Reference   (1M iters): " << ms_b << " ms" << endl;
}

// =============================================================================
// main: Run all demonstrations
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "  Chapter 14: \"具体的优化主题\" (Specific Optimization Topics)" << endl;
    cout << "  Consolidated examples from example_14_1a through 14_30" << endl;
    cout << "============================================================" << endl;
    cout << endl;

    demo_14_1();
    cout << endl;

    demo_14_2();
    cout << endl;

    demo_14_3();
    cout << endl;

    demo_14_4();
    cout << endl;

    demo_14_5();
    cout << endl;

    demo_14_6();
    cout << endl;

    demo_14_7();
    cout << endl;

    demo_14_8();
    cout << endl;

    demo_14_9();
    cout << endl;

    demo_14_10();
    cout << endl;

    demo_14_11();
    cout << endl;

    demo_14_12();
    cout << endl;

    demo_14_13();
    cout << endl;

    demo_14_14();
    cout << endl;

    demo_14_15();
    cout << endl;

    demo_14_16();
    cout << endl;

    demo_14_17();
    cout << endl;

    demo_14_18();
    cout << endl;

    demo_14_19();
    cout << endl;

    demo_14_21();
    cout << endl;

    demo_14_22();
    cout << endl;

    demo_14_23();
    cout << endl;

    demo_14_24();
    cout << endl;

    demo_14_25();
    cout << endl;

    demo_14_26();
    cout << endl;

    demo_14_27();
    cout << endl;

    demo_14_28();
    cout << endl;

    demo_14_29();
    cout << endl;

    demo_14_30();
    cout << endl;

    cout << "Done." << endl;
    return 0;
}
