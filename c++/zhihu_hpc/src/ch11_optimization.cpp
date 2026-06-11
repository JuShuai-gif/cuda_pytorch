#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace std::chrono;

const int DATA_SIZE = 10000000;

template <typename F>
double measure_time(const char* label, int iterations, F fn) {
    auto start = high_resolution_clock::now();
    for (int k = 0; k < iterations; ++k) {
        fn();
    }
    auto end = high_resolution_clock::now();
    double elapsed = duration_cast<nanoseconds>(end - start).count() / 1e6 / iterations;
    cout << "  " << left << setw(40) << label << ": " << fixed << setprecision(3) << elapsed
         << " ms (avg over " << iterations << " runs)" << endl;
    return elapsed;
}

// --- Example 11.1a: Sequential addition (a + b + c + d) ---
// Chained additions create a dependency chain; each ADD must wait for the previous one.
float example_11_1a(float a, float b, float c, float d) {
    float y;
    y = a + b + c + d;
    return y;
}

// --- Example 11.1b: Parenthesized addition ((a + b) + (c + d)) ---
// Grouping allows the CPU to execute (a+b) and (c+d) in parallel before combining.
float example_11_1b(float a, float b, float c, float d) {
    float y;
    y = (a + b) + (c + d);
    return y;
}

// --- Example 11.2a: Simple reduction loop (loop-carried dependency) ---
// sum is updated each iteration, creating a dependency that prevents ILP.
float example_11_2a(const vector<float>& list) {
    float sum = 0.0f;
    for (size_t i = 0; i < list.size(); ++i) {
        sum += list[i];
    }
    return sum;
}

// --- Example 11.2b: Two-accumulator reduction (breaks dependency) ---
// Using two independent accumulators allows the CPU to overlap additions.
float example_11_2b(const vector<float>& list) {
    size_t size = list.size();
    float sum1 = 0.0f, sum2 = 0.0f;
    for (size_t i = 0; i < size; i += 2) {
        sum1 += list[i];
        sum2 += list[i + 1];
    }
    return sum1 + sum2;
}

// --- Example 11.3: Compute (a[i] + b[i])^2 ---
// The temp variable decouples the add from the multiply, exposing ILP.
void example_11_3(const vector<float>& a, const vector<float>& b, vector<float>& c) {
    size_t size = a.size();
    for (size_t i = 0; i < size; ++i) {
        float temp = a[i] + b[i];
        c[i] = temp * temp;
    }
}

// --- Variant of 11.3 without temp (combined expression) for comparison ---
void example_11_3_no_temp(const vector<float>& a, const vector<float>& b, vector<float>& c) {
    size_t size = a.size();
    for (size_t i = 0; i < size; ++i) {
        c[i] = (a[i] + b[i]) * (a[i] + b[i]);
    }
}

// --- 4-accumulator variant of 11.2 for comparison ---
float example_11_2c_4acc(const vector<float>& list) {
    size_t size = list.size();
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    for (size_t i = 0; i < size; i += 4) {
        sum1 += list[i];
        sum2 += list[i + 1];
        sum3 += list[i + 2];
        sum4 += list[i + 3];
    }
    return (sum1 + sum2) + (sum3 + sum4);
}

int main() {
    cout << "=== Chapter 11: Out-of-Order Execution - Performance Comparison ===" << endl;
    cout << "Data size: " << DATA_SIZE << " elements" << endl << endl;

    // --- Setup shared test data ---
    const int ITER_SCALAR = 200000000;
    const int ITER_ARRAY = 20;

    float fa = 1.0f, fb = 2.0f, fc = 3.0f, fd = 4.0f;
    vector<float> listA(DATA_SIZE);
    vector<float> listB(DATA_SIZE);
    vector<float> listC(DATA_SIZE);

    for (int i = 0; i < DATA_SIZE; ++i) {
        listA[i] = static_cast<float>(i % 100) * 0.01f;
        listB[i] = static_cast<float>((i + 37) % 100) * 0.01f;
    }

    // --- Section 1: Scalar addition (11.1) ---
    cout << "--- 11.1: Scalar Addition (chained vs. parenthesized) ---" << endl;

    volatile float sink;  // Prevent compiler from optimizing away the computation

    measure_time("11.1a y=a+b+c+d (chained)", ITER_SCALAR,
                 [&]() { sink = example_11_1a(fa, fb, fc, fd); });

    measure_time("11.1b y=(a+b)+(c+d) (parallel)", ITER_SCALAR,
                 [&]() { sink = example_11_1b(fa, fb, fc, fd); });

    (void)sink;

    cout << endl;

    // --- Section 2: Reduction loop (11.2) ---
    cout << "--- 11.2: Array Reduction (loop-carried vs. multi-accumulator) ---" << endl;

    volatile float result;

    measure_time("11.2a single accumulator (chained)", ITER_ARRAY,
                 [&]() { result = example_11_2a(listA); });

    measure_time("11.2b two accumulators", ITER_ARRAY, [&]() { result = example_11_2b(listA); });

    measure_time("11.2c four accumulators", ITER_ARRAY,
                 [&]() { result = example_11_2c_4acc(listA); });

    (void)result;

    cout << endl;

    // --- Section 3: Element-wise computation (11.3) ---
    cout << "--- 11.3: Element-wise Compute (with/without temp variable) ---" << endl;

    measure_time("11.3 with temp variable (ILP)", ITER_ARRAY,
                 [&]() { example_11_3(listA, listB, listC); });

    measure_time("11.3 without temp (combined)", ITER_ARRAY,
                 [&]() { example_11_3_no_temp(listA, listB, listC); });

    cout << endl;

    // --- Sanity check: verify results are correct ---
    cout << "--- Correctness Checks ---" << endl;

    // 11.1: both should produce 10.0
    float r1a = example_11_1a(1.0f, 2.0f, 3.0f, 4.0f);
    float r1b = example_11_1b(1.0f, 2.0f, 3.0f, 4.0f);
    cout << "  11.1a result: " << r1a << ", 11.1b result: " << r1b
         << " (match: " << (abs(r1a - r1b) < 0.0001f ? "YES" : "NO") << ")" << endl;

    // 11.2: all reduction methods should give the same sum
    float r2a = example_11_2a(listA);
    float r2b = example_11_2b(listA);
    float r2c = example_11_2c_4acc(listA);
    cout << "  11.2a sum: " << r2a << ", 11.2b sum: " << r2b << ", 11.2c sum: " << r2c
         << " (close: " << (abs(r2a - r2b) < 0.01f && abs(r2a - r2c) < 0.01f ? "YES" : "NO") << ")"
         << endl;

    // 11.3: verify first few elements
    vector<float> c1(DATA_SIZE), c2(DATA_SIZE);
    example_11_3(listA, listB, c1);
    example_11_3_no_temp(listA, listB, c2);
    bool match = true;
    for (int i = 0; i < 10 && match; ++i) {
        if (abs(c1[i] - c2[i]) > 0.0001f)
            match = false;
    }
    cout << "  11.3 first 10 elements match: " << (match ? "YES" : "NO") << endl;

    return 0;
}
