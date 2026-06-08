// Chapter 8: Compiler Optimizations
// Demonstrates how compilers optimize C++ code and how to help them.
// Compile: see CMakeLists.txt (ch08_demo target)

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>

// ---- Example 8.1a: Function inlining ----
static inline float Square(float a) {
    return a * a;
}

float Parabola(float x) {
    return Square(x) + 1.0f;
}

// ---- Example 8.4: Constant folding ----
double ConstantFoldingDemo() {
    double a = std::sin(0.8); // sin(0.8) computed at compile time by some compilers
    return a;
}

// ---- Example 8.5a: Common subexpression elimination ----
void CommonSubexprDemo(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i) {
        // The compiler may eliminate the common subexpression
        out[i] = a[i] * b[i] + a[i] * b[i];
    }
}

// ---- Example 8.7: Loop-invariant code motion ----
double LoopInvariantDemo(const double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        // sqrt(2.0) is loop-invariant; compiler moves it outside
        sum += x[i] * std::sqrt(2.0);
    }
    return sum;
}

// ---- Example 8.13a: Induction variables ----
void InductionVarDemo(const double *a, double b, double *out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] * b;
    }
}

// ---- Example 8.18: Pointer aliasing barrier ----
void NoAliasDemo(const double *__restrict a,
                 const double *__restrict b,
                 double *__restrict c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// ---- Example 8.23b: Recurrence using induction variables ----
// Evaluate polynomial: y = sum( a[i] * x^i ) using Horner-like method
double PolynomialEval(const double *a, int n, double x) {
    double y = 0.0;
    double xn = 1.0;
    for (int i = 0; i < n; ++i) {
        y += a[i] * xn;
        xn *= x;
    }
    return y;
}

// ---- Example 8.24: const helps optimization ----
int ArraySum(const int *arr, int n) {
    const int ArraySize = n;
    int sum = 0;
    for (int i = 0; i < ArraySize; ++i) {
        sum += arr[i];
    }
    return sum;
}

// ---- Main ----
int main() {
    std::cout << "=== Chapter 8: Compiler Optimizations ===\n\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Parabola(3.0): " << Parabola(3.0f) << "\n";
    std::cout << "Constant folding sin(0.8): " << ConstantFoldingDemo() << "\n";

    double a_arr[] = {1.0, 2.0, 3.0, 4.0};
    double b_arr[] = {5.0, 6.0, 7.0, 8.0};
    double c_arr[4] = {};

    CommonSubexprDemo(a_arr, b_arr, c_arr, 4);
    std::cout << "CSE result[0]: " << c_arr[0] << "\n";

    NoAliasDemo(a_arr, b_arr, c_arr, 4);
    std::cout << "NoAlias result[0]: " << c_arr[0] << "\n";

    double poly_coeff[] = {1.0, 2.0, 3.0}; // 1 + 2x + 3x^2
    std::cout << "Poly eval x=2.0: " << PolynomialEval(poly_coeff, 3, 2.0) << "\n";

    int nums[] = {1, 2, 3, 4, 5};
    std::cout << "ArraySum: " << ArraySum(nums, 5) << "\n";

    std::cout << "\nAll chapter 8 checks passed.\n";
    return 0;
}
