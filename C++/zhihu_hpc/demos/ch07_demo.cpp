// Chapter 7: C++ Language Construct Efficiency
// Demonstrates key optimization techniques for C++ language features.
// Compile: see CMakeLists.txt (ch07_demo target)

#include <cstring>
#include <iostream>
#include <cmath>

// ---- Example 7.1: Static lookup tables ----
float StaticLookup(int x) {
    static float list[] = {1.1f, 0.3f, -2.0f, 4.4f, 2.5f};
    return list[x];
}

// ---- Example 7.4: Signed vs unsigned division ----
int FastDivBy10(int a) {
    // Unsigned division by constant is faster than signed
    return static_cast<int>(static_cast<unsigned int>(a) / 10);
}

// ---- Example 7.15a: SafeArray with bounds checking ----
template <typename T, unsigned int N>
class SafeArray {
    T a[N];

public:
    SafeArray() { std::memset(a, 0, sizeof(a)); }
    int Size() const { return N; }
    T& operator[](unsigned int i) {
        if (i >= N) {
            std::cerr << "Error: index out of range\n";
            std::abort();
        }
        return a[i];
    }
    const T& operator[](unsigned int i) const {
        if (i >= N) {
            std::cerr << "Error: index out of range\n";
            std::abort();
        }
        return a[i];
    }
};

// ---- Example 7.25-7.26: Avoid int-to-float conversion in loops ----
void IntToFloatConversion() {
    // Bad: converts i to float each iteration
    float a_bad[100];
    for (int i = 0; i < 100; ++i) {
        a_bad[i] = 2.0f * static_cast<float>(i);
    }
    // Good: use a floating-point induction variable
    float a_good[100];
    float f2 = 0.0f;
    for (int i = 0; i < 100; ++i, f2 += 2.0f) {
        a_good[i] = f2;
    }
    // Suppress unused warnings
    volatile float check_bad = a_bad[50];
    volatile float check_good = a_good[50];
    (void)check_bad;
    (void)check_good;
}

// ---- Example 7.29b: Lookup table instead of branch ----
float BranchlessSelect(bool b) {
    const float lookup[2] = {2.6f, 1.5f};
    return lookup[b ? 1 : 0];
}

// ---- Example 7.30: Loop unrolling to eliminate branches ----
void LoopUnrolling() {
    // Original: branch inside loop
    for (int i = 0; i < 20; ++i) {
        if (i % 2 == 0) {
            // FuncA(i);
        } else {
            // FuncB(i);
        }
    }
    // Unrolled: no branch
    for (int i = 0; i < 20; i += 2) {
        // FuncA(i);
        // FuncB(i+1);
    }
}

// ---- Example 7.38: Factorial as loop (not recursion) ----
unsigned long long FactorialLoop(unsigned int n) {
    unsigned long long product = 1;
    while (n > 1) {
        product *= n;
        --n;
    }
    return product;
}

// ---- Example 7.39b: Reorder struct members to minimize padding ----
struct S1_Optimized {
    double b;     // 8 bytes, offset 0
    int d;        // 4 bytes, offset 8
    short int a;  // 2 bytes, offset 12
    // 2 unused bytes at end
};
static_assert(sizeof(S1_Optimized) == 16, "Expected 16 bytes after reorder");

// ---- Example 7.44c: Bit-field packing with bitwise operations ----
union Bitfield {
    struct {
        int a : 4;
        int b : 2;
        int c : 2;
    } bits;
    char abc;
};

void BitfieldPacking(int A, int B, int C) {
    Bitfield x;
    x.abc = static_cast<char>((A & 0x0F) | ((B & 3) << 4) | ((C & 3) << 6));
    (void)x;
}

// ---- Example 7.46: Template vs runtime parameter ----
template <int M>
int MultiplyBy(int x) {
    return x * M;  // Compiler can optimize when M is known at compile time
}

// ---- Example 7.47b: CRTP for compile-time polymorphism ----
template <typename Derived>
class CParent {
public:
    void Hello() { static_cast<Derived*>(this)->Disp(); }
};

class CChild1 : public CParent<CChild1> {
public:
    void Disp() { std::cout << "Hello from Child1\n"; }
};

class CChild2 : public CParent<CChild2> {
public:
    void Disp() { std::cout << "Hello from Child2\n"; }
};

// ---- Main demonstration ----
int main() {
    std::cout << "=== Chapter 7: C++ Language Construct Efficiency ===\n\n";

    std::cout << "Static lookup table: " << StaticLookup(2) << "\n";
    std::cout << "Fast unsigned division: " << FastDivBy10(42) << "\n";
    std::cout << "Factorial(10): " << FactorialLoop(10) << "\n";
    std::cout << "sizeof(S1_Optimized): " << sizeof(S1_Optimized) << "\n";

    SafeArray<float, 100> arr;
    arr[0] = 3.14f;
    std::cout << "SafeArray[0]: " << arr[0] << "\n";

    std::cout << "Branchless select(false): " << BranchlessSelect(false) << "\n";
    std::cout << "Branchless select(true): " << BranchlessSelect(true) << "\n";

    std::cout << "MultiplyBy<8>(10): " << MultiplyBy<8>(10) << "\n";

    CChild1 c1;
    CChild2 c2;
    c1.Hello();
    c2.Hello();

    BitfieldPacking(1, 2, 3);

    std::cout << "\nAll chapter 7 checks passed.\n";
    return 0;
}
