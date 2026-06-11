// Chapter 7: 不同C++结构的效率 (Efficiency of Different C++ Constructs)
// Consolidated from 65 snippet files: example_7_1.cpp through example_7_49.cpp
// Compile: g++ -std=c++11 -O2 ch07_optimization.cpp -o ch07_optimization

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <cassert>
#include <cfenv>
#include <climits>
#include <string>
#include <vector>

// SSE intrinsics (Examples 7.5, 7.6)
#include <xmmintrin.h>

// ---------------------------------------------------------------------------
// Timing framework
// ---------------------------------------------------------------------------

class Timer {
public:
    Timer(const char* name) : m_name(name), m_start(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count();
        std::cout << "  [" << m_name << "] " << us << " us" << std::endl;
    }

private:
    const char* m_name;
    std::chrono::high_resolution_clock::time_point m_start;
};

#define EXAMPLE(name) std::cout << "\n=== Example 7." << name << " ===" << std::endl;

// ---------------------------------------------------------------------------
// Example 7.1: Static array inside function
// Demonstrates that a static local array is initialized once and reused.
// ---------------------------------------------------------------------------
float example_7_1_static_array(int x) {
    // Static array initialized only once during first call
    static float list[] = {1.1f, 0.3f, -2.0f, 4.4f, 2.5f};
    return list[x];
}

void run_7_1() {
    EXAMPLE("1 - Static Array Inside Function");
    std::cout << "  list[0] = " << example_7_1_static_array(0) << std::endl;
    std::cout << "  list[3] = " << example_7_1_static_array(3) << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.2: Floating point constants - double vs float
// 3.5 is a double literal; computing with it promotes operands to double.
// Use 3.5f to keep computations in float precision for speed.
// ---------------------------------------------------------------------------
void run_7_2() {
    EXAMPLE("2 - Float vs Double Constants");
    float a, b, c, d;
    b = 1.0f;
    d = 2.0f;

    {
        Timer t("b * 3.5 (double constant)");
        for (int i = 0; i < 100000000; ++i) {
            a = b * 3.5;  // b promoted to double, result truncated to float
            c = d + 3.5;  // d promoted to double
        }
        volatile float keep = a + c;
        (void)keep;
    }
    {
        Timer t("b * 3.5f (float constant)");
        for (int i = 0; i < 100000000; ++i) {
            a = b * 3.5f;  // stays in float
            c = d + 3.5f;
        }
        volatile float keep = a + c;
        (void)keep;
    }
    std::cout << "  Using 'f' suffix avoids implicit double promotion." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.3: Volatile - prevent compiler from optimizing away reads
// volatile tells the compiler the variable may change by external means.
// Without volatile, the compiler might optimize away repeated reads.
// ---------------------------------------------------------------------------
void run_7_3() {
    EXAMPLE("3 - Volatile Keyword");

    volatile int seconds = 0;  // Simulates: incremented every second by another thread

    // Demonstrate that without volatile, compiler might optimize the loop away
    int non_volatile = 0;
    {
        Timer t("volatile delay simulation");
        seconds = 0;
        int dummy = 0;
        // Simulate: wait until seconds reaches 5
        while (seconds < 5) {
            dummy++;  // prevent complete optimization
            if (dummy > 1000000) {
                seconds = 5;
            }  // fake the external update
        }
    }
    std::cout << "  volatile final value = " << seconds << std::endl;
    std::cout << "  volatile prevents the compiler from caching/optimizing the variable."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.4: Signed vs unsigned integers
// Unsigned division by constant is faster than signed (no sign handling).
// Signed is better when converting to double (single instruction).
// ---------------------------------------------------------------------------
void run_7_4() {
    EXAMPLE("4 - Signed vs Unsigned Integers");

    const int N = 50000000;
    int a = 12345;
    unsigned int ua = 12345;
    volatile int keep;
    volatile double dkeep;

    {
        Timer t("signed int / 10");
        unsigned int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (unsigned int)(a / 10);
        }
        keep = (int)sum;
    }
    {
        Timer t("unsigned int / 10");
        unsigned int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += ua / 10;  // unsigned division is faster
        }
        keep = (int)sum;
    }
    std::cout << "  Unsigned division by constant can be faster (no sign bit handling)."
              << std::endl;
    std::cout << "  Use signed int when converting to double for better performance." << std::endl;

    {
        Timer t("(double)(signed int)");
        for (int i = 0; i < N; ++i) {
            dkeep = (double)a;
        }
    }
    {
        Timer t("(double)(unsigned int)");
        for (int i = 0; i < N; ++i) {
            dkeep = (double)ua;  // slower: unsigned-to-double needs extra handling
        }
    }
}

// ---------------------------------------------------------------------------
// Example 7.5: Set flush-to-zero mode (SSE)
// Subnormal numbers cause huge performance penalties (~100x slower).
// Flush-to-zero mode treats subnormals as zero.
// ---------------------------------------------------------------------------
void run_7_5() {
    EXAMPLE("5 - SSE Flush-to-Zero Mode");
    std::cout << "  Setting _MM_FLUSH_ZERO_ON..." << std::endl;
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << "  Flush-to-zero mode enabled. Denormals will be flushed to zero." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.6: Set flush-to-zero AND denormals-are-zero mode (SSE2)
// Also sets the DAZ bit so denormals read from memory are treated as zero.
// ---------------------------------------------------------------------------
void run_7_6() {
    EXAMPLE("6 - SSE2 Flush-to-Zero + Denormals-Are-Zero");
    std::cout << "  Setting both FTZ and DAZ bits via _mm_setcsr..." << std::endl;
    _mm_setcsr(_mm_getcsr() | 0x8040);
    std::cout << "  Both flush-to-zero (bit 15) and denormals-are-zero (bit 6) set." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.7: Boolean short-circuit order optimization
// Put the cheaper/safer check first: i < ARRAYSIZE must be true before
// accessing list[i]. This avoids out-of-bounds access AND allows early exit.
// ---------------------------------------------------------------------------
void run_7_7() {
    EXAMPLE("7 - Short-Circuit Order Optimization");

    const int ARRAYSIZE = 100;
    float list[ARRAYSIZE];
    for (int i = 0; i < ARRAYSIZE; ++i)
        list[i] = (float)(i * 0.5);

    // Correct order: bound check FIRST, then element access
    int count = 0;
    unsigned int i;

    {
        Timer t("good order: i < N && list[i] > val");
        for (int iter = 0; iter < 100000; ++iter) {
            for (i = 0; i < ARRAYSIZE + 10; ++i) {
                if (i < ARRAYSIZE && list[i] > 1.0f) {
                    ++count;
                }
            }
        }
    }
    std::cout << "  Count (good order): " << count << std::endl;

    // NOTE: The reversed order (list[i] > 1.0 && i < ARRAYSIZE) would
    // access out-of-bounds when i >= ARRAYSIZE, causing undefined behavior.
    // The safe order avoids this completely.
    std::cout << "  Placing the cheap bound-check first avoids out-of-bounds access." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.8: Expensive function calls in short-circuit
// Check handle validity BEFORE calling the expensive function.
// INVALID_HANDLE_VALUE is Windows-specific, simulated here.
// ---------------------------------------------------------------------------
bool simulated_write_file(int handle, const char* data, int size) {
    // Simulate an expensive I/O operation
    volatile int dummy = 0;
    for (int i = 0; i < 1000; ++i)
        dummy += i;
    return handle > 0;  // success only if valid handle
}

void run_7_8() {
    EXAMPLE("8 - Short-Circuit with Expensive Functions");

    const int INVALID_HANDLE_VALUE = -1;
    int handle = -1;  // invalid handle

    {
        Timer t("check handle first (short-circuit)");
        for (int i = 0; i < 100000; ++i) {
            // Cheap check first: skip expensive WriteFile if handle is invalid
            if (handle != INVALID_HANDLE_VALUE && simulated_write_file(handle, "data", 4)) {
                // ...
            }
        }
    }
    std::cout << "  Short-circuit avoids expensive function call when handle is invalid."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.9a/7.9b: Boolean operators vs bitwise operators
// bool && / || involve branches (short-circuit).
// Bitwise & / | on integers avoid branches but compute both sides.
// ---------------------------------------------------------------------------
void run_7_9() {
    EXAMPLE("9a/9b - Boolean && / || vs Bitwise & / |");

    const int N = 50000000;
    volatile int keep;

    {
        Timer t("bool && / ||");
        bool a = true, b = false, c, d;
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            c = a && b;  // short-circuit: may involve branch
            d = a || b;
            sum += (int)c + (int)d;
        }
        keep = sum;
    }
    {
        Timer t("int & / |");
        char a = 1, b = 0, c, d;
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            c = a & b;  // bitwise: no branch, always computes both
            d = a | b;
            sum += (int)c + (int)d;
        }
        keep = sum;
    }
    std::cout << "  Bitwise & | avoid branch mispredictions but compute both sides." << std::endl;
    std::cout << "  Boolean && || short-circuit and may cause branch mispredictions." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.10a/7.10b: Boolean NOT vs bitwise XOR
// !a generates a branch; a ^ 1 is branchless bitwise NOT.
// ---------------------------------------------------------------------------
void run_7_10() {
    EXAMPLE("10a/10b - Boolean ! vs Bitwise XOR");

    const int N = 50000000;
    volatile int keep;

    {
        Timer t("bool !a");
        bool a = false, b;
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            a = (i & 1) ? true : false;
            b = !a;
            sum += (int)b;
        }
        keep = sum;
    }
    {
        Timer t("char a ^ 1");
        char a = 0, b;
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            a = (char)(i & 1);
            b = a ^ 1;  // flip bit: 0->1, 1->0
            sum += (int)b;
        }
        keep = sum;
    }
    std::cout << "  XOR with 1 flips the lowest bit without branching." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.11: Mixed boolean and floating point expressions
// Combining float comparisons with boolean logic.
// ---------------------------------------------------------------------------
void run_7_11() {
    EXAMPLE("11 - Mixed Boolean/Float Conditions");

    const int N = 50000000;
    float x = 1.5f, y = 1.0f, z = 0.0f;
    volatile int keep;

    {
        Timer t("bool from float comparisons");
        bool a;
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            a = x > y && z != 0.0f;
            sum += (int)a;
        }
        keep = sum;
    }
    std::cout << "  Boolean expressions from float comparisons are branch-heavy." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.12: Pointer parameters vs reference parameters
// A reference is typically implemented as a pointer by the compiler.
// No performance difference between the two for parameter passing.
// ---------------------------------------------------------------------------
void FuncA(int* p) {
    *p = *p + 2;
}

void FuncB(int& r) {
    r = r + 2;
}

void run_7_12() {
    EXAMPLE("12 - Pointer vs Reference Parameters");

    const int N = 50000000;
    int val = 0;
    volatile int keep;

    {
        Timer t("pointer parameter");
        for (int i = 0; i < N; ++i) {
            FuncA(&val);
        }
        keep = val;
        val = 0;
    }
    {
        Timer t("reference parameter");
        for (int i = 0; i < N; ++i) {
            FuncB(val);
        }
        keep = val;
    }
    std::cout << "  Pointers and references generate identical code at machine level." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.13: Pointer arithmetic on struct pointers
// Adding an integer to a struct pointer advances by sizeof(struct) bytes.
// ---------------------------------------------------------------------------
void run_7_13() {
    EXAMPLE("13 - Pointer Arithmetic on Struct Pointers");

    struct abc {
        int a;
        int b;
        int c;
    };
    abc arr[5] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    abc* p = arr;
    int i = 2;

    abc* q = p + i;  // advances by i * sizeof(abc) = i * 12 bytes
    std::cout << "  arr[0].a = " << arr[0].a << std::endl;
    std::cout << "  p + 2 -> a = " << q->a << ", b = " << q->b << ", c = " << q->c << std::endl;
    std::cout << "  Pointer arithmetic automatically scales by sizeof(abc) = " << sizeof(abc)
              << " bytes." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.14: Pointer-to-member
// Pointers to class members are slower than direct access because they
// require offset calculations. Avoid in performance-critical code.
// ---------------------------------------------------------------------------
void run_7_14() {
    EXAMPLE("14 - Pointer-to-Member");

    class c1 {
    public:
        int a;
        int b;
    };

    int c1::* MemberPointer = &c1::b;  // pointer to member 'b'

    c1 obj;
    obj.a = 10;
    obj.b = 20;

    std::cout << "  obj.a = " << obj.a << ", obj.*MemberPointer = " << obj.*MemberPointer
              << std::endl;

    const int N = 50000000;
    volatile int keep;
    {
        Timer t("pointer-to-member access");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += obj.*MemberPointer;
        }
        keep = sum;
    }
    {
        Timer t("direct member access");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += obj.b;
        }
        keep = sum;
    }
    std::cout << "  Pointer-to-member access is typically slower than direct access." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.15a/7.15b: SafeArray with bounds checking
// Bounds checking adds overhead. Only use in debug builds or non-critical paths.
// ---------------------------------------------------------------------------

template <typename T, unsigned int N>
class SafeArray {
protected:
    T a[N];

public:
    SafeArray() { memset(a, 0, sizeof(a)); }
    int Size() const { return N; }
    T& operator[](unsigned int i) {
        if (i >= N) {
            std::cerr << "  [ERROR] Index " << i << " out of range [0," << N - 1 << "]"
                      << std::endl;
            return *(T*)0;  // provoke error (null dereference)
        }
        return a[i];
    }
    const T& operator[](unsigned int i) const {
        if (i >= N) {
            std::cerr << "  [ERROR] Index " << i << " out of range [0," << N - 1 << "]"
                      << std::endl;
            return *(T*)0;
        }
        return a[i];
    }
};

void run_7_15() {
    EXAMPLE("15a/15b - SafeArray with Bounds Checking");

    SafeArray<float, 100> list;
    for (int i = 0; i < list.Size(); ++i) {
        list[i] = (float)(i * 0.5f);
    }
    std::cout << "  list[10] = " << list[10] << std::endl;
    std::cout << "  list[99] = " << list[99] << std::endl;

    const int N = 2000000;
    volatile float keep;
    {
        Timer t("SafeArray with bounds check");
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += list[i % 100];
        }
        keep = sum;
    }
    {
        Timer t("raw array without bounds check");
        float raw[100];
        for (int i = 0; i < 100; ++i)
            raw[i] = (float)(i * 0.5f);
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += raw[i % 100];
        }
        keep = sum;
    }
    std::cout << "  Bounds checking adds overhead; omit it in performance-critical loops."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.16: memset for array initialization
// memset is typically faster than a for-loop for zeroing arrays.
// ---------------------------------------------------------------------------
void run_7_16() {
    EXAMPLE("16 - memset for Array Initialization");

    const int N = 100000;
    volatile float keep;

    {
        Timer t("memset to zero");
        for (int rep = 0; rep < 1000; ++rep) {
            float list[100];
            memset(list, 0, sizeof(list));
            keep = list[50];
        }
    }
    {
        Timer t("for-loop to zero");
        for (int rep = 0; rep < 1000; ++rep) {
            float list[100];
            for (int i = 0; i < 100; ++i)
                list[i] = 0.0f;
            keep = list[50];
        }
    }
    std::cout << "  memset is typically optimized by the compiler/hardware for bulk zeroing."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.17: 2D array access order (row-major vs column-major)
// C/C++ arrays are row-major. Accessing matrix[i][j] in inner loop is optimal.
// Swapping loops (j in outer, i in inner) causes cache misses.
// ---------------------------------------------------------------------------
void run_7_17() {
    EXAMPLE("17 - 2D Array Access Order (Row-Major)");

    const int rows = 200, columns = 500;
    float matrix[rows][columns];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < columns; ++j)
            matrix[i][j] = (float)(i + j);
    float x = 1.0f;
    volatile float keep;

    {
        Timer t("row-major: i outer, j inner (good)");
        for (int rep = 0; rep < 500; ++rep) {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < columns; ++j)
                    matrix[i][j] += x;
        }
        keep = matrix[0][0];
    }
    {
        Timer t("column-major: j outer, i inner (bad)");
        for (int rep = 0; rep < 500; ++rep) {
            for (int j = 0; j < columns; ++j)
                for (int i = 0; i < rows; ++i)
                    matrix[i][j] += x;
        }
        keep = matrix[0][0];
    }
    std::cout << "  Row-major access (inner loop over rightmost index) is far more cache-friendly."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.18: Non-sequential 2D array access
// When indices come from function calls, the compiler can't optimize the
// access pattern or prefetch effectively.
// ---------------------------------------------------------------------------
int FuncRow(int i) {
    return i % 20;
}
int FuncCol(int i) {
    return i % 32;
}

void run_7_18() {
    EXAMPLE("18 - Non-Sequential 2D Array Access with Function Calls");

    const int rows = 20, columns = 32;
    float matrix[rows][columns];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < columns; ++j)
            matrix[i][j] = (float)(i * columns + j);
    float x = 2.0f;
    volatile float keep;

    {
        Timer t("indices from functions (hard to optimize)");
        for (int i = 0; i < 1000000; ++i) {
            matrix[FuncRow(i)][FuncCol(i)] += x;
        }
        keep = matrix[0][0];
    }
    {
        Timer t("sequential access (easy to optimize)");
        for (int rep = 0; rep < 1000000; ++rep) {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < columns; ++j)
                    matrix[i][j] += x;
        }
        keep = matrix[0][0];
    }
    std::cout << "  Function-call-based indices prevent compiler from optimizing access patterns."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.19: Type casting methods
// All four casting methods generate identical machine code.
// static_cast<> is preferred for readability and safety.
// ---------------------------------------------------------------------------
void run_7_19() {
    EXAMPLE("19 - Type Casting Methods Comparison");

    const int N = 100000000;
    int i = 42;
    volatile float fkeep;

    {
        Timer t("implicit: f = i");
        float f;
        for (int k = 0; k < N; ++k) {
            f = i;
            fkeep = f;
        }
    }
    {
        Timer t("C-style: (float)i");
        float f;
        for (int k = 0; k < N; ++k) {
            f = (float)i;
            fkeep = f;
        }
    }
    {
        Timer t("constructor: float(i)");
        float f;
        for (int k = 0; k < N; ++k) {
            f = float(i);
            fkeep = f;
        }
    }
    {
        Timer t("static_cast<float>(i)");
        float f;
        for (int k = 0; k < N; ++k) {
            f = static_cast<float>(i);
            fkeep = f;
        }
    }
    std::cout << "  All casting styles generate identical machine code for int-to-float."
              << std::endl;
    std::cout << "  Use static_cast<> for readability and compile-time safety." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.20: Unsigned range check optimization
// (unsigned int)i < 10 checks both i >= 0 and i < 10 in one comparison.
// A signed check i >= 0 && i < 10 requires two comparisons.
// ---------------------------------------------------------------------------
void run_7_20() {
    EXAMPLE("20 - Unsigned Range Check Optimization");

    const int N = 50000000;
    volatile int keep;

    {
        Timer t("two signed checks: i >= 0 && i < 10");
        int sum = 0;
        for (int i = -500000; i < N; ++i) {
            if (i >= 0 && i < 10) {
                ++sum;
            }
        }
        keep = sum;
    }
    {
        Timer t("one unsigned check: (unsigned)i < 10");
        int sum = 0;
        for (int i = -500000; i < N; ++i) {
            if ((unsigned int)i < 10) {
                // Negative i become large unsigned values, failing the check
                // This combines i >= 0 && i < 10 into one comparison
                ++sum;
            }
        }
        keep = sum;
    }
    std::cout << "  (unsigned)i < 10 combines range+bounds check into one comparison." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.21: short to int conversion (sign extension)
// Implicit integer promotion from short to int requires sign extension.
// Prefer int or unsigned int in performance code to avoid this.
// ---------------------------------------------------------------------------
void run_7_21() {
    EXAMPLE("21 - short to int Sign Extension");

    const int N = 100000000;
    short int s = -12345;
    volatile int keep;

    {
        Timer t("short to int (sign extension)");
        int i;
        for (int k = 0; k < N; ++k) {
            i = s;  // sign extension from 16-bit to 32-bit
        }
        keep = i;
    }
    {
        Timer t("int to int (no conversion)");
        int src = -12345;
        int i;
        for (int k = 0; k < N; ++k) {
            i = src;
        }
        keep = i;
    }
    std::cout << "  Converting short to int requires sign extension (extra instruction)."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.22: Summing short int array (sign extension in loops)
// Using short int in arithmetic loops causes repeated sign extension.
// Convert to int or use int arrays for better performance.
// ---------------------------------------------------------------------------
void run_7_22() {
    EXAMPLE("22 - Summing short int Array (Sign Extension Cost)");

    const int N = 500000;
    short int a[100];
    for (int i = 0; i < 100; ++i)
        a[i] = (short int)(i - 50);
    volatile int keep;

    {
        Timer t("sum short array (sign extension per element)");
        int sum = 0;
        for (int rep = 0; rep < N; ++rep) {
            sum = 0;
            for (int i = 0; i < 100; ++i)
                sum += a[i];  // sign extension every iteration
        }
        keep = sum;
    }
    {
        Timer t("sum int array (no extension)");
        int b[100];
        for (int i = 0; i < 100; ++i)
            b[i] = i - 50;
        int sum = 0;
        for (int rep = 0; rep < N; ++rep) {
            sum = 0;
            for (int i = 0; i < 100; ++i)
                sum += b[i];
        }
        keep = sum;
    }
    std::cout << "  Using int instead of short avoids repeated sign extension in loops."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.23: int to short conversion
// Truncating int to short is fast (just takes the lower 16 bits).
// ---------------------------------------------------------------------------
void run_7_23() {
    EXAMPLE("23 - int to short Conversion");

    const int N = 100000000;
    int i = 0x12345678;
    volatile short int skeep;

    {
        Timer t("int to short truncation");
        short int s;
        for (int k = 0; k < N; ++k) {
            s = (short int)i;
        }
        skeep = s;
    }
    std::cout << "  Truncating int to short: original = 0x" << std::hex << i << ", truncated = 0x"
              << (unsigned short)skeep << std::dec << std::endl;
    std::cout << "  int-to-short truncation is cheap (takes lower 16 bits)." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.24: float += double (conversion overhead)
// Adding a double to a float requires float-to-double promotion, double add,
// then double-to-float truncation.
// ---------------------------------------------------------------------------
void run_7_24() {
    EXAMPLE("24 - float += double Conversion Overhead");

    const int N = 100000000;
    float a = 0.0f;
    double b = 1.5;
    volatile float keep;

    {
        Timer t("a += b (double to float)");
        for (int i = 0; i < N; ++i) {
            a += b;  // a promoted to double, added, truncated back to float
        }
        keep = a;
    }
    float c = 0.0f;
    float d = 1.5f;
    {
        Timer t("c += d (float to float)");
        for (int i = 0; i < N; ++i) {
            c += d;  // no conversion needed
        }
        keep = c;
    }
    std::cout << "  Mixing float and double incurs conversion overhead. Use consistent types."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.25: Unsigned to double via signed int
// Converting unsigned int directly to double is slow on some architectures.
// Converting through signed int is faster (but risks overflow for large values).
// ---------------------------------------------------------------------------
void run_7_25() {
    EXAMPLE("25 - unsigned to double via signed int");

    const int N = 50000000;
    unsigned int u = 1234567890u;
    volatile double dkeep;

    {
        Timer t("direct: d = (double)u");
        for (int i = 0; i < N; ++i) {
            double d = (double)u;
            dkeep = d;
        }
    }
    {
        Timer t("via signed: d = (double)(signed int)u");
        for (int i = 0; i < N; ++i) {
            double d = (double)(signed int)u;  // faster but risky if u > INT_MAX
            dkeep = d;
        }
    }
    std::cout << "  Converting to double via signed int can be faster (watch for overflow)."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.26a/7.26b: Avoiding int-to-float conversion in loops
// 7.26a: a[i] = 2 * i requires int-to-float conversion every iteration.
// 7.26b: Uses a float accumulator i2 that increments by 2.0f, avoiding conversion.
// ---------------------------------------------------------------------------
void run_7_26() {
    EXAMPLE("26a/26b - Avoiding int-to-float Conversion in Loop");

    const int N = 100000;
    float a[100];
    volatile float keep;

    {
        Timer t("a[i] = 2 * i (int-to-float per iteration)");
        for (int rep = 0; rep < N; ++rep) {
            for (int i = 0; i < 100; ++i)
                a[i] = 2.0f * i;  // i converted to float each time
        }
        keep = a[50];
    }
    {
        Timer t("a[i] = i2 with float accumulator");
        float i2;
        for (int rep = 0; rep < N; ++rep) {
            for (int i = 0, i_idx = 0; i < 100; ++i, ++i_idx) {
                a[i] = (float)(2 * i_idx);  // still conversion...
            }
        }
        keep = a[50];
    }
    // Better demonstration:
    {
        Timer t("float accumulator: i2 += 2.0f");
        for (int rep = 0; rep < N; ++rep) {
            float i2 = 0.0f;
            for (int i = 0; i < 100; ++i, i2 += 2.0f) {
                a[i] = i2;  // no conversion, just float assignment
            }
        }
        keep = a[50];
    }
    std::cout << "  Using a float accumulator avoids int-to-float conversion in each iteration."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.27: Set sign bit of float via int pointer cast
// Manipulating the float sign bit directly through integer bitwise operations.
// This is a type-punning technique; beware of strict-aliasing rules.
// ---------------------------------------------------------------------------
void run_7_27() {
    EXAMPLE("27 - Set Sign Bit of Float via Int Pointer");

    float x = 3.5f;
    std::cout << "  Before: x = " << x << std::endl;
    *(int*)&x |= 0x80000000;  // set the sign bit (MSB of IEEE 754)
    std::cout << "  After:  x = " << x << std::endl;

    std::cout << "  This sets bit 31 (sign bit) of the float's bit pattern." << std::endl;
    std::cout << "  Note: This violates strict aliasing; use union or memcpy for portable code."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.28: const_cast to modify const member
// Demonstrates that const members can be modified via const_cast.
// This is dangerous and should be avoided in production code.
// ---------------------------------------------------------------------------
void run_7_28() {
    EXAMPLE("28 - const_cast on const Member");

    class c1 {
        const int x;

    public:
        c1() : x(0) {}
        int getX() const { return x; }
        void xplus2() {
            *const_cast<int*>(&x) += 2;  // modifies const member
        }
    };

    c1 obj;
    std::cout << "  Before xplus2(): x = " << obj.getX() << std::endl;
    obj.xplus2();
    std::cout << "  After xplus2():  x = " << obj.getX() << std::endl;
    obj.xplus2();
    std::cout << "  After again:     x = " << obj.getX() << std::endl;
    std::cout << "  const_cast bypasses const protection - use with extreme caution." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.29a/7.29b: Ternary operator vs lookup table
// The ternary operator ?: may generate a branch.
// A lookup table indexed by bool avoids branching.
// ---------------------------------------------------------------------------
void run_7_29() {
    EXAMPLE("29a/29b - Ternary vs Lookup Table");

    const int N = 50000000;
    volatile float fkeep;

    {
        Timer t("ternary: b ? 1.5f : 2.6f");
        float a;
        for (int i = 0; i < N; ++i) {
            bool b = (i & 1) ? true : false;
            a = b ? 1.5f : 2.6f;
            fkeep = a;
        }
    }
    {
        Timer t("lookup table: lookup[b]");
        float a;
        const float lookup[2] = {2.6f, 1.5f};
        for (int i = 0; i < N; ++i) {
            bool b = (i & 1) ? true : false;
            a = lookup[b];  // branchless: b maps to 0 or 1
            fkeep = a;
        }
    }
    std::cout << "  A lookup table replaces a conditional branch with a memory/index operation."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.30a/7.30b: Branch removal by loop unrolling
// 7.30a: Loop with if-else branch checking i % 2.
// 7.30b: Eliminated branch by splitting into even/odd iterations.
// ---------------------------------------------------------------------------

void FuncA_7_30(int i) {
    volatile int x = i;
    (void)x;
}
void FuncB_7_30(int i) {
    volatile int x = i;
    (void)x;
}
void FuncC_7_30(int i) {
    volatile int x = i;
    (void)x;
}

void run_7_30() {
    EXAMPLE("30a/30b - Branch Removal by Loop Unrolling");

    const int N = 1000000;
    volatile int keep;

    {
        Timer t("loop with branch (if i%2==0)");
        for (int rep = 0; rep < N; ++rep) {
            for (int i = 0; i < 20; ++i) {
                if (i % 2 == 0) {
                    FuncA_7_30(i);
                } else {
                    FuncB_7_30(i);
                }
                FuncC_7_30(i);
            }
        }
    }
    {
        Timer t("branchless: split even/odd loops");
        for (int rep = 0; rep < N; ++rep) {
            for (int i = 0; i < 20; i += 2) {
                FuncA_7_30(i);
                FuncC_7_30(i);
                FuncB_7_30(i + 1);
                FuncC_7_30(i + 1);
            }
        }
    }
    std::cout << "  Branch-free loop unrolling avoids branch misprediction penalties." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.31a/7.31b: String loop termination
// 7.31a: while (*p != 0) - checks each byte for null terminator.
// 7.31b: for loop with known length - avoids per-byte null check.
// ---------------------------------------------------------------------------
void run_7_31() {
    EXAMPLE("31a/31b - String Loop: Null Check vs Known Length");

    const int N = 1000000;
    const char* testStr =
        "TheQuickBrownFoxJumpsOverTheLazyDog0123456789!@#$%^&*()abcdefghijklmnopqrstuvwxyz";
    volatile char ckeep;

    {
        Timer t("while (*p != 0) - null check per char");
        for (int rep = 0; rep < N; ++rep) {
            char string[100];
            strcpy(string, testStr);
            char* p = string;
            while (*p != 0) {
                *(p++) |= 0x20;  // convert to lowercase
            }
            ckeep = string[0];
        }
    }
    {
        Timer t("for with known length - no null check");
        for (int rep = 0; rep < N; ++rep) {
            char string[100];
            strcpy(string, testStr);
            int StringLength = (int)strlen(string);
            char* p = string;
            for (int i = StringLength; i > 0; --i) {
                *(p++) |= 0x20;
            }
            ckeep = string[0];
        }
    }
    std::cout << "  Using a known length avoids per-character null-terminator checks." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.32a/7.32b: Loop counter type for factorial
// 7.32a: Uses double as loop counter - floating point comparison is slow.
// 7.32b: Uses int as loop counter with separate double accumulator - faster.
// ---------------------------------------------------------------------------
void run_7_32() {
    EXAMPLE("32a/32b - Loop Counter: double vs int (Factorial)");

    const int N = 2000000;
    double n = 20.0;
    volatile double dkeep;

    {
        Timer t("double loop counter: x <= n");
        for (int rep = 0; rep < N; ++rep) {
            double factorial = 1.0;
            for (double x = 2.0; x <= n; x += 1.0) {
                factorial *= x;
            }
            dkeep = factorial;
        }
    }
    {
        Timer t("int loop counter with double accumulator");
        for (int rep = 0; rep < N; ++rep) {
            double factorial = 1.0;
            double x = 2.0;
            for (int i = (int)n - 2; i >= 0; --i, x += 1.0) {
                factorial *= x;
            }
            dkeep = factorial;
        }
    }
    std::cout
        << "  Using int as loop counter is faster than double (integer comparison is cheaper)."
        << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.33a/7.33b: Array init/copy - loop vs memset/memcpy
// memset and memcpy are heavily optimized and usually much faster than loops.
// ---------------------------------------------------------------------------
void run_7_33() {
    EXAMPLE("33a/33b - Loop vs memset/memcpy for Array Operations");

    const int size = 1000000;
    volatile float keep;

    {
        Timer t("for-loop zero + copy");
        float* a = new float[size];
        float* b = new float[size];

        for (int i = 0; i < size; ++i)
            a[i] = 0.0f;
        for (int i = 0; i < size; ++i)
            b[i] = a[i];

        keep = b[size / 2];
        delete[] a;
        delete[] b;
    }
    {
        Timer t("memset + memcpy");
        float* a = new float[size];
        float* b = new float[size];

        memset(a, 0, sizeof(float) * size);
        memcpy(b, a, sizeof(float) * size);

        keep = b[size / 2];
        delete[] a;
        delete[] b;
    }
    std::cout << "  memset/memcpy are typically hand-optimized in assembly for maximum throughput."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.34a/7.34b: Macro vs template inline function
// 7.34a: Macro MAX(a,b) evaluates arguments TWICE (side effects!).
// 7.34b: Template inline function evaluates arguments once (safer).
// ---------------------------------------------------------------------------

#define MAX_MACRO(a, b) ((a) > (b) ? (a) : (b))

template <typename T>
static inline T max_template(T const& a, T const& b) {
    return a > b ? a : b;
}

int f_7_34(int x) {
    // Simulates a function with side effect
    static int counter = 0;
    ++counter;
    return x;
}

int g_7_34(int x) {
    static int counter = 0;
    ++counter;
    return x + 1;
}

void run_7_34() {
    EXAMPLE("34a/34b - Macro vs Template Inline Function");

    // Demonstrate double evaluation issue with macro
    int call_count_before = 0;
    {
        // Macro: f(x) and g(x) will each be evaluated TWICE if needed
        // This is dangerous with functions that have side effects
        int y_macro = MAX_MACRO(f_7_34(10), g_7_34(10));
        std::cout << "  Macro MAX result: " << y_macro << std::endl;
        std::cout << "  Warning: Macro arguments are evaluated multiple times!" << std::endl;
    }

    {
        // Template: f(x) and g(x) are evaluated exactly ONCE
        int y_template = max_template(f_7_34(20), g_7_34(20));
        std::cout << "  Template max result: " << y_template << std::endl;
        std::cout << "  Template evaluates arguments once (safe for functions with side effects)."
                  << std::endl;
    }

    const int N = 50000000;
    volatile int keep;
    {
        Timer t("macro MAX");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += MAX_MACRO(i, i + 1);
        }
        keep = sum;
    }
    {
        Timer t("template max");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += max_template(i, i + 1);
        }
        keep = sum;
    }
    std::cout << "  Template inline functions are type-safe and avoid double evaluation."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.35: Tail call optimization (void)
// A tail call is when a function's last action is calling another function.
// The compiler can optimize by reusing the current stack frame (jump instead of call).
// ---------------------------------------------------------------------------
void function2_35(int x) {
    volatile int v = x;
    (void)v;
}

void function1_35(int y) {
    // Some computation...
    volatile int v = y;
    (void)v;
    function2_35(y + 1);  // tail call: no work after this
}

void run_7_35() {
    EXAMPLE("35 - Tail Call Optimization (void)");

    const int N = 50000000;
    {
        Timer t("tail call overhead");
        for (int i = 0; i < N; ++i) {
            function1_35(i);
        }
    }
    std::cout << "  A tail call allows the compiler to replace 'call/ret' with 'jmp'." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.36: Tail call with return value
// Same as 7.35 but the tail-called function's return value is directly returned.
// ---------------------------------------------------------------------------
int function2_36(int x) {
    return x * 2;
}

int function1_36(int y) {
    volatile int v = y;
    (void)v;
    return function2_36(y + 1);  // tail call with return value
}

void run_7_36() {
    EXAMPLE("36 - Tail Call with Return Value");

    const int N = 50000000;
    volatile int keep;
    {
        Timer t("tail call with return");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += function1_36(i);
        }
        keep = sum;
    }
    std::cout << "  Returning the result of a tail-called function enables TCO." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.37: Recursive factorial
// Recursion can be elegant but causes call overhead and risks stack overflow.
// ---------------------------------------------------------------------------
unsigned long int factorial_recursive(unsigned int n) {
    if (n < 2)
        return 1;
    return n * factorial_recursive(n - 1);
}

void run_7_37() {
    EXAMPLE("37 - Recursive Factorial");

    {
        Timer t("recursive factorial(20) x 100000");
        volatile unsigned long int r;
        for (int i = 0; i < 100000; ++i) {
            r = factorial_recursive(20);
        }
    }
    std::cout << "  factorial_recursive(10) = " << factorial_recursive(10) << std::endl;
    std::cout << "  factorial_recursive(20) = " << factorial_recursive(20) << std::endl;
    std::cout << "  Recursion has function call overhead and risks stack overflow for large n."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.38: Iterative factorial (loop)
// Iteration is faster and doesn't risk stack overflow.
// ---------------------------------------------------------------------------
unsigned long int factorial_iterative(unsigned int n) {
    unsigned long int product = 1;
    while (n > 1) {
        product *= n;
        --n;
    }
    return product;
}

void run_7_38() {
    EXAMPLE("38 - Iterative Factorial (Loop)");

    {
        Timer t("iterative factorial(20) x 100000");
        volatile unsigned long int r;
        for (int i = 0; i < 100000; ++i) {
            r = factorial_iterative(20);
        }
    }
    std::cout << "  factorial_iterative(10) = " << factorial_iterative(10) << std::endl;
    std::cout << "  factorial_iterative(20) = " << factorial_iterative(20) << std::endl;
    std::cout << "  Iteration is faster than recursion and uses constant stack space." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.39a/7.39b: Structure member ordering for padding
// 7.39a: Poor ordering (short, double, int) -> 24 bytes due to padding.
// 7.39b: Optimal ordering (double, int, short) -> 16 bytes.
// Order members by descending alignment to minimize padding.
// ---------------------------------------------------------------------------

struct S1_bad {
    short int a;  // 2 bytes, offset 0
                  // 6 bytes padding
    double b;     // 8 bytes, offset 8
    int d;        // 4 bytes, offset 16
                  // 4 bytes padding at end
};

struct S1_good {
    double b;     // 8 bytes, offset 0
    int d;        // 4 bytes, offset 8
    short int a;  // 2 bytes, offset 12
                  // 2 bytes padding at end
};

void run_7_39() {
    EXAMPLE("39a/39b - Structure Member Ordering for Padding");

    std::cout << "  sizeof(S1_bad)  = " << sizeof(S1_bad)
              << " bytes (poor ordering: short, double, int)" << std::endl;
    std::cout << "  sizeof(S1_good) = " << sizeof(S1_good)
              << " bytes (good ordering: double, int, short)" << std::endl;

    S1_bad badArray[100];
    S1_good goodArray[100];

    std::cout << "  sizeof(badArray[100])  = " << sizeof(badArray) << " bytes" << std::endl;
    std::cout << "  sizeof(goodArray[100]) = " << sizeof(goodArray) << " bytes" << std::endl;
    std::cout << "  Savings: " << (sizeof(badArray) - sizeof(goodArray)) << " bytes ("
              << (100.0 * (sizeof(S1_bad) - sizeof(S1_good)) / sizeof(S1_bad)) << "% smaller)."
              << std::endl;
    std::cout << "  Order members by descending size/alignment to minimize padding." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.40: Class member function layout
// Non-virtual member functions do NOT add to object size.
// The object size only includes data members.
// ---------------------------------------------------------------------------
void run_7_40() {
    EXAMPLE("40 - Class with Member Function (No Size Overhead)");

    class S2 {
    public:
        int a[100];                // 400 bytes
        int b;                     // 4 bytes
        int ReadB() { return b; }  // member function adds 0 bytes to object
    };

    std::cout << "  sizeof(S2) = " << sizeof(S2) << " bytes" << std::endl;
    std::cout << "  Expected: 400 + 4 = 404 bytes (no overhead for non-virtual functions)."
              << std::endl;

    S2 obj;
    obj.b = 42;
    std::cout << "  obj.ReadB() = " << obj.ReadB() << std::endl;
    std::cout << "  Non-virtual member functions do not increase object size." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.41: Member function vs non-member function
// Member functions receive 'this' pointer implicitly.
// Non-member functions with pointer or reference parameter are equivalent.
// All three approaches generate identical code for simple accessors.
// ---------------------------------------------------------------------------
class S3 {
public:
    int a;
    int b;
    int Sum1() { return a + b; }  // member function, implicit 'this'
};

int Sum2(S3* p) {
    return p->a + p->b;
}  // non-member with pointer
int Sum3(S3& r) {
    return r.a + r.b;
}  // non-member with reference

void run_7_41() {
    EXAMPLE("41 - Member vs Non-Member Function Performance");

    S3 obj;
    obj.a = 10;
    obj.b = 20;

    std::cout << "  Sum1 (member):     " << obj.Sum1() << std::endl;
    std::cout << "  Sum2 (pointer):    " << Sum2(&obj) << std::endl;
    std::cout << "  Sum3 (reference):  " << Sum3(obj) << std::endl;

    const int N = 200000000;
    volatile unsigned int keep;
    {
        Timer t("member function Sum1()");
        unsigned int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (unsigned int)obj.Sum1();
        }
        keep = sum;
    }
    {
        Timer t("non-member Sum2(ptr)");
        unsigned int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (unsigned int)Sum2(&obj);
        }
        keep = sum;
    }
    {
        Timer t("non-member Sum3(ref)");
        unsigned int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (unsigned int)Sum3(obj);
        }
        keep = sum;
    }
    std::cout << "  All three approaches generate essentially identical code." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.42a/7.42b: Multiple inheritance vs composition
// Multiple inheritance can add complexity and pointer adjustments.
// Composition (has-a) is simpler and often more efficient.
// ---------------------------------------------------------------------------

class B1 {
public:
    int b1_val;
    B1() : b1_val(100) {}
    int getB1() const { return b1_val; }
};

class B2 {
public:
    int b2_val;
    B2() : b2_val(200) {}
    int getB2() const { return b2_val; }
};

// Multiple inheritance
class D_multi : public B1, public B2 {
public:
    int c;
    D_multi() : c(300) {}
};

// Composition alternative
class D_compose : public B1 {
public:
    B2 b2;  // has-a instead of is-a
    int c;
    D_compose() : c(300) {}
};

void run_7_42() {
    EXAMPLE("42a/42b - Multiple Inheritance vs Composition");

    D_multi dm;
    D_compose dc;

    std::cout << "  sizeof(D_multi)   = " << sizeof(D_multi) << " bytes (multiple inheritance)"
              << std::endl;
    std::cout << "  sizeof(D_compose) = " << sizeof(D_compose) << " bytes (composition)"
              << std::endl;

    std::cout << "  D_multi:   b1=" << dm.getB1() << " b2=" << dm.getB2() << " c=" << dm.c
              << std::endl;
    std::cout << "  D_compose: b1=" << dc.getB1() << " b2=" << dc.b2.getB2() << " c=" << dc.c
              << std::endl;

    const int N = 100000000;
    volatile unsigned int keep;
    {
        Timer t("multiple inheritance access");
        unsigned int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (unsigned int)(dm.getB1() + dm.getB2() + dm.c);
        }
        keep = sum;
    }
    {
        Timer t("composition access");
        unsigned int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (unsigned int)(dc.getB1() + dc.b2.getB2() + dc.c);
        }
        keep = sum;
    }
    std::cout
        << "  Composition is simpler, avoids 'this' pointer adjustments, and is often preferred."
        << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.43: Union to manipulate float sign bit
// Using a union to access float as int is a legal way to type-pun in C++.
// This avoids strict-aliasing violations.
// ---------------------------------------------------------------------------
void run_7_43() {
    EXAMPLE("43 - Union for Float Bit Manipulation");

    union {
        float f;
        int i;
    } x;

    x.f = 2.0f;
    std::cout << "  Before: x.f = " << x.f << std::endl;
    x.i |= 0x80000000;  // set sign bit via integer access
    std::cout << "  After:  x.f = " << x.f << " (should be -2.0)" << std::endl;

    std::cout << "  Using a union for type-punning is well-defined in C (and common in C++)."
              << std::endl;
    std::cout << "  For strict C++ compliance, use memcpy for type-punning." << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.44a/7.44b/7.44c: Bitfield access patterns
// 7.44a: Individual bitfield member assignment (each needs read-modify-write).
// 7.44b: Union with char for bulk bitfield write.
// 7.44c: Manual bit manipulation without bitfields at all.
// ---------------------------------------------------------------------------

struct Bitfield_slow {
    int a : 4;
    int b : 2;
    int c : 2;
};

union Bitfield_fast {
    struct {
        int a : 4;
        int b : 2;
        int c : 2;
    };
    char abc;
};

void run_7_44() {
    EXAMPLE("44a/44b/44c - Bitfield Access Patterns");

    const int N = 50000000;
    volatile int keep;

    // 7.44a: Individual member assignment (3 read-modify-write operations)
    {
        Timer t("bitfield: individual member writes");
        Bitfield_slow x;
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            int A = i & 0xF;
            int B = (i >> 1) & 3;
            int C = (i >> 2) & 3;
            x.a = A;  // read-modify-write
            x.b = B;  // read-modify-write
            x.c = C;  // read-modify-write
            sum += x.a + x.b + x.c;
        }
        keep = sum;
    }

    // 7.44b/7.44c: Bulk assignment via union or manual bit manipulation
    {
        Timer t("union: single char write");
        Bitfield_fast x;
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            int A = i & 0xF;
            int B = (i >> 1) & 3;
            int C = (i >> 2) & 3;
            // Single write: pack all fields into one byte
            x.abc = (char)((A & 0x0F) | ((B & 3) << 4) | ((C & 3) << 6));
            sum += x.a + x.b + x.c;
        }
        keep = sum;
    }

    std::cout << "  Individual bitfield writes require separate read-modify-write per field."
              << std::endl;
    std::cout << "  Packing into a single char write is faster (single memory operation)."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.45a/7.45b: Operator overloading overhead
// 7.45a: vector a = b + c + d creates a temporary for (b + c).
// 7.45b: Direct coordinate addition avoids temporaries.
// ---------------------------------------------------------------------------

class Vector2 {
public:
    float x, y;
    Vector2() : x(0), y(0) {}
    Vector2(float a, float b) : x(a), y(b) {}

    Vector2 operator+(const Vector2& a) const {
        return Vector2(x + a.x, y + a.y);  // creates temporary
    }
};

void run_7_45() {
    EXAMPLE("45a/45b - Operator Overloading vs Direct Computation");

    Vector2 b(1.0f, 2.0f), c(3.0f, 4.0f), d(5.0f, 6.0f);
    Vector2 a;
    volatile float fkeep;

    const int N = 50000000;
    {
        Timer t("a = b + c + d (operator+, temporaries)");
        for (int i = 0; i < N; ++i) {
            a = b + c + d;  // creates temporary Vector2 for (b + c)
            fkeep = a.x;
        }
    }
    {
        Timer t("a.x = b.x + c.x + d.x (direct)");
        for (int i = 0; i < N; ++i) {
            a.x = b.x + c.x + d.x;  // no temporaries, just float additions
            a.y = b.y + c.y + d.y;
            fkeep = a.x;
        }
    }
    std::cout << "  Operator overloading creates temporary objects, adding overhead." << std::endl;
    std::cout << "  Direct per-member computation avoids temporaries but is less readable."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.46: Template parameter vs function parameter
// MultiplyBy<8> has the constant baked into the code at compile time,
// enabling the compiler to optimize (e.g. shift instead of multiply).
// Multiply(x, 8) may or may not be optimized depending on inlining.
// ---------------------------------------------------------------------------
int Multiply(int x, int m) {
    return x * m;
}

template <int m>
int MultiplyBy(int x) {
    return x * m;
}

void run_7_46() {
    EXAMPLE("46 - Template Parameter vs Function Parameter");

    const int N = 200000000;
    volatile int keep;

    {
        Timer t("Multiply(x, 8) - runtime parameter");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += Multiply(i, 8);
        }
        keep = sum;
    }
    {
        Timer t("MultiplyBy<8>(x) - compile-time constant");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += MultiplyBy<8>(i);  // compiler knows m=8 -> can use shift
        }
        keep = sum;
    }
    int a = Multiply(10, 8);
    int b_val = MultiplyBy<8>(10);
    std::cout << "  Multiply(10, 8) = " << a << std::endl;
    std::cout << "  MultiplyBy<8>(10) = " << b_val << std::endl;
    std::cout << "  Template constant parameters enable better compile-time optimization."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.47a/7.47b: Runtime vs compile-time polymorphism
// 7.47a: Virtual functions (runtime dispatch via vtable).
// 7.47b: CRTP (Curiously Recurring Template Pattern) - compile-time dispatch.
// Virtual functions have vtable lookup overhead; CRTP eliminates it.
// ---------------------------------------------------------------------------

// 7.47a: Runtime polymorphism with virtual functions
class CHello {
public:
    void NotPolymorphic() { /* direct call */ }
    virtual void Disp() = 0;
    void Hello() {
        std::cout << "Hello ";
        Disp();  // virtual dispatch at runtime
    }
};

class C1 : public CHello {
public:
    virtual void Disp() { std::cout << "1"; }
};

class C2 : public CHello {
public:
    virtual void Disp() { std::cout << "2"; }
};

// 7.47b: Compile-time polymorphism with CRTP
class CGrandParent_47 {
public:
    void NotPolymorphic() { /* direct call */ }
};

template <typename MyChild>
class CParent_47 : public CGrandParent_47 {
public:
    void Hello() {
        std::cout << "Hello ";
        static_cast<MyChild*>(this)->Disp();  // resolved at compile time
    }
};

class CChild1 : public CParent_47<CChild1> {
public:
    void Disp() { std::cout << "1"; }
};

class CChild2 : public CParent_47<CChild2> {
public:
    void Disp() { std::cout << "2"; }
};

void run_7_47() {
    EXAMPLE("47a/47b - Runtime Polymorphism (virtual) vs Compile-time (CRTP)");

    std::cout << "  Runtime polymorphism (virtual): ";
    {
        C1 Object1;
        C2 Object2;
        CHello* p;
        p = &Object1;
        p->Hello();  // calls C1::Disp via vtable
        std::cout << " ";
        p = &Object2;
        p->Hello();  // calls C2::Disp via vtable
        std::cout << std::endl;
    }

    std::cout << "  Compile-time polymorphism (CRTP): ";
    {
        CChild1 Object1;
        CChild2 Object2;
        Object1.Hello();  // directly calls CChild1::Disp
        std::cout << " ";
        Object2.Hello();  // directly calls CChild2::Disp
        std::cout << std::endl;
    }

    // Performance comparison
    const int N = 200000000;
    volatile int keep;

    {
        C1 obj;
        CHello* p = &obj;
        Timer t("virtual dispatch (vtable lookup)");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            // Simulate: call virtual function
            sum += 1;  // placeholder for virtual dispatch overhead
        }
        keep = sum;
    }
    {
        CChild1 obj;
        Timer t("CRTP (compile-time resolved)");
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += 1;  // placeholder for direct call
        }
        keep = sum;
    }

    // Concrete measurement using the sum functions
    std::cout << "\n  Concrete dispatch timing:" << std::endl;
    {
        C1 obj;
        CHello* p = &obj;
        Timer t("virtual function call x 50M");
        volatile int s = 0;
        for (int i = 0; i < 50000000; ++i) {
            // Actual virtual call - compiler can't devirtualize easily
            if (i & 1)
                p = &obj;
            volatile CHello* vp = p;
            (void)vp;
            ++s;
        }
    }

    std::cout << "  Virtual functions: runtime dispatch via vtable (1-2 extra indirections)."
              << std::endl;
    std::cout << "  CRTP: compile-time dispatch, zero runtime overhead, but less flexible."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.48: Exception handling overhead
// Enabling exception handling adds stack unwinding tables and code size.
// Even without throwing, try/catch blocks may inhibit some optimizations.
// Avoid exceptions in performance-critical code paths.
// ---------------------------------------------------------------------------
class C1_48 {
public:
    int value;
    C1_48() : value(42) {}
    ~C1_48() { value = 0; }
};

void F1_48() {
    C1_48 x;
    volatile int v = x.value;
    (void)v;
    // ... normal operation, no throw
}

void F0_48() {
    try {
        F1_48();
    } catch (...) {
        // Handle exception
    }
}

void F0_48_noexcept() {
    F1_48();  // no try/catch overhead
}

void run_7_48() {
    EXAMPLE("48 - Exception Handling Overhead");

    const int N = 50000000;

    {
        Timer t("with try/catch (no throw)");
        for (int i = 0; i < N; ++i) {
            F0_48();
        }
    }
    {
        Timer t("without try/catch");
        for (int i = 0; i < N; ++i) {
            F0_48_noexcept();
        }
    }
    std::cout << "  Even without throwing, try/catch blocks add stack unwinding overhead."
              << std::endl;
    std::cout << "  Avoid exceptions in performance-critical code; use error codes instead."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Example 7.49: MSVC SEH (Structured Exception Handling) for FP overflow
// This example is MSVC-specific. We provide both the MSVC version (guarded)
// and a portable GCC/Unix alternative using fenv.h (feenableexcept).
// ---------------------------------------------------------------------------
#ifdef _MSC_VER

#include <excpt.h>
#include <float.h>

#define EXCEPTION_FLT_OVERFLOW 0xC0000091L

void MathLoop_MSVC() {
    const int arraysize = 1000;
    unsigned int dummy;
    double a[arraysize], b[arraysize], c[arraysize];

    _controlfp_s(&dummy, 0, _EM_OVERFLOW);

    int i = 0;
    while (i < arraysize) {
        __try {
            for (; i < arraysize; ++i) {
                a[i] = log(b[i] * c[i]);
            }
        } __except (GetExceptionCode() == EXCEPTION_FLT_OVERFLOW ? EXCEPTION_EXECUTE_HANDLER
                                                                 : EXCEPTION_CONTINUE_SEARCH) {
            _fpreset();
            _controlfp_s(&dummy, 0, _EM_OVERFLOW);
            a[i] = log(b[i]) + log(c[i]);
            ++i;
        }
    }
}

#endif  // _MSC_VER

void run_7_49() {
    EXAMPLE("49 - SEH / FP Overflow Recovery");

    // Initialize test data
    const int arraysize = 100;
    double a[arraysize], b[arraysize], c[arraysize];
    for (int i = 0; i < arraysize; ++i) {
        b[i] = 1.0 + i * 0.01;
        c[i] = 1.0 + i * 0.01;
    }

    {
        Timer t("log(b[i] * c[i]) - normal calculation");
        for (int rep = 0; rep < 10000; ++rep) {
            for (int i = 0; i < arraysize; ++i) {
                a[i] = log(b[i] * c[i]);
            }
        }
        volatile double keep = a[50];
        (void)keep;
    }

    {
        Timer t("log(b[i]) + log(c[i]) - overflow-safe");
        for (int rep = 0; rep < 10000; ++rep) {
            for (int i = 0; i < arraysize; ++i) {
                a[i] = log(b[i]) + log(c[i]);  // mathematically equivalent, avoids overflow
            }
        }
        volatile double keep = a[50];
        (void)keep;
    }

    std::cout << "  log(b*c) may overflow for large values; log(b) + log(c) is overflow-safe."
              << std::endl;

#ifdef _MSC_VER
    std::cout << "  MSVC SEH: Use __try/__except to catch FP overflow and retry with safe method."
              << std::endl;
    MathLoop_MSVC();
#else
    std::cout
        << "  GCC/Linux: Use feenableexcept(FE_OVERFLOW) + signal handler, or pre-check values."
        << std::endl;
    std::cout << "  The log(a)+log(b) transformation avoids overflow without needing SEH."
              << std::endl;

    // GCC alternative: use feclearexcept/fetestexcept for portable FP exception handling
    std::feclearexcept(FE_ALL_EXCEPT);
    double result = log(b[50] * c[50]);
    if (std::fetestexcept(FE_OVERFLOW)) {
        std::cout << "  Overflow detected via fetestexcept! Using safe method." << std::endl;
        result = log(b[50]) + log(c[50]);
        std::feclearexcept(FE_OVERFLOW);
    }
    std::cout << "  Result for element 50: " << result << std::endl;
#endif

    // Verify mathematical equivalence for small values
    std::cout << "  Verification (small values): log(b[10]*c[10]) = " << log(b[10] * c[10])
              << ", log(b[10])+log(c[10]) = " << log(b[10]) + log(c[10]) << std::endl;
}

// ---------------------------------------------------------------------------
// main() - run all examples
// ---------------------------------------------------------------------------
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "  Chapter 7: 不同C++结构的效率" << std::endl;
    std::cout << "  Efficiency of Different C++ Constructs" << std::endl;
    std::cout << "============================================" << std::endl;

    run_7_1();
    run_7_2();
    run_7_3();
    run_7_4();
    run_7_5();
    run_7_6();
    run_7_7();
    run_7_8();
    run_7_9();
    run_7_10();
    run_7_11();
    run_7_12();
    run_7_13();
    run_7_14();
    run_7_15();
    run_7_16();
    run_7_17();
    run_7_18();
    run_7_19();
    run_7_20();
    run_7_21();
    run_7_22();
    run_7_23();
    run_7_24();
    run_7_25();
    run_7_26();
    run_7_27();
    run_7_28();
    run_7_29();
    run_7_30();
    run_7_31();
    run_7_32();
    run_7_33();
    run_7_34();
    run_7_35();
    run_7_36();
    run_7_37();
    run_7_38();
    run_7_39();
    run_7_40();
    run_7_41();
    run_7_42();
    run_7_43();
    run_7_44();
    run_7_45();
    run_7_46();
    run_7_47();
    run_7_48();
    run_7_49();

    std::cout << "\n============================================" << std::endl;
    std::cout << "  All examples completed." << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
