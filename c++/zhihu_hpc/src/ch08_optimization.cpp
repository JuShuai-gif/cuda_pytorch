#include <iostream>
#include <cmath>
#include <cstring>

// ============================================================================
// Chapter 8: Compiler Optimizations (编译器中的优化)
// Consolidated from 40 individual example snippets.
// Compile: g++ -std=c++11 -O2 ch08_optimization.cpp -o ch08_optimization
// ============================================================================

#ifdef __GNUC__
#define PURE_FN __attribute__((const))
#else
#define PURE_FN
#endif

// ============================================================================
// Helper: print banner for each example section
// ============================================================================
static void print_banner(const char* title) {
    std::cout << "\n========== " << title << " ==========\n";
}

// ============================================================================
// Example 8.1a/8.1b: Function Inlining
// The call to square(x) is inlined, replacing it with x*x.
// Before: parabola calls square(x)
// After:  parabola computes x*x directly
// ============================================================================
static float square_8_1(float a) {
    return a * a;
}

static float parabola_8_1a(float x) {
    return square_8_1(x) + 1.0f;
}

static float parabola_8_1b(float x) {
    return x * x + 1.0f;
}

static void example_8_1() {
    print_banner("8.1a/8.1b: Function Inlining");
    float x = 3.0f;
    std::cout << "parabola(before inline) = " << parabola_8_1a(x) << "\n";
    std::cout << "parabola(after inline)  = " << parabola_8_1b(x) << "\n";
}

// ============================================================================
// Example 8.2a/8.2b: Constant Folding
// 2.0/3.0 is computed at compile time.
// Before: a = b + 2.0/3.0
// After:  a = b + 0.666666666666666666667
// ============================================================================
static double constant_folding_8_2a(double b) {
    double a;
    a = b + 2.0 / 3.0;
    return a;
}

static double constant_folding_8_2b(double b) {
    double a;
    a = b + 0.666666666666666666667;
    return a;
}

static void example_8_2() {
    print_banner("8.2a/8.2b: Constant Folding");
    double b = 1.0;
    std::cout << "fold 2.0/3.0 at runtime = " << constant_folding_8_2a(b) << "\n";
    std::cout << "pre-folded constant    = " << constant_folding_8_2b(b) << "\n";
}

// ============================================================================
// Example 8.3a/8.3b: Constant Propagation
// parabola(2.0f) returns 5.0f, then b = a + 1.0f = 6.0f.
// Before: a = parabola(2.0f); b = a + 1.0f;
// After:  a = 5.0f; b = 6.0f;
// ============================================================================
static float parabola_8_3(float x) {
    return x * x + 1.0f;
}

static void constant_prop_8_3a(float& a, float& b) {
    a = parabola_8_3(2.0f);
    b = a + 1.0f;
}

static void constant_prop_8_3b(float& a, float& b) {
    a = 5.0f;
    b = 6.0f;
}

static void example_8_3() {
    print_banner("8.3a/8.3b: Constant Propagation");
    float a1, b1, a2, b2;
    constant_prop_8_3a(a1, b1);
    constant_prop_8_3b(a2, b2);
    std::cout << "before propagate: a=" << a1 << " b=" << b1 << "\n";
    std::cout << "after propagate:  a=" << a2 << " b=" << b2 << "\n";
}

// ============================================================================
// Example 8.4: Constant Folding of Standard Functions
// sin(0.8) is computed at compile time.
// ============================================================================
static void example_8_4() {
    print_banner("8.4: Constant Folding of Standard Functions");
    double a = sin(0.8);
    std::cout << "sin(0.8) folded at compile time = " << a << "\n";
}

// ============================================================================
// Example 8.5a/8.5b: Inlining of Indirect Operations Through Pointers
// Plus2(&a) is inlined to a += 2.
// Before: Plus2(&a)
// After:  a += 2
// ============================================================================
static void plus2_8_5(int* p) {
    *p = *p + 2;
}

static void indirect_inline_8_5a(int& a) {
    plus2_8_5(&a);
}

static void indirect_inline_8_5b(int& a) {
    a += 2;
}

static void example_8_5() {
    print_banner("8.5a/8.5b: Inlining of Indirect Operations");
    int a1 = 10, a2 = 10;
    indirect_inline_8_5a(a1);
    indirect_inline_8_5b(a2);
    std::cout << "before inline (via ptr): a=" << a1 << "\n";
    std::cout << "after inline (direct):   a=" << a2 << "\n";
}

// ============================================================================
// Example 8.6a/8.6b: Common Subexpression Elimination (CSE)
// (a+1) appears twice; compiler computes it once.
// Before: b = (a+1)*(a+1); c = (a+1)/4;
// After:  temp = a+1; b = temp*temp; c = temp/4;
// ============================================================================
static void cse_8_6a(int a, int& b, int& c) {
    b = (a + 1) * (a + 1);
    c = (a + 1) / 4;
}

static void cse_8_6b(int a, int& b, int& c) {
    int temp = a + 1;
    b = temp * temp;
    c = temp / 4;
}

static void example_8_6() {
    print_banner("8.6a/8.6b: Common Subexpression Elimination");
    int b1, c1, b2, c2;
    cse_8_6a(5, b1, c1);
    cse_8_6b(5, b2, c2);
    std::cout << "before CSE: b=" << b1 << " c=" << c1 << "\n";
    std::cout << "after CSE:  b=" << b2 << " c=" << c2 << "\n";
}

// ============================================================================
// Example 8.7: Register Variables / Write Buffer
// Values loaded from memory are kept in registers; writes to x[]
// go through a write buffer to hide memory latency.
// ============================================================================
static int example_8_7_func(int a, int x[]) {
    int b, c;
    x[0] = a;
    b = a + 1;
    x[1] = b;
    c = b + 1;
    return c;
}

static void example_8_7() {
    print_banner("8.7: Register Variables / Write Buffer");
    int x[2] = {0, 0};
    int result = example_8_7_func(5, x);
    std::cout << "result=" << result << " x[0]=" << x[0] << " x[1]=" << x[1] << "\n";
}

// ============================================================================
// Example 8.8a/8.8b: Code Hoisting
// "z = y + 1;" is common to both branches; hoisted outside.
// Before: y+1 computed inside each branch
// After:  y+1 computed once after the if-else
// ============================================================================
static void code_hoist_8_8a(double x, bool b, double& y, double& z) {
    if (b) {
        y = sin(x);
        z = y + 1.;
    } else {
        y = cos(x);
        z = y + 1.;
    }
}

static void code_hoist_8_8b(double x, bool b, double& y, double& z) {
    if (b) {
        y = sin(x);
    } else {
        y = cos(x);
    }
    z = y + 1.;
}

static void example_8_8() {
    print_banner("8.8a/8.8b: Code Hoisting");
    double y1, z1, y2, z2;
    code_hoist_8_8a(1.0, true, y1, z1);
    code_hoist_8_8b(1.0, true, y2, z2);
    std::cout << "before hoist: y=" << y1 << " z=" << z1 << "\n";
    std::cout << "after hoist:  y=" << y2 << " z=" << z2 << "\n";
}

// ============================================================================
// Example 8.9b: When Common Code Inside if-else Can't Be Hoisted
// "return a + 1" looks common but 'a' is modified differently
// in each branch, so the common code cannot be hoisted safely.
// ============================================================================
static int no_hoist_8_9b(int a, bool b) {
    if (b) {
        a = a * 2;
        return a + 1;
    } else {
        a = a * 3;
        return a + 1;
    }
}

static void example_8_9() {
    print_banner("8.9b: Common Code That Cannot Be Hoisted");
    std::cout << "b=true:  " << no_hoist_8_9b(5, true) << "\n";
    std::cout << "b=false: " << no_hoist_8_9b(5, false) << "\n";
}

// ============================================================================
// Example 8.10a/8.10b: Dead Code Elimination
// if(true) makes the else branch dead.
// Before: if(true) { a = b; } else { a = c; }
// After:  a = b;
// ============================================================================
static int dead_code_elim_8_10a(int b, int) {
    int a;
    if (true) {
        a = b;
    } else {
        a = 0;  // dead code, never executed
    }
    return a;
}

static int dead_code_elim_8_10b(int b, int) {
    return b;
}

static void example_8_10() {
    print_banner("8.10a/8.10b: Dead Code Elimination");
    std::cout << "before DCE: a=" << dead_code_elim_8_10a(42, 99) << "\n";
    std::cout << "after DCE:  a=" << dead_code_elim_8_10b(42, 99) << "\n";
}

// ============================================================================
// Example 8.11a/8.11b: Branch Merging
// Two if(b) blocks are merged into one.
// Before: two separate if(b) { ... } else { ... }
// After:  single if(b) { ... } else { ... }
// ============================================================================
static int branch_merge_8_11a(int a, bool b) {
    if (b) {
        a = a * 2;
    } else {
        a = a * 3;
    }
    if (b) {
        return a + 1;
    } else {
        return a - 1;
    }
}

static int branch_merge_8_11b(int a, bool b) {
    if (b) {
        a = a * 2;
        return a + 1;
    } else {
        a = a * 3;
        return a - 1;
    }
}

static void example_8_11() {
    print_banner("8.11a/8.11b: Branch Merging");
    std::cout << "before merge (b=true):  " << branch_merge_8_11a(5, true) << "\n";
    std::cout << "after merge (b=true):   " << branch_merge_8_11b(5, true) << "\n";
    std::cout << "before merge (b=false): " << branch_merge_8_11a(5, false) << "\n";
    std::cout << "after merge (b=false):  " << branch_merge_8_11b(5, false) << "\n";
}

// ============================================================================
// Example 8.12a/8.12b: Loop Unrolling
// A loop of 2 iterations is completely unrolled.
// Before: for (i=0; i<2; i++) a[i] = i+1;
// After:  a[0] = 1; a[1] = 2;
// ============================================================================
static void loop_unroll_8_12a(int a[2]) {
    for (int i = 0; i < 2; i++)
        a[i] = i + 1;
}

static void loop_unroll_8_12b(int a[2]) {
    a[0] = 1;
    a[1] = 2;
}

static void example_8_12() {
    print_banner("8.12a/8.12b: Loop Unrolling");
    int a1[2] = {0, 0}, a2[2] = {0, 0};
    loop_unroll_8_12a(a1);
    loop_unroll_8_12b(a2);
    std::cout << "before unroll: [" << a1[0] << ", " << a1[1] << "]\n";
    std::cout << "after unroll:  [" << a2[0] << ", " << a2[1] << "]\n";
}

// ============================================================================
// Example 8.13a/8.13b: Loop-Invariant Code Motion (LICM)
// b*b+1 does not change in the loop; moved outside.
// Before: a[i] = b*b+1 inside loop
// After:  temp = b*b+1; a[i] = temp inside loop
// ============================================================================
static void licm_8_13a(int a[100], int b) {
    for (int i = 0; i < 100; i++) {
        a[i] = b * b + 1;
    }
}

static void licm_8_13b(int a[100], int b) {
    int temp = b * b + 1;
    for (int i = 0; i < 100; i++) {
        a[i] = temp;
    }
}

static void example_8_13() {
    print_banner("8.13a/8.13b: Loop-Invariant Code Motion");
    int a1[100], a2[100];
    licm_8_13a(a1, 5);
    licm_8_13b(a2, 5);
    std::cout << "before LICM: a[0]=" << a1[0] << " a[99]=" << a1[99] << "\n";
    std::cout << "after LICM:  a[0]=" << a2[0] << " a[99]=" << a2[99] << "\n";
}

// ============================================================================
// Example 8.14a/8.14b: Induction Variables
// i*9+3 is replaced by a running sum: temp += 9.
// Before: a[i] = i * 9 + 3
// After:  a[i] = temp; temp += 9
// ============================================================================
static void induction_8_14a(int a[100]) {
    for (int i = 0; i < 100; i++) {
        a[i] = i * 9 + 3;
    }
}

static void induction_8_14b(int a[100]) {
    int temp = 3;
    for (int i = 0; i < 100; i++) {
        a[i] = temp;
        temp += 9;
    }
}

static void example_8_14() {
    print_banner("8.14a/8.14b: Induction Variables");
    int a1[100], a2[100];
    induction_8_14a(a1);
    induction_8_14b(a2);
    // Verify equivalence
    bool ok = true;
    for (int i = 0; i < 100; i++) {
        if (a1[i] != a2[i]) {
            ok = false;
            break;
        }
    }
    std::cout << "a[0]=" << a1[0] << " a[1]=" << a1[1] << " a[99]=" << a1[99] << "\n";
    std::cout << "induction variable equivalence: " << (ok ? "PASS" : "FAIL") << "\n";
}

// ============================================================================
// Example 8.15a/8.15b: Pointerization of Struct Arrays
// Array indexing is converted to pointer walking.
// Before: list[i].a = 1.0
// After:  temp->a = 1.0; temp++
// ============================================================================
struct S1_8_15 {
    double a;
    double b;
};

static void pointerize_8_15a(S1_8_15 list[100]) {
    for (int i = 0; i < 100; i++) {
        list[i].a = 1.0;
        list[i].b = 2.0;
    }
}

static void pointerize_8_15b(S1_8_15 list[100]) {
    S1_8_15* temp;
    for (temp = &list[0]; temp < &list[100]; temp++) {
        temp->a = 1.0;
        temp->b = 2.0;
    }
}

static void example_8_15() {
    print_banner("8.15a/8.15b: Pointerization of Struct Arrays");
    S1_8_15 list1[100] = {}, list2[100] = {};
    pointerize_8_15a(list1);
    pointerize_8_15b(list2);
    std::cout << "index: list[0].a=" << list1[0].a << " list[0].b=" << list1[0].b << "\n";
    std::cout << "ptr:   list[0].a=" << list2[0].a << " list[0].b=" << list2[0].b << "\n";
}

// ============================================================================
// Example 8.16: Instruction-Level Parallel Scheduling
// Independent add chains (x = a+b+c and y = d+e+f) can be
// interleaved by the CPU to exploit instruction-level parallelism.
// ============================================================================
static void example_8_16() {
    print_banner("8.16: Instruction-Level Parallel Scheduling");
    float a = 1.0f, b = 2.0f, c = 3.0f, d = 4.0f, e = 5.0f, f = 6.0f;
    float x = a + b + c;
    float y = d + e + f;
    std::cout << "x=" << x << " y=" << y << "\n";
    std::cout << "Two independent add chains exploit ILP\n";
}

// ============================================================================
// Example 8.17: Integer Promotion Efficiency
// char arithmetic promotes to int. Using int directly avoids
// repeated sign extension.
// ============================================================================
static void example_8_17() {
    print_banner("8.17: Integer Promotion Efficiency");
    char a = -100, b = 100, c = 100, y;
    y = a + b + c;
    std::cout << "char arithmetic result: " << static_cast<int>(y) << "\n";
    std::cout << "chars promote to int for arithmetic\n";
}

// ============================================================================
// Example 8.18: Float Associativity
// (a+b)+c loses precision when a+b is large and c is small.
// Order of floating-point addition matters.
// ============================================================================
static void example_8_18() {
    print_banner("8.18: Float Associativity and Precision");
    float a = -1.0E8f, b = 1.0E8f, c = 1.23456f, y1, y2;
    y1 = a + b + c;    // (a+b) = 0, then +c = c
    y2 = a + (b + c);  // (b+c) ≈ b, then a+b ≈ 0
    std::cout << "(a+b)+c = " << y1 << " (correct)\n";
    std::cout << "a+(b+c) = " << y2 << " (may lose c)\n";
    std::cout << "Compiler respects parens; -ffast-math may re-associate\n";
}

// ============================================================================
// Example 8.19: Devirtualization
// When the concrete type is known (C1 obj1), the virtual call
// can be resolved statically.
// ============================================================================
class C0_8_19 {
public:
    virtual int f() { return 100; }
};

class C1_8_19 : public C0_8_19 {
public:
    virtual int f() { return 200; }
};

static void example_8_19() {
    print_banner("8.19: Devirtualization");
    C1_8_19 obj1;
    C0_8_19* p = &obj1;
    int result = p->f();  // Virtual call to C1::f - compiler can devirtualize
    std::cout << "virtual call result: " << result << "\n";
    std::cout << "Compiler can devirtualize since concrete type is known\n";
}

// ============================================================================
// Example 8.20: Cross-Module Inlining (Link-Time Optimization)
// Func1 from module1 is inlined into Func2 in module2.
// Requires LTO (flto) or whole-program optimization.
// ============================================================================
static int func1_8_20(int x) {
    return x * x + 1;
}

static int func2_8_20() {
    int a = func1_8_20(2);
    return a;
}

static void example_8_20() {
    print_banner("8.20: Cross-Module Inlining");
    int result = func2_8_20();
    std::cout << "func2 result: " << result << "\n";
    std::cout << "With LTO, func1 would be inlined into func2\n";
}

// ============================================================================
// Example 8.21: Pointer Aliasing Problem
// When a[] and *p may alias (point to overlapping memory),
// the compiler cannot hoist *p out of the loop.
// ============================================================================
static void func1_8_21(int a[], int* p) {
    for (int i = 0; i < 100; i++) {
        a[i] = *p + 2;
    }
}

static void example_8_21() {
    print_banner("8.21: Pointer Aliasing Problem");
    int list[100];
    func1_8_21(list, &list[8]);  // list and p overlap - aliasing!
    std::cout << "list[0]=" << list[0] << " list[8]=" << list[8] << " list[9]=" << list[9] << "\n";
    std::cout << "Aliasing prevents hoisting *p out of the loop\n";
}

// ============================================================================
// Example 8.22: __attribute__((const)) Pure Function
// Marking a function as pure (no side effects, result depends
// only on arguments) allows CSE of redundant calls.
// ============================================================================
static double func1_8_22(double x) PURE_FN;

static double func1_8_22(double x) {
    return x * x + 2.0;
}

static double func2_8_22(double x) {
    return func1_8_22(x) * func1_8_22(x) + 1.;
}

static void example_8_22() {
    print_banner("8.22: __attribute__((const)) Pure Function");
    double result = func2_8_22(3.0);
    std::cout << "func2(3.0) = " << result << "\n";
    std::cout << "func1 is pure; compiler can eliminate redundant call\n";
}

// ============================================================================
// Example 8.23a/8.23b: Induction Variable Polynomial Optimization
// The polynomial A*x*x + B*x + C is computed with recurrence
// relations to avoid multiplication in each iteration.
// Before: Table[x] = A*x*x + B*x + C
// After:  Table[x] = Y; Y += Z; Z += A2
// ============================================================================
static void polynomial_induction_8_23a(double Table[100]) {
    const double A = 1.1, B = 2.2, C = 3.3;
    for (int x = 0; x < 100; x++) {
        Table[x] = A * x * x + B * x + C;
    }
}

static void polynomial_induction_8_23b(double Table[100]) {
    const double A = 1.1, B = 2.2, C = 3.3;
    const double A2 = A + A;
    double Y = C;
    double Z = A + B;
    for (int x = 0; x < 100; x++) {
        Table[x] = Y;
        Y += Z;
        Z += A2;
    }
}

static void example_8_23() {
    print_banner("8.23a/8.23b: Induction Variable Polynomial Optimization");
    double t1[100], t2[100];
    polynomial_induction_8_23a(t1);
    polynomial_induction_8_23b(t2);
    bool ok = true;
    for (int i = 0; i < 100; i++) {
        if (std::abs(t1[i] - t2[i]) > 1e-10) {
            ok = false;
            break;
        }
    }
    std::cout << "Table[0]=" << t1[0] << " Table[1]=" << t1[1] << " Table[99]=" << t1[99] << "\n";
    std::cout << "induction equivalence: " << (ok ? "PASS" : "FAIL") << "\n";
}

// ============================================================================
// Example 8.24: Integer Constant Propagation
// const int ArraySize is known at compile time and propagated
// into the loop bound, enabling further optimizations.
// ============================================================================
static const int kArraySize_8_24 = 1000;

static void example_8_24() {
    print_banner("8.24: Integer Constant Propagation");
    int List[kArraySize_8_24] = {};
    for (int i = 0; i < kArraySize_8_24; i++)
        List[i]++;
    std::cout << "List[0]=" << List[0] << " List[999]=" << List[kArraySize_8_24 - 1] << "\n";
    std::cout << "ArraySize=1000 known at compile time\n";
}

// ============================================================================
// Example 8.25: static const float Initialization
// log(2.0) is computed once at program init, not per call.
// ============================================================================
static void example_8_25_func() {
    static const double log2 = log(2.0);
    // use log2 in subsequent operations
    double result = log2 * 10.0;
    std::cout << "  log2 * 10.0 = " << result << "\n";
}

static void example_8_25() {
    print_banner("8.25: static const float Initialization");
    std::cout << "log(2.0) computed once at first call:\n";
    example_8_25_func();
    example_8_25_func();
    std::cout << "static const: computed only once\n";
}

// ============================================================================
// Example 8.26a/8.26b: Induction Variable Replacing Slow Division
// The expression i/2 is slow. Induction variable replaces it
// with a counter that increments every 2 iterations.
// Before: a[i] = r + i/2
// After:  step by 2, use Induction variable
// ============================================================================
static void slow_division_8_26a(int a[], int& r) {
    for (int i = 0; i < 100; i++) {
        a[i] = r + i / 2;
    }
}

static void induction_division_8_26b(int a[], int& r) {
    int Induction = r;
    for (int i = 0; i < 100; i += 2) {
        a[i] = Induction;
        a[i + 1] = Induction;
        Induction++;
    }
}

static void example_8_26() {
    print_banner("8.26a/8.26b: Induction Variable Replacing Slow Division");
    int a1[100], a2[100];
    int r = 10;
    slow_division_8_26a(a1, r);
    induction_division_8_26b(a2, r);
    bool ok = true;
    for (int i = 0; i < 100; i++) {
        if (a1[i] != a2[i]) {
            ok = false;
            break;
        }
    }
    std::cout << "a[0]=" << a1[0] << " a[1]=" << a1[1] << " a[99]=" << a1[99] << "\n";
    std::cout << "induction equivalence: " << (ok ? "PASS" : "FAIL") << "\n";
}

// ============================================================================
// main: run all examples
// ============================================================================
int main() {
    std::cout << "Chapter 8: Compiler Optimizations (编译器中的优化)\n";
    std::cout << "Demonstrating transformations the compiler performs at -O2\n";

    example_8_1();
    example_8_2();
    example_8_3();
    example_8_4();
    example_8_5();
    example_8_6();
    example_8_7();
    example_8_8();
    example_8_9();
    example_8_10();
    example_8_11();
    example_8_12();
    example_8_13();
    example_8_14();
    example_8_15();
    example_8_16();
    example_8_17();
    example_8_18();
    example_8_19();
    example_8_20();
    example_8_21();
    example_8_22();
    example_8_23();
    example_8_24();
    example_8_25();
    example_8_26();

    std::cout << "\nAll examples completed successfully.\n";
    return 0;
}
