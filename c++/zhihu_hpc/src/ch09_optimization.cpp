// Chapter 9: 优化内存访问 (Optimizing Memory Access)
// Consolidated examples: 9.1a, 9.1b, 9.2a, 9.2b, 9.4, 9.5a, 9.5b, 9.6a, 9.6b
//
// Compile: g++ -std=c++11 -O2 -msse2 ch09_optimization.cpp -o ch09_optimization

#include <xmmintrin.h>  // SSE/MMX intrinsics (_mm_stream_pi, _mm_empty)
#include <cstring>      // std::memset
#include <chrono>       // std::chrono::high_resolution_clock
#include <iostream>     // std::cout, std::endl
#include <iomanip>      // std::setw
#include <cstdlib>      // std::srand, std::rand
#include <ctime>        // std::time

// ============================================================================
// Simple high-resolution timer
// ============================================================================
class Timer {
public:
    void start() { _start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - _start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point _start;
};

// ============================================================================
// Helper function used in Example 9.1a and 9.1b
// ============================================================================
int Func(int x) {
    return x * 3 + 1;
}

// ============================================================================
// Global data for Example 9.1a and 9.1b
// ============================================================================
const int SIZE_9_1 = 1024;

// 9.1a: Separate arrays (Structure of Arrays / SoA)
int a_s[SIZE_9_1];
int b_s[SIZE_9_1];

// 9.1b: Struct of arrays (Array of Structures / AoS)
struct Sab {
    int a;
    int b;
};
Sab ab[SIZE_9_1];

// ============================================================================
// Example 9.1a: Separate arrays - each field has its own contiguous array.
// This is cache-friendly when only one field is accessed at a time.
// ============================================================================
double example_9_1a() {
    // Initialize array a_s
    for (int i = 0; i < SIZE_9_1; i++) {
        a_s[i] = i;
    }

    Timer t;
    t.start();

    // Access pattern: sequential reads from a_s, sequential writes to b_s.
    // Only the 'a' values are in cache; 'b' values are separate.
    for (int i = 0; i < SIZE_9_1; i++) {
        b_s[i] = Func(a_s[i]);
    }

    double elapsed = t.elapsed_ms();

    // Prevent compiler from optimizing away the loop
    volatile int sink = 0;
    for (int i = 0; i < SIZE_9_1; i++) {
        sink += b_s[i];
    }
    (void)sink;

    return elapsed;
}

// ============================================================================
// Example 9.1b: Struct of arrays - each element contains both fields together.
// This can waste cache bandwidth when only one field is accessed.
// ============================================================================
double example_9_1b() {
    // Initialize struct array
    for (int i = 0; i < SIZE_9_1; i++) {
        ab[i].a = i;
    }

    Timer t;
    t.start();

    // Access pattern: each struct occupies two ints in memory.
    // When reading ab[i].a, the cache line also loads ab[i].b (unused).
    // This wastes half the cache bandwidth compared to separate arrays.
    for (int i = 0; i < SIZE_9_1; i++) {
        ab[i].b = Func(ab[i].a);
    }

    double elapsed = t.elapsed_ms();

    // Prevent compiler from optimizing away the loop
    volatile int sink = 0;
    for (int i = 0; i < SIZE_9_1; i++) {
        sink += ab[i].b;
    }
    (void)sink;

    return elapsed;
}

// ============================================================================
// Helper functions for Example 9.2a and 9.2b
// ============================================================================
void F1(int x[]) {
    for (int i = 0; i < 1000; i++) {
        x[i] = i * 2;
    }
}

void F2(float x[]) {
    for (int i = 0; i < 1000; i++) {
        x[i] = static_cast<float>(i) * 0.5f;
    }
}

// ============================================================================
// Example 9.2a: Separate local arrays in conditional branches.
// Both int a[1000] and float b[1000] are allocated on the stack,
// even though only one is used. Total stack waste: 1000*4 + 1000*4 = 8000 bytes.
// ============================================================================
void example_9_2a(bool y) {
    if (y) {
        int a[1000];  // allocated on stack regardless of condition
        F1(a);
    } else {
        float b[1000];  // allocated on stack regardless of condition
        F2(b);
    }
}

// ============================================================================
// Example 9.2b: Union to save stack space.
// The union ensures only max(sizeof(int[1000]), sizeof(float[1000])) = 4000 bytes
// are allocated on the stack instead of 8000 bytes.
// ============================================================================
void example_9_2b(bool y) {
    union {
        int a[1000];
        float b[1000];
    };
    if (y) {
        F1(a);
    } else {
        F2(b);
    }
}

// ============================================================================
// Example 9.4: Row-major traversal of a 2D matrix.
// In C/C++, 2D arrays are stored in row-major order.
// Traversing by rows first (outer loop: row, inner loop: column) ensures
// sequential memory access, maximizing cache line utilization.
// ============================================================================
double example_9_4() {
    const int NUMROWS = 100;
    const int NUMCOLUMNS = 100;
    int matrix[NUMROWS][NUMCOLUMNS];

    Timer t;
    t.start();

    // Row-major traversal: outer loop over rows, inner loop over columns.
    // This accesses matrix elements sequentially in memory.
    for (int row = 0; row < NUMROWS; row++) {
        for (int column = 0; column < NUMCOLUMNS; column++) {
            matrix[row][column] = row + column;
        }
    }

    double elapsed = t.elapsed_ms();

    // Prevent compiler from optimizing away
    volatile int sink = 0;
    for (int row = 0; row < NUMROWS; row++) {
        for (int column = 0; column < NUMCOLUMNS; column++) {
            sink += matrix[row][column];
        }
    }
    (void)sink;

    return elapsed;
}

// ============================================================================
// Example 9.5a: Simple matrix transpose (in-place).
// Uses a straightforward double loop swapping elements below the diagonal.
// Size: 64x64 to keep the demonstration fast.
// ============================================================================
const int SIZE_9_5A = 64;

void transpose_simple(double a[SIZE_9_5A][SIZE_9_5A]) {
#define swapd_simple(x, y) \
    {                      \
        double temp = x;   \
        x = y;             \
        y = temp;          \
    }
    for (int r = 1; r < SIZE_9_5A; r++) {
        for (int c = 0; c < r; c++) {
            swapd_simple(a[r][c], a[c][r]);
        }
    }
#undef swapd_simple
}

double example_9_5a() {
    // Allocate matrix aligned to cache line boundary (64 bytes)
    double matrix[SIZE_9_5A][SIZE_9_5A] __attribute__((aligned(64)));

    // Initialize matrix with deterministic values
    for (int r = 0; r < SIZE_9_5A; r++) {
        for (int c = 0; c < SIZE_9_5A; c++) {
            matrix[r][c] = static_cast<double>(r * SIZE_9_5A + c);
        }
    }

    Timer t;
    t.start();
    transpose_simple(matrix);
    double elapsed = t.elapsed_ms();

    // Prevent compiler from optimizing away
    volatile double sink = 0.0;
    for (int r = 0; r < SIZE_9_5A; r++) {
        for (int c = 0; c < SIZE_9_5A; c++) {
            sink += matrix[r][c];
        }
    }
    (void)sink;

    return elapsed;
}

// ============================================================================
// Example 9.5b: Tiled (blocked) matrix transpose to avoid cache contention.
// When the matrix size is a multiple of the cache size (critical stride),
// simple transpose suffers from cache line evictions. Tiling breaks the
// operation into smaller blocks that fit in cache.
// Uses SIZE=512 to trigger the tiled path (SIZE > 256 && SIZE % 128 == 0).
// ============================================================================
const int SIZE_9_5B = 512;

void transpose_tiled(double a[SIZE_9_5B][SIZE_9_5B]) {
#define swapd_tiled(x, y) \
    {                     \
        double temp = x;  \
        x = y;            \
        y = temp;         \
    }

    // Check if level-2 cache contention will occur:
    if (SIZE_9_5B > 256 && SIZE_9_5B % 128 == 0) {
        // Cache contention expected. Use square tiling (blocking):
        const int TILESIZE = 8;  // TILESIZE must divide SIZE evenly

        // Process off-diagonal tiles
        for (int r1 = 0; r1 < SIZE_9_5B; r1 += TILESIZE) {
            for (int c1 = 0; c1 < r1; c1 += TILESIZE) {
                // Swap elements inside the (r1, c1) tile block
                for (int r2 = r1; r2 < r1 + TILESIZE; r2++) {
                    for (int c2 = c1; c2 < c1 + TILESIZE; c2++) {
                        swapd_tiled(a[r2][c2], a[c2][r2]);
                    }
                }
            }

            // Handle the half-square at the diagonal
            for (int r2 = r1 + 1; r2 < r1 + TILESIZE; r2++) {
                for (int c2 = r1; c2 < r2; c2++) {
                    swapd_tiled(a[r2][c2], a[c2][r2]);
                }
            }
        }
    } else {
        // No cache contention. Use the simple method from 9.5a.
        for (int r = 1; r < SIZE_9_5B; r++) {
            for (int c = 0; c < r; c++) {
                swapd_tiled(a[r][c], a[c][r]);
            }
        }
    }
#undef swapd_tiled
}

double example_9_5b() {
    // Use dynamic allocation for the large matrix (512x512 doubles = 2 MB)
    // to avoid stack overflow. alignas ensures cache-line alignment.
    double (*matrix)[SIZE_9_5B] = new double[SIZE_9_5B][SIZE_9_5B]();

    // Initialize matrix
    for (int r = 0; r < SIZE_9_5B; r++) {
        for (int c = 0; c < SIZE_9_5B; c++) {
            matrix[r][c] = static_cast<double>(r * SIZE_9_5B + c);
        }
    }

    Timer t;
    t.start();
    transpose_tiled(matrix);
    double elapsed = t.elapsed_ms();

    // Prevent compiler from optimizing away
    volatile double sink = 0.0;
    for (int r = 0; r < SIZE_9_5B; r++) {
        for (int c = 0; c < SIZE_9_5B; c++) {
            sink += matrix[r][c];
        }
    }
    (void)sink;

    delete[] matrix;
    return elapsed;
}

// ============================================================================
// Example 9.6a: Transpose and copy matrix (normal load/store).
// Reads from b in row-major order, writes transposed to a.
// Both source and destination are accessed, potentially polluting cache.
// ============================================================================
const int SIZE_9_6 = 512;

void transpose_copy_normal(double a[SIZE_9_6][SIZE_9_6], double b[SIZE_9_6][SIZE_9_6]) {
    for (int r = 0; r < SIZE_9_6; r++) {
        for (int c = 0; c < SIZE_9_6; c++) {
            a[c][r] = b[r][c];
        }
    }
}

double example_9_6a() {
    double (*a)[SIZE_9_6] = new double[SIZE_9_6][SIZE_9_6]();
    double (*b)[SIZE_9_6] = new double[SIZE_9_6][SIZE_9_6]();

    // Initialize source matrix b
    for (int r = 0; r < SIZE_9_6; r++) {
        for (int c = 0; c < SIZE_9_6; c++) {
            b[r][c] = static_cast<double>(r * SIZE_9_6 + c);
        }
    }

    Timer t;
    t.start();
    transpose_copy_normal(a, b);
    double elapsed = t.elapsed_ms();

    // Prevent compiler from optimizing away
    volatile double sink = 0.0;
    for (int r = 0; r < SIZE_9_6; r++) {
        for (int c = 0; c < SIZE_9_6; c++) {
            sink += a[r][c];
        }
    }
    (void)sink;

    delete[] a;
    delete[] b;
    return elapsed;
}

// ============================================================================
// Example 9.6b: Transpose and copy using non-temporal store (MOVNTQ).
// _mm_stream_pi writes 64 bits (one double) directly to memory,
// bypassing the cache hierarchy. This avoids polluting the cache with
// the destination matrix, leaving more cache for the source data.
// ============================================================================

// Non-temporal store for a single double using MMX MOVNTQ instruction.
// The data is written directly to memory without loading the cache line first.
static inline void StoreNTD(double* dest, double const& source) {
    _mm_stream_pi(reinterpret_cast<__m64*>(dest), *reinterpret_cast<__m64 const*>(&source));
    _mm_empty();  // Clear MMX state before returning to x87 FPU
}

void transpose_copy_nt(double a[SIZE_9_6][SIZE_9_6], double b[SIZE_9_6][SIZE_9_6]) {
    for (int r = 0; r < SIZE_9_6; r++) {
        for (int c = 0; c < SIZE_9_6; c++) {
            StoreNTD(&a[c][r], b[r][c]);
        }
    }
}

double example_9_6b() {
    double (*a)[SIZE_9_6] = new double[SIZE_9_6][SIZE_9_6]();
    double (*b)[SIZE_9_6] = new double[SIZE_9_6][SIZE_9_6]();

    // Initialize source matrix b
    for (int r = 0; r < SIZE_9_6; r++) {
        for (int c = 0; c < SIZE_9_6; c++) {
            b[r][c] = static_cast<double>(r * SIZE_9_6 + c);
        }
    }

    Timer t;
    t.start();
    transpose_copy_nt(a, b);
    double elapsed = t.elapsed_ms();

    // Prevent compiler from optimizing away
    volatile double sink = 0.0;
    for (int r = 0; r < SIZE_9_6; r++) {
        for (int c = 0; c < SIZE_9_6; c++) {
            sink += a[r][c];
        }
    }
    (void)sink;

    delete[] a;
    delete[] b;
    return elapsed;
}

// ============================================================================
// Main: run all examples with timing comparisons
// ============================================================================
int main() {
    std::cout << "=====================================================" << std::endl;
    std::cout << "  Chapter 9: 优化内存访问 (Optimizing Memory Access)" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << std::endl;

    // ---- Example 9.1: AoS vs SoA ----
    std::cout << "--- Example 9.1: AoS vs SoA ---" << std::endl;
    std::cout << "  Array size: " << SIZE_9_1 << " elements" << std::endl;

    double t_9_1a = example_9_1a();
    std::cout << "  9.1a (Separate arrays / SoA): " << t_9_1a << " ms" << std::endl;

    double t_9_1b = example_9_1b();
    std::cout << "  9.1b (Struct of arrays / AoS): " << t_9_1b << " ms" << std::endl;

    if (t_9_1b > 0.0) {
        std::cout << "  Ratio (AoS/SoA): " << (t_9_1b / t_9_1a) << "x" << std::endl;
    }
    std::cout << std::endl;

    // ---- Example 9.2: Stack space with union ----
    std::cout << "--- Example 9.2: Stack space optimization with union ---" << std::endl;
    std::cout << "  Running 9.2a (branch with separate arrays):" << std::endl;
    example_9_2a(true);
    example_9_2a(false);
    std::cout << "  Running 9.2b (branch with union):" << std::endl;
    example_9_2b(true);
    example_9_2b(false);
    std::cout << "  9.2a allocates 8000 bytes on stack (int[1000] + float[1000])." << std::endl;
    std::cout << "  9.2b allocates 4000 bytes on stack (union)." << std::endl;
    std::cout << std::endl;

    // ---- Example 9.4: Row-major traversal ----
    std::cout << "--- Example 9.4: Row-major traversal ---" << std::endl;
    std::cout << "  Matrix: 100x100 int" << std::endl;
    double t_9_4 = example_9_4();
    std::cout << "  Row-major fill time: " << t_9_4 << " ms" << std::endl;
    std::cout << std::endl;

    // ---- Example 9.5a: Simple transpose ----
    std::cout << "--- Example 9.5a: Simple matrix transpose ---" << std::endl;
    std::cout << "  Matrix: " << SIZE_9_5A << "x" << SIZE_9_5A << " double" << std::endl;
    double t_9_5a = example_9_5a();
    std::cout << "  Simple transpose time: " << t_9_5a << " ms" << std::endl;
    std::cout << std::endl;

    // ---- Example 9.5b: Tiled transpose ----
    std::cout << "--- Example 9.5b: Tiled matrix transpose ---" << std::endl;
    std::cout << "  Matrix: " << SIZE_9_5B << "x" << SIZE_9_5B << " double" << std::endl;
    std::cout << "  Condition: SIZE > 256 && SIZE % 128 == 0 => tiled path active" << std::endl;
    double t_9_5b = example_9_5b();
    std::cout << "  Tiled transpose time: " << t_9_5b << " ms" << std::endl;
    std::cout << std::endl;

    // ---- Example 9.6a: Normal transpose-copy ----
    std::cout << "--- Example 9.6a: Transpose-copy (normal) ---" << std::endl;
    std::cout << "  Matrix: " << SIZE_9_6 << "x" << SIZE_9_6 << " double" << std::endl;
    double t_9_6a = example_9_6a();
    std::cout << "  Normal store time: " << t_9_6a << " ms" << std::endl;
    std::cout << std::endl;

    // ---- Example 9.6b: Non-temporal transpose-copy ----
    std::cout << "--- Example 9.6b: Transpose-copy (non-temporal store) ---" << std::endl;
    std::cout << "  Matrix: " << SIZE_9_6 << "x" << SIZE_9_6 << " double" << std::endl;
    std::cout << "  Using _mm_stream_pi (MOVNTQ) to bypass cache" << std::endl;
    double t_9_6b = example_9_6b();
    std::cout << "  Non-temporal store time: " << t_9_6b << " ms" << std::endl;

    if (t_9_6a > 0.0) {
        std::cout << "  Ratio (NT/normal): " << (t_9_6b / t_9_6a) << "x" << std::endl;
    }
    std::cout << std::endl;

    // ---- Summary ----
    std::cout << "=====================================================" << std::endl;
    std::cout << "  All examples completed successfully." << std::endl;
    std::cout << "=====================================================" << std::endl;

    return 0;
}
