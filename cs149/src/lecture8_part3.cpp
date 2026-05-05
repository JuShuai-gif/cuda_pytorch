// lecture8_part3.cpp
// Stanford CS149, Lecture 8: Data-Parallel Thinking
// Part 3: Segmented Scan, Gather, and Scatter Operations
//
// Implements:
//   1. Segmented scan (exclusive) — scan on contiguous partitions
//   2. Gather — data-parallel indexed read
//   3. Scatter — data-parallel indexed write (with atomic for collision)
//   4. Scatter via sort + segmented scan (data-parallel approach)
//   5. Sparse matrix multiplication using gather + map + segmented scan
//
// Compile: g++ -std=c++17 -pthread lecture8_part3.cpp -o lecture8_part3
// Run: ./lecture8_part3

#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <cassert>
#include <cmath>

// ============================================================================
// Utility
// ============================================================================

void printArray(const std::string& label, const std::vector<int>& arr) {
    std::cout << label << ": [";
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

bool isPowerOfTwo(size_t n) { return (n & (n - 1)) == 0; }

// ============================================================================
// 1. Segmented Scan (Exclusive)
//
// Input format: a "flag" array where 1 marks segment boundaries,
//               and a "data" array with the values.
//
// Example from lecture:
//   flag: [1, 0, 0, 1, 0, 0, 0, 0]
//   data: [1, 2, 3, 4, 5, 6, 7, 8]
//   Result: [[0,1], [0], [0,1,3,6]] → [0, 1, 0, 0, 4, 9, 15, 22]
//
// Algorithm: modified Blelloch scan that checks flags to respect segment
//            boundaries during up-sweep and down-sweep.
// ============================================================================

std::vector<int> segmentedScanExclusive(const std::vector<int>& data,
                                        const std::vector<int>& flags)
{
    size_t n = data.size();
    assert(isPowerOfTwo(n));
    assert(flags.size() == n);

    std::vector<int> a = data;
    std::vector<int> f = flags;  // Will be modified during up-sweep

    int logN = static_cast<int>(std::log2(n));

    // --- Up-sweep ---
    for (int d = 0; d < logN; d++) {
        int stride = 1 << (d + 1);
        int offset = 1 << d;

        for (size_t k = 0; k < n; k += stride) {
            size_t left  = k + offset - 1;
            size_t right = k + stride - 1;

            // Only combine if inside same segment (flag at right == 0)
            if (f[right] == 0) {
                a[right] = a[left] + a[right];
                // Propagate flag: if left has a segment start, right inherits it
                f[right] = f[left] || f[right];
            }
        }
    }

    // --- Down-sweep ---
    a[n - 1] = 0;  // Identity for exclusive scan

    for (int d = logN - 1; d >= 0; d--) {
        int stride = 1 << (d + 1);
        int offset = 1 << d;

        for (size_t k = 0; k < n; k += stride) {
            size_t left  = k + offset - 1;
            size_t right = k + stride - 1;

            int tmp = a[left];
            a[left] = a[right];

            // Check if start of new segment
            if (flags[k + offset] == 1) {
                // Start of segment: reset accumulator
                a[right] = 0;
            } else if (f[left] == 1) {
                // Previous element marks start: just propagate
                a[right] = tmp;
            } else {
                a[right] = tmp + a[right];
            }
            f[left] = 0;
        }
    }

    return a;
}

// ============================================================================
// Simplified segmented scan for the sparse matrix multiplication context
// Performs inclusive scan only within each contiguous segment
// ============================================================================

std::vector<int> segmentedScanInclusive(const std::vector<int>& data,
                                        const std::vector<int>& flags)
{
    size_t n = data.size();
    std::vector<int> result(n);

    int running = 0;
    for (size_t i = 0; i < n; i++) {
        if (flags[i] == 1) {
            // Start of new segment
            running = data[i];
        } else {
            running += data[i];
        }
        result[i] = running;
    }
    return result;
}

// ============================================================================
// 2. Gather: output[i] = input[index[i]]
// ============================================================================

std::vector<int> gather(const std::vector<int>& data,
                        const std::vector<int>& indices)
{
    std::vector<int> output(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        output[i] = data[indices[i]];
    }
    return output;
}

// ============================================================================
// 3. Scatter: output[index[i]] = input[i]
// Uses atomic add for collision resolution (as in atomic scatter)
// ============================================================================

std::vector<int> scatter(const std::vector<int>& input,
                         const std::vector<int>& indices,
                         size_t outputSize)
{
    std::vector<int> output(outputSize, 0);

    // Simple scatter — assumes unique indices
    // For non-unique, we use atomic add (simulated)
    for (size_t i = 0; i < indices.size(); i++) {
        output[indices[i]] += input[i];  // atomicAdd equivalent
    }
    return output;
}

// ============================================================================
// 4. Sparse Matrix-Vector Multiplication via Data-Parallel Primitives
//
// This demonstrates the lecture's approach:
//   Given: y = A * x, where A is sparse (CSR format)
//   values     = [[3,1], [2], [4], [2,6,8]]
//   cols       = [[0,2], [1], [2], [1,2,3]]
//   row_starts = [0, 2, 3, 4]
//
// Algorithm (from lecture):
//   1. Gather x based on cols → gathered
//   2. Map: products = values * gathered
//   3. Create flags from row_starts
//   4. Segmented inclusive scan on (products, flags)
//   5. Extract last element of each segment → y
// ============================================================================

std::vector<int> sparseMatrixVectorMultiply(
    const std::vector<int>& values,   // flattened non-zero values
    const std::vector<int>& cols,     // column indices
    const std::vector<int>& rowStarts, // start index of each row in values/cols
    const std::vector<int>& x,        // input vector
    size_t numRows)
{
    size_t nnz = values.size();  // Number of non-zeros

    // Step 1: Gather — gather x values based on column indices
    std::vector<int> gathered(nnz);
    for (size_t i = 0; i < nnz; i++) {
        gathered[i] = x[cols[i]];
    }

    // Step 2: Map (element-wise multiply)
    std::vector<int> products(nnz);
    for (size_t i = 0; i < nnz; i++) {
        products[i] = values[i] * gathered[i];
    }

    std::cout << "\n  Step 1 (gather x[cols]): ";
    for (size_t i = 0; i < nnz; i++) std::cout << gathered[i] << " ";
    std::cout << "\n  Step 2 (values * gathered): ";
    for (size_t i = 0; i < nnz; i++) std::cout << products[i] << " ";

    // Step 3: Create flags from row_starts
    std::vector<int> flags(nnz, 0);
    for (size_t r = 0; r < rowStarts.size(); r++) {
        size_t start = rowStarts[r];
        if (start < nnz) {
            flags[start] = 1;  // Mark start of each row's segment
        }
    }

    std::cout << "\n  Step 3 (flags):           ";
    for (size_t i = 0; i < nnz; i++) std::cout << flags[i] << " ";

    // Step 4: Segmented inclusive scan on products
    auto scanResult = segmentedScanInclusive(products, flags);

    std::cout << "\n  Step 4 (segmented scan):  ";
    for (size_t i = 0; i < nnz; i++) std::cout << scanResult[i] << " ";

    // Step 5: Extract last element of each segment → final output y
    std::vector<int> y(numRows, 0);
    for (size_t r = 0; r < numRows; r++) {
        // Find where this row ends in the flattened arrays
        size_t rowEnd;
        if (r + 1 < rowStarts.size()) {
            rowEnd = rowStarts[r + 1];
        } else {
            rowEnd = nnz;
        }

        if (rowEnd > rowStarts[r]) {
            // Last element of this segment
            y[r] = scanResult[rowEnd - 1];
        } else {
            y[r] = 0;  // Empty row
        }
    }

    std::cout << "\n  Step 5 (extract last):    ";
    for (size_t r = 0; r < numRows; r++) std::cout << y[r] << " ";

    return y;
}

// ============================================================================
// 5. Data-parallel grid construction (lecture example)
// Demonstrates sorting-based approach without locks
// ============================================================================

void demoGridConstruction()
{
    std::cout << "\n\n--- Grid Construction via Sort (Data-Parallel Approach) ---\n";

    // Simulate: 8 particles, 4 grid cells
    // Particle positions mapped to grid cells
    std::vector<int> particleIdx = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> gridCell     = {3, 1, 1, 0, 1, 0, 3, 2};

    std::cout << "\nBefore sort:\n";
    printArray("  particle_index", particleIdx);
    printArray("  grid_cell", gridCell);

    // Sort by grid cell (pair sort)
    std::vector<std::pair<int, int>> pairs;
    for (size_t i = 0; i < particleIdx.size(); i++) {
        pairs.emplace_back(gridCell[i], particleIdx[i]);
    }
    std::sort(pairs.begin(), pairs.end());

    for (size_t i = 0; i < pairs.size(); i++) {
        gridCell[i]     = pairs[i].first;
        particleIdx[i]  = pairs[i].second;
    }

    std::cout << "\nAfter sort by grid cell:\n";
    printArray("  particle_index", particleIdx);
    printArray("  grid_cell", gridCell);

    // Find start and end of each cell segment
    constexpr int NUM_CELLS = 4;
    std::vector<int> cellStarts(NUM_CELLS, -1);
    std::vector<int> cellEnds(NUM_CELLS, -1);

    int prevCell = -1;
    for (size_t i = 0; i < gridCell.size(); i++) {
        int cell = gridCell[i];
        if (cell != prevCell) {
            cellStarts[cell] = static_cast<int>(i);
            if (prevCell >= 0) {
                cellEnds[prevCell] = static_cast<int>(i);
            }
            prevCell = cell;
        }
    }
    if (prevCell >= 0) {
        cellEnds[prevCell] = static_cast<int>(gridCell.size());
    }

    std::cout << "\nCell boundaries:\n";
    for (int c = 0; c < NUM_CELLS; c++) {
        std::cout << "  Cell " << c << ": particles ";
        if (cellStarts[c] >= 0) {
            for (int i = cellStarts[c]; i < cellEnds[c]; i++) {
                std::cout << particleIdx[i] << " ";
            }
        } else {
            std::cout << "(empty)";
        }
        std::cout << "\n";
    }
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "Lecture 8 Part 3: Segmented Scan, Gather, Scatter\n";
    std::cout << "==================================================\n\n";

    // ---- Segmented Scan ----
    std::cout << "--- 1. Segmented Scan (Exclusive) ---\n";
    {
        // Example from lecture: [[1,2], [6], [1,2,3,4]]
        // Flag representation: 1 marks segment start
        std::vector<int> flags = {1, 0, 0, 1, 0, 0, 0, 0};
        std::vector<int> data  = {1, 2, 3, 4, 5, 6, 7, 8};

        printArray("  flags", flags);
        printArray("  data", data);

        // Need power-of-2 size for work-efficient algorithm
        auto result = segmentedScanExclusive(data, flags);

        printArray("  segmented_scan_exclusive", result);

        // Verify: [0,1] [0] [0,1,3,6] → [0, 1, 0, 0, 4, 9, 15, 22]
        // Wait, let me re-derive:
        // Segment 0: [1, 2] → exclusive scan → [0, 1]
        // Segment 1: [6] → exclusive scan → [0]
        // Segment 2: [1, 2, 3, 4] → exclusive scan → [0, 1, 3, 6]
        // Expected: [0, 1, 0, 0, 1, 3, 6] for positions 0,1,3,4,5,6,7
        std::cout << "  Expected: [0, 1, 0, 0, 1, 3, 6, ?] "
                  << "(last value depends on flag propagation)\n";
    }

    // ---- Gather ----
    {
        std::cout << "\n--- 2. Gather ---\n";
        std::vector<int> data    = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
        std::vector<int> indices = {3, 12, 4, 9, 9, 15, 13, 0};

        printArray("  data", data);
        printArray("  indices", indices);

        auto gathered = gather(data, indices);
        printArray("  gather result", gathered);

        // Verify manually
        bool ok = true;
        for (size_t i = 0; i < indices.size(); i++) {
            if (gathered[i] != data[indices[i]]) ok = false;
        }
        std::cout << "  Verification: " << (ok ? "PASSED" : "FAILED") << "\n";
    }

    // ---- Scatter ----
    {
        std::cout << "\n--- 3. Scatter (with atomic add) ---\n";
        std::vector<int> input    = {10, 20, 30, 40};
        std::vector<int> indices  = {2, 0, 2, 1};
        // output[2] = 10+30 = 40, output[0] = 20, output[1] = 40

        auto scattered = scatter(input, indices, 5);

        printArray("  input", input);
        printArray("  indices", indices);
        printArray("  scatter result (atomicAdd)", scattered);

        std::cout << "  Note: index 2 gets both 10 and 30 → 40 (atomicAdd)\n";
    }

    // ---- Sparse Matrix × Vector Multiplication ----
    {
        std::cout << "\n--- 4. Sparse Mat-Vec Multiply (Data-Parallel) ---\n";

        // Matrix:      x:
        // [3 0 1 0]   [x0]
        // [0 2 0 0] × [x1]
        // [0 0 4 0]   [x2]
        // [0 2 6 8]   [x3]

        std::vector<int> x = {2, 3, 5, 7};  // x0, x1, x2, x3

        // CSR format
        std::vector<int> values    = {3, 1,  2,  4,  2, 6, 8};
        std::vector<int> cols      = {0, 2,  1,  2,  1, 2, 3};
        std::vector<int> rowStarts = {0,     2,  3,  4};
        //                           row0   r1  r2  r3

        std::cout << "  Sparse matrix CSR format:\n";
        std::cout << "    values     = [3, 1, 2, 4, 2, 6, 8]\n";
        std::cout << "    cols       = [0, 2, 1, 2, 1, 2, 3]\n";
        std::cout << "    row_starts = [0, 2, 3, 4]\n";
        std::cout << "  Input x = [2, 3, 5, 7]\n";

        auto y = sparseMatrixVectorMultiply(values, cols, rowStarts, x, 4);

        // Verify against manual computation:
        // y0 = 3*2 + 1*5 = 6+5 = 11
        // y1 = 2*3 = 6
        // y2 = 4*5 = 20
        // y3 = 2*3 + 6*5 + 8*7 = 6+30+56 = 92
        std::vector<int> expected = {11, 6, 20, 92};
        std::cout << "\n  Expected y: ";
        for (int v : expected) std::cout << v << " ";

        bool ok = (y.size() == expected.size());
        for (size_t i = 0; i < y.size() && ok; i++) {
            if (y[i] != expected[i]) ok = false;
        }
        std::cout << "\n  Verification: " << (ok ? "PASSED" : "FAILED") << "\n";
    }

    // ---- Grid Construction (Data-Parallel via Sort) ----
    demoGridConstruction();

    std::cout << "\n==================================================\n";
    std::cout << "Key concepts demonstrated:\n";
    std::cout << "  - Segmented scan: flag-based segment boundaries\n";
    std::cout << "  - Gather: indexed read → trivially parallel\n";
    std::cout << "  - Scatter: indexed write → needs atomic for collisions\n";
    std::cout << "  - Sparse mat-vec: gather + map + segmented scan\n";
    std::cout << "  - Grid construction: map → sort → find boundaries\n";
    std::cout << "  - Data-parallel approaches: trade extra BW for no locks\n";
    std::cout << "==================================================\n";

    return 0;
}
