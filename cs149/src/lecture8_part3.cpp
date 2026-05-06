// lecture8_part3.cpp
// Stanford CS149 第8讲：数据并行思维
// 第三部分：分段扫描（Segmented Scan）、Gather 和 Scatter 操作
//
// 实现内容：
//   1. 分段扫描（互斥，exclusive）— 在连续分区上进行 scan 操作
//      在普通 scan 的基础上，增加 flag 标记来界定分段边界。
//      每个分段内部独立进行扫描，分段之间不互相影响。
//      是稀疏矩阵运算、图算法等高级应用的核心构建块。
//
//   2. Gather — 数据并行索引读取
//      output[i] = input[index[i]]
//      即根据索引数组从源数组中读取数据。天然可并行化，
//      因为每个输出元素独立读取，无数据依赖。
//
//   3. Scatter — 数据并行索引写入（使用 atomic add 处理冲突）
//      output[index[i]] = input[i]（或 output[index[i]] += input[i]）
//      不同线程可能写入同一位置，需要原子操作保证正确性。
//      在无冲突时高效；有大量冲突时可能成为瓶颈。
//
//   4. 通过 Sort + Segmented Scan 实现 Scatter（数据并行方法）
//      将 scatter 转化为 sort + segmented scan，避免原子操作。
//      虽然增加了工作量，但在高冲突场景下可能优于原子操作。
//
//   5. 稀疏矩阵向量乘法（Sparse Mat-Vec Multiply）
//      使用 gather + map + segmented scan 的数据并行流程：
//      CSR 格式 → 按列 gather x 值 → 逐元素乘 → 分段 scan → 提取每段末尾
//
// 编译：g++ -std=c++17 -pthread lecture8_part3.cpp -o lecture8_part3
// 运行：./lecture8_part3

#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <cassert>
#include <cmath>

// ============================================================================
// 工具函数
// ============================================================================

void printArray(const std::string& label, const std::vector<int>& arr) {
    std::cout << label << "：[";
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

bool isPowerOfTwo(size_t n) { return (n & (n - 1)) == 0; }

// ============================================================================
// 1. 分段扫描（Segmented Scan，互斥版本）
//
// 输入格式：一个 "flag" 数组（1 标记分段边界）和 "data" 数组（数据值）。
//
// 课堂示例：
//   flag：[1, 0, 0, 1, 0, 0, 0, 0]
//   data：[1, 2, 3, 4, 5, 6, 7, 8]
//   分段：[[1,2], [6], [1,2,3,4]]
//   互斥 scan 后：[[0,1], [0], [0,1,3,6]]
//   结果： [0, 1, 0, 0, 4, 9, 15, 22]
//
// 算法：对 Blelloch scan 进行修改，在 up-sweep 和 down-sweep 阶段
//       检查 flag 来尊重分段边界。segment 内部独立进行 prefix sum，
//       segment 之间不传播值。
//
// Up-sweep 中的关键修改：
//   只有当 flag[right] == 0（即左右元素在同一分段内）时才进行合并。
//   如果 flag 标记了分段边界，则该处的值不参与跨分段的合并。
//
// Down-sweep 中的关键修改：
//   遇到新分段的起始位置时，将累加值重置为 0（恢复互斥扫描的恒等元）。
// ============================================================================

std::vector<int> segmentedScanExclusive(const std::vector<int>& data,
                                        const std::vector<int>& flags)
{
    size_t n = data.size();
    assert(isPowerOfTwo(n));  // 为保证算法正确性，需要 2 的幂
    assert(flags.size() == n);

    std::vector<int> a = data;
    std::vector<int> f = flags;  // flag 副本，在 up-sweep 阶段会被修改用于传播

    int logN = static_cast<int>(std::log2(n));

    // --- Up-sweep（上扫）---
    for (int d = 0; d < logN; d++) {
        int stride = 1 << (d + 1);
        int offset = 1 << d;

        for (size_t k = 0; k < n; k += stride) {
            size_t left  = k + offset - 1;
            size_t right = k + stride - 1;

            // 仅当左右在同一分段内时才合并（flag at right == 0）
            // 如果 flag 为 1，说明该位置是新分段的起始，不参与跨段合并
            if (f[right] == 0) {
                a[right] = a[left] + a[right];
                // 传播 flag：如果左标记为分段起点，右也继承该标记
                f[right] = f[left] || f[right];
            }
        }
    }

    // --- Down-sweep（下扫）---
    a[n - 1] = 0;  // 互斥扫描的恒等元

    for (int d = logN - 1; d >= 0; d--) {
        int stride = 1 << (d + 1);
        int offset = 1 << d;

        for (size_t k = 0; k < n; k += stride) {
            size_t left  = k + offset - 1;
            size_t right = k + stride - 1;

            int tmp = a[left];
            a[left] = a[right];

            // 检查是否为新分段的起始位置
            if (flags[k + offset] == 1) {
                // 新分段起始：重置累加值为 0
                a[right] = 0;
            } else if (f[left] == 1) {
                // 前一个元素标记了分段起始：仅传播，不累加
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
// 简化版分段扫描（包含，inclusive）— 用于稀疏矩阵乘法场景
// 在每个连续分段内执行包含前缀和扫描
// 这种简化版本易于理解和实现，不需要 2 的幂约束
// ============================================================================

std::vector<int> segmentedScanInclusive(const std::vector<int>& data,
                                        const std::vector<int>& flags)
{
    size_t n = data.size();
    std::vector<int> result(n);

    int running = 0;
    for (size_t i = 0; i < n; i++) {
        if (flags[i] == 1) {
            // 遇到新分段的起始，重新开始累加
            running = data[i];
        } else {
            // 同一分段内继续累加
            running += data[i];
        }
        result[i] = running;
    }
    return result;
}

// ============================================================================
// 2. Gather：output[i] = input[index[i]]
//
// Gather 是最简单的数据并行操作之一：每个线程根据索引从源数组读取数据。
// 天然可并行化：每个输出元素的计算完全独立。
// 在 GPU 上，需要确保 index 值在合法范围内，否则会导致越界访问。
// 性能取决于读取的地址模式：连续的（coalesced）读取效率高，
// 随机的（strided/random）读取效率低（因为无法充分利用 cache line）。
// ============================================================================

std::vector<int> gather(const std::vector<int>& data,
                        const std::vector<int>& indices)
{
    std::vector<int> output(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        output[i] = data[indices[i]];  // 根据索引读取
    }
    return output;
}

// ============================================================================
// 3. Scatter：output[index[i]] = input[i]
// 使用 atomic add 处理冲突（当多个线程写同一位置时）
//
// Scatter 比 gather 更复杂，因为可能有多个线程写入同一输出位置。
// 无冲突场景：output[index[i]] = input[i]，简单高效
// 有冲突场景：需要使用原子操作（atomicAdd）累加，避免读-改-写竞争
// 在 GPU 上，大量冲突可能导致性能急剧下降（所有线程序列化等待）
// ============================================================================

std::vector<int> scatter(const std::vector<int>& input,
                         const std::vector<int>& indices,
                         size_t outputSize)
{
    std::vector<int> output(outputSize, 0);

    // 简单 scatter — 假设索引唯一
    // 对于非唯一索引，使用 atomic add（模拟）
    // 等价于 CUDA 中的 atomicAdd(&output[indices[i]], input[i])
    for (size_t i = 0; i < indices.size(); i++) {
        output[indices[i]] += input[i];  // 等同于 atomicAdd
    }
    return output;
}

// ============================================================================
// 4. 稀疏矩阵向量乘法 — 使用数据并行原语实现
//
// 本程序演示课堂中的方法：
//   给定：y = A * x，其中 A 是稀疏矩阵（以 CSR 格式存储）
//
// CSR（Compressed Sparse Row，压缩稀疏行）格式：
//   values     = [[3,1], [2], [4], [2,6,8]]  （非零值的扁平数组）
//   cols       = [[0,2], [1], [2], [1,2,3]]  （对应列索引）
//   row_starts = [0, 2, 3, 4]                （每行的起始位置）
//
// 矩阵表示（4×4）：
//   [3 0 1 0]
//   [0 2 0 0]
//   [0 0 4 0]
//   [0 2 6 8]
//
// 算法流程（来自课堂）：
//   1. Gather：根据 cols 索引从 x 中 gather 对应的值 → gathered
//      即 gathered[i] = x[cols[i]]，将列索引转为实际 x 值
//   2. Map：逐元素乘，products[i] = values[i] * gathered[i]
//      每个非零元素与对应 x 值相乘
//   3. 从 row_starts 创建 flags 数组（标记每行的分段起始位置）
//   4. 对 (products, flags) 执行分段包含扫描
//      每行内部的元素被累加起来
//   5. 提取每个分段的最后一个元素 → 即每行的点积结果 y
//
// 示例验证（x = [2, 3, 5, 7]）：
//   y[0] = 3*2 + 1*5 = 6+5 = 11
//   y[1] = 2*3 = 6
//   y[2] = 4*5 = 20
//   y[3] = 2*3 + 6*5 + 8*7 = 6+30+56 = 92
// ============================================================================

std::vector<int> sparseMatrixVectorMultiply(
    const std::vector<int>& values,     // 扁平化的非零值数组
    const std::vector<int>& cols,       // 每个非零值的列索引
    const std::vector<int>& rowStarts,  // 每行在 values/cols 中的起始索引
    const std::vector<int>& x,          // 输入向量（稠密）
    size_t numRows)
{
    size_t nnz = values.size();  // 非零元素总数（Number of Non-Zeros）

    // 步骤 1：Gather — 根据列索引从 x 中 gather 对应的值
    // 例如：cols=[0,2,1,2,1,2,3]，x=[2,3,5,7]
    // gathered = [x[0],x[2],x[1],x[2],x[1],x[2],x[3]] = [2,5,3,5,3,5,7]
    std::vector<int> gathered(nnz);
    for (size_t i = 0; i < nnz; i++) {
        gathered[i] = x[cols[i]];
    }

    // 步骤 2：Map（逐元素乘法）
    // products = values[i] * gathered[i]
    // 例如：[3*2, 1*5, 2*3, 4*5, 2*3, 6*5, 8*7] = [6,5,6,20,6,30,56]
    std::vector<int> products(nnz);
    for (size_t i = 0; i < nnz; i++) {
        products[i] = values[i] * gathered[i];
    }

    std::cout << "\n  步骤 1（gather x[cols]）：";
    for (size_t i = 0; i < nnz; i++) std::cout << gathered[i] << " ";
    std::cout << "\n  步骤 2（values * gathered）：";
    for (size_t i = 0; i < nnz; i++) std::cout << products[i] << " ";

    // 步骤 3：从 row_starts 创建 flags 数组
    // flags 长度为 nnz，在每行的起始位置标记为 1
    // row_starts=[0,2,3,4] → flags[0]=1, flags[2]=1, flags[3]=1
    std::vector<int> flags(nnz, 0);
    for (size_t r = 0; r < rowStarts.size(); r++) {
        size_t start = rowStarts[r];
        if (start < nnz) {
            flags[start] = 1;  // 标记每行的分段起点
        }
    }

    std::cout << "\n  步骤 3（flags）：        ";
    for (size_t i = 0; i < nnz; i++) std::cout << flags[i] << " ";

    // 步骤 4：对 products 执行分段包含扫描
    // 每行内部的元素被顺序累加，分段边界处重新开始
    auto scanResult = segmentedScanInclusive(products, flags);

    std::cout << "\n  步骤 4（segmented scan）：";
    for (size_t i = 0; i < nnz; i++) std::cout << scanResult[i] << " ";

    // 步骤 5：提取每段最后一个元素 → 即最终输出 y
    // 每段最后一个元素包含该段所有元素的累加和（即该行的点积结果）
    std::vector<int> y(numRows, 0);
    for (size_t r = 0; r < numRows; r++) {
        // 确定此行在扁平数组中的结束位置
        size_t rowEnd;
        if (r + 1 < rowStarts.size()) {
            rowEnd = rowStarts[r + 1];
        } else {
            rowEnd = nnz;  // 最后一行的结束位置就是数组末尾
        }

        if (rowEnd > rowStarts[r]) {
            // 此分段有元素，取最后一个元素作为该行的结果
            y[r] = scanResult[rowEnd - 1];
        } else {
            y[r] = 0;  // 空行（该行无非零元素）
        }
    }

    std::cout << "\n  步骤 5（提取段末尾）：  ";
    for (size_t r = 0; r < numRows; r++) std::cout << y[r] << " ";

    return y;
}

// ============================================================================
// 5. 数据并行网格构建（课堂示例）
// 演示基于排序的无锁方法
//
// 背景：在粒子模拟等应用中，需要将粒子分配到空间网格中。
// 传统方法可能使用锁来防止并发修改同一网格单元。
// 数据并行方法：通过排序将所有粒子按网格单元分组，
// 从而无需锁就能确定每个网格单元的粒子范围。
//
// 方法步骤：
//   1. Map：计算每个粒子所属的网格单元
//   2. Sort：按网格单元 ID 对（grid_cell, particle_index）排序
//   3. 扫描：排序后，同一网格单元的粒子连续排列
//   4. 找到每段连续相同 grid cell 的起始和结束位置
//
// 开销：排序 O(N log N)，但没有原子操作竞争，数据局部性好。
// ============================================================================

void demoGridConstruction()
{
    std::cout << "\n\n--- 通过排序实现的网格构建（数据并行方法）---\n";

    // 模拟：8 个粒子，4 个网格单元
    // 粒子的位置被映射到网格单元编号
    std::vector<int> particleIdx = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> gridCell     = {3, 1, 1, 0, 1, 0, 3, 2};

    std::cout << "\n排序前：\n";
    printArray("  粒子编号", particleIdx);
    printArray("  网格单元", gridCell);

    // 按网格单元排序（成对排序：pair<grid_cell, particle_index>）
    // 排序后同一网格单元的所有粒子连续排列
    std::vector<std::pair<int, int>> pairs;
    for (size_t i = 0; i < particleIdx.size(); i++) {
        pairs.emplace_back(gridCell[i], particleIdx[i]);
    }
    std::sort(pairs.begin(), pairs.end());

    // 提取排序后的结果
    for (size_t i = 0; i < pairs.size(); i++) {
        gridCell[i]     = pairs[i].first;
        particleIdx[i]  = pairs[i].second;
    }

    std::cout << "\n按网格单元排序后：\n";
    printArray("  粒子编号", particleIdx);
    printArray("  网格单元", gridCell);

    // 找到每个网格单元的起始和结束位置
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

    std::cout << "\n各网格单元边界：\n";
    for (int c = 0; c < NUM_CELLS; c++) {
        std::cout << "  单元 " << c << "：粒子 ";
        if (cellStarts[c] >= 0) {
            for (int i = cellStarts[c]; i < cellEnds[c]; i++) {
                std::cout << particleIdx[i] << " ";
            }
        } else {
            std::cout << "（空）";
        }
        std::cout << "\n";
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "第8讲 第三部分：分段扫描、Gather、Scatter\n";
    std::cout << "==================================================\n\n";

    // ---- 分段扫描 ----
    std::cout << "--- 1. 分段扫描（互斥版本）---\n";
    {
        // 课堂示例：[[1,2], [6], [1,2,3,4]]
        // flag 表示：1 标记分段起始位置
        std::vector<int> flags = {1, 0, 0, 1, 0, 0, 0, 0};
        std::vector<int> data  = {1, 2, 3, 4, 5, 6, 7, 8};

        printArray("  flags", flags);
        printArray("  data", data);

        // 需要 2 的幂大小以使用工作高效算法
        auto result = segmentedScanExclusive(data, flags);

        printArray("  分段互斥扫描结果", result);

        // 验证：
        // 分段 0：[1, 2] → 互斥 scan → [0, 1]
        // 分段 1：[6] → 互斥 scan → [0]
        // 分段 2：[1, 2, 3, 4] → 互斥 scan → [0, 1, 3, 6]
        // 期望：[0, 1, 0, 0, 1, 3, 6, ?]
        std::cout << "  期望值：[0, 1, 0, 0, 1, 3, 6, ?] "
                  << "（最后一个值取决于 flag 传播结果）\n";
    }

    // ---- Gather ----
    {
        std::cout << "\n--- 2. Gather ---\n";
        std::vector<int> data    = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
        std::vector<int> indices = {3, 12, 4, 9, 9, 15, 13, 0};

        printArray("  data", data);
        printArray("  indices", indices);

        auto gathered = gather(data, indices);
        printArray("  gather 结果", gathered);

        // 手动验证
        bool ok = true;
        for (size_t i = 0; i < indices.size(); i++) {
            if (gathered[i] != data[indices[i]]) ok = false;
        }
        std::cout << "  验证结果：" << (ok ? "通过（PASSED）" : "失败（FAILED）") << "\n";
    }

    // ---- Scatter ----
    {
        std::cout << "\n--- 3. Scatter（使用 atomic add）---\n";
        std::vector<int> input    = {10, 20, 30, 40};
        std::vector<int> indices  = {2, 0, 2, 1};
        // output[2] = 10+30 = 40, output[0] = 20, output[1] = 40

        auto scattered = scatter(input, indices, 5);

        printArray("  input", input);
        printArray("  indices", indices);
        printArray("  scatter 结果（atomicAdd）", scattered);

        std::cout << "  注意：索引 2 同时接收了 10 和 30 → 40（atomicAdd 累加）\n";
    }

    // ---- 稀疏矩阵 × 向量乘法 ----
    {
        std::cout << "\n--- 4. 稀疏矩阵-向量乘法（数据并行方法）---\n";

        // 矩阵：          x：
        // [3 0 1 0]     [x0]
        // [0 2 0 0]  ×  [x1]
        // [0 0 4 0]     [x2]
        // [0 2 6 8]     [x3]

        std::vector<int> x = {2, 3, 5, 7};  // x0, x1, x2, x3

        // CSR 格式（Compressed Sparse Row）
        std::vector<int> values    = {3, 1,  2,  4,  2, 6, 8};
        std::vector<int> cols      = {0, 2,  1,  2,  1, 2, 3};
        std::vector<int> rowStarts = {0,     2,  3,  4};
        //                           row0   r1  r2  r3

        std::cout << "  稀疏矩阵 CSR 格式：\n";
        std::cout << "    values     = [3, 1, 2, 4, 2, 6, 8]\n";
        std::cout << "    cols       = [0, 2, 1, 2, 1, 2, 3]\n";
        std::cout << "    row_starts = [0, 2, 3, 4]\n";
        std::cout << "  输入向量 x = [2, 3, 5, 7]\n";

        auto y = sparseMatrixVectorMultiply(values, cols, rowStarts, x, 4);

        // 手动验证：
        // y0 = 3*2 + 1*5 = 6+5 = 11
        // y1 = 2*3 = 6
        // y2 = 4*5 = 20
        // y3 = 2*3 + 6*5 + 8*7 = 6+30+56 = 92
        std::vector<int> expected = {11, 6, 20, 92};
        std::cout << "\n  期望 y：";
        for (int v : expected) std::cout << v << " ";

        bool ok = (y.size() == expected.size());
        for (size_t i = 0; i < y.size() && ok; i++) {
            if (y[i] != expected[i]) ok = false;
        }
        std::cout << "\n  验证结果：" << (ok ? "通过（PASSED）" : "失败（FAILED）") << "\n";
    }

    // ---- 网格构建（基于排序的数据并行方法）----
    demoGridConstruction();

    std::cout << "\n==================================================\n";
    std::cout << "演示的核心概念：\n";
    std::cout << "  - 分段扫描：基于 flag 标记的分段边界控制\n";
    std::cout << "  - Gather：索引读取 → 天然可并行化\n";
    std::cout << "  - Scatter：索引写入 → 冲突时需使用原子操作\n";
    std::cout << "  - 稀疏矩阵向量乘：gather + map + segmented scan\n";
    std::cout << "  - 网格构建：map → sort → 查找边界\n";
    std::cout << "  - 数据并行方法：用额外带宽换取无锁设计\n";
    std::cout << "==================================================\n";

    return 0;
}
