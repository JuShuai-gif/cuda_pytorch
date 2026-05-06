// lecture8_part2.cpp
// Stanford CS149 第8讲：数据并行思维
// 第二部分：并行 Scan（前缀和 / Prefix Sum）算法
//
// 实现课堂讨论的五种 Scan（前缀和）变体：
//
//   1. 顺序 Scan（基线，O(N) 工作量，O(N) span）
//      朴素顺序方法：output[i] = output[i-1] + input[i]。
//      简单但不可并行化，span 为 O(N)。
//
//   2. 朴素并行包含 Scan（O(N log N) 工作量，O(log N) span）
//      每次迭代 stride 翻倍：a[k] = a[k-2^d] + a[k]（当 k ≥ 2^d 时）。
//      虽然工作量大，但 SIMD 利用率高，适合 warp 级并行。
//
//   3. 工作高效的互斥 Scan（Blelloch 算法）（O(N) 工作量，O(log N) span）
//      分两个阶段：
//        - Up-sweep（上扫）：构建二叉树，每个内部节点存储子节点之和
//        - Down-sweep（下扫）：从根向下传播前缀和
//      这是最优的并行 scan 算法，工作量和顺序算法同级。
//
//   4. SIMD 风格的 Warp Scan（O(N log N) 工作量，小 N 表现更优）
//      虽然工作量更大，但在 CUDA warp 内部（32 个线程）执行时
//      SIMD 利用率更高，因为每步的 control flow 在 warp 内是统一的。
//
//   5. 多核 Scan（partition + 顺序 scan + 添加基数）
//      课堂展示的 2+ 核心方法。~1.5N 工作量，具有完美的空间局部性。
//      先各自顺序 scan，再合并 partial sums 作为后续块的基数。
//
// 为什么 Scan 是核心原语：
//   Scan 看似是顺序操作，但可以通过分治策略高效并行化。
//   它是许多高级数据并行算法的基础构建块（如 filter、segmented scan、
//   稀疏矩阵乘法中的 scatter 等）。
//
// 编译：g++ -std=c++17 -pthread lecture8_part2.cpp -o lecture8_part2
// 运行：./lecture8_part2

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cassert>

// ============================================================================
// 工具函数：检查 N 是否为 2 的幂
// Blelloch 工作高效算法要求输入大小为 2 的幂
// 使用位运算技巧：(n & (n-1)) == 0 当且仅当 n 是 2 的幂
// ============================================================================

bool isPowerOfTwo(size_t n) { return (n & (n - 1)) == 0; }

// 返回不小于 n 的最小 2 的幂（用于填充数组）
// 例如 nextPowerOfTwo(5) = 8, nextPowerOfTwo(12) = 16
size_t nextPowerOfTwo(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;  // 持续左移直到 ≥ n
    return p;
}

// 计算 floor(log2(n))，即 n 的二进制最高位的位置
// 例如 ilog2(8)=3, ilog2(15)=3, ilog2(1)=0
int ilog2(size_t n) {
    int log = 0;
    while (n >>= 1) log++;
    return log;
}

// ============================================================================
// 数组打印辅助函数
// ============================================================================

void printArray(const std::string& label, const std::vector<int>& arr) {
    std::cout << label << "：[";
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// ============================================================================
// 1. 顺序 Scan（基线实现）
//
// 包含扫描（Inclusive Scan）：output[i] = sum(input[0..i])
//   output[0] = input[0]
//   output[i] = output[i-1] + input[i]  （i > 0）
//
// 互斥扫描（Exclusive Scan）：output[i] = sum(input[0..i-1])
//   output[0] = 0
//   output[i] = output[i-1] + input[i-1]  （i > 0）
//
// 互斥扫描是更多算法的构建基础（如 filter 的偏移量计算）。
// ============================================================================

std::vector<int> sequentialScan(const std::vector<int>& input) {
    std::vector<int> output(input.size());
    if (input.empty()) return output;

    output[0] = input[0];  // 包含扫描：第一个元素即自身
    for (size_t i = 1; i < input.size(); i++) {
        output[i] = output[i - 1] + input[i];  // 利用前一个已累加的结果
    }
    return output;
}

std::vector<int> sequentialScanExclusive(const std::vector<int>& input) {
    std::vector<int> output(input.size());
    if (input.empty()) return output;

    output[0] = 0;  // 互斥扫描的第一个元素始终是 0（单位元）
    int running = 0;
    for (size_t i = 0; i < input.size() - 1; i++) {
        running += input[i];
        output[i + 1] = running;
    }
    return output;
}

// ============================================================================
// 2. 朴素并行包含 Scan（O(N log N) 工作量，O(log N) span）
// 课堂幻灯片算法：每一步将 stride 翻倍
//
// 伪代码：
//   for d = 0 to log2(N)-1:
//     forall k in parallel: if k >= 2^d:
//       a[k] = a[k-2^d] + a[k]
//
// 关键洞察：第 d 轮后，位置 k 存储的是 a[k-2^d+1..k] 的和。
// 经过 d 轮的传播，a[k] 最终包含 a[0..k] 的全部和。
//
// 虽然总工作量比顺序扫描多（O(N log N) vs O(N)），
// 但在 SIMD 硬件上，每一步内的所有加法都可以并行执行，
// 并且 control flow 在每个 warp 内是统一的（无分支发散）。
// ============================================================================

void naiveParallelScan(std::vector<int>& a) {
    size_t n = a.size();
    if (n == 0) return;

    for (int d = 0; d < ilog2(n); d++) {
        int stride = 1 << d;  // 2^d：当前阶段的偏移距离

        // 并行 forall k — 使用多线程模拟
        int numThreads = static_cast<int>(n) / 2;
        numThreads = std::max(1, std::min(numThreads, 8));
        int chunkSize = (static_cast<int>(n) + numThreads - 1) / numThreads;

        std::vector<std::thread> workers;
        for (int w = 0; w < numThreads; w++) {
            int start = w * chunkSize;
            int end   = std::min(start + chunkSize, static_cast<int>(n));

            workers.emplace_back([&a, stride, start, end]() {
                for (int k = start; k < end; k++) {
                    // 只有距离足够远的元素才需要相加
                    if (k >= stride) {
                        a[k] = a[k - stride] + a[k];
                    }
                }
            });
        }
        for (auto& t : workers) t.join();
    }
}

// ============================================================================
// 3. 工作高效的互斥 Scan（Blelloch 算法）
// O(N) 工作量，O(log N) span — 理论上最优的并行 scan 算法
//
// Up-sweep（上扫 / Reduce 阶段）— 构建部分和树：
//   将相邻元素配对求和，重复此过程直到得到一个总的"和"。
//   仿照：for d=0 to log2(N)-1:
//           forall k=0 to N-1 by 2^(d+1):
//             a[k + 2^(d+1) - 1] = a[k + 2^d - 1] + a[k + 2^(d+1) - 1]
//
// Down-sweep（下扫 / 分发阶段）— 向下传播前缀和：
//   将根节点的总和逐步分发到各子节点。
//   a[N-1] = 0
//   for d=log2(N)-1 down to 0:
//     forall k=0 to N-1 by 2^(d+1):
//       tmp = a[k + 2^d - 1]
//       a[k + 2^d - 1] = a[k + 2^(d+1) - 1]
//       a[k + 2^(d+1) - 1] = tmp + a[k + 2^(d+1) - 1]
//
// 直观理解：Up-sweep 构建一棵二叉树，叶子存储原数据，
// 内部节点存储子树和。Down-sweep 从根开始，将前缀和逐层
// 向下传递到叶子节点。
// ============================================================================

void workEfficientScan(std::vector<int>& a) {
    size_t n = a.size();
    if (n < 2) return;
    assert(isPowerOfTwo(n));  // 确保输入大小为 2 的幂

    int logN = ilog2(n);

    // --- Up-sweep（上扫 / 归约阶段）---
    for (int d = 0; d < logN; d++) {
        int stride   = 1 << (d + 1);  // 2^(d+1)：当前组的跨度
        int offset   = 1 << d;         // 2^d：左子节点距离左边界的位置

        std::vector<std::thread> workers;
        for (size_t k = 0; k < n; k += stride) {
            workers.emplace_back([&a, k, offset, stride, n]() {
                (void)n;
                size_t left  = k + offset - 1;   // 左子节点
                size_t right = k + stride - 1;    // 右子节点（父节点位置）
                a[right] = a[left] + a[right];    // 左+右 → 父
            });
        }
        for (auto& t : workers) t.join();
    }

    // --- Down-sweep（下扫 / 分发阶段）---
    a[n - 1] = 0;  // 将最后一个元素置为恒等元（互斥扫描），
                   // 这对应于构建树根节点的前缀和为 0

    for (int d = logN - 1; d >= 0; d--) {
        int stride   = 1 << (d + 1);
        int offset   = 1 << d;

        std::vector<std::thread> workers;
        for (size_t k = 0; k < n; k += stride) {
            workers.emplace_back([&a, k, offset, stride]() {
                size_t left  = k + offset - 1;
                size_t right = k + stride - 1;
                int tmp      = a[left];            // 保存左子节点的值
                a[left]      = a[right];           // 左子接收来自父的前缀和
                a[right]     = tmp + a[right];     // 右子 = 原左子 + 父前缀和
            });
        }
        for (auto& t : workers) t.join();
    }
}

// ============================================================================
// 4. SIMD 风格的 Warp Scan（朴素 O(N log N)，但 SIMD 利用率更好）
//
// 这是在 CUDA warp 内部用于 32 元素 scan 的版本。
// 尽管工作量更大（O(N log N) vs O(N)），但它更好地映射到 SIMD 硬件，
// 因为每一步的 control flow 在 warp 内部是统一的，没有分支发散。
//
// 在 GPU 上，warp 内部的 32 个 lane 共享一个程序计数器（PC），
// 因此我们使用类似朴素并行 scan 的指数步长方法，
// 确保所有 32 个 lane 在每个阶段执行相同的指令。
// ============================================================================

std::vector<int> warpScan(const std::vector<int>& input) {
    size_t n = input.size();
    std::vector<int> ptr = input;  // 原地计算（in-place）
    std::vector<int> result(n, 0);

    int steps = ilog2(n);  // 对于 n=32，steps=5（2^5=32）

    // 朴素并行 scan：每一步 stride 翻倍
    for (int i = 0; i < steps; i++) {
        int shift = 1 << i;  // 2^i
        for (size_t idx = 0; idx < n; idx++) {
            int lane = static_cast<int>(idx);
            if (lane >= shift) {
                ptr[idx] = ptr[idx - shift] + ptr[idx];
            }
        }
    }

    // 从包含扫描结果提取互斥扫描结果
    // 互斥扫描：result[0] = 0, result[i] = inclusive[i-1]
    for (size_t i = 0; i < n; i++) {
        result[i] = (i > 0) ? ptr[i - 1] : 0;
    }

    return result;
}

// ============================================================================
// 5. 多核 Scan（partition + 顺序 scan + 添加基数）
//
// 这是课堂展示的 2+ 核心方法。约 1.5N 工作量，具有完美的空间局部性。
//
// 算法步骤：
//   步骤 1（并行）：每个核心对自己负责的 chunk 执行顺序包含扫描
//   步骤 2（顺序）：计算 partial sums 的互斥前缀和作为各 chunk 的基数
//   步骤 3（并行）：将基数加到各 chunk 的所有元素上（除第一个 chunk）
//
// 直观理解：先各自扫描每个 chunk，然后计算"前面的 chunk 总和"
//（基数），再将基数加到后续 chunk 的每个元素上。
// ============================================================================

std::vector<int> multiCoreScan(const std::vector<int>& input,
                               size_t numWorkers)
{
    size_t n = input.size();
    std::vector<int> output(n, 0);

    if (numWorkers == 0) numWorkers = 1;

    // 分区：将数组等分为 numWorkers 块
    size_t chunkSize = (n + numWorkers - 1) / numWorkers;

    // 步骤 1：每个 worker 对自己负责的 chunk 执行顺序包含扫描
    std::vector<int> partialSums(numWorkers, 0);

    // 处理最后一个 chunk 可能较小的情况
    std::vector<size_t> chunkStarts(numWorkers);
    std::vector<size_t> chunkSizes(numWorkers);
    size_t pos = 0;
    for (size_t w = 0; w < numWorkers; w++) {
        chunkStarts[w] = pos;
        chunkSizes[w]  = (pos + chunkSize <= n) ? chunkSize : (n > pos ? n - pos : 0);
        pos += chunkSizes[w];
    }

    std::vector<std::thread> workers;
    for (size_t w = 0; w < numWorkers; w++) {
        workers.emplace_back([&input, &output, &partialSums, &chunkStarts,
                              &chunkSizes, w]() {
            size_t start = chunkStarts[w];
            size_t size  = chunkSizes[w];
            if (size == 0) return;

            // 在 chunk 内执行顺序包含扫描
            output[start] = input[start];
            for (size_t i = 1; i < size; i++) {
                output[start + i] = output[start + i - 1] + input[start + i];
            }
            partialSums[w] = output[start + size - 1];  // 该 chunk 的总和
        });
    }
    for (auto& t : workers) t.join();

    // 步骤 2：计算各 chunk 的基数（partial sums 的互斥前缀和）
    // bases[w] = 前 w 个 chunk 的所有元素之和
    std::vector<int> bases(numWorkers, 0);
    int runningBase = 0;
    for (size_t w = 0; w < numWorkers; w++) {
        bases[w]   = runningBase;
        runningBase += partialSums[w];
    }

    // 步骤 3：将基数加到各 chunk 的每个元素上（第一个 chunk 的基数为 0）
    workers.clear();
    for (size_t w = 1; w < numWorkers; w++) {
        workers.emplace_back([&output, &bases, &chunkStarts, &chunkSizes, w]() {
            size_t start = chunkStarts[w];
            size_t size  = chunkSizes[w];
            for (size_t i = 0; i < size; i++) {
                output[start + i] += bases[w];  // 每个元素加上基数
            }
        });
    }
    for (auto& t : workers) t.join();

    return output;
}

// ============================================================================
// 验证函数：逐元素比较结果与期望值
// ============================================================================

bool verify(const std::string& name,
            const std::vector<int>& result,
            const std::vector<int>& expected)
{
    if (result.size() != expected.size()) {
        std::cout << "  " << name << "：失败（FAILED）— 大小不匹配\n";
        return false;
    }
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != expected[i]) {
            std::cout << "  " << name << "：失败（FAILED）— 索引 " << i
                      << " 处不匹配（得到 " << result[i] << "，期望 " << expected[i] << "）\n";
            return false;
        }
    }
    std::cout << "  " << name << "：通过（PASSED）\n";
    return true;
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "第8讲 第二部分：并行 Scan（前缀和）算法\n";
    std::cout << "==================================================\n\n";

    // 测试数据
    std::vector<int> data = {3, 8, 4, 6, 3, 9, 2, 8};
    // 期望包含扫描结果：[3, 11, 15, 21, 24, 33, 35, 43]
    // 期望互斥扫描结果：[0, 3, 11, 15, 21, 24, 33, 35]

    printArray("输入数组", data);

    // 1. 顺序基线
    auto seqInclusive = sequentialScan(data);
    auto seqExclusive = sequentialScanExclusive(data);
    std::cout << "\n--- 基线：顺序扫描 ---\n";
    printArray("  包含扫描", seqInclusive);
    printArray("  互斥扫描", seqExclusive);

    // 2. 朴素并行扫描（O(N log N)）
    {
        std::cout << "\n--- 朴素并行扫描（O(N log N)）---\n";
        std::vector<int> naive = data;
        naiveParallelScan(naive);
        printArray("  结果", naive);
        verify("朴素包含扫描", naive, seqInclusive);
    }

    // 3. 工作高效扫描（O(N)）— 要求输入大小为 2 的幂
    {
        std::cout << "\n--- 工作高效扫描（Blelloch，O(N)）---\n";
        // 填充到 2 的幂
        size_t paddedN = nextPowerOfTwo(data.size());
        std::vector<int> padded(paddedN, 0);
        std::copy(data.begin(), data.end(), padded.begin());

        workEfficientScan(padded);

        // 提取原始大小部分的结果（后面是填充的 0，忽略）
        std::vector<int> blelloch(data.size());
        std::copy(padded.begin(), padded.begin() + data.size(),
                  blelloch.begin());
        printArray("  结果", blelloch);
        verify("Blelloch 互斥扫描", blelloch, seqExclusive);
    }

    // 4. Warp 扫描（SIMD 风格）
    {
        std::cout << "\n--- Warp 扫描（SIMD 风格，适用于小 N）---\n";
        // 使用 32 个元素进行 warp 扫描（warp 大小）
        std::vector<int> warpData(32, 0);
        for (size_t i = 0; i < data.size(); i++) warpData[i] = data[i];

        auto warpResult = warpScan(warpData);

        // 仅显示相关部分
        std::vector<int> warpSubset(data.size());
        std::copy(warpResult.begin(), warpResult.begin() + data.size(),
                  warpSubset.begin());
        printArray("  结果（互斥扫描）", warpSubset);

        // 生成填充后 warp 数据的期望互斥扫描结果
        std::vector<int> warpExpected(warpData.size());
        if (!warpData.empty()) {
            warpExpected[0] = 0;
            int running = 0;
            for (size_t i = 0; i < warpData.size() - 1; i++) {
                running += warpData[i];
                warpExpected[i + 1] = running;
            }
        }
        std::vector<int> warpExpectedSubset(data.size());
        std::copy(warpExpected.begin(), warpExpected.begin() + data.size(),
                  warpExpectedSubset.begin());
        verify("Warp 互斥扫描", warpSubset, warpExpectedSubset);
    }

    // 5. 多核扫描
    {
        std::cout << "\n--- 多核扫描（分区 + 合并）---\n";
        size_t numWorkers = 3;

        auto mcResult = multiCoreScan(data, numWorkers);
        printArray("  包含扫描（" + std::to_string(numWorkers) + " 个核心）", mcResult);
        verify("多核包含扫描", mcResult, seqInclusive);
    }

    // 6. 大规模数组上的性能对比
    {
        std::cout << "\n--- 大规模数组扫描（N=2^20 = " << (1 << 20) << "）---\n";
        size_t largeN = 1 << 20;  // 1,048,576
        std::vector<int> largeData(largeN);
        // 全 1 数组：方便验证扫描后第 i 个位置的值应为 i+1
        for (size_t i = 0; i < largeN; i++) largeData[i] = 1;

        // 顺序扫描
        auto t0 = std::chrono::high_resolution_clock::now();
        auto largeSeq = sequentialScan(largeData);
        auto t1 = std::chrono::high_resolution_clock::now();
        double timeSeq = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 工作高效扫描（Blelloch）
        std::vector<int> blellochData = largeData;
        auto t2 = std::chrono::high_resolution_clock::now();
        workEfficientScan(blellochData);
        auto t3 = std::chrono::high_resolution_clock::now();
        double timeBlelloch = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // 多核扫描（8 个 worker）
        auto t4 = std::chrono::high_resolution_clock::now();
        auto largeMC = multiCoreScan(largeData, 8);
        auto t5 = std::chrono::high_resolution_clock::now();
        double timeMC = std::chrono::duration<double, std::milli>(t5 - t4).count();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  顺序扫描：          " << timeSeq << " 毫秒\n";
        std::cout << "  工作高效扫描：      " << timeBlelloch << " 毫秒\n";
        std::cout << "  多核扫描（8 核）：  " << timeMC << " 毫秒\n";

        // 验证多核扫描的正确性
        bool mcOk = true;
        for (size_t i = 0; i < largeN && mcOk; i++) {
            if (largeMC[i] != largeSeq[i]) mcOk = false;
        }
        std::cout << "  多核扫描正确性：    " << (mcOk ? "通过（PASSED）" : "失败（FAILED）") << "\n";
    }

    std::cout << "\n==================================================\n";
    std::cout << "演示的核心概念：\n";
    std::cout << "  - 朴素扫描：O(N log N) 工作量，SIMD 利用率更好\n";
    std::cout << "  - Blelloch 扫描：O(N) 工作量，up-sweep + down-sweep\n";
    std::cout << "  - Warp 扫描：SIMD 友好，适合小数组（32 元素）\n";
    std::cout << "  - 多核扫描：partition + 顺序 scan + 添加基数\n";
    std::cout << "  - 不同级别的硬件采用不同的并行策略\n";
    std::cout << "==================================================\n";

    return 0;
}
