// lecture8_part1.cpp
// Stanford CS149 第8讲：数据并行思维
// 第一部分：Map 和 Reduce（Fold）操作
//
// 实现核心的数据并行原语（primitives）：
//   - map：将函数应用到所有元素（天然可并行化）
//     对每个元素独立执行相同的纯函数，元素之间无依赖，因此
//     可以完美并行。在 GPU 上每个线程处理一个元素。
//   - reduce/fold：使用二元结合运算符组合所有元素
//     通过分治策略并行化：先将数据分块，每块局部归约，
//     再将部分结果合并。要求运算符满足结合律（associative）。
//   - filter：选择满足谓词条件的元素
//     两阶段处理：先统计各块匹配数（确定输出偏移量），
//     再将匹配元素写入正确位置。需要前缀和计算各块的输出偏移。
//   - 数据并行直方图：通过 map → sort → count 实现
//     将模运算映射为 bin ID，排序后连续块即为同一 bin 的元素集合。
//
// 数据并行的核心思想：
//   将问题分解为对大规模数据集合的独立操作（如 map、filter）
//   或可以使用结合律并行化的操作（如 reduce、scan）。
//   关键是识别出可以安全并行执行的计算模式。
//
// 编译：g++ -std=c++17 -pthread lecture8_part1.cpp -o lecture8_part1
// 运行：./lecture8_part1

#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <functional>
#include <chrono>
#include <cassert>

// ============================================================================
// 工具函数：将工作划分为多个块（chunk）用于并行执行
// 上取整除法：chunkSize = ⌈total/numWorkers⌉
// 返回每个 worker 负责的 [start, end) 范围
// ============================================================================

struct WorkRange {
    size_t start;
    size_t end;
};

std::vector<WorkRange> partitionWork(size_t total, size_t numWorkers) {
    std::vector<WorkRange> ranges(numWorkers);
    size_t chunkSize = (total + numWorkers - 1) / numWorkers;  // 上取整
    for (size_t i = 0; i < numWorkers; i++) {
        ranges[i].start = i * chunkSize;
        ranges[i].end   = std::min(ranges[i].start + chunkSize, total);
    }
    return ranges;
}

// ============================================================================
// 1. 并行 Map 操作
// 将一元函数 f 应用到输入序列的每个元素，结果写入输出序列。
// 天然可并行化的原因：f 是纯函数（无副作用），
// 每个元素的计算完全独立，不依赖任何其他元素的结果。
// 在 GPU 上，这直接转化为每个 thread 处理一个元素。
// ============================================================================

template<typename InputIt, typename OutputIt, typename UnaryFunc>
void parallelMap(InputIt first, InputIt last, OutputIt d_first,
                 UnaryFunc f, size_t numThreads)
{
    size_t n = static_cast<size_t>(std::distance(first, last));
    auto ranges = partitionWork(n, numThreads);

    std::vector<std::thread> workers;
    for (const auto& r : ranges) {
        workers.emplace_back([first, d_first, f, r]() {
            for (size_t i = r.start; i < r.end; i++) {
                // 对每个元素独立应用函数 f — 完全无依赖，完美并行
                *(d_first + i) = f(*(first + i));
            }
        });
    }

    for (auto& t : workers) t.join();
}

// ============================================================================
// 2. 并行 Reduce（Fold）操作
// 使用二元结合运算符将所有元素合并为一个结果。
// 采用两阶段方法：
//   阶段一：每个 worker 计算自己负责部分的局部结果
//   阶段二：合并所有部分结果（仅 numThreads 个元素，很快）
//
// 前提条件：运算符 f 必须满足结合律（associative），即
//   f(f(a, b), c) == f(a, f(b, c))，例如加法、乘法、求最大/最小值。
// 非满足结合律的运算（如减法、除法）不能使用此方法。
// ============================================================================

template<typename InputIt, typename T, typename BinaryFunc>
T parallelReduce(InputIt first, InputIt last, T identity, BinaryFunc f,
                 size_t numThreads)
{
    size_t n = static_cast<size_t>(std::distance(first, last));
    auto ranges = partitionWork(n, numThreads);

    // 阶段一：每个 worker 计算其部分结果
    // 每个线程独立累加自己负责的 chunk，没有线程间通信
    std::vector<T> partials(numThreads, identity);
    std::vector<std::thread> workers;

    for (size_t w = 0; w < numThreads; w++) {
        workers.emplace_back([first, f, identity, w, &ranges, &partials]() {
            T local = identity;  // 初始值为单位元
            for (size_t i = ranges[w].start; i < ranges[w].end; i++) {
                local = f(local, *(first + i));  // 累积应用运算符
            }
            partials[w] = local;
        });
    }

    for (auto& t : workers) t.join();

    // 阶段二：合并部分结果（顺序执行 — 只有 numThreads 个元素）
    T result = identity;
    for (size_t w = 0; w < numThreads; w++) {
        result = f(result, partials[w]);
    }
    return result;
}

// ============================================================================
// 3. 并行 Filter 操作
// 筛选满足谓词条件的元素，输出到新的序列中。
// 采用两阶段方法（因为输出大小事先未知）：
//   阶段一：统计各块中匹配元素的数量（用于分配输出空间）
//   阶段二：将匹配元素按偏移量写入正确位置
//
// 阶段一需要前缀和（prefix sum/exclusive scan）来计算每个块的输出起始偏移量，
// 这是数据并行中常见的模式：先确定写入位置，再并行写入。
// ============================================================================

template<typename InputIt, typename OutputIt, typename Predicate>
size_t parallelFilter(InputIt first, InputIt last, OutputIt d_first,
                      Predicate pred, size_t numThreads)
{
    size_t n = static_cast<size_t>(std::distance(first, last));
    auto ranges = partitionWork(n, numThreads);

    // 阶段一：计算每个块中匹配元素个数的前缀和，以确定输出偏移量
    std::vector<size_t> matchCounts(numThreads, 0);
    std::vector<std::thread> workers;

    for (size_t w = 0; w < numThreads; w++) {
        workers.emplace_back([first, pred, w, &ranges, &matchCounts]() {
            size_t count = 0;
            for (size_t i = ranges[w].start; i < ranges[w].end; i++) {
                if (pred(*(first + i))) count++;
            }
            matchCounts[w] = count;
        });
    }
    for (auto& t : workers) t.join();

    // 计算各块输出的偏移量（matchCounts 的互斥前缀和/exclusive prefix sum）
    // 例如 matchCounts = [3, 2, 4] → offsets = [0, 3, 5]
    std::vector<size_t> offsets(numThreads, 0);
    for (size_t w = 1; w < numThreads; w++) {
        offsets[w] = offsets[w - 1] + matchCounts[w - 1];
    }
    size_t totalMatches = offsets.back() + matchCounts.back();

    // 阶段二：将匹配元素在计算好的偏移量处写入输出
    // 每个 worker 从自己负责的偏移量开始顺序写入
    workers.clear();
    for (size_t w = 0; w < numThreads; w++) {
        workers.emplace_back([first, d_first, pred, w, &ranges, &offsets]() {
            size_t pos = offsets[w];  // 该块的输出起始位置
            for (size_t i = ranges[w].start; i < ranges[w].end; i++) {
                if (pred(*(first + i))) {
                    *(d_first + pos) = *(first + i);
                    pos++;
                }
            }
        });
    }
    for (auto& t : workers) t.join();

    return totalMatches;
}

// ============================================================================
// 4. 数据并行直方图：通过 Map + Sort 实现
//
// 算法步骤：
//   1. Map：将每个元素映射为对应的 bin 编号（使用取模运算）
//   2. Sort：按 bin 编号排序（同 bin 的元素连续排列）
//   3. Count：统计每段连续 bin 的元素数量
//
// 与使用 atomicAdd 的直方图不同，这种方法避免了原子操作冲突，
// 但代价是需要排序（O(N log N)）。在 GPU 上，如果原子操作竞争严重，
// 排序方法可能反而更快。
// ============================================================================

std::vector<int> parallelHistogram(const std::vector<int>& data,
                                   int numBins, size_t numThreads)
{
    size_t n = data.size();

    // 步骤 1：Map — 把每个元素映射到对应的 bin 编号
    std::vector<int> binIds(n);
    parallelMap(data.begin(), data.end(), binIds.begin(),
                [numBins](int v) {
                    int bin = v % numBins;
                    return (bin < 0) ? bin + numBins : bin;  // 处理负数
                },
                numThreads);

    // 步骤 2：排序 binIds
    // 在真实的 GPU 实现中，会使用高效的并行排序算法（如 radix sort）
    // 此处为了简化使用 std::sort（顺序排序）
    std::vector<int> sortedBinIds = binIds;
    std::sort(sortedBinIds.begin(), sortedBinIds.end());

    // 步骤 3：统计每个 bin 的元素数量（利用已排序数据）
    // 排序后相同 bin 的元素连续排列，只需扫描一次即可完成统计
    std::vector<int> histogram(numBins, 0);
    for (size_t i = 0; i < sortedBinIds.size(); i++) {
        histogram[sortedBinIds[i]]++;
    }

    return histogram;
}

// ============================================================================
// 演示函数
// ============================================================================

void demoMap()
{
    std::cout << "--- 1. 并行 Map ---\n";

    std::vector<int> input  = {3, 8, 4, 6, 3, 9, 2, 8};
    std::vector<int> output(input.size());

    // f(x) = x + 10（与课堂示例相同）
    // 每个元素独立加 10，无依赖关系，完美并行
    parallelMap(input.begin(), input.end(), output.begin(),
                [](int x) { return x + 10; }, 4);

    std::cout << "输入：      ";
    for (int v : input)  std::cout << v << " ";
    std::cout << "\n";
    std::cout << "map +10 后：";
    for (int v : output) std::cout << v << " ";
    std::cout << "\n";

    // 使用 std::transform 验证结果
    std::vector<int> expected(input.size());
    std::transform(input.begin(), input.end(), expected.begin(),
                   [](int x) { return x + 10; });
    bool ok = (output == expected);
    std::cout << "验证结果：" << (ok ? "通过（PASSED）" : "失败（FAILED）") << "\n\n";
}

void demoReduce()
{
    std::cout << "--- 2. 并行 Reduce（Fold）---\n";

    std::vector<int> data = {3, 8, 4, 6, 3, 9, 2, 8};

    // fold 10 (+) data = 10+3+8+4+6+3+9+2+8 = 53
    // 初始值 10 是加法的单位元，所有元素通过加法结合合并
    int result = parallelReduce(data.begin(), data.end(), 10,
                                std::plus<int>(), 4);

    std::cout << "数据：";
    for (int v : data) std::cout << v << " ";
    std::cout << "\n";
    std::cout << "fold 10 (+) data = " << result << "\n";

    // 验证
    int expected = 10 + std::accumulate(data.begin(), data.end(), 0);
    std::cout << "验证结果：" << (result == expected ? "通过（PASSED）" : "失败（FAILED）")
              << "（期望值 " << expected << "）\n\n";
}

void demoFilter()
{
    std::cout << "--- 3. 并行 Filter ---\n";

    std::vector<int> data = {3, 8, 4, 6, 3, 9, 2, 8};
    std::vector<int> output(data.size());

    // 仅保留偶数（x % 2 == 0）
    size_t count = parallelFilter(data.begin(), data.end(), output.begin(),
                                  [](int x) { return x % 2 == 0; }, 4);
    output.resize(count);  // 调整输出大小到实际匹配数量

    std::cout << "输入：       ";
    for (int v : data) std::cout << v << " ";
    std::cout << "\n";
    std::cout << "filter 偶数：";
    for (size_t i = 0; i < count; i++) std::cout << output[i] << " ";
    std::cout << "\n";
    std::cout << "（过滤掉了 " << data.size() - count << " 个元素）\n\n";
}

void demoHistogram()
{
    std::cout << "--- 4. 并行直方图（Map + Sort 方法）---\n";

    std::vector<int> data = {0, 3, 4, 1, 9, 2, 8, 4, 1, 7,
                             5, 6, 2, 3, 9, 0, 1, 5, 8, 4};
    constexpr int NUM_BINS = 10;

    auto hist = parallelHistogram(data, NUM_BINS, 4);

    std::cout << "数据：";
    for (int v : data) std::cout << v << " ";
    std::cout << "\n直方图（通过 map+sort 的数据并行方法计算）：\n";
    for (int b = 0; b < NUM_BINS; b++) {
        std::cout << "  bin[" << b << "]：" << hist[b] << "  ";
        for (int i = 0; i < hist[b]; i++) std::cout << "#";
        std::cout << "\n";
    }

    // 直接统计验证
    bool ok = true;
    for (int b = 0; b < NUM_BINS; b++) {
        int expected = std::count_if(data.begin(), data.end(),
                                     [b](int v) { return v % NUM_BINS == b; });
        if (hist[b] != expected) ok = false;
    }
    std::cout << "验证结果：" << (ok ? "通过（PASSED）" : "失败（FAILED）") << "\n\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "第8讲 第一部分：Map、Reduce、Filter、Histogram\n";
    std::cout << "==================================================\n\n";

    demoMap();
    demoReduce();
    demoFilter();
    demoHistogram();

    std::cout << "==================================================\n";
    std::cout << "演示的核心概念：\n";
    std::cout << "  - map：无副作用的纯函数 → 天然可并行化\n";
    std::cout << "  - reduce：结合运算符合并 → 先局部计算再合并\n";
    std::cout << "  - filter：两阶段方法（计数+前缀和确定偏移量）\n";
    std::cout << "  - histogram：map f → sort → count（适合并行执行）\n";
    std::cout << "==================================================\n";

    return 0;
}
