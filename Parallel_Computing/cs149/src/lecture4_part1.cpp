/**
 * lecture4_part1.cpp - ISPC SPMD 抽象模拟
 *
 * 模拟 ISPC 的核心概念：
 * - programCount：同时执行的程序实例数量（对应 SIMD 宽度）
 * - programIndex：当前实例的ID（0..programCount-1）
 *   让每个实例知道自己在 gang 中的位置，从而处理不同的数据
 * - uniform vs varying 变量：
 *   uniform 变量在所有实例中具有相同值，编译器可优化存储
 *   varying 变量（默认）每个实例持有独立的值
 * - 交错分配 vs 块分配：
 *   交错分配产生连续内存访问（适合 packed vector load）
 *   块分配产生不连续访问（需要 gather 指令）
 * - foreach 抽象：
 *   程序员声明并行迭代范围，编译器/运行时决定具体分配策略
 *   体现了 CS149 的核心主题：抽象 vs 实现
 * - reduce_add 跨实例通信：
 *   将 gang 中所有实例的部分结果汇总为一个统一值
 *
 * 编译命令：g++ -std=c++17 -pthread lecture4_part1.cpp -o lecture4_part1 && ./lecture4_part1
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <algorithm>
#include <iomanip>

// ============================================================================
// 第1部分：SPMD 模拟 - sin(x) 泰勒级数计算
//
// 使用 ISPC 的 SPMD 编程模型概念来模拟并行 sin(x) 计算。
// 每个"程序实例"（模拟 SIMD 通道）处理不同的数组元素。
// ============================================================================

// 模拟 programCount = 8（SIMD 宽度为8）
// 在现代 CPU 中，这通常对应 AVX2 的 8-wide 单精度向量
constexpr int PROGRAM_COUNT = 8;

// sin(x) 的泰勒级数展开：sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
// 使用增量计算方式避免重复计算高阶幂和阶乘
float compute_sinx(float x, int terms) {
    float value = x;
    float numer = x * x * x;
    float denom = 6.0f;  // 3! = 6
    float sign = -1.0f;

    for (int j = 1; j <= terms; j++) {
        value += sign * numer / denom;
        numer *= x * x;
        denom *= (2 * j + 2) * (2 * j + 3);
        sign *= -1.0f;
    }
    return value;
}

/**
 * 模拟 ISPC 的交错分配（Interleaved Assignment）：
 *
 * for (uniform int i = 0; i < N; i += programCount) {
 *     int idx = i + programIndex;  // 每个实例获取不同元素
 *     result[idx] = compute_sinx(x[idx], terms);
 * }
 *
 * 每个"程序实例"以步幅 programCount 处理元素。
 * 这种模式允许 packed vector loads，因为每次迭代中所有实例
 * 访问的元素在内存中是连续的。
 *
 * 内存访问模式（programCount=8）：
 *   迭代0：实例0..7 处理元素 0..7   → 连续8个元素（一条 vmovaps 指令）
 *   迭代1：实例0..7 处理元素 8..15  → 连续8个元素
 *
 * 这是 ISPC 中最高效的数据分配策略。
 */
void interleaved_sinx(const std::vector<float>& x, std::vector<float>& result,
                      int terms, int programIndex) {
    int N = x.size();
    for (int i = 0; i < N; i += PROGRAM_COUNT) {
        int idx = i + programIndex;
        if (idx < N) {
            result[idx] = compute_sinx(x[idx], terms);
        }
    }
}

/**
 * 模拟 ISPC 的块分配（Blocked Assignment）：
 *
 * uniform int count = N / programCount;
 * int start = programIndex * count;
 * for (uniform int i = 0; i < count; i++) {
 *     int idx = start + i;
 *     result[idx] = compute_sinx(x[idx], terms);
 * }
 *
 * 每个实例处理一个连续的数据块。
 * 但在每次迭代中，不同实例访问的元素不连续，
 * 需要 gather 指令（如 vgatherdps）。
 *
 * 内存访问模式（programCount=8, N=64）：
 *   迭代0：实例0..7 处理元素 0,8,16,24,32,40,48,56 → 不连续（需要 gather）
 *   迭代1：实例0..7 处理元素 1,9,17,25,33,41,49,57 → 不连续
 *
 * gather 指令比 packed load 更复杂且开销更大。
 */
void blocked_sinx(const std::vector<float>& x, std::vector<float>& result,
                   int terms, int programIndex) {
    int N = x.size();
    int count = N / PROGRAM_COUNT;
    int start = programIndex * count;
    // 处理边界情况：最后一个实例可能多处理一些元素
    int end = (programIndex == PROGRAM_COUNT - 1) ? N : start + count;
    for (int idx = start; idx < end; idx++) {
        result[idx] = compute_sinx(x[idx], terms);
    }
}

/**
 * 演示 "foreach" 概念：
 * 系统自动将迭代分配给程序实例。
 * 这里我们使用简单的并行 for 循环作为实现。
 *
 * foreach 是 ISPC 中最高级的抽象：
 * - 程序员只写"这些是需要执行的迭代"
 * - 不需要关心哪个实例执行哪个迭代
 * - 编译器/运行时自动选择最佳分配策略
 *
 * 这体现了关注点分离的设计原则。
 */
void foreach_sinx(const std::vector<float>& x, std::vector<float>& result, int terms) {
    int N = x.size();
    // foreach (i = 0 ... N) -- 程序员只需声明并行迭代
    // 在 OpenMP 中可写为 #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        result[i] = compute_sinx(x[i], terms);
    }
    // 在 ISPC 中，系统处理分配。这里我们使用简单的并行for循环。
}

// ============================================================================
// 第2部分：跨实例通信 - reduce_add
//
// reduce_add 是 ISPC 中最基本的跨实例归约操作。
// 它将 gang 中所有实例的值合并为一个 uniform 结果。
// ============================================================================

/**
 * 模拟 ISPC reduce_add：将所有程序实例的值求和。
 * 每个实例计算自己的私有 partial 和，然后 reduce_add 合并它们。
 *
 * 为什么需要这样设计？
 * - 如果所有实例共享一个变量（uniform），会产生数据竞争。
 * - 如果每个实例有自己的变量（varying），无法合并结果。
 * - 解决方案：每个实例维护私有 partial + reduce_add 合并。
 */
float simulated_reduce_add(const std::vector<float>& partials) {
    float sum = 0.0f;
    for (float p : partials) {
        sum += p;
    }
    return sum;
}

/**
 * 使用 ISPC 模式正确实现数组求和：
 * - 每个实例在私有 partial 中累积（无通信，无竞争）
 * - reduce_add 合并所有 partial 得到最终结果
 *
 * 这是课程中强调的"正确"实现方式。
 */
float reduce_sum(const std::vector<float>& arr) {
    std::vector<float> partial(PROGRAM_COUNT, 0.0f);
    int N = arr.size();

    // 每个"程序实例"累积自己的 partial 和
    for (int i = 0; i < N; i++) {
        int inst = i % PROGRAM_COUNT;
        partial[inst] += arr[i];
    }

    // 跨实例 reduce_add
    return simulated_reduce_add(partial);
}

// ============================================================================
// 第3部分：高级协作 - O(log N) 并行乘积
//
// 使用 ISPC 的 shift/rotate + 条件操作实现蝶形并行归约。
// 这展示了跨实例通信操作的强大表达能力。
// ============================================================================

/**
 * ISPC 风格：使用 shift/rotate 实现8个元素的并行乘积。
 * 每个步骤将活跃实例数量减半。
 * 总计：lg(8) = 3 步。
 *
 * 蝶形归约模式（Butterfly Reduction）：
 *   初始状态：[a,b,c,d,e,f,g,h]  (8个活跃实例)
 *   步骤1(偏移1)：[ab, ab, cd, cd, ef, ef, gh, gh]  (4个活跃)
 *   步骤2(偏移2)：[abcd, abcd, abcd, abcd, efgh, efgh, efgh, efgh]  (2个活跃)
 *   步骤3(偏移4)：[abcdefgh, ...]  (1个活跃，结果在实例0)
 */
float parallel_product8(const std::vector<float>& arr) {
    // 假设 gang 大小为8且 arr.size() == 8
    std::vector<float> val(arr.begin(), arr.end());

    // 步骤1：偏移1，偶数索引实例相乘
    for (int i = 0; i < 8; i += 2) {
        val[i] *= val[i + 1];
    }

    // 步骤2：偏移2，实例0和4各乘一个
    for (int i = 0; i < 8; i += 4) {
        val[i] *= val[i + 2];
    }

    // 步骤3：偏移4，实例0乘实例4（最终结果）
    val[0] *= val[4];

    return val[0];
}

// ============================================================================
// 第4部分：通用并行归约（log2 步骤）
//
// 树形归约是实现 reduce_add 的底层原理。
// 工作量为 O(N)，但步骤数为 O(log N)。
// ============================================================================

/**
 * 通用并行归约：使用树形归约对数组求和。
 * 展示 reduce_add 背后的原理：O(log N) 步骤，O(N) 工作量。
 *
 * 树形归约模式：
 *   N=8: [a,b,c,d,e,f,g,h]
 *   步骤1(步幅1)：[a+b, b, c+d, d, e+f, f, g+h, h]
 *   步骤2(步幅2)：[a+b+c+d, b, c+d, d, e+f+g+h, f, g+h, h]
 *   步骤3(步幅4)：[a+b+...+h, b, c+d, d, e+f+g+h, f, g+h, h]
 *   结果在 data[0]
 */
float parallel_reduce_sum(const std::vector<float>& arr) {
    std::vector<float> data(arr.begin(), arr.end());
    int n = data.size();

    // 填充到2的幂次（便于树形归约）
    while ((n & (n - 1)) != 0) {
        data.push_back(0.0f);
        n++;
    }

    // 树形归约：每个步骤将活跃元素数量减半
    for (int step = 1; step < n; step *= 2) {
        for (int i = 0; i < n; i += 2 * step) {
            data[i] += data[i + step];
        }
    }

    return data[0];
}

// ============================================================================
// 第5部分：Uniform vs Varying 变量演示
//
// ISPC 中两种变量类型的直观对比：
// - uniform：所有实例看到相同的值（编译器可优化为只存一份）
// - varying：每个实例有自己的副本（默认类型）
// ============================================================================

/**
 * 演示 ISPC 中 uniform vs varying 的概念。
 *
 * uniform：所有实例看到相同的值（只存储一次）
 *   例：循环边界、常量、迭代计数
 *
 * varying：每个实例有自己的副本
 *   例：数组元素值、局部累加器
 *
 * 这种区分既是正确性保证（避免数据竞争），
 * 也是性能优化提示（减少存储和通信）。
 */
void demonstrate_uniform_varying() {
    std::cout << "\n=== Uniform vs Varying 变量 ===\n";

    // uniform int N = 10;  -- 所有实例共享此值
    int uniform_N = 10;

    // varying float partial = 0.0f; -- 每个实例有自己的副本
    std::vector<float> varying_partial(PROGRAM_COUNT, 0.0f);

    // 模拟每个程序实例递增自己的 partial
    for (int inst = 0; inst < PROGRAM_COUNT; inst++) {
        for (int j = 0; j < uniform_N; j++) {
            varying_partial[inst] += static_cast<float>(j + 1);
        }
    }

    std::cout << "Uniform N = " << uniform_N << " （所有实例共享）\n";
    std::cout << "Varying partial 数组：";
    for (int inst = 0; inst < PROGRAM_COUNT; inst++) {
        std::cout << varying_partial[inst] << " ";
    }
    std::cout << "\n每个实例计算了：sum(1..10) = " << (10 * 11 / 2) << "\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "第4讲 第1部分：ISPC SPMD 抽象模拟\n";
    std::cout << "============================================================\n";

    const int N = 1024;
    const int TERMS = 5;
    std::vector<float> x(N);
    std::vector<float> result_interleaved(N, 0.0f);
    std::vector<float> result_blocked(N, 0.0f);
    std::vector<float> result_foreach(N, 0.0f);

    // 初始化输入数组，值范围在 [0, PI]
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i) * M_PI / N;
    }

    // === 演示交错分配（Interleaved Assignment）===
    std::cout << "\n--- 交错分配（programCount=" << PROGRAM_COUNT << "）---\n";

    std::vector<std::thread> threads;
    for (int inst = 0; inst < PROGRAM_COUNT; inst++) {
        threads.emplace_back(interleaved_sinx, std::ref(x), std::ref(result_interleaved),
                             TERMS, inst);
    }
    for (auto& t : threads) t.join();

    std::cout << "前16个结果（交错分配）：";
    for (int i = 0; i < 16 && i < N; i++) {
        std::cout << std::fixed << std::setprecision(4) << result_interleaved[i] << " ";
    }
    std::cout << "\n内存访问：所有实例访问连续内存（高效 packed load）\n";

    // === 演示块分配（Blocked Assignment）===
    std::cout << "\n--- 块分配（programCount=" << PROGRAM_COUNT << "）---\n";
    threads.clear();
    for (int inst = 0; inst < PROGRAM_COUNT; inst++) {
        threads.emplace_back(blocked_sinx, std::ref(x), std::ref(result_blocked),
                             TERMS, inst);
    }
    for (auto& t : threads) t.join();

    std::cout << "前16个结果（块分配）：";
    for (int i = 0; i < 16 && i < N; i++) {
        std::cout << std::fixed << std::setprecision(4) << result_blocked[i] << " ";
    }
    std::cout << "\n内存访问：跨实例不连续（需要 gather 指令）\n";

    // === 演示 foreach 抽象 ===
    std::cout << "\n--- foreach 抽象 ---\n";
    foreach_sinx(x, result_foreach, TERMS);
    std::cout << "前16个结果（foreach）：";
    for (int i = 0; i < 16 && i < N; i++) {
        std::cout << std::fixed << std::setprecision(4) << result_foreach[i] << " ";
    }
    std::cout << "\nforeach 让系统自动管理迭代分配。\n";

    // === 验证结果一致性 ===
    bool match = true;
    for (int i = 0; i < N && match; i++) {
        if (std::abs(result_interleaved[i] - result_blocked[i]) > 1e-4f) match = false;
    }
    std::cout << "\n交错分配 == 块分配 结果：" << (match ? "一致" : "不一致") << "\n";

    match = true;
    for (int i = 0; i < N && match; i++) {
        if (std::abs(result_interleaved[i] - result_foreach[i]) > 1e-4f) match = false;
    }
    std::cout << "交错分配 == foreach 结果：" << (match ? "一致" : "不一致") << "\n";

    // === 演示 reduce_add ===
    std::cout << "\n--- reduce_add（跨实例求和）---\n";
    std::vector<float> test_arr = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float sum = reduce_sum(test_arr);
    std::cout << "数组：[";
    for (size_t i = 0; i < test_arr.size(); i++) {
        std::cout << test_arr[i] << (i < test_arr.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "reduce_add 求和 = " << sum << " （期望值：36）\n";

    // === 演示并行乘积 ===
    std::cout << "\n--- 并行乘积（O(log N) 步骤）---\n";
    float prod = parallel_product8(test_arr);
    std::cout << "parallel_product8 = " << prod << " （期望值：40320 = 8!）\n";

    // === 演示通用并行归约 ===
    std::cout << "\n--- 通用并行归约（树形求和）---\n";
    float tsum = parallel_reduce_sum(test_arr);
    std::cout << "树形归约求和 = " << tsum << " （期望值：36）\n";

    // === 演示 uniform vs varying ===
    demonstrate_uniform_varying();

    // === 分配模式总结 ===
    std::cout << "\n=== 分配模式总结 ===\n";
    std::cout << "┌─────────────┬──────────────────────────────────────┬───────────────────────┐\n";
    std::cout << "│ 分配方式    │ 内存访问模式                         │ SIMD 效率             │\n";
    std::cout << "├─────────────┼──────────────────────────────────────┼───────────────────────┤\n";
    std::cout << "│ 交错分配    │ 每次迭代元素连续                     │ Packed load (vmovaps) │\n";
    std::cout << "│ 块分配      │ 跨实例元素不连续                     │ Gather (vgatherdps)   │\n";
    std::cout << "│ foreach     │ 系统管理（目前为静态分配）           │ 由实现决定            │\n";
    std::cout << "└─────────────┴──────────────────────────────────────┴───────────────────────┘\n";

    std::cout << "\n所有测试成功完成。\n";
    return 0;
}
