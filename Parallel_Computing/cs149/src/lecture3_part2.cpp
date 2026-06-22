// lecture3_part2.cpp - SPMD/ISPC 编程模型模拟
// =============================================================================
// CS149 第3讲核心概念：
//   - ISPC：Intel SPMD Program Compiler，一种将SPMD语义编译为SIMD指令的编译器
//   - SPMD（Single Program, Multiple Data）：单程序多数据编程模型
//     程序员编写一个程序，该程序由多个"程序实例"同时运行
//   - Gang（组）抽象：programCount 个程序实例并发执行
//     每个 gang 映射到一组 SIMD 硬件通道
//   - programCount：gang 中同时执行的程序实例数量（对应 SIMD 宽度）
//   - programIndex：当前程序实例的ID（0 到 programCount-1）
//     用于让每个实例知道自己的身份，从而处理不同的数据
//   - uniform（统一变量）：所有程序实例持有相同的值
//     编译器优化：只需存储一份，不需要为每个实例复制
//   - varying（变化变量，默认）：每个程序实例的值可能不同
//     这是ISPC的默认变量类型，每个实例有独立的副本
//   - 交错分配（Interleaved Assignment）：
//     idx = i + programIndex，产生连续向量加载（vmovaps）
//     内存访问模式友好，适合SIMD硬件
//   - 块分配（Blocked Assignment）：
//     start = programIndex * count，需要 gather/scatter 操作
//     内存访问不连续，性能较差
//   - foreach：ISPC 的并行迭代抽象
//     程序员声明"这些是 gang 必须执行的迭代"，
//     编译器/运行时决定如何将迭代分配给程序实例
//   - reduce_add()：跨实例归约原语，将所有实例的部分和汇总
//     生成 uniform 类型的结果
//   - 抽象 vs. 实现：SPMD 是编程模型（软件抽象），
//     SIMD 是硬件实现（硬件指令，如 AVX2、Neon）
//
// 编译命令：g++ -std=c++17 -O2 lecture3_part2.cpp -o lecture3_part2
// =============================================================================

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <numeric>

// =============================================================================
// ISPC 风格的 Gang 抽象模拟
//
// 在真正的 ISPC 中，Gang 是一组并发执行的程序实例。
// 本模拟使用 C++ 类来近似表示 ISPC 的 Gang 概念：
// - programCount 个实例同时执行
// - 每个实例有自己的局部（varying）变量
// - 支持 reduce_add 等跨实例操作
// =============================================================================
class ISPCGang {
public:
    int programCount; // gang 中的实例数量（对应 SIMD 宽度）

    explicit ISPCGang(int count) : programCount(count) {
        instances_.resize(count);
    }

    // 每个实例存储其局部（varying）变量
    // 在真实 ISPC 中，这些变量位于 SIMD 寄存器的不同通道中
    struct Instance {
        int programIndex;    // 当前实例在 gang 中的编号
        float value = 0.0f;  // 实例的局部值（varying）
        float numer = 0.0f;  // 泰勒级数的分子（varying）
        float partial = 0.0f;// 部分和（用于 reduce_add）
        // ... 其他每实例状态
    };

    Instance& instance(int idx) { return instances_[idx]; }
    const Instance& instance(int idx) const { return instances_[idx]; }

    // 跨实例归约：将所有实例的 partial 值求和
    // 在真实 ISPC 中，这会被编译为高效的 SIMD 归约指令
    float reduce_add(float* partials) {
        float sum = 0.0f;
        for (int i = 0; i < programCount; i++) {
            sum += partials[i];
        }
        return sum;
    }

    // 屏障同步：所有实例在此处同步（概念上的）
    // 在真实 ISPC 中，这是由 SIMD 锁步执行自然保证的
    void barrier() {
        // 在真实 ISPC 中，SIMD 锁步执行天然保证了屏障语义
    }

private:
    std::vector<Instance> instances_;
};

// ---------------------------------------------------------------------------
// sin(x) 的泰勒展开（与第2讲中的函数相同）
//
// sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
// 使用增量计算避免重复计算幂和阶乘
// ---------------------------------------------------------------------------
float sin_taylor(float x, int terms) {
    float value = x;
    float numer = x * x * x;
    float denom = 6.0f;  // 3! = 6
    int sign = -1;
    for (int j = 1; j <= terms; j++) {
        value += sign * numer / denom;
        numer *= x * x;
        denom *= (2 * j + 2) * (2 * j + 3);
        sign *= -1;
    }
    return value;
}

// ---------------------------------------------------------------------------
// ISPC 风格 sinx：交错分配
//
// 对应课程中的 ispc_sinx() 函数，使用 programCount 和 programIndex。
// 交错分配策略：
// - 程序实例0处理元素 0, 8, 16, ...（步幅 = programCount）
// - 程序实例1处理元素 1, 9, 17, ...
// - 依此类推
//
// 优点：每次迭代中所有程序实例访问连续的内存地址，
// 可以使用高效的 packed vector load（vmovaps）指令。
// ---------------------------------------------------------------------------
void ispc_sinx_interleaved(int N, int terms, const float* x, float* result) {
    const int PROGRAM_COUNT = 8;
    ISPCGang gang(PROGRAM_COUNT);

    // 模拟 gang 执行 ISPC 函数
    // 在 ISPC 中：for (uniform int i=0; i<N; i+=programCount)
    for (int i = 0; i < N; i += PROGRAM_COUNT) {
        // 所有 programCount 个实例并行执行（SIMD）
        for (int pi = 0; pi < PROGRAM_COUNT && (i + pi) < N; pi++) {
            int idx = i + pi; // idx = i + programIndex（交错模式）
            float value = x[idx];
            float numer = x[idx] * x[idx] * x[idx];
            // uniform 变量（所有实例使用相同值）
            float denom = 6.0f;
            float sign_val = -1.0f;

            for (int j = 1; j <= terms; j++) {
                value += sign_val * numer / denom;
                numer *= x[idx] * x[idx];
                denom *= (2 * j + 2) * (2 * j + 3);
                sign_val *= -1.0f;
            }
            result[idx] = value;
        }
    }
}

// ---------------------------------------------------------------------------
// ISPC 风格 sinx：块分配（课程中的版本2）
//
// 与交错分配不同，块分配给每个程序实例分配一个连续的数据块：
// - 程序实例0处理元素 0, 1, 2, ...（连续块）
// - 程序实例1处理下一个块
// - 依此类推
//
// 缺点：在每次迭代中，不同实例访问非连续的内存地址，
// 需要使用 gather 指令（如 vgatherdps），性能较差。
// ---------------------------------------------------------------------------
void ispc_sinx_blocked(int N, int terms, const float* x, float* result) {
    const int PROGRAM_COUNT = 8;
    int count = N / PROGRAM_COUNT; // uniform int count = N / programCount

    // 模拟：每个实例处理一个连续的数据块
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        int start = pi * count; // int start = programIndex * count

        for (int j = 0; j < count; j++) {
            int idx = start + j;
            float value = x[idx];
            float numer = x[idx] * x[idx] * x[idx];
            float denom = 6.0f;
            float sign_val = -1.0f;

            for (int k = 1; k <= terms; k++) {
                value += sign_val * numer / denom;
                numer *= x[idx] * x[idx];
                denom *= (2 * k + 2) * (2 * k + 3);
                sign_val *= -1.0f;
            }
            result[idx] = value;
        }
    }
}

// ---------------------------------------------------------------------------
// ISPC foreach 抽象模拟
//
// foreach 是 ISPC 中最高级的并行抽象：
// - 程序员只需声明并行迭代的范围
// - 编译器/运行时决定具体的分配策略
//
// 这体现了 CS149 的核心主题：抽象（程序员看到什么）
// 与实现（底层如何运行）的分离。
// ---------------------------------------------------------------------------
void ispc_foreach_sinx(int N, int terms, const float* x, float* result) {
    // foreach 实现：将迭代交错分配给程序实例
    const int PROGRAM_COUNT = 8;

    for (int loop_i = 0; loop_i < N; loop_i += PROGRAM_COUNT) {
        for (int pi = 0; pi < PROGRAM_COUNT && (loop_i + pi) < N; pi++) {
            int i = loop_i + pi;
            float value = x[i];
            float numer = x[i] * x[i] * x[i];
            float denom = 6.0f;
            float sign_val = -1.0f;

            for (int j = 1; j <= terms; j++) {
                value += sign_val * numer / denom;
                numer *= x[i] * x[i];
                denom *= (2 * j + 2) * (2 * j + 3);
                sign_val *= -1.0f;
            }
            result[i] = value;
        }
    }
}

// ---------------------------------------------------------------------------
// ISPC reduce_add 模拟：对数组所有元素求和
//
// 对应课程中正确的 sum_array 实现：
// - 每个程序实例在私有 partial 变量中累积（无竞争条件）
// - 最后使用 reduce_add 将所有 partial 合并
//
// 为什么其他实现是错误的？
// 1. varying float sum：每个实例有自己的 sum，但无法合并它们
// 2. uniform float sum：所有实例共享一个 sum，但 x[i] 对每个实例不同
//    → 数据竞争（多个实例同时写同一个变量）
// ---------------------------------------------------------------------------
float ispc_sum_array(int N, const float* x) {
    const int PROGRAM_COUNT = 8;
    std::vector<float> partials(PROGRAM_COUNT, 0.0f);

    // foreach (i = 0 ... N)
    for (int loop_i = 0; loop_i < N; loop_i += PROGRAM_COUNT) {
        for (int pi = 0; pi < PROGRAM_COUNT && (loop_i + pi) < N; pi++) {
            int i = loop_i + pi;
            partials[pi] += x[i]; // 每个实例在自己的私有 partial 中累积
        }
    }

    // reduce_add：跨实例求和
    float sum = 0.0f;
    for (float p : partials) sum += p;
    return sum;
}

// ---------------------------------------------------------------------------
// ISPC 跨实例操作：reduce_min 模拟
//
// 跨实例操作允许 gang 内的程序实例之间进行通信。
// reduce_min 找出所有实例中的最小值。
// ---------------------------------------------------------------------------
float ispc_reduce_min(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    float min_val = values[0];
    for (float v : values) min_val = std::min(min_val, v);
    return min_val;
}

// ---------------------------------------------------------------------------
// ISPC shift/rotate 操作：将值传递给实例 i+offset
//
// rotate 是 ISPC 中的关键跨实例通信操作之一。
// 它将每个实例的值"左旋"或"右旋"指定偏移量。
// 这对于实现并行归约（如并行乘积）至关重要。
// ---------------------------------------------------------------------------
std::vector<float> ispc_rotate(const std::vector<float>& values, int offset) {
    int n = static_cast<int>(values.size());
    std::vector<float> result(n);
    for (int i = 0; i < n; i++) {
        result[(i + offset) % n] = values[i];
    }
    return result;
}

// ---------------------------------------------------------------------------
// ISPC broadcast：将一个实例的值广播到所有实例
//
// broadcast 操作将一个程序实例的值复制到 gang 中的所有其他实例。
// 这在需要让所有实例共享某个计算结果时非常有用。
// ---------------------------------------------------------------------------
float ispc_broadcast(const std::vector<float>& values, int index) {
    assert(index >= 0 && index < static_cast<int>(values.size()));
    return values[index];
}

// ---------------------------------------------------------------------------
// 8个元素的并行乘积：在 log2(8) = 3 步内完成
//
// 对应课程中的 vec8product 示例。
// 使用 shift + 条件乘法的蝶形归约模式：
//   步骤1：偏移1，偶数索引实例相乘
//   步骤2：偏移2，每4个中第1个相乘
//   步骤3：偏移4，每8个中第1个相乘（最终结果在实例0中）
//
// 这种模式展示了如何利用跨实例通信实现 O(log N) 的并行归约。
// ---------------------------------------------------------------------------
float ispc_vec8product(const float* x) {
    const int PROGRAM_COUNT = 8;
    std::vector<float> val(PROGRAM_COUNT);

    // 步骤1：每个实例加载自己的值
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        val[pi] = x[pi];
    }

    // 步骤2：偏移1，偶数索引实例相乘
    auto val2 = ispc_rotate(val, 1);
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        if (pi % 2 == 0) val[pi] = val[pi] * val2[pi];
    }

    // 步骤3：偏移2，每4个中第1个相乘
    val2 = ispc_rotate(val, 2);
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        if (pi % 4 == 0) val[pi] = val[pi] * val2[pi];
    }

    // 步骤4：偏移4，每8个中第1个相乘（最终结果在实例0中）
    val2 = ispc_rotate(val, 4);
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        if (pi % 8 == 0) val[pi] = val[pi] * val2[pi];
    }

    return val[0];
}

// ---------------------------------------------------------------------------
// 演示交错分配 vs 块分配（programCount=8）
//
// 展示两种分配策略下程序实例处理的数据元素分布：
// - 交错分配：每个实例处理步幅为 programCount 的元素
//   → 每次迭代产生连续的内存访问（适合 SIMD packed load）
// - 块分配：每个实例处理连续的数据块
//   → 跨实例的内存访问不连续（需要 gather 指令）
// ---------------------------------------------------------------------------
void demo_assignment_strategies(int N) {
    std::cout << "[1] 交错分配 vs 块分配（programCount=8）\n" << std::endl;

    const int PC = 8;
    int elements_per_instance = N / PC;

    // 展示每个程序实例处理的元素
    std::cout << "    交错分配（idx = i + programIndex）：\n";
    std::cout << "    ";
    for (int pi = 0; pi < PC; pi++) {
        std::cout << "PI" << pi << "     ";
    }
    std::cout << "\n    ";
    for (int pi = 0; pi < PC; pi++) std::cout << "--------";
    std::cout << std::endl;

    for (int loop_i = 0; loop_i < N; loop_i += PC) {
        std::cout << "    ";
        for (int pi = 0; pi < PC; pi++) {
            int idx = loop_i + pi;
            if (idx < N)
                std::cout << std::setw(4) << idx << "    ";
            else
                std::cout << "  -     ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n    → 连续内存访问：向量加载（vmovaps）高效执行\n" 
              << std::endl;

    // 块分配策略
    std::cout << "    块分配（start = programIndex × count）：\n";
    std::cout << "    ";
    for (int pi = 0; pi < PC; pi++) {
        std::cout << "PI" << pi << "     ";
    }
    std::cout << "\n    ";
    for (int pi = 0; pi < PC; pi++) std::cout << "--------";
    std::cout << std::endl;

    int count = N / PC;
    for (int j = 0; j < count; j++) {
        std::cout << "    ";
        for (int pi = 0; pi < PC; pi++) {
            int idx = pi * count + j;
            std::cout << std::setw(4) << idx << "    ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n    → 非连续访问：需要 gather 指令（vgatherdps）\n";
    std::cout << "    → gather 指令更复杂且开销更大\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 正确 vs 错误的 ISPC sum 实现
//
// 展示 ISPC 中求和操作的三种实现方式及其问题：
// 1. 正确：每个实例使用私有 partial，最后 reduce_add 合并
// 2. 错误（varying）：每个实例有独立的 sum，但无法合并
// 3. 错误（uniform）：所有实例共享 sum，导致数据竞争
// ---------------------------------------------------------------------------
void demo_ispc_sum() {
    std::cout << "[2] ISPC sum：正确与错误实现\n" << std::endl;

    const int N = 1024;
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) x[i] = static_cast<float>(i + 1);

    float expected = static_cast<float>(N * (N + 1) / 2);

    // 正确实现：每个实例使用私有 partial，然后 reduce_add
    float result = ispc_sum_array(N, x.data());
    std::cout << "    正确实现（私有 partial + reduce_add）：\n";
    std::cout << "    求和 = " << std::fixed << std::setprecision(0) << result 
              << " （期望值：" << expected << "） ✓\n" << std::endl;

    std::cout << "    为什么错误版本会失败：\n";
    std::cout << "    - sum 声明为 'float'（varying）：每个实例有自己的 sum，\n"
              << "      但没有办法合并它们\n";
    std::cout << "    - sum 声明为 'uniform float'：所有实例共享一个 sum，\n"
              << "      但 x[i] 对每个实例不同 → 数据竞争\n";
    std::cout << "    - 这两种情况在 ISPC 中都会产生编译时类型错误\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 演示 vec8product（O(log N) 并行乘积）
//
// 展示如何使用 ISPC 的跨实例通信操作（rotate + 条件运算）
// 在 O(log N) 步骤内完成 N 个数的并行乘积。
// ---------------------------------------------------------------------------
void demo_vec8product() {
    std::cout << "[3] 高级 ISPC 协作：vec8product\n" << std::endl;

    float x[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float expected = 1*2*3*4*5*6*7*8.0f; // 40320.0

    float result = ispc_vec8product(x);

    std::cout << "    输入： [1, 2, 3, 4, 5, 6, 7, 8]\n";
    std::cout << "    乘积 = " << std::fixed << std::setprecision(0) << result 
              << " （期望值：" << expected << "）\n";
    std::cout << "    步骤数：lg(8) = 3（使用偏移+条件乘法）\n" 
              << std::endl;
}

// =============================================================================
// 主函数：按顺序展示课程第3讲中ISPC/SPMD编程模型的核心概念
// =============================================================================
int main() {
    std::cout << "=== CS149 第3讲：ISPC/SPMD 编程模型模拟 ===\n" << std::endl;

    const int N = 64; // 使用较小的 N 以便清晰展示

    // ---- 第1部分：交错分配 vs 块分配 ----
    demo_assignment_strategies(N);

    // ---- 第2部分：ISPC 求和 ----
    demo_ispc_sum();

    // ---- 第3部分：vec8product ----
    demo_vec8product();

    // ---- 第4部分：ISPC 核心概念总结 ----
    std::cout << "[4] ISPC 核心概念总结\n" << std::endl;
    std::cout << "    ┌─────────────────────┬──────────────────────────────────────┐\n";
    std::cout << "    │ programCount        │ 每个 gang 中的程序实例数量           │\n";
    std::cout << "    │ programIndex        │ 当前实例的ID（0..PC-1）              │\n";
    std::cout << "    │ uniform             │ 所有实例共享相同值（优化提示）        │\n";
    std::cout << "    │ varying（默认）     │ 每个实例的值可能不同                 │\n";
    std::cout << "    │ foreach             │ 并行迭代（由 gang 调度执行）         │\n";
    std::cout << "    │ reduce_add()        │ 跨实例求和（结果为 uniform）         │\n";
    std::cout << "    │ broadcast()         │ 将一个实例的值发送给所有实例         │\n";
    std::cout << "    │ rotate()            │ 将值传递给实例 i+offset              │\n";
    std::cout << "    │ SPMD                │ 编程抽象（软件层面）                 │\n";
    std::cout << "    │ SIMD                │ 硬件实现（AVX2、Neon 等）            │\n";
    std::cout << "    └─────────────────────┴──────────────────────────────────────┘\n" << std::endl;

    // ---- 第5部分：关键要点 ----
    std::cout << "[5] 第3讲（ISPC）关键要点\n" << std::endl;
    std::cout << "    - SPMD：程序员从 programCount 个逻辑线程的角度思考\n";
    std::cout << "    - SIMD：编译器生成向量指令（AVX2、Neon 等）\n";
    std::cout << "    - 交错分配：适合连续内存访问（向量加载高效）\n";
    std::cout << "    - 块分配：可能需要 gather/scatter（开销更大）\n";
    std::cout << "    - foreach：提升抽象层次（关注迭代而非实例）\n";
    std::cout << "    - uniform 变量：性能优化，不影响正确性\n";
    std::cout << "    - 跨实例操作实现了 gang 内部的通信\n";
    std::cout << "    - 抽象 vs. 实现 是理解并行编程的关键\n";
    std::cout << "    - ISPC tasks：用于多核并行的独立机制\n";

    return 0;
}
