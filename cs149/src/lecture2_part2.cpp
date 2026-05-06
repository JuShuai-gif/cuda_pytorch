// lecture2_part2.cpp - SIMD 执行与条件掩码模拟
// =============================================================================
// CS149 第2讲核心概念：
//   - SIMD：单指令多数据（Single Instruction, Multiple Data）
//     核心思想：将管理一条指令流的成本和复杂度分摊到多个 ALU 上。
//     一条指令同时操作多个数据元素，减少取指和解码的开销。
//
//   - AVX 内在函数（Intrinsic）：
//     类型：__m256（256 位向量，可存储 8 个 float）
//     操作：_mm256_load_ps（加载）、_mm256_mul_ps（乘法）、_mm256_add_ps（加法）等
//     内在函数是编译器内置的，直接映射到对应的 CPU 向量指令。
//
//   - SIMD 中的条件执行：使用掩码（Mask）屏蔽（丢弃）部分 ALU 的输出
//     当程序中存在 if/else 分支时，不是所有数据元素走相同的路径。
//     SIMD 的处理方式：执行 THEN 分支时屏蔽 ELSE 通道（让它们空转），
//     然后对 ELSE 分支也做同样处理。最终用掩码合并结果。
//
//   - 一致执行（Coherent）：所有数据元素执行相同的指令序列
//     理想情况：100% SIMD 利用率，无需掩码。
//     示例：对所有元素计算 sin(x) 泰勒展开——迭代次数相同，操作相同。
//
//   - 分支发散执行（Divergent）：不同数据元素有不同的控制流
//     需要掩码处理 → 被屏蔽的通道空转 → 吞吐量降低。
//     最坏情况：只有 1/WIDTH 的通道有效 → 效率 = 1/WIDTH。
//
//   - 显式 SIMD（CPU）：编译器生成向量指令
//     程序员可以：1) 让编译器自动向量化 2) 使用内在函数手动控制
//
//   - 隐式 SIMD（GPU）：硬件在多个线程上运行相同指令
//     GPU 的 warp/wavefront 就是隐式 SIMD 的实现，
//     程序员写标量代码，硬件在 SIMD 通道上同时执行。
//
// 编译: g++ -std=c++17 -O2 lecture2_part2.cpp -o lecture2_part2
// =============================================================================

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <cassert>
#include <random>
#include <chrono>

// ---------------------------------------------------------------------------
// SIMD 向量抽象（模拟 8-wide SIMD，类似 AVX2）
// 每个向量包含 8 个类型为 T 的元素
//
// 这个类模拟了 AVX 内在函数的核心操作：
//   - load：从内存加载 8 个元素
//   - store：将 8 个元素写入内存
//   - +、*、/：逐元素运算
//   - broadcast：将标量复制到所有通道
// ---------------------------------------------------------------------------
template<typename T, int WIDTH = 8>
class SIMDVector {
public:
    static constexpr int width = WIDTH;
    T data[WIDTH];

    SIMDVector() {
        for (int i = 0; i < WIDTH; i++) data[i] = 0;
    }

    explicit SIMDVector(T val) {
        for (int i = 0; i < WIDTH; i++) data[i] = val;
    }

    // 从内存加载（模拟 _mm256_load_ps）
    static SIMDVector load(const T* ptr) {
        SIMDVector v;
        for (int i = 0; i < WIDTH; i++) v.data[i] = ptr[i];
        return v;
    }

    // 写入内存（模拟 _mm256_store_ps）
    void store(T* ptr) const {
        for (int i = 0; i < WIDTH; i++) ptr[i] = data[i];
    }

    // 逐元素乘法
    SIMDVector operator*(const SIMDVector& other) const {
        SIMDVector result;
        for (int i = 0; i < WIDTH; i++) result.data[i] = data[i] * other.data[i];
        return result;
    }

    // 逐元素加法
    SIMDVector operator+(const SIMDVector& other) const {
        SIMDVector result;
        for (int i = 0; i < WIDTH; i++) result.data[i] = data[i] + other.data[i];
        return result;
    }

    // 逐元素除法
    SIMDVector operator/(const SIMDVector& other) const {
        SIMDVector result;
        for (int i = 0; i < WIDTH; i++) result.data[i] = data[i] / other.data[i];
        return result;
    }

    // 将标量广播到所有通道（模拟 _mm256_set1_ps）
    static SIMDVector broadcast(T val) {
        return SIMDVector(val);
    }

    void print(const char* label = "") const {
        if (label[0]) std::cout << label << " = [";
        else std::cout << "[";
        for (int i = 0; i < WIDTH; i++) {
            std::cout << std::setw(6) << std::setprecision(1) << std::fixed << data[i];
            if (i < WIDTH - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
};

// ---------------------------------------------------------------------------
// SIMD 条件执行掩码
// 掩码是位向量：bit i = 1 表示通道 i 处于活跃状态
//
// 在真实的 SIMD 硬件中，掩码由比较指令产生（如 _mm256_cmp_ps），
// 然后用于后续的掩码移动指令（如 _mm256_mask_mul_ps）。
//
// 被屏蔽的通道仍然会执行指令，但结果会被丢弃。
// 这意味着即使只有 1 个通道活跃，全部 8 个通道仍在消耗功耗和周期。
// ---------------------------------------------------------------------------
class SIMDMask {
public:
    unsigned int bits; // 8 位掩码，用于 8-wide SIMD

    SIMDMask() : bits(0) {}
    explicit SIMDMask(unsigned int b) : bits(b) {}

    // 从比较结果创建掩码：如果 pred[i] 为 true，则通道 i 活跃
    template<int WIDTH>
    static SIMDMask from_comparison(const bool pred[WIDTH]) {
        unsigned int m = 0;
        for (int i = 0; i < WIDTH; i++) {
            if (pred[i]) m |= (1u << i);
        }
        return SIMDMask(m);
    }

    // 统计活跃通道数（popcount = population count）
    // __builtin_popcount 是 GCC 内置函数，直接编译为 CPU 的 popcnt 指令
    int popcount() const {
        return __builtin_popcount(bits);
    }

    // 掩码取反（~mask 产生对立掩码，用于 ELSE 分支）
    SIMDMask operator~() const {
        return SIMDMask(~bits & 0xFF);
    }

    bool operator[](int i) const {
        return (bits >> i) & 1;
    }

    void print() const {
        std::cout << "[";
        for (int i = 0; i < 8; i++) {
            std::cout << ((bits >> i) & 1 ? "T" : "F");
            if (i < 7) std::cout << ",";
        }
        std::cout << "]（" << popcount() << "/8 活跃）\n";
    }
};

// ---------------------------------------------------------------------------
// 模拟第 2 讲中的条件执行示例：
//
//   forall (int i from 0 to N) {
//       float t = x[i];
//       <无条件代码>
//       if (t > 0.0) {
//           t = t * t;          // 活跃通道：t > 0 的通道
//           t = t * 50.0;
//           t = t + 100.0;
//       } else {
//           t = t + 30.0;       // 活跃通道：t <= 0 的通道
//           t = t / 10.0;
//       }
//       <恢复无条件代码>
//       y[i] = t;
//   }
//
// 关键观察：
//   - THEN 分支执行 3 个操作，ELSE 分支执行 2 个操作
//   - 无论有多少通道走哪个分支，所有通道都执行所有操作
//   - 效率 = 有用的操作数 / 总操作数
// ---------------------------------------------------------------------------
void demo_simd_conditional_execution() {
    using Vec = SIMDVector<float, 8>;

    std::cout << "[1] SIMD 条件执行（第 2 讲示例）\n" << std::endl;

    // 输入数据：混合正值和负值
    float input[8] = {-1.0f, 0.5f, -0.3f, 2.0f, -0.8f, 1.5f, 0.0f, -0.1f};

    Vec t = Vec::load(input);
    std::cout << "    输入值：\n    ";
    t.print();

    // ---- 无条件代码（所有通道活跃） ----
    std::cout << "\n    [无条件：全部 8 个通道活跃]\n";

    // ---- 条件分支：if (t > 0.0) ----
    bool pred_true[8], pred_false[8];
    for (int i = 0; i < 8; i++) {
        pred_true[i] = (t.data[i] > 0.0f);
        pred_false[i] = !pred_true[i];
    }

    SIMDMask mask_true = SIMDMask::from_comparison<8>(pred_true);
    SIMDMask mask_false = ~mask_true;

    std::cout << "    条件 (t > 0.0)：";
    mask_true.print();

    // 模拟 THEN 分支：只有掩码为 true 的通道执行
    // （在真实 SIMD 中，被屏蔽的通道仍执行但结果被丢弃）
    Vec t_branch = t;
    std::cout << "\n    [THEN 分支：" << mask_true.popcount() << " 个活跃通道]\n";
    
    // t = t * t（受掩码控制）
    for (int i = 0; i < 8; i++) {
        if (mask_true[i]) t_branch.data[i] = t_branch.data[i] * t_branch.data[i];
    }
    // t = t * 50.0
    for (int i = 0; i < 8; i++) {
        if (mask_true[i]) t_branch.data[i] = t_branch.data[i] * 50.0f;
    }
    // t = t + 100.0
    for (int i = 0; i < 8; i++) {
        if (mask_true[i]) t_branch.data[i] = t_branch.data[i] + 100.0f;
    }

    std::cout << "    THEN 执行后：";
    t_branch.print();

    // 模拟 ELSE 分支：只有掩码为 false 的通道执行
    std::cout << "\n    [ELSE 分支：" << mask_false.popcount() << " 个活跃通道]\n";
    
    // t = t + 30.0
    for (int i = 0; i < 8; i++) {
        if (mask_false[i]) t_branch.data[i] = t_branch.data[i] + 30.0f;
    }
    // t = t / 10.0
    for (int i = 0; i < 8; i++) {
        if (mask_false[i]) t_branch.data[i] = t_branch.data[i] / 10.0f;
    }

    std::cout << "    ELSE 执行后：";
    t_branch.print();

    // 计算效率
    // 总操作数 = 8 通道 × 5 操作（3 个 THEN + 2 个 ELSE）= 40
    // 有用操作数 = THEN 活跃通道 × 3 + ELSE 活跃通道 × 2
    int total_ops = 8 * 5;
    int useful_ops = mask_true.popcount() * 3 + mask_false.popcount() * 2;
    double efficiency = static_cast<double>(useful_ops) / total_ops * 100.0;

    std::cout << "\n    效率：" << std::fixed << std::setprecision(1) 
              << efficiency << "%（" << useful_ops << "/" << total_ops 
              << " 次有用操作）\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 最坏情况的分支发散执行演示：
// 只有 1/8 的通道走 THEN 分支 → 效率 = 1/8（仅考虑 THEN 时）
//
// 在最坏情况下：
//   THEN 分支：1 个通道活跃 × 3 操作 = 3 个有用操作 / 24 总操作 = 12.5%
//   整体：1×3 + 7×2 = 17 有用 / 40 总操作 = 42.5%
//
// 注意：整体效率包含了两个分支，所以看起来不算太低。
// 真正的 SIMD 效率问题出现在嵌套 if 或只有极少数通道需要大量工作的情况下。
// ---------------------------------------------------------------------------
void demo_worst_case_divergence() {
    using Vec = SIMDVector<float, 8>;

    std::cout << "[2] 最坏情况的分支发散（8 个通道中仅 1 个发散）\n" << std::endl;

    // 只有通道 0 为正，其余全为负
    float input[8] = {1.0f, -0.5f, -0.3f, -2.0f, -0.8f, -1.5f, -0.2f, -0.1f};

    Vec t = Vec::load(input);
    std::cout << "    输入：";
    t.print();

    bool pred[8];
    for (int i = 0; i < 8; i++) pred[i] = (t.data[i] > 0.0f);
    SIMDMask mask = SIMDMask::from_comparison<8>(pred);
    std::cout << "    掩码：";
    mask.print();

    // 模拟 THEN 分支的 3 个操作（1 个通道活跃 → 24 次总操作中 3 次有用）
    int total_ops = 8 * 3;
    int useful_ops = mask.popcount() * 3;
    double eff = static_cast<double>(useful_ops) / total_ops * 100.0;

    std::cout << "\n    THEN 效率：" << std::fixed << std::setprecision(1) 
              << eff << "%（" << useful_ops << "/" << total_ops 
              << " 次有用操作）\n";

    // 整体：THEN（3 操作）+ ELSE（2 操作）= 5 × 8 = 40 总操作
    // 有用：1 个通道走 THEN = 3，7 个通道走 ELSE = 14，合计 17
    int total_all = 8 * 5;
    int useful_all = mask.popcount() * 3 + (8 - mask.popcount()) * 2;
    double eff_all = static_cast<double>(useful_all) / total_all * 100.0;

    std::cout << "    整体效率：" << eff_all << "%（" 
              << useful_all << "/" << total_all << "）\n" << std::endl;

    std::cout << "    关键洞察：即使在最坏的分支发散情况下（1/8），\n"
              << "    整体效率仍达 42.5%，因为两个分支都需要执行。\n"
              << "    真正的 SIMD 效率噩梦是嵌套 if/else 链中\n"
              << "    只有极少数通道执行最昂贵的路径。\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 对比一致执行与分支发散执行
//
// 一致执行（适合 SIMD）：
//   - 所有通道走同样的代码路径
//   - 100% SIMD 利用率
//   - 分支预测也好（所有通道都预测同一方向）
//
// 分支发散执行（不适合 SIMD）：
//   - 不同通道走不同的代码路径
//   - 部分通道被屏蔽 → 利用率下降
//   - 分支预测困难（随机数据导致分支不可预测）
//
// 注意：分支发散对多核执行不是问题，
// 因为每个核可以独立取指和解码不同的指令。
// ---------------------------------------------------------------------------
void demo_coherent_vs_divergent() {
    std::cout << "[3] 一致执行 vs. 分支发散执行\n" << std::endl;

    std::cout << "    ┌──────────────────────────────────────────────────────┐\n";
    std::cout << "    │ 一致执行（Coherent，对 SIMD 有利）：                  │\n";
    std::cout << "    │   所有通道走相同的代码路径                           │\n";
    std::cout << "    │   → 100% SIMD 利用率                                 │\n";
    std::cout << "    │   示例：对所有元素计算 sin(x) 泰勒展开                │\n";
    std::cout << "    │   （迭代次数相同，操作相同）                          │\n";
    std::cout << "    ├──────────────────────────────────────────────────────┤\n";
    std::cout << "    │ 分支发散执行（Divergent，对 SIMD 不利）：             │\n";
    std::cout << "    │   不同通道走不同的代码路径                           │\n";
    std::cout << "    │   → 通道被屏蔽 → 吞吐量降低                          │\n";
    std::cout << "    │   示例：按元素条件判断（if x[i] > 0）                 │\n";
    std::cout << "    │   最坏情况：1/WIDTH 的峰值性能                        │\n";
    std::cout << "    └──────────────────────────────────────────────────────┘\n" << std::endl;

    std::cout << "    注意：分支发散对多核执行不是问题，\n";
    std::cout << "    因为每个核可以独立取指和解码不同的指令。\n" << std::endl;

    // 工作模拟器：一致数据 vs. 发散数据的吞吐量对比
    const int N = 1'000'000;
    std::vector<float> coherent_data(N, 1.0f);  // 全部一致 → 分支可预测
    std::vector<float> divergent_data(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; i++) divergent_data[i] = dist(rng);  // 随机 → 分支不可预测

    // 一致数据：所有数据相同 → 分支高度可预测
    {
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        float sum = 0;
        for (int i = 0; i < N; i++) {
            if (coherent_data[i] > 0) {
                sum += coherent_data[i];
            } else {
                sum -= coherent_data[i];
            }
        }
        auto end = high_resolution_clock::now();
        double time1 = duration_cast<microseconds>(end - start).count() / 1000.0;
        std::cout << "    一致数据（分支可预测）：" 
                  << std::fixed << std::setprecision(1) << time1 << " ms，sum="
                  << sum << std::endl;
    }

    // 发散数据：随机数据 → 分支不可预测
    {
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        float sum = 0;
        for (int i = 0; i < N; i++) {
            if (divergent_data[i] > 0) {
                sum += divergent_data[i];
            } else {
                sum -= divergent_data[i];
            }
        }
        auto end = high_resolution_clock::now();
        double time2 = duration_cast<microseconds>(end - start).count() / 1000.0;
        std::cout << "    发散数据（分支不可预测）：" 
                  << std::fixed << std::setprecision(1) << time2 << " ms，sum="
                  << sum << std::endl;
    }

    std::cout << "\n    这演示了分支预测在超标量 CPU 上如何\n"
              << "    与一致/发散数据模式交互。\n"
              << "    （SIMD 掩码避免了分支预测问题）\n" << std::endl;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 第2讲：SIMD 执行与条件掩码 ===\n" << std::endl;

    demo_simd_conditional_execution();
    demo_worst_case_divergence();
    demo_coherent_vs_divergent();

    // ---- 附加：SIMD 术语参考 ----
    std::cout << "[4] SIMD 术语参考\n" << std::endl;
    std::cout << "    ┌──────────────────────┬───────────────────────────────────────┐\n";
    std::cout << "    │ Intel AVX2           │ 256 位，8×32 位 或 4×64 位           │\n";
    std::cout << "    │ Intel AVX512         │ 512 位，16×32 位                     │\n";
    std::cout << "    │ ARM Neon             │ 128 位，4×32 位                      │\n";
    std::cout << "    │ 显式 SIMD             │ 编译器生成向量指令                   │\n";
    std::cout << "    │ 隐式 SIMD（GPU）      │ 硬件在多个线程上运行相同指令          │\n";
    std::cout << "    │ 一致执行              │ 所有数据元素的指令序列相同            │\n";
    std::cout << "    │ 分支发散执行           │ 每个数据元素有不同的控制流           │\n";
    std::cout << "    └──────────────────────┴───────────────────────────────────────┘\n";

    return 0;
}
