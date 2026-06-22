/**
 * lecture4_part2.cpp - 阿姆达尔定律（Amdahl's Law）：加速比分析
 *
 * 模拟并可视化阿姆达尔定律：
 * - 加速比 ≤ 1/S（其中 S 是串行执行的比例）
 *   这意味着即使有无限个处理器，加速比也被串行部分严格限制
 * - 展示微小的串行区域如何严重限制大规模并行系统的可扩展性
 *   例如：1% 的串行代码将理论最大加速比限制在100倍
 * - 展示并行化开销的影响（例如：合并 partial 和的开销）
 *   实际加速比往往远低于理论最大值
 *
 * 阿姆达尔定律的数学表达式：
 *   speedup(P) = 1 / (S + (1 - S) / P)
 *   其中 S = 串行部分占比，P = 处理器数量
 *
 * 当 P → ∞ 时，speedup → 1/S
 * 这就是为什么即使只有1%的串行代码，最大加速比也只有100倍。
 *
 * 编译命令：g++ -std=c++17 lecture4_part2.cpp -o lecture4_part2 && ./lecture4_part2
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <random>
#include <numeric>

// ============================================================================
// 第1部分：阿姆达尔定律核心计算
// ============================================================================

/**
 * 阿姆达尔定律：给定串行比例 S 和处理器数量 P，
 * 计算最大理论加速比。
 *
 * speedup(P) = 1 / (S + (1 - S) / P)
 *
 * 直观理解：
 * - S：必须串行执行的部分（无法并行化）
 * - (1-S)/P：可并行部分在 P 个处理器上的执行时间
 * - 分母是加速后的总执行时间，分子是原始串行执行时间
 *
 * 关键推论：
 * - 当 P → ∞ 时，speedup → 1/S（串行部分成为瓶颈）
 * - 即使 S 很小（如 1%），最大加速比也只有 100 倍
 * - 这意味着追求大规模并行之前，必须先最小化串行部分
 */
double amdahl_speedup(double S, int P) {
    return 1.0 / (S + (1.0 - S) / P);
}

void print_amdahl_table() {
    std::cout << "\n=== 阿姆达尔定律：最大加速比 ===\n\n";
    std::cout << "┌────────┬──────────────────────────────────────────┐\n";
    std::cout << "│   P    │ S=0.01时加速比  S=0.05  S=0.1  S=0.5   │\n";
    std::cout << "├────────┼──────────────────────────────────────────┤\n";

    std::vector<double> serial_fractions = {0.01, 0.05, 0.1, 0.5};
    std::vector<int> processors = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (int P : processors) {
        std::cout << "│ " << std::setw(6) << P << " │";
        for (double S : serial_fractions) {
            double sp = amdahl_speedup(S, P);
            std::cout << "  " << std::setw(5) << std::fixed << std::setprecision(2) << sp;
        }
        std::cout << " │\n";
    }
    std::cout << "└────────┴──────────────────────────────────────────┘\n";

    std::cout << "\n关键洞察：当 S=0.01（1%串行），即使 P 无限大，最大加速比也仅约100倍。\n";
}

// ============================================================================
// 第2部分：图像处理示例（来自课程讲义）
// ============================================================================

/**
 * 模拟课程中的两步 N×N 图像处理示例：
 * 步骤1：将所有像素亮度乘以2（可完全并行化，工作量为 N²）
 * 步骤2：计算所有像素的平均值（部分可并行化，需要串行合并）
 *
 * 这个例子展示了阿姆达尔定律在实际问题中的应用：
 * - 方案1：步骤1并行 + 步骤2串行 → 加速比受限于2
 * - 方案2：两步都并行 + 合并partial和 → 加速比 → P（当 N >> P）
 */
class ImageProcessor {
private:
    int N;
    std::vector<double> pixels;

    // 模拟工作量的延迟
    void simulated_work(double ops) {
        volatile double x = 0.0;
        for (long i = 0; i < static_cast<long>(ops * 10); i++) {
            x += std::sin(static_cast<double>(i) * 0.001);
        }
    }

public:
    ImageProcessor(int size) : N(size), pixels(size * size) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& p : pixels) p = dist(rng);
    }

    // 顺序实现：T_seq = 2 × N²
    double sequential() {
        auto start = std::chrono::high_resolution_clock::now();

        // 步骤1：将所有像素亮度乘以2（N² 次操作）
        for (int i = 0; i < N * N; i++) {
            pixels[i] *= 2.0;
        }

        // 步骤2：计算平均值（N² 次操作）
        double sum = 0.0;
        for (int i = 0; i < N * N; i++) {
            sum += pixels[i];
        }
        double avg = sum / (N * N);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "  顺序执行：avg=" << avg << "  时间=" << elapsed.count() << "s\n";
        return elapsed.count();
    }

    /**
     * 方案1：步骤1并行，步骤2串行
     *
     * 分析：
     * - 步骤1 并行执行时间 = N² / P
     * - 步骤2 串行执行时间 = N²（无法并行化）
     * - 总时间 = N²/P + N²
     * - 当 P → ∞ 时，加速比 → (2×N²)/(N²) = 2
     *
     * 无论有多少处理器，加速比都无法超过2！
     */
    double attempt1(int P) {
        double t1 = static_cast<double>(N * N) / P;  // 步骤1并行时间
        double t2 = static_cast<double>(N * N);        // 步骤2串行时间
        double speedup = (2.0 * N * N) / (t1 + t2);
        return speedup;
    }

    /**
     * 方案2：步骤1并行，步骤2也并行（计算partial和+合并）
     *
     * 分析：
     * - 步骤1 并行执行时间 = N² / P
     * - 步骤2 并行partial和 + 串行合并 = N²/P + P
     * - 总时间 = 2×N²/P + P
     * - 当 N >> P 时，P 项可忽略，加速比 → P
     *
     * 这展示了通过重新设计算法来减少串行比例的重要性。
     */
    double attempt2(int P) {
        double t1 = static_cast<double>(N * N) / P;       // 步骤1并行时间
        double t2 = static_cast<double>(N * N) / P + P;    // 步骤2并行+合并
        double speedup = (2.0 * N * N) / (t1 + t2);
        return speedup;
    }
};

// ============================================================================
// 第3部分：带并行化开销的阿姆达尔定律模拟
// ============================================================================

/**
 * 模拟一个并行程序，其中：
 * - S 比例是固有串行部分
 * - (1-S) 比例是完全可并行部分
 * - O 是由于并行管理（同步、通信等）引入的开销
 *
 * 带开销的阿姆达尔定律：
 * speedup(P) = 1 / (S + (1-S)/P + overhead(P))
 *
 * 开销通常随 P 增长（例如树形归约的开销 ∝ log(P)），
 * 这意味着即使串行部分很小，开销也可能在大规模时成为瓶颈。
 */
double amdahl_with_overhead(double S, int P, double overhead_per_task) {
    double parallel_portion = 1.0 - S;
    double parallel_time = parallel_portion / P;
    // 树形归约等操作的通信开销通常为 O(log P)
    double overhead = overhead_per_task * std::log2(P);
    return 1.0 / (S + parallel_time + overhead);
}

void analyze_overhead_impact() {
    std::cout << "\n=== 带并行化开销的阿姆达尔定律 ===\n\n";
    std::cout << "┌────────┬────────────────────────────────────────┐\n";
    std::cout << "│   P    │ 无开销    O=0.001    O=0.01    O=0.1   │\n";
    std::cout << "├────────┼────────────────────────────────────────┤\n";

    double S = 0.01;  // 1% 串行
    std::vector<double> overheads = {0.0, 0.001, 0.01, 0.1};

    for (int P : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        std::cout << "│ " << std::setw(6) << P << " │";
        for (double ov : overheads) {
            double sp = amdahl_with_overhead(S, P, ov);
            std::cout << "  " << std::setw(7) << std::fixed << std::setprecision(2) << sp;
        }
        std::cout << " │\n";
    }
    std::cout << "└────────┴────────────────────────────────────────┘\n";
    std::cout << "\n观察：即使每个并行任务的开销很小，当 P 很大时，\n";
    std::cout << "由于开销随 log(P) 或 P 增长，也会严重限制可扩展性。\n";
}

// ============================================================================
// 第4部分：Summit 超级计算机示例
//
// Summit 是美国最快的超级计算机之一，拥有惊人的并行能力，
// 但阿姆达尔定律告诉我们，即使如此多的ALU也无法突破串行瓶颈。
// ============================================================================

void summit_example() {
    std::cout << "\n=== Summit 超级计算机规模分析 ===\n\n";

    // Summit：27,648 个 GPU × 5,376 ALU/GPU = 148,635,648 个 ALU
    long long alus = 148635648LL;

    std::cout << "Summit 超级计算机：" << alus << " 个并行 ALU\n\n";

    std::vector<double> serial_fractions = {0.1, 0.01, 0.001, 0.0001, 0.00001};
    std::cout << "┌───────────┬──────────────┬───────────────────────────┐\n";
    std::cout << "│ 串行比例  │ 最大加速比   │ 有效ALU利用率              │\n";
    std::cout << "├───────────┼──────────────┼───────────────────────────┤\n";

    for (double S : serial_fractions) {
        double sp = amdahl_speedup(S, alus);
        double utilized = sp / alus * 100.0;
        std::cout << "│ " << std::setw(9) << std::fixed << std::setprecision(4) << S * 100 << "%"
                  << " │ " << std::setw(12) << std::fixed << std::setprecision(1) << sp
                  << " │ " << std::setw(16) << std::fixed << std::setprecision(6) << utilized
                  << "%    │\n";
    }
    std::cout << "└───────────┴──────────────┴───────────────────────────┘\n";
    std::cout << "\n关键洞察：当串行代码占0.1%时，在拥有1.48亿并行操作能力的\n";
    std::cout << "机器上，最大加速比也仅约1000倍！\n";
}

// ============================================================================
// 第5部分：测量实际并行加速比
//
// 理论分析很重要，但实际测量更关键。
// 真实系统中的开销（线程创建、缓存竞争、同步等）
// 往往比理论模型预测的更大。
// ============================================================================

/**
 * 通过使用不同数量的线程运行工作负载来测量实际加速比。
 * 这展示了理论（阿姆达尔定律）与实际的差距。
 */
double parallel_workload(int N, int P) {
    std::vector<std::thread> threads;
    std::vector<double> partial_sums(P, 0.0);

    auto worker = [&](int tid) {
        int chunk = N / P;
        int start = tid * chunk;
        int end = (tid == P - 1) ? N : start + chunk;
        double local = 0.0;
        for (int i = start; i < end; i++) {
            local += std::sqrt(static_cast<double>(i + 1));
        }
        partial_sums[tid] = local;
    };

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < P; t++) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) t.join();

    // 串行归约（这是阿姆达尔定律中的"S"部分）
    double total = 0.0;
    for (double s : partial_sums) total += s;

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void measure_speedup() {
    std::cout << "\n=== 测量实际加速比（sqrt 求和）===\n\n";
    const int N = 1000000;
    const int HW_THREADS = static_cast<int>(std::thread::hardware_concurrency());

    std::cout << "可用硬件线程数：" << HW_THREADS << "\n";
    std::cout << "问题规模：N = " << N << "\n\n";

    // 测量串行时间（基准）
    double t1 = parallel_workload(N, 1);
    std::cout << "┌────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "│   P    │  时间 (s)    │   加速比     │   效率 (%)    │\n";
    std::cout << "├────────┼──────────────┼──────────────┼──────────────┤\n";

    for (int P = 1; P <= HW_THREADS && P <= 16; P++) {
        double tP = parallel_workload(N, P);
        double speedup = t1 / tP;
        double efficiency = speedup / P * 100.0;
        std::cout << "│ " << std::setw(6) << P
                  << " │  " << std::setw(10) << std::fixed << std::setprecision(6) << tP
                  << " │  " << std::setw(8) << std::fixed << std::setprecision(3) << speedup
                  << "  │  " << std::setw(8) << std::fixed << std::setprecision(1) << efficiency
                  << "%  │\n";
    }
    std::cout << "└────────┴──────────────┴──────────────┴──────────────┘\n";
    std::cout << "\n效率下降原因：串行归约、线程开销、阿姆达尔定律限制。\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "第4讲 第2部分：阿姆达尔定律 -- 加速比分析\n";
    std::cout << "============================================================\n";

    // 第1部分：打印阿姆达尔定律加速比表
    print_amdahl_table();

    // 第2部分：图像处理示例
    std::cout << "\n=== 图像处理示例（N×N 像素）===\n";
    ImageProcessor img(100);
    double t_seq = img.sequential();

    std::cout << "\n方案1（步骤1并行，步骤2串行）：\n";
    for (int P : {1, 2, 4, 8, 16, 32}) {
        double sp = img.attempt1(P);
        std::cout << "  P=" << P << "：加速比 ≤ " << std::fixed << std::setprecision(2) << sp << "\n";
    }
    std::cout << "  → 加速比上限为2（步骤2为串行）\n";

    std::cout << "\n方案2（两步都并行，合并partial和）：\n";
    for (int P : {1, 2, 4, 8, 16, 32}) {
        double sp = img.attempt2(P);
        std::cout << "  P=" << P << "：加速比 ≈ " << std::fixed << std::setprecision(2) << sp << "\n";
    }
    std::cout << "  → 当 N >> P 时，加速比 → P（近线性扩展）\n";

    // 第3部分：开销影响
    analyze_overhead_impact();

    // 第4部分：Summit 示例
    summit_example();

    // 第5部分：测量实际加速比
    measure_speedup();

    std::cout << "\n=== 阿姆达尔定律关键要点 ===\n";
    std::cout << "1. 加速比 ≤ 1/S，其中 S = 串行比例\n";
    std::cout << "2. 微小的串行区域会严重限制大规模并行性\n";
    std::cout << "3. 并行化开销（同步、通信）增加了有效的 S\n";
    std::cout << "4. 在扩展到多处理器之前，务必先最小化串行部分\n";
    std::cout << "5. 实际测量，不要仅依赖理论 -- 真实开销很重要\n";

    std::cout << "\n所有测试成功完成。\n";
    return 0;
}
