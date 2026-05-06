/**
 * lecture5_part1.cpp - 静态任务分配与动态任务分配
 *
 * 演示 CS149 第5讲的核心概念：
 * - 静态分配（static assignment）：工作均匀划分，成本可预测
 * - 动态分配（dynamic assignment）：线程在运行时通过共享计数器争抢任务
 * - 半静态分配（semi-static assignment）：周期性重新平衡
 * - 任务粒度（task granularity）：细粒度 vs 粗粒度
 * - 工作队列模型（work queue model）
 *
 * 本程序使用质数测试作为负载（质数测试的执行时间不可预测），
 * 以此来展示动态分配在何时优于静态分配。
 *
 * 关键概念详解：
 * ─────────────────────────────────────────────────────────────
 * 【静态分配】
 *   在任务开始前，将总工作量均匀划分成 P 份（P=线程数），每个线程
 *   处理固定的一段。当所有任务成本相同时，这很高效；但如果有少数
 *   "困难任务"集中在某个线程的区间内，该线程会拖慢整体进度。
 *
 * 【动态分配 - 共享计数器】
 *   维护一个全局原子计数器，每个线程循环执行 fetch_add 来获取
 *   下一个任务索引。任务成本不均时，快的线程自然处理更多任务，
 *   实现了自动负载均衡。代价是每次 fetch_add 都有同步开销。
 *
 * 【任务粒度权衡】
 *   细粒度（granularity=1）：每次只抓取1个任务，负载均衡最佳，
 *     但原子操作开销最高。
 *   粗粒度（granularity=20）：每次抓取一批任务，减少原子操作，
 *     但可能导致最后一个线程处理过多"困难任务"。
 *
 * 【半静态分配】
 *   当任务的近期成本可以预测时（如粒子模拟中粒子缓慢移动），
 *   应用周期性对执行情况进行性能剖析，然后重新调整分配。
 *   在两次调整之间，分配是"静态"的。
 *
 * 编译：g++ -std=c++17 -pthread lecture5_part1.cpp -o lecture5_part1 && ./lecture5_part1
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <algorithm>
#include <atomic>
#include <cmath>

// ============================================================================
// 第一部分：模拟工作负载 - 质数测试
// ============================================================================

/**
 * 模拟质数测试，执行时间各不相同。
 * 数值越大，计算量越大（如果调度器事先不知道输入值的分布，
 * 则每个任务的成本从调度器的视角来看是不可预测的）。
 */
bool test_primality(long long n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    // 试除法 - 工作量与 sqrt(n) 成正比
    long long limit = static_cast<long long>(std::sqrt(static_cast<double>(n)));
    for (long long i = 3; i <= limit; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

// ============================================================================
// 第二部分：静态分配
// ============================================================================

/**
 * 静态分配：每个线程获得一个固定的、预先确定的连续任务区间。
 * 当所有任务成本相同时表现良好。
 * 当任务成本不可预测时表现差（出现负载不均衡）。
 */
std::vector<bool> static_assignment(const std::vector<long long>& inputs,
                                     int num_threads) {
    int N = inputs.size();
    std::vector<bool> results(N, false);

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        int start = t * (N / num_threads);
        int end = (t == num_threads - 1) ? N : start + (N / num_threads);
        threads.emplace_back([&inputs, &results, start, end]() {
            for (int i = start; i < end; i++) {
                results[i] = test_primality(inputs[i]);
            }
        });
    }
    for (auto& th : threads) th.join();
    return results;
}

// ============================================================================
// 第三部分：动态分配（共享计数器方式）
// ============================================================================

/**
 * 使用共享原子计数器实现动态分配。
 * 每个线程循环获取下一个可用的工作项。
 * 当任务成本不一时，负载均衡更优。
 * 代价：原子递增操作带来的同步开销。
 */
std::vector<bool> dynamic_assignment_counter(const std::vector<long long>& inputs,
                                              int num_threads) {
    int N = inputs.size();
    std::vector<bool> results(N, false);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&inputs, &results, &counter, N]() {
            while (true) {
                int i = counter.fetch_add(1);
                if (i >= N) break;
                results[i] = test_primality(inputs[i]);
            }
        });
    }
    for (auto& th : threads) th.join();
    return results;
}

// ============================================================================
// 第四部分：可调粒度的动态分配
// ============================================================================

/**
 * 具有可调粒度的动态分配。
 * GRANULARITY = 1: 细粒度（每进入一次临界区处理1个元素）
 * GRANULARITY = 10: 粗粒度（每进入一次临界区处理10个元素）
 *
 * 权衡：
 * - 细粒度：负载均衡更好，但同步开销更高
 * - 粗粒度：同步开销更低，但负载均衡可能更差
 */
std::vector<bool> dynamic_assignment_granular(const std::vector<long long>& inputs,
                                               int num_threads, int granularity) {
    int N = inputs.size();
    std::vector<bool> results(N, false);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&inputs, &results, &counter, N, granularity]() {
            while (true) {
                int i = counter.fetch_add(granularity);
                if (i >= N) break;
                int end = std::min(i + granularity, N);
                for (int j = i; j < end; j++) {
                    results[j] = test_primality(inputs[j]);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    return results;
}

// ============================================================================
// 第五部分：工作队列模型
// ============================================================================

/**
 * 简单的共享工作队列。
 * 任务被推入队列，工作线程从队列中拉取。
 * 这是最简单的工作队列模型 - 单个队列，多个工作线程。
 *
 * 注意：当多个线程同时访问单个队列时会产生竞争。
 * 使用分布式队列（每个工作线程一个队列）+ 工作窃取的方式更优。
 */
class SimpleWorkQueue {
private:
    std::mutex mtx;
    std::vector<int> tasks;
    int next_task;

public:
    SimpleWorkQueue() : next_task(0) {}

    void add_task(int task) {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.push_back(task);
    }

    bool get_task(int& task) {
        std::lock_guard<std::mutex> lock(mtx);
        if (next_task >= static_cast<int>(tasks.size())) return false;
        task = tasks[next_task++];
        return true;
    }

    size_t size() const { return tasks.size() - next_task; }
};

// ============================================================================
// 第六部分：性能基准测试
// ============================================================================

struct BenchmarkResult {
    double time_seconds;
    int tasks_completed;
    double imbalance_ratio;  // 线程间最大时间 / 最小时间
};

/**
 * 生成混合了"简单"和"困难"质数测试的负载。
 * 在使用静态分配时，这种不均匀的负载会导致负载不均衡。
 */
std::vector<long long> generate_workload(int N, bool balanced) {
    std::vector<long long> data(N);
    std::mt19937 rng(42);

    if (balanced) {
        // 所有任务成本相近（数值集中在 1000 附近，质数测试时间大致相同）
        std::uniform_int_distribution<long long> dist(900, 1100);
        for (int i = 0; i < N; i++) data[i] = dist(rng);
    } else {
        // 不均匀负载：个别很大的数值（高成本）散落在小数值之间
        // 每隔 50 个元素放一个"困难任务"
        for (int i = 0; i < N; i++) {
            if (i % 50 == 0) {
                // 困难任务：对大数进行质数测试（约 1000 倍的工作量）
                data[i] = 1000000 + (rng() % 10000);
            } else {
                // 简单任务：对小数的质数测试
                data[i] = 100 + (rng() % 200);
            }
        }
    }
    return data;
}

template<typename F>
BenchmarkResult benchmark(F&& fn, const std::string& label,
                           const std::vector<long long>& inputs, int num_threads) {
    auto start = std::chrono::high_resolution_clock::now();
    auto results = fn(inputs, num_threads);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    int completed = static_cast<int>(results.size());

    std::cout << "  " << std::left << std::setw(35) << label
              << " 时间=" << std::fixed << std::setprecision(4) << elapsed << "秒"
              << "  任务数=" << completed << "\n";
    return {elapsed, completed, 0.0};
}

// ============================================================================
// 第七部分：负载不均衡的可视化解释
// ============================================================================

void explain_load_imbalance() {
    std::cout << "\n=== 负载不均衡演示 ===\n\n";

    std::cout << "静态分配（分块），P=4 个线程：\n";
    std::cout << "  P1: [简单, 简单, 简单, 困难] → 耗时 3 个单位\n";
    std::cout << "  P2: [简单, 简单, 简单, 简单] → 耗时 4 个单位\n";
    std::cout << "  P3: [简单, 简单, 简单, 简单] → 耗时 4 个单位\n";
    std::cout << "  P4: [简单, 简单, 简单, 简单] → 耗时 4 个单位\n\n";
    std::cout << "  P1 最后在 t=4 时完成，但 P2-P4 从 t=3 就开始空闲。\n";
    std::cout << "  结果：25% 的空闲时间 → 等效的串行比例 S=0.25！\n\n";

    std::cout << "【解释】Amdahl 定律指出，加速比受限于串行部分。\n";
    std::cout << "  即使只有 1/16 的困难任务，静态分配导致的空闲时间\n";
    std::cout << "  相当于引入了人工的「串行瓶颈」。\n\n";

    std::cout << "动态分配（共享计数器），P=4 个线程：\n";
    std::cout << "  P1: [简单, 简单, 困难]  P2: [简单, 简单, 简单, 简单]\n";
    std::cout << "  P3: [简单, 简单, 简单, 简单]  P4: [简单, 简单, 简单]\n\n";
    std::cout << "  工作自然均衡分配，所有线程大约在 t=4 时完成。\n";
    std::cout << "  P1 虽然抓到了困难任务，但它处理的总任务数更少，\n";
    std::cout << "  这使得所有线程几乎同时完成。\n";
}

// ============================================================================
// 第八部分：半静态分配概念
// ============================================================================

void explain_semi_static() {
    std::cout << "\n=== 半静态分配 ===\n\n";

    std::cout << "核心概念：当近期未来的工作成本可以预测时使用。\n";
    std::cout << "  - 应用程序周期性对执行情况进行性能剖析\n";
    std::cout << "  - 根据近期性能重新调整任务分配\n";
    std::cout << "  - 在两次调整之间，分配是「静态」的\n";
    std::cout << "  - 结合了静态分配的低开销和动态分配的适应性\n\n";

    std::cout << "来自课程的实际例子：\n";
    std::cout << "  - 粒子模拟：当粒子缓慢移动时，重新分配区域\n";
    std::cout << "  - 自适应网格：当物体位置改变时重新划分网格，重新分配区域\n";
    std::cout << "  - 成本函数：预估的下一次工作成本 ≈ 最近一次的工作成本\n";
    std::cout << "    （利用了工作成本的时间局部性）\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "第5讲 第一部分：静态 vs 动态任务分配\n";
    std::cout << "============================================================\n";

    const int NUM_THREADS = 4;
    const int N_BALANCED = 200;
    const int N_UNBALANCED = 200;

    // === 生成工作负载 ===
    auto balanced_wl = generate_workload(N_BALANCED, true);
    auto unbalanced_wl = generate_workload(N_UNBALANCED, false);

    // === 基准测试：均衡负载 ===
    std::cout << "\n--- 均衡负载（成本相似的任务） ---\n";
    benchmark([](const auto& in, int p) { return static_assignment(in, p); },
              "静态分配（分块）", balanced_wl, NUM_THREADS);
    benchmark([](const auto& in, int p) { return dynamic_assignment_counter(in, p); },
              "动态分配（计数器）", balanced_wl, NUM_THREADS);

    std::cout << "\n  观察：当成本均衡时，静态 ≈ 动态。\n";
    std::cout << "  静态分配开销更低（无需每个元素的原子操作）。\n";
    std::cout << "  这说明：如果工作负载可预测，静态分配是最优解。\n";

    // === 基准测试：不均衡负载 ===
    std::cout << "\n--- 不均衡负载（混合简单/困难任务） ---\n";
    benchmark([](const auto& in, int p) { return static_assignment(in, p); },
              "静态分配（分块）", unbalanced_wl, NUM_THREADS);
    benchmark([](const auto& in, int p) { return dynamic_assignment_counter(in, p); },
              "动态分配（计数器，细粒度）", unbalanced_wl, NUM_THREADS);
    benchmark([](const auto& in, int p) { return dynamic_assignment_granular(in, p, 5); },
              "动态分配（粒度=5）", unbalanced_wl, NUM_THREADS);
    benchmark([](const auto& in, int p) { return dynamic_assignment_granular(in, p, 20); },
              "动态分配（粒度=20）", unbalanced_wl, NUM_THREADS);

    std::cout << "\n  观察：当成本不均衡时，动态 >> 静态。\n";
    std::cout << "  较粗的粒度降低了同步开销，但可能牺牲负载均衡效果。\n";
    std::cout << "  需要在同步开销与负载均衡之间找到最优的粒度取值。\n";

    // === 验证正确性 ===
    std::cout << "\n--- 正确性验证 ---\n";
    auto ref_results = static_assignment(unbalanced_wl, 1);  // 单线程串行执行作为参考
    auto dyn_results = dynamic_assignment_counter(unbalanced_wl, NUM_THREADS);

    bool correct = (ref_results.size() == dyn_results.size());
    for (size_t i = 0; i < ref_results.size() && correct; i++) {
        correct = (ref_results[i] == dyn_results[i]);
    }
    std::cout << "  静态(1线程) == 动态(4线程): " << (correct ? "是" : "否") << "\n";

    // === 负载不均衡解释 ===
    explain_load_imbalance();

    // === 半静态分配解释 ===
    explain_semi_static();

    // === 任务粒度权衡总结 ===
    std::cout << "\n=== 任务粒度权衡 ===\n";
    std::cout << "┌──────────────┬───────────────────┬─────────────────────┐\n";
    std::cout << "│ 粒度         │ 负载均衡          │ 同步开销            │\n";
    std::cout << "├──────────────┼───────────────────┼─────────────────────┤\n";
    std::cout << "│ 细(1)        │ 最佳              │ 最高（每个元素）    │\n";
    std::cout << "│ 中(5-20)     │ 良好              │ 适中                │\n";
    std::cout << "│ 粗(100+)     │ 可能较差           │ 最低                │\n";
    std::cout << "└──────────────┴───────────────────┴─────────────────────┘\n";
    std::cout << "\n【关键结论】实际应用中，动态分配 + 适中的粒度（5-20）\n";
    std::cout << "  通常是通用场景下的最佳选择。\n";

    std::cout << "\n所有测试成功完成。\n";
    return 0;
}
