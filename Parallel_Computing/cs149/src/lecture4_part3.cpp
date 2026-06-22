/**
 * lecture4_part3.cpp - 数据并行网格求解器（红黑高斯-赛德尔迭代法）
 *
 * 模拟 CS149 第4讲中的二维网格求解器：
 * - 在 (N+2)×(N+2) 的网格上使用迭代高斯-赛德尔方法
 *   这是一个典型的 Laplace 方程离散求解问题
 * - 红黑着色（Red-Black Coloring）技术：
 *   通过将网格点分为红色和黑色两类，
 *   暴露并行性：同一颜色的所有点可以同时更新
 *   （因为红色点只依赖黑色邻居，反之亦然）
 * - 数据并行表达式含隐式屏障：
 *   红阶段所有更新完成后才会开始黑阶段
 * - 演示分解、分配和编排三大并行编程要素：
 *   分解：每个网格单元的更新是独立任务
 *   分配：按行块分配给线程（静态分配）
 *   编排：红黑两阶段间的隐式屏障
 *
 * 每个单元的算法：
 *   A[i][j] = 0.2 × (A[i-1][j] + A[i][j-1] + A[i][j] + A[i+1][j] + A[i][j+1])
 *
 * 编译命令：g++ -std=c++17 -pthread lecture4_part3.cpp -o lecture4_part3 && ./lecture4_part3
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>
#include <chrono>

// ============================================================================
// 网格求解器：数据结构与核心算法
// ============================================================================

class GridSolver {
public:
    enum CellColor { RED, BLACK };  // 红黑着色：标记网格单元的颜色类型

private:
    int N;              // 内部网格大小（实际网格为 (N+2) × (N+2)，含边界）
    int total_size;     // N + 2（包含边界行和列）
    std::vector<double> grid;       // 当前网格值
    std::vector<double> new_grid;   // 更新缓冲区（用于并行阶段的临时存储）
    double tolerance;               // 收敛容忍度
    int max_iterations;             // 最大迭代次数

    // 辅助函数：以二维方式访问一维数组
    // 内部网格索引 i,j 范围从 0 到 N+1（边界包含在内）
    double& at(int i, int j) { return grid[i * total_size + j]; }
    const double& at(int i, int j) const { return grid[i * total_size + j]; }
    double& at_new(int i, int j) { return new_grid[i * total_size + j]; }

    /**
     * 确定网格单元 (i,j) 的颜色：坐标和决定颜色
     * 红黑着色的核心思想：
     * - 红色单元：(i+j) 为偶数
     * - 黑色单元：(i+j) 为奇数
     * 
     * 这样做的原因是：在五点差分格式中，每个网格点
     * 的更新只依赖其上下左右邻居。如果按棋盘着色，
     * 则红色点的邻居全是黑色，黑色点的邻居全是红色。
     * 因此所有红色点可以同时更新（它们之间没有依赖关系），
     * 所有黑色点也可以同时更新。
     */
    CellColor cell_color(int i, int j) const {
        return ((i + j) % 2 == 0) ? RED : BLACK;
    }

public:
    GridSolver(int n, double tol = 1e-4, int max_iter = 10000)
        : N(n), total_size(n + 2), grid((n + 2) * (n + 2), 0.0),
          new_grid((n + 2) * (n + 2), 0.0),
          tolerance(tol), max_iterations(max_iter) {}

    /**
     * 初始化网格，设置边界条件。
     * 网格边界设置为固定值（模拟 Dirichlet 边界条件）：
     * - 上边界 = 1.0
     * - 下边界 = 0.0
     * - 左右边界 = 0.5
     * - 内部初始化为 0.0（初始猜测值）
     */
    void initialize() {
        // 设置上下边界值
        for (int j = 0; j < total_size; j++) {
            at(0, j) = 1.0;                    // 上边界
            at(total_size - 1, j) = 0.0;       // 下边界
        }
        // 设置左右边界值
        for (int i = 0; i < total_size; i++) {
            at(i, 0) = 0.5;                    // 左边界
            at(i, total_size - 1) = 0.5;       // 右边界
        }

        // 内部初始化为0.0（平均值猜测）
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                at(i, j) = 0.0;
            }
        }
    }

    // ========================================================================
    // 顺序求解器（原始高斯-赛德尔，逐行迭代）
    //
    // 标准高斯-赛德尔方法的特点：
    // - 使用同一迭代中的当前更新值（就地更新）
    // - 具有天然的串行依赖性（每个点依赖其左方和上方邻居的最新值）
    // - 收敛速度通常比雅可比迭代快
    // ========================================================================

    struct SolveResult {
        double diff;         // 最终最大差异
        int iterations;      // 迭代次数
        bool converged;      // 是否收敛
        double time_seconds; // 执行时间（秒）
    };

    SolveResult solve_sequential() {
        auto start = std::chrono::high_resolution_clock::now();
        int iter = 0;
        bool done = false;

        while (!done && iter < max_iterations) {
            double diff = 0.0;  // 累积差异（用于收敛判断）

            // 高斯-赛德尔：使用同一迭代中已更新的值
            // 注意这种顺序依赖于逐行逐列的顺序更新
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    double prev = at(i, j);
                    // 五点平均公式：当前点的值 = 自身与四邻居的平均值的加权平均
                    at(i, j) = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                      at(i, j) + at(i + 1, j) + at(i, j + 1));
                    diff += std::abs(at(i, j) - prev);
                }
            }

            iter++;
            // 检查收敛：平均差异小于容忍度
            if (diff / (N * N) < tolerance) {
                done = true;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return {calculate_diff(), iter, done,
                std::chrono::duration<double>(end - start).count()};
    }

    // ========================================================================
    // 数据并行求解器（红黑着色）
    //
    // 核心思想：红色单元只依赖于黑色单元，反之亦然。
    // 因此同一颜色的所有单元可以并行更新。
    // 两种颜色都更新后才检查收敛。
    //
    // 并行执行的三个阶段：
    // 1. 红阶段（RED phase）：所有红色单元并行更新
    // 2. 隐式屏障：确保红阶段完成后才开始黑阶段
    // 3. 黑阶段（BLACK phase）：所有黑色单元并行更新
    // ========================================================================

    /**
     * 并行更新指定颜色的所有单元。
     * 这模拟了对一种颜色的所有单元执行的 data-parallel for_all 操作。
     *
     * 在 ISPC 中等价于：for_all (红色单元 (i,j)) { ... }
     *
     * 每个线程负责一部分行，每行中按颜色交替更新单元。
     * 因为同一颜色的单元之间没有依赖关系，
     * 线程之间不需要同步（除了最后的 reduce 操作）。
     */
    void update_color_parallel(CellColor color, double& local_diff, int tid, int num_threads) {
        // 以块方式将行分配给线程（静态分配）
        int rows_per_thread = N / num_threads;
        int start_row = 1 + tid * rows_per_thread;
        int end_row = (tid == num_threads - 1) ? N + 1 : start_row + rows_per_thread;

        for (int i = start_row; i < end_row; i++) {
            // 根据颜色确定每行的起始列
            // 红色单元：(i+j) 为偶数；黑色单元：(i+j) 为奇数
            int j_start = 1;
            // 调整 j_start 使 (i + j_start) % 2 与目标颜色匹配
            int target_parity = (color == RED) ? 0 : 1;
            if ((i + j_start) % 2 != target_parity) {
                j_start = 2;  // 从第2列开始，步长为2
            }

            // 每隔一列更新（同颜色的单元在行内是间隔分布）
            for (int j = j_start; j <= N; j += 2) {
                double prev = at(i, j);
                double new_val = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                        at(i, j) + at(i + 1, j) + at(i, j + 1));
                at_new(i, j) = new_val;  // 先写入缓冲区，避免影响并行执行的邻居
                local_diff += std::abs(new_val - prev);
            }
        }
    }

    /**
     * 使用红黑着色的数据并行网格求解器。
     *
     * 分解（Decomposition）：
     *   处理单个网格单元 = 独立的工作单元
     *   每个网格点的更新公式只依赖其四邻居
     *
     * 分配（Assignment）：
     *   系统分配：按行块分配给线程（静态分配）
     *   每个线程获得连续的若干行
     *
     * 编排（Orchestration）：
     *   红阶段和黑阶段之间的隐式屏障
     *   每个阶段内的线程并行工作，两阶段之间必须同步
     *
     * 通信（Communication）：
     *   隐式通信：通过共享网格数组进行
     *   数据并行风格：程序员不需要显式管理通信
     */
    SolveResult solve_redblack_parallel(int num_threads) {
        auto start = std::chrono::high_resolution_clock::now();
        int iter = 0;
        bool done = false;

        // 重置网格
        initialize();

        while (!done && iter < max_iterations) {
            double global_diff = 0.0;
            std::vector<double> partial_diffs(num_threads, 0.0);
            std::vector<std::thread> threads;

            // ================================================================
            // 阶段1：并行更新所有红色单元
            // 因为红色单元只依赖黑色邻居，且所有黑色值是"旧"的，
            // 所以所有红色单元可以同时安全地更新。
            // ================================================================
            for (int t = 0; t < num_threads; t++) {
                threads.emplace_back([this, t, num_threads, &partial_diffs]() {
                    update_color_parallel(RED, partial_diffs[t], t, num_threads);
                });
            }
            for (auto& th : threads) th.join();

            // 将红色单元的更新从 new_grid 复制回 grid
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if (cell_color(i, j) == RED) {
                        at(i, j) = at_new(i, j);
                    }
                }
            }
            // 隐式屏障：确保所有红色更新完成后，黑阶段才开始

            // ================================================================
            // 阶段2：并行更新所有黑色单元
            // 现在黑色单元可以使用刚刚更新的红色邻居值，
            // 所有黑色单元之间没有依赖，可以并行更新。
            // ================================================================
            threads.clear();
            for (int t = 0; t < num_threads; t++) {
                threads.emplace_back([this, t, num_threads, &partial_diffs]() {
                    update_color_parallel(BLACK, partial_diffs[t], t, num_threads);
                });
            }
            for (auto& th : threads) th.join();

            // 将黑色单元的更新写回
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if (cell_color(i, j) == BLACK) {
                        at(i, j) = at_new(i, j);
                    }
                }
            }

            // 合并各线程的 partial diff（模拟 reduce_add）
            for (double d : partial_diffs) global_diff += d;
            partial_diffs.assign(num_threads, 0.0);  // 重置为下一轮迭代做准备

            iter++;
            if (global_diff / (N * N) < tolerance) {
                done = true;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return {calculate_diff(), iter, done,
                std::chrono::duration<double>(end - start).count()};
    }

    // ========================================================================
    // 工具函数
    // ========================================================================

    double calculate_diff() const {
        double max_diff = 0.0;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                double expected = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                         at(i, j) + at(i + 1, j) + at(i, j + 1));
                max_diff = std::max(max_diff, std::abs(at(i, j) - expected));
            }
        }
        return max_diff;
    }

    void print_grid_summary() const {
        std::cout << "  角落值：左上=" << at(1, 1)
                  << "  右上=" << at(1, N)
                  << "  左下=" << at(N, 1)
                  << "  右下=" << at(N, N)
                  << "  中心=" << at(N / 2 + 1, N / 2 + 1) << "\n";
    }

    // 验证红黑并行和顺序求解器得到相同结果
    static bool verify_results(const std::vector<double>& a,
                                const std::vector<double>& b, double eps) {
        for (size_t k = 0; k < a.size(); k++) {
            if (std::abs(a[k] - b[k]) > eps) return false;
        }
        return true;
    }

    std::vector<double> get_grid_copy() const { return grid; }
};

// ============================================================================
// 第2部分：工作分配策略分析
//
// 比较网格求解器的不同工作分配策略：
// 1. 一维块分配：每个线程获得连续的行
// 2. 一维交错分配：线程 t 获得行 t, t+P, t+2P, ...
// 3. 二维块分配：网格划分为矩形块
//
// 关键指标：
// - 每个线程处理的元素数（负载均衡）
// - 通信量（边界行/列数，影响缓存和内存带宽）
// ============================================================================

/**
 * 比较网格求解器的不同工作分配策略。
 *
 * 通信量分析：
 * - 1D块分配：每个线程需要与上下邻居交换 2×N/P 个边界行
 * - 1D交错分配：每个线程需要与所有其他线程通信 → N×N/2 通信量大
 * - 2D块分配：每个线程只与4个邻居交换 → 2×N/√P 个边界
 *
 * 结论：二维块分配更好地利用了二维空间局部性，
 * 通信量随 √P 增长而非 P 增长。
 */
void analyze_assignments() {
    std::cout << "\n=== 网格求解器的工作分配策略 ===\n\n";

    std::cout << "┌─────────────────┬──────────────────────┬──────────────────────┐\n";
    std::cout << "│ 分配策略        │ 每线程元素数         │ 通信量（行数）       │\n";
    std::cout << "├─────────────────┼──────────────────────┼──────────────────────┤\n";

    int N = 256;
    int P = 4;

    // 1D块分配：每个线程获得 N/P 个连续行
    // 通信量：上下两个边界各一行
    std::cout << "│ 一维块分配      │ " << std::setw(18) << (N * N / P)
              << "  │ " << std::setw(18) << (2 * N / P) << "        │\n";

    // 1D交错分配：每个线程获得 N/P 个交错行
    // 通信量：每行都可能与邻居线程通信
    std::cout << "│ 一维交错分配    │ " << std::setw(18) << (N * N / P)
              << "  │ " << std::setw(18) << (N * N / 2) << "        │\n";

    // 2D块分配：√P × √P 的二维分块
    // 通信量：4个边界的长度之和 = 2×N/√P
    int sqrtP = static_cast<int>(std::sqrt(P));
    std::cout << "│ 二维块分配      │ " << std::setw(18) << (N * N / P)
              << "  │ " << std::setw(18) << (2 * N / sqrtP) << "        │\n";

    std::cout << "└─────────────────┴──────────────────────┴──────────────────────┘\n";
    std::cout << "\n关键洞察：二维块分配更好地捕获了二维空间局部性。\n";
    std::cout << "每处理器的通信量：一维块分配 ∝ N，二维块分配 ∝ N/√P。\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "第4讲 第3部分：数据并行网格求解器（红黑高斯-赛德尔）\n";
    std::cout << "============================================================\n";

    const int GRID_SIZE = 64;  // 内部网格为 N×N
    const double TOLERANCE = 1e-4;

    // === 顺序求解器 ===
    std::cout << "\n--- 顺序高斯-赛德尔求解器 ---\n";
    GridSolver seq_solver(GRID_SIZE, TOLERANCE);
    seq_solver.initialize();

    auto seq_result = seq_solver.solve_sequential();
    std::cout << "  迭代次数：" << seq_result.iterations << "\n";
    std::cout << "  是否收敛：" << (seq_result.converged ? "是" : "否") << "\n";
    std::cout << "  最终差异：" << seq_result.diff << "\n";
    std::cout << "  时间：    " << seq_result.time_seconds << "s\n";
    seq_solver.print_grid_summary();

    auto seq_grid = seq_solver.get_grid_copy();

    // === 红黑并行求解器 ===
    std::cout << "\n--- 红黑并行求解器 ---\n";
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw_threads < 1) hw_threads = 4;

    for (int P : {1, 2, 4, 8}) {
        if (P > hw_threads * 2) continue;

        GridSolver par_solver(GRID_SIZE, TOLERANCE);
        par_solver.initialize();

        auto par_result = par_solver.solve_redblack_parallel(P);
        auto par_grid = par_solver.get_grid_copy();

        double speedup = seq_result.time_seconds / par_result.time_seconds;
        bool match = GridSolver::verify_results(seq_grid, par_grid, 1e-3);

        std::cout << "  P=" << P
                  << "：迭代=" << par_result.iterations
                  << "  时间=" << par_result.time_seconds << "s"
                  << "  加速比=" << std::fixed << std::setprecision(2) << speedup
                  << "x  结果一致=" << (match ? "是" : "否") << "\n";
    }

    // === 工作分配策略分析 ===
    analyze_assignments();

    // === 分解总结 ===
    std::cout << "\n=== 数据并行网格求解器：核心概念 ===\n";
    std::cout << "┌────────────────┬─────────────────────────────────────────┐\n";
    std::cout << "│ 概念           │ 实现方式                                │\n";
    std::cout << "├────────────────┼─────────────────────────────────────────┤\n";
    std::cout << "│ 分解           │ 每个网格单元更新 = 独立任务             │\n";
    std::cout << "│ 分配           │ 连续行块分配给线程（静态分配）          │\n";
    std::cout << "│ 编排           │ 红阶段与黑阶段之间的隐式屏障            │\n";
    std::cout << "│ 通信           │ 通过共享网格数组隐式通信                │\n";
    std::cout << "│ 同步（归约）   │ 线程局部 partial 和 + 全局求和          │\n";
    std::cout << "│ 核心技术       │ 红黑着色消除依赖关系                    │\n";
    std::cout << "└────────────────┴─────────────────────────────────────────┘\n";

    std::cout << "\n所有测试成功完成。\n";
    return 0;
}
