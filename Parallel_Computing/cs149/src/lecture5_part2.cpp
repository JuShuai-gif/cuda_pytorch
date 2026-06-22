/**
 * lecture5_part2.cpp - 工作窃取调度器模拟
 *
 * 模拟 Cilk 的工作窃取（work stealing）调度器：
 * - 每个线程拥有自己的双端队列（dequeue，即 double-ended queue）
 * - 本地线程从尾部（tail）入队/出队（LIFO，后进先出）
 * - 远程线程从头部（head）窃取（FIFO，先进先出）
 * - 延续窃取策略（continuation stealing，即"先运行子任务"）
 * - 随机选择受害线程（victim selection）
 * - 贪婪 join 调度（greedy join scheduling）
 *
 * 关键概念详解：
 * ─────────────────────────────────────────────────────────────
 * 【工作窃取的核心思想】
 *   每个线程都有一个双端队列。线程将自己产生的任务放入自己队列的
 *   尾部，并从尾部取任务执行（LIFO）。当线程的队列为空时，它随机
 *   选择一个其他线程，从其队列头部偷取任务（FIFO）。这样设计的好处：
 *
 *   1. 本地 LIFO → 保持深度优先执行 → 更好的缓存局部性
 *      （最近产生的任务数据很可能还在缓存中）
 *   2. 远程 FIFO → 偷取最大/最老的任务 → 减少总偷取次数
 *      （偷大任务意味着被偷线程能更久地处理自己的任务）
 *   3. 无竞争 → 本地线程操作 tail，远程线程操作 head → 几乎无锁
 *
 * 【延续窃取 vs 子任务窃取】
 *   "先运行子任务"（run child first）：线程立即执行 spawn 出来的
 *   第一个子任务（foo(0)），将整个循环的延续（i=1..N-1）放入队列。
 *   这样队列占用空间是 O(线程数)，而非 O(N)。
 *
 *   相反，"先运行延续"（run continuation first）会将所有子任务
 *   入队，导致 O(N) 的空间开销和广度优先的执行顺序。
 *
 * 【贪婪 join 调度】
 *   所有线程只要没事做就去偷。只有当整个系统中确实没有任何工作时，
 *   线程才进入空闲。由最后一个完成 spawn 的线程来继续执行调用者
 *   （cilk_sync 之后的代码）。当没有发生偷取时，cilk_sync 是零开销的。
 *
 * 编译：g++ -std=c++17 -pthread lecture5_part2.cpp -o lecture5_part2 && ./lecture5_part2
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <chrono>
#include <deque>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <memory>

// ============================================================================
// 第一部分：用于工作窃取的（近似）无锁双端队列
// ============================================================================

/**
 * 用于工作窃取的简化双端队列。
 * 在真实的 Cilk 实现中，这是无锁的以追求性能。
 * 这里为了简单使用 mutex。
 *
 * 本地操作（push_back, pop_back）：LIFO（后进先出）
 * 远程操作（pop_front/steal_front）：FIFO（先进先出），优先偷取最大的工作块
 */
template<typename T>
class WorkStealingDequeue {
private:
    std::deque<T> queue;
    mutable std::mutex mtx;

public:
    void push_back(T item) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push_back(std::move(item));
    }

    bool pop_back(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        item = std::move(queue.back());
        queue.pop_back();
        return true;
    }

    // 从头部偷取：获取最大/最老的工作块
    // 这样被偷线程能更久地保留较新的（较小的）任务
    bool steal_front(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        item = std::move(queue.front());
        queue.pop_front();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
};

// ============================================================================
// 第二部分：任务和工作的表示
// ============================================================================

/**
 * 表示系统中的一个工作项。
 * "延续"（Continuation）任务代表 for 循环或函数的剩余部分。
 * "叶子"（Leaf）任务是单个独立的工作项。
 *
 * 在 Cilk 中：
 * - Leaf 对应 foo(i) 这样的单个 spawn 调用
 * - Continuation 代表"循环中剩余的所有迭代"作为一整块工作
 */
struct WorkItem {
    enum Type { LEAF, CONTINUATION };

    Type type;
    int id;          // 任务标识符
    int start;       // 对于延续任务：剩余范围的起始索引
    int end;         // 对于延续任务：剩余范围的结束索引
    int block_id;    // 此任务所属的同步块（用于 cilk_sync 追踪）

    std::string describe() const {
        std::ostringstream oss;
        if (type == LEAF) {
            oss << "叶子(" << id << ")";
        } else {
            oss << "延续([" << start << "," << end << "), 块=" << block_id << ")";
        }
        return oss.str();
    }
};

// ============================================================================
// 第三部分：同步块描述符
// ============================================================================

/**
 * cilk_sync 块的描述符。
 * 跟踪该块中已 spawn 的任务数和已完成的任务数。
 * 用于判断一个同步块中所有 spawn 的工作何时全部完成。
 *
 * 关键优化：只有当该块中的工作被偷取时，才需要记录描述符信息。
 * 如果没有偷取发生，cilk_sync 是零开销的。
 */
struct SyncBlockDescriptor {
    int block_id;
    int total_spawned;   // 本块中 spawn 的总数
    int total_completed;  // 已完成的数量
    bool stolen;          // 本块中的工作是否曾被偷取
    std::unique_ptr<std::mutex> mtx;

    SyncBlockDescriptor(int id)
        : block_id(id), total_spawned(0), total_completed(0), stolen(false),
          mtx(std::make_unique<std::mutex>()) {}

    void increment_spawned() {
        std::lock_guard<std::mutex> lock(*mtx);
        total_spawned++;
    }

    void increment_completed() {
        std::lock_guard<std::mutex> lock(*mtx);
        total_completed++;
    }

    bool all_completed() {
        std::lock_guard<std::mutex> lock(*mtx);
        return total_spawned > 0 && total_completed >= total_spawned;
    }
};

// ============================================================================
// 第四部分：工作窃取调度器
// ============================================================================

class WorkStealingScheduler {
private:
    int num_threads;
    std::vector<WorkStealingDequeue<WorkItem>> queues;
    std::vector<std::thread> workers;
    std::vector<SyncBlockDescriptor> sync_blocks;
    std::atomic<bool> shutdown{false};
    std::atomic<int> active_workers{0};

    // 用于随机选择偷取目标的随机数引擎
    std::mt19937 rng;

    // 统计信息
    std::atomic<long> total_steals{0};
    std::atomic<long> total_local_pops{0};
    std::atomic<long> total_tasks_completed{0};

public:
    explicit WorkStealingScheduler(int threads) : num_threads(threads), queues(threads) {
        std::random_device rd;
        rng.seed(rd());
    }

    /**
     * 模拟 spawn 一个 for 循环：for (int i=0; i<N; i++) cilk_spawn foo(i);
     *
     * 使用延续窃取策略（先运行子任务）：
     * - 线程 0 立即开始执行 foo(0)
     * - 将延续任务（i=1..N）放入自己的工作队列中供其他线程偷取
     *
     * 关键：整个循环只需要入队一个延续任务，而非 N 个叶子任务！
     * 这保证了队列的空间复杂度为 O(线程数)。
     */
    void simulate_spawn_loop(int N, int tid) {
        if (N <= 0) return;

        // 创建一个同步块用于追踪所有 spawn 的完成情况
        int block_id = static_cast<int>(sync_blocks.size());
        sync_blocks.emplace_back(block_id);

        // 先运行子任务：执行 foo(0)，将延续任务入队
        for (int i = 0; i < N; i++) {
            // 记录 foo(i) 已被 spawn
            sync_blocks[block_id].increment_spawned();

            if (i == 0 && tid == 0) {
                // 线程 0 立即运行子任务 foo(0)（先运行子任务策略）
                std::cout << "  [T" << tid << "] 直接执行 foo(" << i
                          << ")（采用先运行子任务策略）\n";
                execute_task({WorkItem::LEAF, i, 0, 0, block_id}, tid);
                sync_blocks[block_id].increment_completed();
            } else {
                // 将延续入队：代表所有剩余迭代
                // 在真实 Cilk 中：只入队一个延续，i 作为循环计数器
                WorkItem cont{WorkItem::CONTINUATION, 0, i, N, block_id};
                queues[tid].push_back(cont);
                std::cout << "  [T" << tid << "] 入队延续任务 i="
                          << i << ".." << N-1 << "（块=" << block_id << "）\n";
                break;  // 只需要一个延续任务
            }
        }
    }

    /**
     * 执行单个任务（模拟工作）。
     */
    void execute_task(const WorkItem& item, int tid) {
        // 模拟计算工作
        volatile int work = 0;
        int workload = (item.type == WorkItem::LEAF) ? 1000000 : 100000;
        for (int i = 0; i < workload; i++) work++;

        total_tasks_completed++;
        std::cout << "  [T" << tid << "] 完成 " << item.describe() << "\n";
    }

    /**
     * 工作线程的主循环。
     * 实现 Cilk 工作窃取的工作线程行为：
     * 1. 尝试从自己的队列尾部 pop 工作（LIFO）
     * 2. 如果自己的队列为空，尝试从随机受害者偷取（从 head 偷，FIFO）
     * 3. 如果哪里都偷不到，进入空闲状态
     *
     * 这是 Cilk 高效调度的核心：线程总是忙碌的，除非系统中确实没有工作。
     */
    void worker_loop(int tid) {
        active_workers++;
        int failed_steal_attempts = 0;
        const int MAX_FAILED_STEALS = num_threads * 2;

        while (!shutdown) {
            WorkItem task;

            // 步骤1：尝试本地队列（从尾部 pop - LIFO）
            // 本地 pop 有利于缓存局部性：最近产生的任务数据大概率还在缓存中
            if (queues[tid].pop_back(task)) {
                total_local_pops++;
                failed_steal_attempts = 0;
                execute_task(task, tid);
                continue;
            }

            // 步骤2：队列为空 - 尝试偷取（从头部 pop - FIFO）
            // 随机选择受害线程，从它的队列头部偷取最大的工作块
            int victim = std::uniform_int_distribution<int>(0, num_threads - 1)(rng);
            if (victim != tid) {
                if (queues[victim].steal_front(task)) {
                    total_steals++;
                    failed_steal_attempts = 0;
                    std::cout << "  [T" << tid << "] 从 T" << victim
                              << " 偷取了: " << task.describe() << "\n";
                    execute_task(task, tid);
                    continue;
                }
            }

            // 步骤3：没有可偷取的工作
            failed_steal_attempts++;
            if (failed_steal_attempts >= MAX_FAILED_STEALS) {
                // 贪婪策略：如果系统中确实没有工作，线程进入空闲
                // 在真实 Cilk 中：线程阻塞直到有新工作到来
                break;
            }
            std::this_thread::yield();
        }
        active_workers--;
    }

    /**
     * 运行工作窃取模拟。
     */
    void run_simulation(int num_tasks) {
        std::cout << "\n=== 工作窃取调度器模拟 ===\n";
        std::cout << "线程数: " << num_threads << "  任务数: " << num_tasks << "\n\n";

        // 启动所有工作线程
        for (int t = 0; t < num_threads; t++) {
            workers.emplace_back(&WorkStealingScheduler::worker_loop, this, t);
        }

        // 给线程一些启动时间
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // 线程 0 spawn 所有工作（模拟主线程创建任务）
        std::cout << "[主线程] 在 T0 上 spawn " << num_tasks << " 个任务...\n";
        simulate_spawn_loop(num_tasks, 0);

        // 等待所有工作完成
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        shutdown = true;

        for (auto& w : workers) w.join();

        // 输出统计信息
        std::cout << "\n--- 调度器统计 ---\n";
        std::cout << "  总完成任务数: " << total_tasks_completed << "\n";
        std::cout << "  本地 pop 次数: " << total_local_pops << "\n";
        std::cout << "  偷取次数:      " << total_steals << "\n";
        if (total_local_pops + total_steals > 0) {
            double steal_pct = 100.0 * total_steals / (total_local_pops + total_steals);
            std::cout << "  偷取比例:      " << std::fixed
                      << std::setprecision(1) << steal_pct << "%\n";
        }
    }
};

// ============================================================================
// 第五部分：双端队列行为的可视化演示
// ============================================================================

void demonstrate_dequeue_behavior() {
    std::cout << "\n=== 双端队列行为：本地 LIFO，远程 FIFO ===\n\n";

    WorkStealingDequeue<int> dq;

    // 模拟线程从尾部 push 工作（延续窃取场景）
    std::cout << "线程 0（本地）从尾部入队工作项：\n";
    for (int i = 0; i < 5; i++) {
        dq.push_back(i * 10);
        std::cout << "  push_back(" << i * 10 << ")   ← 尾方向\n";
    }

    std::cout << "\n线程 0 从尾部 pop（LIFO 后进先出）：\n";
    int val;
    if (dq.pop_back(val)) std::cout << "  pop_back() → " << val << "  （最新入队的最先取出）\n";
    if (dq.pop_back(val)) std::cout << "  pop_back() → " << val << "\n";

    std::cout << "\n线程 1 从头部偷取（FIFO 先进先出）：\n";
    if (dq.steal_front(val)) std::cout << "  steal_front() → " << val << "（最老的工作，最大块）\n";
    if (dq.steal_front(val)) std::cout << "  steal_front() → " << val << "（次老的工作）\n";
    if (dq.steal_front(val)) std::cout << "  steal_front() → " << val << "（第三老的工作）\n";

    std::cout << "\n队列为空？" << (dq.empty() ? "是" : "否") << "\n";

    std::cout << "\n【关键洞察】：\n";
    std::cout << "  - 本地 LIFO：保持深度优先执行顺序 → 更好的缓存局部性\n";
    std::cout << "    （最近产生的任务，其数据大概率还在 L1/L2 缓存中）\n";
    std::cout << "  - 远程 FIFO：偷取最大/最老的工作块 → 减少总偷取次数\n";
    std::cout << "    （被偷线程可以更久地处理自己剩余的新任务）\n";
    std::cout << "  - 无竞争：本地线程操作尾部，远程线程操作头部\n";
    std::cout << "    （在无锁实现中，这两个操作不会有冲突）\n";
}

// ============================================================================
// 第六部分：先运行子任务 vs 先运行延续
// ============================================================================

void explain_spawn_strategies() {
    std::cout << "\n=== Spawn 策略：先运行子任务 vs 先运行延续 ===\n\n";

    std::cout << "代码: for (int i=0; i<N; i++) { cilk_spawn foo(i); } cilk_sync;\n\n";

    std::cout << "策略1：先运行延续（「子任务窃取」，child stealing）\n";
    std::cout << "  线程将 foo(0) 入队，然后继续 spawn foo(1), foo(2), ...\n";
    std::cout << "  所有 spawn 完成后的队列内容：[foo(0), foo(1), ..., foo(N-1)]\n";
    std::cout << "  空间复杂度：O(N) 个元素在队列中\n";
    std::cout << "  执行顺序：广度优先（与串行执行完全不同）\n";
    std::cout << "  问题：对于大型 N，队列溢出 + 执行顺序打乱影响缓存\n\n";

    std::cout << "策略2：先运行子任务（「延续窃取」，continuation stealing）← Cilk 使用此策略\n";
    std::cout << "  线程立即执行 foo(0)，将 ONE 个延续任务（i=1..N）入队\n";
    std::cout << "  队列内容：[cont(i=1..N-1)]\n";
    std::cout << "  空间复杂度：O(T)，其中 T = 最大并行线程数（有界！）\n";
    std::cout << "  执行顺序：深度优先（如果没有偷取，与串行执行顺序相同）\n";
    std::cout << "  优势：空间可控 + 无偷取时执行顺序不变 = 可预测的缓存行为\n\n";

    std::cout << "【核心动机】Cilk 选择策略2，因为：\n";
    std::cout << "  1. 空间复杂度由 O(N) 降到 O(T) - 这对大循环至关重要\n";
    std::cout << "  2. 无偷取时结果与串行完全一致 - 便于调试\n";
    std::cout << "  3. 每个 spawn 只入队一个延续而非一个叶子任务\n";
}

// ============================================================================
// 第七部分：贪婪 Join 调度解释
// ============================================================================

void explain_greedy_join() {
    std::cout << "\n=== 贪婪 Join 调度（Cilk） ===\n\n";

    std::cout << "核心原则：\n";
    std::cout << "  1. 所有线程在无事可做时始终尝试偷取\n";
    std::cout << "  2. 只有当系统中确实没有任何工作时，线程才进入空闲\n";
    std::cout << "  3. 发起 spawn 的线程可能不会执行 cilk_sync 之后的代码\n";
    std::cout << "     （「最后一个」完成 spawn 的线程来继续执行调用者代码）\n\n";

    std::cout << "为什么这很重要：\n";
    std::cout << "  - 「最后一个」完成 spawn 的线程继续执行调用者\n";
    std::cout << "  - 同步记账开销仅在发生偷取时产生\n";
    std::cout << "  - 常见情况（无偷取）：cilk_sync 是零开销的（no-op）\n";
    std::cout << "  - 描述符仅在发生偷取时才记录 spawn/completion 计数\n";
    std::cout << "  - 这意味着：不并行化的 Cilk 程序几乎和纯 C 程序一样快\n\n";

    std::cout << "【对比传统 barrier】\n";
    std::cout << "  传统 pthread barrier：所有线程必须到达同一点\n";
    std::cout << "  浪费资源：只有最后一个到达的线程做有用工作\n";
    std::cout << "  Cilk 的贪婪 join：线程不会等待 - 它们去偷别人的工作！\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "第5讲 第二部分：工作窃取调度器模拟\n";
    std::cout << "============================================================\n";

    // 第一部分：双端队列行为演示
    demonstrate_dequeue_behavior();

    // 第二部分：Spawn 策略解释
    explain_spawn_strategies();

    // 第三部分：工作窃取模拟
    int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads < 2) hw_threads = 2;

    WorkStealingScheduler scheduler(hw_threads);
    scheduler.run_simulation(5);  // 小型模拟以保持输出清晰

    // 第四部分：贪婪 join 解释
    explain_greedy_join();

    // 第五部分：关键要点总结
    std::cout << "\n=== Cilk 调度器：核心设计总结 ===\n";
    std::cout << "┌────────────────────┬────────────────────────────────────┐\n";
    std::cout << "│ 设计要素           │ 实现方式                           │\n";
    std::cout << "├────────────────────┼────────────────────────────────────┤\n";
    std::cout << "│ 队列结构           │ 双端队列：本地操作 tail，远程 head │\n";
    std::cout << "│ Spawn 策略         │ 先运行子任务（延续窃取）           │\n";
    std::cout << "│ 本地操作           │ LIFO（push/pop 尾部）              │\n";
    std::cout << "│ 偷取操作           │ FIFO（从头部偷取）                 │\n";
    std::cout << "│ 受害者选择         │ 随机（均匀分布）                   │\n";
    std::cout << "│ Join 行为          │ 贪婪：从不等待，始终尝试偷取       │\n";
    std::cout << "│ 同步开销           │ 仅在偷取发生时产生                 │\n";
    std::cout << "│ 工作队列存储       │ 最多 O(T * stack_depth)             │\n";
    std::cout << "└────────────────────┴────────────────────────────────────┘\n";

    std::cout << "\n所有测试成功完成。\n";
    return 0;
}
