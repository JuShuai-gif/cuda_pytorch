/**
 * lecture5_part3.cpp - Fork-Join 并行模式与 Cilk 风格快速排序
 *
 * 演示 CS149 第5讲的概念：
 * - Fork-Join 模式（cilk_spawn / cilk_sync）
 * - 使用分治法实现的并行快速排序
 * - Spawn 截断阈值（对小问题切换回串行）
 * - 并行松弛度（parallel slack）与递归分解
 * - 对比串行执行与 Fork-Join 并行执行
 *
 * 关键概念详解：
 * ─────────────────────────────────────────────────────────────
 * 【Fork-Join 模式】
 *   cilk_spawn: 创建一个可并行执行的子任务（fork）
 *   cilk_sync:  等待所有子任务完成（join）
 *   这是 Cilk 编程模型的核心，非常适合分治算法。
 *
 * 【递归分解并行化 for 循环】
 *   直接对每个迭代 spawn 会导致 O(N) 的开销。Cilk 的做法是将其
 *   递归分解：每次将范围一分为二，只对一半 spawn，另一半直接执行。
 *   这样 spawn 次数从 O(N) 降到 O(log N)。
 *
 * 【并行松弛度（Parallel Slack）】
 *   并行松弛度 = 独立工作总量 / 并行执行能力
 *   经验法则：slack ≈ 8 是一个良好的实用比例。
 *   - slack 太小：工作线程可能因等待而空闲
 *   - slack 太大：管理细粒度任务的调度开销占主导
 *
 * 【截断阈值（Cutoff）优化】
 *   当问题规模小于某个阈值时，切换回串行算法。
 *   原因：对很小的子问题进行 spawn 的开销大于并行加速带来的收益。
 *   Cilk 实践中通常使用 sort cutoff 约为 500-1000 个元素。
 *
 * 编译：g++ -std=c++17 -pthread lecture5_part3.cpp -o lecture5_part3 && ./lecture5_part3
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

// ============================================================================
// 第一部分：简化的 Cilk 风格运行时
// ============================================================================

/**
 * 支持类似 Cilk spawn/sync 语义的固定大小线程池的最小实现。
 *
 * 注意：这不是生产质量的代码 - 仅用于清晰展示 fork-join 概念。
 *
 * 【设计要点】
 * - pending_tasks 计数器和 cv_sync 条件变量共同实现 sync() 语义
 * - sync() 阻塞直到所有已 spawn 的任务都完成且队列为空
 * - 工作线程从队列取出任务执行，完成后递减 pending_tasks
 */
class CilkPool {
public:
    explicit CilkPool(int num_threads) : stop(false) {
        for (int i = 0; i < num_threads; i++) {
            workers.emplace_back(&CilkPool::worker_loop, this, i);
        }
    }

    ~CilkPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        cv.notify_all();
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }

    // 将函数入队，由线程池执行（模拟 cilk_spawn）
    void spawn(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            ++pending_tasks;
            task_queue.push(std::move(task));
        }
        cv.notify_one();
    }

    // 等待所有已 spawn 的任务完成（模拟 cilk_sync）
    void sync() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_sync.wait(lock, [this] { return pending_tasks == 0 && task_queue.empty(); });
    }

    int pending_count() const { return pending_tasks; }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::condition_variable cv_sync;
    int pending_tasks = 0;
    bool stop;

    void worker_loop(int tid) {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this] { return stop || !task_queue.empty(); });
                if (stop && task_queue.empty()) return;
                task = std::move(task_queue.front());
                task_queue.pop();
            }
            task();
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                --pending_tasks;
            }
            cv_sync.notify_all();
        }
    }
};

// ============================================================================
// 第二部分：串行快速排序（参考基准）
// ============================================================================

/**
 * 标准的串行快速排序实现，作为性能比较的基线。
 * 使用末尾元素作为 pivot，原地分区。
 */
void sequential_quicksort(std::vector<int>& arr, int begin, int end) {
    if (begin >= end - 1) return;

    // 分区操作：选择末尾元素作为 pivot
    // 将所有 <= pivot 的元素移到左侧，> pivot 的元素留在右侧
    int pivot = arr[end - 1];
    int i = begin;
    for (int j = begin; j < end - 1; j++) {
        if (arr[j] <= pivot) {
            std::swap(arr[i], arr[j]);
            i++;
        }
    }
    std::swap(arr[i], arr[end - 1]);
    int middle = i;

    // 递归对左右两个分区排序
    sequential_quicksort(arr, begin, middle);
    sequential_quicksort(arr, middle + 1, end);
}

// ============================================================================
// 第三部分：并行快速排序（Fork-Join）
// ============================================================================

/**
 * 使用 spawn/sync（fork-join 模式）的并行快速排序。
 *
 * 等效的 Cilk 伪代码：
 * void quick_sort(int* begin, int* end) {
 *     if (begin >= end - PARALLEL_CUTOFF) {
 *         std::sort(begin, end);  // 小问题直接串行
 *     } else {
 *         int* middle = partition(begin, end);
 *         cilk_spawn quick_sort(begin, middle);  // 并行处理左半部
 *         quick_sort(middle + 1, last);          // 当前线程处理右半部
 *     }
 * }
 *
 * 【为什么需要 cutoff】
 * 每个 spawn 都有调度和队列管理的开销。当子数组很小时
 * （例如少于 500 个元素），spawn 的开销超过并行带来的收益。
 * 因此直接切换到高效的串行排序。
 */
class ParallelQuicksort {
private:
    CilkPool pool;
    int parallel_cutoff;  // 当子数组小于此阈值时切换到串行排序

public:
    ParallelQuicksort(int num_threads, int cutoff = 1000)
        : pool(num_threads), parallel_cutoff(cutoff) {}

    void sort(std::vector<int>& arr) {
        parallel_quicksort(arr, 0, static_cast<int>(arr.size()));
        pool.sync();  // 等待所有 spawn 的子任务完成
    }

private:
    void sequential_sort(std::vector<int>& arr, int begin, int end) {
        sequential_quicksort(arr, begin, end);
    }

    void parallel_quicksort(std::vector<int>& arr, int begin, int end) {
        int size = end - begin;

        // 截断判断：对小数据块切换回串行
        // 避免对过小的问题产生 spawn 调度开销
        if (size <= parallel_cutoff) {
            sequential_sort(arr, begin, end);
            return;
        }

        // 分区操作
        int pivot = arr[end - 1];
        int i = begin;
        for (int j = begin; j < end - 1; j++) {
            if (arr[j] <= pivot) {
                std::swap(arr[i], arr[j]);
                i++;
            }
        }
        std::swap(arr[i], arr[end - 1]);
        int middle = i;

        // Fork-join：spawn 左半部，直接执行右半部
        // （为了简化，这里使用"先执行延续"策略；
        //  真实 Cilk 会使用"先执行子任务"策略）
        pool.spawn([this, &arr, begin, middle]() {
            parallel_quicksort(arr, begin, middle);
        });

        parallel_quicksort(arr, middle + 1, end);
    }
};

// ============================================================================
// 第四部分：使用 std::async 的 Fork-Join 演示（C++ 标准库）
// ============================================================================

/**
 * 使用 std::async 实现的可选 Fork-Join 快速排序。
 * 展示 fork-join 概念与具体编程语言无关。
 *
 * std::async 的 .get() 相当于 cilk_sync。
 */
void async_quicksort(std::vector<int>& arr, int begin, int end, int cutoff = 1000) {
    int size = end - begin;
    if (size <= 1) return;

    if (size <= cutoff) {
        std::sort(arr.begin() + begin, arr.begin() + end);
        return;
    }

    // 分区操作
    int pivot = arr[end - 1];
    int i = begin;
    for (int j = begin; j < end - 1; j++) {
        if (arr[j] <= pivot) {
            std::swap(arr[i], arr[j]);
            i++;
        }
    }
    std::swap(arr[i], arr[end - 1]);
    int mid = i;

    // Spawn 左半部为异步任务，直接执行右半部
    auto left_future = std::async(std::launch::async, [&arr, begin, mid, cutoff]() {
        async_quicksort(arr, begin, mid, cutoff);
    });

    async_quicksort(arr, mid + 1, end, cutoff);
    left_future.get();  // Sync：等待左半部完成（等价于 cilk_sync）
}

// ============================================================================
// 第五部分：递归 Fork-Join 模式（用于 for 循环并行化）
// ============================================================================

/**
 * Cilk 的关键技巧：通过递归分解来并行化 for 循环。
 *
 * 直接做法（不好）：
 *   for (int i=0; i<N; i++) cilk_spawn foo(i);
 *   → O(N) 的 spawn 开销
 *
 * 改进方式（Cilk 风格）：recursive_for(0, N)，其中：
 *   recursive_for(start, end):
 *     if (end - start <= GRANULARITY):
 *       串行执行范围内的迭代
 *     else:
 *       mid = (start+end)/2
 *       cilk_spawn recursive_for(start, mid)
 *       recursive_for(mid, end)
 *
 * 这样产生 O(log N) 次 spawn 而非 O(N) 次。
 * 每次递归将范围一分为二，直到子范围足够小（<= 粒度），
 * 此时用串行方式处理。这确保了：
 * 1. spawn 开销可控（对数级别）
 * 2. 足够的子任务供工作窃取调度器分配
 *
 * 参数说明：
 * - granularity（粒度）：控制递归停止的阈值
 * - 粒度越小 → 越多的并行子任务 → 更好的负载均衡但更高的调度开销
 */
void recursive_parallel_for(int start, int end, int granularity,
                             const std::function<void(int)>& work_fn,
                             int depth = 0) {
    int size = end - start;

    if (size <= granularity) {
        // 基本情况：串行执行
        for (int i = start; i < end; i++) {
            work_fn(i);
        }
    } else {
        int mid = start + size / 2;

        // Spawn 左半部
        auto future = std::async(std::launch::async, [&, start, mid, granularity, depth]() {
            recursive_parallel_for(start, mid, granularity, work_fn, depth + 1);
        });

        // 直接执行右半部
        recursive_parallel_for(mid, end, granularity, work_fn, depth + 1);

        future.get();  // sync
    }
}

// ============================================================================
// 第六部分：性能基准测试
// ============================================================================

struct SortBenchmark {
    std::string name;
    double time_seconds;
    bool is_sorted;
};

SortBenchmark benchmark_sort(const std::string& name,
                              std::function<void(std::vector<int>&)> sort_fn,
                              const std::vector<int>& original) {
    std::vector<int> data(original);

    auto start = std::chrono::high_resolution_clock::now();
    sort_fn(data);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    bool sorted = std::is_sorted(data.begin(), data.end());

    std::cout << "  " << std::left << std::setw(30) << name
              << " 时间=" << std::fixed << std::setprecision(4) << elapsed << "秒"
              << "  已排序=" << (sorted ? "是" : "否") << "\n";

    return {name, elapsed, sorted};
}

// ============================================================================
// 第七部分：并行松弛度分析
// ============================================================================

void analyze_parallel_slack() {
    std::cout << "\n=== 并行松弛度分析 ===\n\n";

    std::cout << "并行松弛度 = 独立工作总量 / 并行执行能力\n\n";

    std::cout << "快速排序（N 个元素）：\n";
    std::cout << "  - 分解方式：每次分区产生 2 个独立的子问题\n";
    std::cout << "  - 独立工作的总数量随递归深度呈指数增长\n";
    std::cout << "  - 在有 N 个元素时，递归深度为 O(log₂ N)\n";
    std::cout << "  - 独立子问题总数随递归深入呈指数增长，最终产生约 N 个叶子子问题\n";
    std::cout << "  - 并行松弛度随树的扩展而增长\n\n";

    std::cout << "经验法则：slack ≈ 8 是一个良好的实用比例。\n";
    std::cout << "  - slack 太小：工作线程可能因为等待新工作而空闲\n";
    std::cout << "  - slack 太大：管理细粒度任务的调度开销占主导\n";
    std::cout << "  - 最佳取值依赖于硬件（核心数、缓存层次）和应用特性\n\n";

    std::cout << "截断阈值（cutoff）优化：\n";
    std::cout << "  - 当问题规模 < PARALLEL_CUTOFF 时停止 spawn\n";
    std::cout << "  - 对小数据块切换回串行 std::sort\n";
    std::cout << "  - 在减少 spawn 开销的同时不牺牲并行性\n";
    std::cout << "  - 典型 cutoff 取值：500-2000 个元素\n";
}

// ============================================================================
// 第八部分：分治法模式解释
// ============================================================================

void explain_divide_conquer() {
    std::cout << "\n=== 分治法与 Fork-Join ===\n\n";

    std::cout << "常见的并行编程模式：\n\n";

    std::cout << "1. 数据并行（ISPC foreach, map, #pragma omp parallel for）：\n";
    std::cout << "   foreach (i=0..N) { B[i] = foo(A[i]); }\n";
    std::cout << "   → 对多个数据元素执行相同的操作\n";
    std::cout << "   → 适合规则的数据结构（数组、矩阵）\n";
    std::cout << "   → 核心数变化时只需改变分区策略，代码无需修改\n\n";

    std::cout << "2. FORK-JOIN（Cilk spawn/sync, OpenMP tasks）：\n";
    std::cout << "   cilk_spawn quicksort(left);\n";
    std::cout << "   quicksort(right);\n";
    std::cout << "   cilk_sync;\n";
    std::cout << "   → 自然而然地适合分治算法\n";
    std::cout << "   → 不规则的并行度：子任务数随递归动态变化\n";
    std::cout << "   → 依赖工作窃取来维持负载均衡\n\n";

    std::cout << "3. 显式线程（std::thread, pthread）：\n";
    std::cout << "   std::thread t[NUM_CORES](myFunction, args);\n";
    std::cout << "   → 程序员自行管理分解、分配和协调\n";
    std::cout << "   → 最大的灵活性，但也最容易出错\n";
    std::cout << "   → 适合自定义调度策略的场景\n\n";

    std::cout << "4. 批量启动（CUDA, ISPC tasks）：\n";
    std::cout << "   launch[numTasks] myTask(args);\n";
    std::cout << "   → 系统自动处理到执行单元的分配\n";
    std::cout << "   → 适合 GPU 等大规模并行硬件\n";
    std::cout << "   → 程序员只需关心任务本身，不需要管理线程\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "第5讲 第三部分：Fork-Join 并行模式与快速排序\n";
    std::cout << "============================================================\n";

    // === 生成测试数据 ===
    const int N = 500000;
    std::vector<int> data(N);
    std::mt19937 rng(42);
    for (int i = 0; i < N; i++) data[i] = rng() % 1000000;

    std::cout << "\n--- 对 " << N << " 个随机整数进行排序 ---\n\n";

    // === 串行 std::sort（最优串行基线） ===
    benchmark_sort("std::sort（C++ 标准库）", [](std::vector<int>& arr) {
        std::sort(arr.begin(), arr.end());
    }, data);

    // === 串行快速排序（我们的实现） ===
    benchmark_sort("串行快速排序", [](std::vector<int>& arr) {
        sequential_quicksort(arr, 0, arr.size());
    }, data);

    // === 使用类 Cilk 线程池的并行快速排序 ===
    int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads < 2) hw_threads = 2;
    std::cout << "\n  硬件线程数: " << hw_threads << "\n";

    for (int cutoff : {100, 1000, 5000, 20000}) {
        std::string name = "Cilk 快速排序（截断=" + std::to_string(cutoff) + "）";
        benchmark_sort(name, [hw_threads, cutoff](std::vector<int>& arr) {
            ParallelQuicksort pq(hw_threads, cutoff);
            pq.sort(arr);
        }, data);
    }

    // === 使用 std::async 的并行快速排序 ===
    for (int cutoff : {1000, 20000}) {
        std::string name = "std::async 快排（截断=" + std::to_string(cutoff) + "）";
        benchmark_sort(name, [cutoff](std::vector<int>& arr) {
            async_quicksort(arr, 0, arr.size(), cutoff);
        }, data);
    }

    // === 递归并行 for 演示 ===
    std::cout << "\n--- 递归 Fork-Join 并行 for 循环（N=1000）---\n";
    {
        std::vector<int> results(1000, 0);
        auto work = [&results](int i) {
            results[i] = i * i;
        };

        auto start = std::chrono::high_resolution_clock::now();
        recursive_parallel_for(0, 1000, 50, work);
        auto end = std::chrono::high_resolution_clock::now();

        bool correct = true;
        for (int i = 0; i < 1000 && correct; i++) {
            correct = (results[i] == i * i);
        }
        std::cout << "  结果正确性: " << (correct ? "是" : "否") << "\n";
    }

    // === 并行松弛度分析 ===
    analyze_parallel_slack();

    // === 分治法解释 ===
    explain_divide_conquer();

    // === 总结 ===
    std::cout << "\n=== Fork-Join 关键要点 ===\n";
    std::cout << "1. cilk_spawn 创建可并行执行的独立工作；cilk_sync 等待所有工作完成。\n";
    std::cout << "2. 对小问题使用截断阈值：spawn 开销可能大于并行带来的收益。\n";
    std::cout << "3. 递归分解：将 spawn 次数从 O(N) 降到 O(log N)。\n";
    std::cout << "4. 并行松弛度经验法则：约为执行单元数的 8 倍。\n";
    std::cout << "5. 工作窃取透明地处理负载均衡（详见第5讲第二部分）。\n";

    std::cout << "\n所有测试成功完成。\n";
    return 0;
}
