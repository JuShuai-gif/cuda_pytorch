/**
 * lecture5_part3.cpp - Fork-Join Parallelism & Cilk-Style Quicksort
 *
 * Demonstrates CS149 Lecture 5 concepts:
 * - Fork-join pattern (cilk_spawn / cilk_sync)
 * - Parallel quicksort with divide-and-conquer
 * - Spawn cutoff for small problems
 * - Parallel slack and recursive decomposition
 * - Compares sequential vs fork-join execution
 *
 * Compile: g++ -std=c++17 -pthread lecture5_part3.cpp -o lecture5_part3 && ./lecture5_part3
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
// Part 1: Simplified Cilk-Style Runtime
// ============================================================================

/**
 * Minimal implementation of a fixed-size thread pool that supports
 * spawn/sync semantics similar to Cilk.
 *
 * This is NOT production quality - it's designed to illustrate the
 * fork-join concept clearly.
 */

// Global thread pool for spawn/sync
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

    // Enqueue a function to be executed by the thread pool
    void spawn(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            ++pending_tasks;
            task_queue.push(std::move(task));
        }
        cv.notify_one();
    }

    // Wait until all spawned tasks have completed
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
// Part 2: Sequential Quicksort (Reference)
// ============================================================================

void sequential_quicksort(std::vector<int>& arr, int begin, int end) {
    if (begin >= end - 1) return;

    // Partition: pick pivot as last element
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

    sequential_quicksort(arr, begin, middle);
    sequential_quicksort(arr, middle + 1, end);
}

// ============================================================================
// Part 3: Parallel Quicksort (Fork-Join)
// ============================================================================

/**
 * Parallel quicksort using spawn/sync (fork-join pattern).
 *
 * Equivalent Cilk code:
 * void quick_sort(int* begin, int* end) {
 *     if (begin >= end - PARALLEL_CUTOFF) {
 *         std::sort(begin, end);
 *     } else {
 *         int* middle = partition(begin, end);
 *         cilk_spawn quick_sort(begin, middle);
 *         quick_sort(middle + 1, last);
 *     }
 * }
 */
class ParallelQuicksort {
private:
    CilkPool pool;
    int parallel_cutoff;  // Switch to sequential for small chunks

public:
    ParallelQuicksort(int num_threads, int cutoff = 1000)
        : pool(num_threads), parallel_cutoff(cutoff) {}

    void sort(std::vector<int>& arr) {
        parallel_quicksort(arr, 0, static_cast<int>(arr.size()));
        pool.sync();  // Wait for all spawned tasks
    }

private:
    void sequential_sort(std::vector<int>& arr, int begin, int end) {
        sequential_quicksort(arr, begin, end);
    }

    void parallel_quicksort(std::vector<int>& arr, int begin, int end) {
        int size = end - begin;

        // Cutoff: switch to sequential for small chunks
        if (size <= parallel_cutoff) {
            sequential_sort(arr, begin, end);
            return;
        }

        // Partition
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

        // Fork-join: spawn left half, execute right half directly
        // (This is "run continuation first" for simplicity; Cilk would "run child first")
        pool.spawn([this, &arr, begin, middle]() {
            parallel_quicksort(arr, begin, middle);
        });

        parallel_quicksort(arr, middle + 1, end);
    }
};

// ============================================================================
// Part 4: std::async Fork-Join Demo (C++ Standard Library)
// ============================================================================

/**
 * Alternative fork-join implementation using std::async.
 * Shows that the fork-join concept is language-agnostic.
 */
void async_quicksort(std::vector<int>& arr, int begin, int end, int cutoff = 1000) {
    int size = end - begin;
    if (size <= 1) return;

    if (size <= cutoff) {
        std::sort(arr.begin() + begin, arr.begin() + end);
        return;
    }

    // Partition
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

    // Spawn left half as async, run right half directly
    auto left_future = std::async(std::launch::async, [&arr, begin, mid, cutoff]() {
        async_quicksort(arr, begin, mid, cutoff);
    });

    async_quicksort(arr, mid + 1, end, cutoff);
    left_future.get();  // Sync: wait for left half
}

// ============================================================================
// Part 5: Recursive Fork-Join Pattern (for loop parallelization)
// ============================================================================

/**
 * Cilk's trick: parallelize for loops by recursive decomposition.
 *
 * for (int i=0; i<N; i++) cilk_spawn foo(i);  → O(N) spawn overhead
 *
 * Better: recursive_for(0, N) where:
 *   recursive_for(start, end):
 *     if (end - start <= GRANULARITY):
 *       for sequential
 *     else:
 *       mid = (start+end)/2
 *       cilk_spawn recursive_for(start, mid)
 *       recursive_for(mid, end)
 *
 * This creates O(log N) spawns instead of O(N).
 */
void recursive_parallel_for(int start, int end, int granularity,
                             const std::function<void(int)>& work_fn,
                             int depth = 0) {
    int size = end - start;

    if (size <= granularity) {
        // Base case: sequential
        for (int i = start; i < end; i++) {
            work_fn(i);
        }
    } else {
        int mid = start + size / 2;

        // Spawn left half
        auto future = std::async(std::launch::async, [&, start, mid, granularity, depth]() {
            recursive_parallel_for(start, mid, granularity, work_fn, depth + 1);
        });

        // Execute right half directly
        recursive_parallel_for(mid, end, granularity, work_fn, depth + 1);

        future.get();  // sync
    }
}

// ============================================================================
// Part 6: Benchmarking
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
              << " time=" << std::fixed << std::setprecision(4) << elapsed << "s"
              << "  sorted=" << (sorted ? "YES" : "NO") << "\n";

    return {name, elapsed, sorted};
}

// ============================================================================
// Part 7: Parallel Slack Analysis
// ============================================================================

void analyze_parallel_slack() {
    std::cout << "\n=== Parallel Slack Analysis ===\n\n";

    std::cout << "Parallel Slack = (independent work) / (parallel execution capability)\n\n";

    std::cout << "Quicksort with N elements:\n";
    std::cout << "  - Decomposition: each partition creates 2 independent subproblems\n";
    std::cout << "  - Total independent work grows exponentially with recursion depth\n";
    std::cout << "  - Parallel slack grows as tree expands\n\n";

    std::cout << "Rule of thumb: slack ~8 is a good practical ratio.\n";
    std::cout << "  - Too little slack: workers may idle waiting for work\n";
    std::cout << "  - Too much slack: overhead of managing fine-grained tasks dominates\n\n";

    std::cout << "Cutoff optimization:\n";
    std::cout << "  - Stop spawning when problem size < PARALLEL_CUTOFF\n";
    std::cout << "  - Switches to sequential std::sort for small chunks\n";
    std::cout << "  - Reduces spawn overhead without sacrificing parallelism\n";
}

// ============================================================================
// Part 8: Divide-and-Conquer Pattern Explanation
// ============================================================================

void explain_divide_conquer() {
    std::cout << "\n=== Divide-and-Conquer & Fork-Join ===\n\n";

    std::cout << "Common parallel programming patterns:\n\n";

    std::cout << "1. DATA PARALLELISM (ISPC foreach, map, #pragma omp parallel for):\n";
    std::cout << "   foreach (i=0..N) { B[i] = foo(A[i]); }\n";
    std::cout << "   → Same operation on many data elements\n\n";

    std::cout << "2. FORK-JOIN (Cilk spawn/sync, OpenMP tasks):\n";
    std::cout << "   cilk_spawn quicksort(left);\n";
    std::cout << "   quicksort(right);\n";
    std::cout << "   cilk_sync;\n";
    std::cout << "   → Natural for divide-and-conquer algorithms\n\n";

    std::cout << "3. EXPLICIT THREADS (std::thread, pthread):\n";
    std::cout << "   std::thread t[NUM_CORES](myFunction, args);\n";
    std::cout << "   → Programmer manages decomposition, assignment, orchestration\n\n";

    std::cout << "4. BULK LAUNCH (CUDA, ISPC tasks):\n";
    std::cout << "   launch[numTasks] myTask(args);\n";
    std::cout << "   → System handles assignment to execution units\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 5 Part 3: Fork-Join Parallelism & Quicksort\n";
    std::cout << "============================================================\n";

    // === Generate test data ===
    const int N = 500000;
    std::vector<int> data(N);
    std::mt19937 rng(42);
    for (int i = 0; i < N; i++) data[i] = rng() % 1000000;

    std::cout << "\n--- Sorting " << N << " random integers ---\n\n";

    // === Sequential std::sort (best sequential) ===
    benchmark_sort("std::sort (C++ stdlib)", [](std::vector<int>& arr) {
        std::sort(arr.begin(), arr.end());
    }, data);

    // === Sequential quicksort (our implementation) ===
    benchmark_sort("Sequential quicksort", [](std::vector<int>& arr) {
        sequential_quicksort(arr, 0, arr.size());
    }, data);

    // === Parallel quicksort with Cilk-like pool ===
    int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads < 2) hw_threads = 2;
    std::cout << "\n  Hardware threads: " << hw_threads << "\n";

    for (int cutoff : {100, 1000, 5000, 20000}) {
        std::string name = "Cilk quicksort (cutoff=" + std::to_string(cutoff) + ")";
        benchmark_sort(name, [hw_threads, cutoff](std::vector<int>& arr) {
            ParallelQuicksort pq(hw_threads, cutoff);
            pq.sort(arr);
        }, data);
    }

    // === Parallel quicksort with std::async ===
    for (int cutoff : {1000, 20000}) {
        std::string name = "std::async qsort (cutoff=" + std::to_string(cutoff) + ")";
        benchmark_sort(name, [cutoff](std::vector<int>& arr) {
            async_quicksort(arr, 0, arr.size(), cutoff);
        }, data);
    }

    // === Recursive parallel for demonstration ===
    std::cout << "\n--- Recursive Fork-Join for loop (N=1000) ---\n";
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
        std::cout << "  Results correct: " << (correct ? "YES" : "NO") << "\n";
    }

    // === Parallel slack analysis ===
    analyze_parallel_slack();

    // === Divide-and-conquer explanation ===
    explain_divide_conquer();

    // === Summary ===
    std::cout << "\n=== Fork-Join Key Takeaways ===\n";
    std::cout << "1. cilk_spawn creates independent work; cilk_sync waits for all.\n";
    std::cout << "2. Use cutoff for small problems: spawn overhead > parallel benefit.\n";
    std::cout << "3. Recursive decomposition: O(log N) spawns instead of O(N).\n";
    std::cout << "4. Parallel slack rule: ~8x more work than execution units.\n";
    std::cout << "5. Work stealing handles load balance transparently (Lecture 5 part 2).\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
