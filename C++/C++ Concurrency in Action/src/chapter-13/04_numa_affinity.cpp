// 04_numa_affinity.cpp — NUMA 拓扑与 CPU 亲和性 (Linux)
// 演示: 检测 NUMA 拓扑、CPU 绑核、线程迁移

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ===== 1. 检测 NUMA 拓扑 (Linux) =====
void detect_numa_topology() {
    std::cout << "=== NUMA 拓扑检测 ===\n";

#ifdef __linux__
    // 读取 /sys 文件系统获取信息
    auto read_sys = [](const std::string& path) -> std::string {
        FILE* fp = fopen(path.c_str(), "r");
        if (!fp) return "N/A";
        char buf[256] = {};
        fgets(buf, sizeof(buf), fp);
        fclose(fp);
        std::string result(buf);
        if (!result.empty() && result.back() == '\n')
            result.pop_back();
        return result;
    };

    long num_cpus = sysconf(_SC_NPROCESSORS_CONF);
    std::cout << "  CPU 核心数: " << num_cpus << "\n";

    // 尝试读取 NUMA 节点数
    int numa_nodes = 0;
    while (true) {
        std::string path =
            "/sys/devices/system/node/node" + std::to_string(numa_nodes);
        if (access(path.c_str(), F_OK) != 0) break;
        ++numa_nodes;
    }
    std::cout << "  NUMA 节点数: " << numa_nodes << "\n";

    for (int n = 0; n < numa_nodes; ++n) {
        std::string cpulist_path =
            "/sys/devices/system/node/node" + std::to_string(n) + "/cpulist";
        std::string cpulist = read_sys(cpulist_path);
        std::cout << "    Node " << n << " CPUs: " << cpulist << "\n";
    }
#else
    std::cout << "  非 Linux 平台，跳过检测\n";
#endif
}

// ===== 2. CPU 绑核 =====
void demo_cpu_pinning() {
    std::cout << "\n=== CPU 绑核演示 ===\n";

#ifdef __linux__
    auto get_current_cpu = []() -> int {
        return sched_getcpu();
    };

    auto pin_to_core = [](int core_id) -> bool {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        return pthread_setaffinity_np(pthread_self(),
                                      sizeof(cpu_set_t), &cpuset) == 0;
    };

    auto print_affinity = []() {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        std::cout << "    亲和性 CPUs: ";
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &cpuset)) std::cout << i << " ";
        }
        std::cout << "\n";
    };

    // 主线程当前运行的 CPU
    std::cout << "  主线程当前 CPU: " << get_current_cpu() << "\n";
    std::cout << "  主线程";
    print_affinity();

    // 创建工作线程并绑定到 CPU 0
    long num_cpus = sysconf(_SC_NPROCESSORS_CONF);
    int target_core = 0;

    std::jthread pinned_thread([&]() {
        pin_to_core(target_core);
        auto start = std::chrono::high_resolution_clock::now();
        int prev_cpu = get_current_cpu();
        int switches = 0;

        // 持续运行，观察是否会迁移
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count() < 500) {
            int current = get_current_cpu();
            if (current != prev_cpu) {
                ++switches;
                prev_cpu = current;
            }
            // 做一些工作防止被优化掉
            volatile int x = 0;
            for (int i = 0; i < 1000; ++i) x += i;
        }

        std::cout << "  绑定到 CPU " << target_core
                  << " 的线程在 500ms 内迁移 " << switches << " 次\n";
        std::cout << "  (若绑定成功，迁移次数应接近 0)\n";
    });

    pinned_thread.join();
#else
    std::cout << "  非 Linux 平台，跳过 CPU 绑核演示\n";
#endif
}

// ===== 3. 线程迁移对性能的影响 =====
void demo_migration_impact() {
    std::cout << "\n=== 线程迁移对缓存的影饷 ===\n";

    const size_t kDataSize = 8 * 1024 * 1024; // 8M ints = 32MB
    std::vector<int> data(kDataSize, 1);

    auto compute_sum = [&data]() -> long long {
        long long sum = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
        }
        return sum;
    };

    // 预热: 确保数据在某个核心的缓存中
    volatile long long warm = compute_sum();
    (void)warm;

    // 测试: 同一线程连续访问（缓存热）
    auto start = std::chrono::high_resolution_clock::now();
    const int kRounds = 10;
    for (int i = 0; i < kRounds; ++i) {
        volatile long long s = compute_sum();
        (void)s;
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    std::cout << "  热缓存 (10 轮求和): " << elapsed.count() << " ms\n";
    std::cout << "  说明: 线程在核心间迁移会清空 L1/L2 缓存，"
              << "导致冷启动开销\n";
}

int main() {
    detect_numa_topology();
    demo_cpu_pinning();
    demo_migration_impact();

    std::cout << "\n提示:\n";
    std::cout << "  - 使用 `taskset -c 0,1 ./program` 绑定进程到特定 CPU\n";
    std::cout << "  - 使用 `numactl --cpunodebind=0 --membind=0 ./program` "
              << "绑定 NUMA 节点\n";
    std::cout << "  - 生产环境中线程绑核可显著降低 tail latency\n";
    return 0;
}
