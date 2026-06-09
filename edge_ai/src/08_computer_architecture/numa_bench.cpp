#include "numa_bench.h"
#include "timer.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

#ifdef HAS_LIBNUMA
#include <numa.h>
#include <numaif.h>
#endif

void demo_numa_awareness() {
    print_header("演示 4: NUMA 节点感知");

#ifdef HAS_LIBNUMA
    if (numa_available() < 0) {
        std::cout << "  此系统上 NUMA 不可用。\n";
        return;
    }

    int num_nodes = numa_num_configured_nodes();
    int num_cpus = numa_num_configured_cpus();
    std::cout << "  NUMA 节点数: " << num_nodes << "\n";
    std::cout << "  CPU 数: " << num_cpus << "\n\n";

    constexpr size_t BUF_SIZE = 64 * 1024 * 1024; // 64MB
    constexpr size_t STEPS = 1'000'000;

    if (num_nodes >= 2) {
        for (int n = 0; n < num_nodes; ++n) {
            std::cout << "  节点 " << n << " CPU: ";
            struct bitmask *bm = numa_allocate_cpumask();
            numa_node_to_cpus(n, bm);
            for (int c = 0; c < num_cpus; ++c) {
                if (numa_bitmask_isbitset(bm, c)) {
                    std::cout << c << " ";
                }
            }
            numa_free_cpumask(bm);
            std::cout << "\n";
        }
        std::cout << "\n";

        // 跨节点延迟测量
        // 在节点 0 上分配
        int64_t *buf_node0 = static_cast<int64_t *>(
            numa_alloc_onnode(BUF_SIZE, 0));
        // 在节点 1 上分配
        int64_t *buf_node1 = static_cast<int64_t *>(
            numa_alloc_onnode(BUF_SIZE, 1));

        if (!buf_node0 || !buf_node1) {
            std::cout << "  无法分配 NUMA 内存。\n";
            if (buf_node0) numa_free(buf_node0, BUF_SIZE);
            if (buf_node1) numa_free(buf_node1, BUF_SIZE);
            return;
        }

        // 触碰页面以提交
        std::memset(buf_node0, 0, BUF_SIZE);
        std::memset(buf_node1, 0, BUF_SIZE);

        // 将当前线程绑定到节点 0 以进行本地测试
        struct bitmask *mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, 0);
        numa_bind(mask); // 之后的内存分配都会分配到节点 0

        std::cout << "  " << std::left << std::setw(20) << "测试"
                  << std::right << std::setw(15) << "时间(ms)"
                  << std::setw(15) << "纳秒/访问" << "\n";
        std::cout << "  " << std::string(50, '-') << "\n";

        for (int local_node = 0; local_node < num_nodes; ++local_node) {
            for (int buf_node = 0; buf_node < num_nodes; ++buf_node) {
                int64_t *buf = (buf_node == 0) ? buf_node0 : buf_node1;

                Timer t;
                t.start();
                volatile long sum = 0;
                for (size_t i = 0; i < STEPS; ++i) {
                    sum += buf[i % (BUF_SIZE / sizeof(int64_t))];
                }
                double ms = t.elapsed_ms();
                double ns = ms * 1e6 / STEPS;
                g_sink = sum;

                std::string label = "线程@N" + std::to_string(local_node)
                                    + "->内存@N" + std::to_string(buf_node);
                std::cout << "  " << std::left << std::setw(20) << label
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(15) << ms
                          << std::fixed << std::setprecision(2)
                          << std::setw(15) << ns << "\n";
            }
        }

        numa_free(buf_node0, BUF_SIZE);
        numa_free(buf_node1, BUF_SIZE);
        numa_free_nodemask(mask);
    } else {
        std::cout << "  单 NUMA 节点系统 - 跳过跨节点测试。\n";
        std::cout << "  本地访问延迟测量:\n\n";

        int64_t *buf = static_cast<int64_t *>(
            numa_alloc_onnode(BUF_SIZE, 0));
        if (buf) {
            std::memset(buf, 0, BUF_SIZE);
            Timer t;
            t.start();
            volatile long sum = 0;
            for (size_t i = 0; i < STEPS; ++i) {
                sum += buf[i % (BUF_SIZE / sizeof(int64_t))];
            }
            double ms = t.elapsed_ms();
            g_sink = sum;
            std::cout << "  本地访问 (节点0): " << std::fixed
                      << std::setprecision(2) << (ms * 1e6 / STEPS)
                      << " ns/次访问\n";
            numa_free(buf, BUF_SIZE);
        }
    }
#else
    std::cout << "  libnuma 不可用。请安装 libnuma-dev:\n";
    std::cout << "    sudo apt install libnuma-dev  (Debian/Ubuntu)\n";
    std::cout << "  然后使用 cmake 重新构建。\n";
#endif
}
