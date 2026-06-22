#include "benchmarks.h"

#include <iostream>
#include <thread>

int main() {
    std::cout << "=====================================================\n";
    std::cout << "  C++ 多线程与实时系统基准测试\n";
    std::cout << "  硬件并发数: "
              << std::thread::hardware_concurrency() << " 线程\n";
    std::cout << "=====================================================\n";

    demo_thread_pool();
    demo_lockfree_queue();
    benchmark_queue_comparison<1024>();
    demo_priority_inversion();
    demo_memory_ordering();
    demo_producer_consumer();

    std::cout << "\n所有演示已完成。\n";
    return 0;
}
