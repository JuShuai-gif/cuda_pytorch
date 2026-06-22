#include "sys_info.h"
#include "timer.h"

#include <iostream>
#include <unistd.h>

void demo_cpu_info() {
    print_header("演示 7: 系统 CPU 与缓存信息");

    std::cout << "  使用 sysconf 查询系统信息:\n\n";

#ifdef _SC_LEVEL1_DCACHE_SIZE
    std::cout << "  L1 数据缓存大小:  "
              << sysconf(_SC_LEVEL1_DCACHE_SIZE) << " 字节\n";
#endif
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
    std::cout << "  L1 数据缓存行:  "
              << sysconf(_SC_LEVEL1_DCACHE_LINESIZE) << " 字节\n";
#endif
#ifdef _SC_LEVEL1_ICACHE_SIZE
    std::cout << "  L1 指令缓存大小:  "
              << sysconf(_SC_LEVEL1_ICACHE_SIZE) << " 字节\n";
#endif
#ifdef _SC_LEVEL2_CACHE_SIZE
    std::cout << "  L2 缓存大小:    "
              << sysconf(_SC_LEVEL2_CACHE_SIZE) << " 字节\n";
#endif
#ifdef _SC_LEVEL2_CACHE_LINESIZE
    std::cout << "  L2 缓存行:    "
              << sysconf(_SC_LEVEL2_CACHE_LINESIZE) << " 字节\n";
#endif
#ifdef _SC_LEVEL3_CACHE_SIZE
    std::cout << "  L3 缓存大小:    "
              << sysconf(_SC_LEVEL3_CACHE_SIZE) << " 字节\n";
#endif
#ifdef _SC_LEVEL3_CACHE_LINESIZE
    std::cout << "  L3 缓存行:    "
              << sysconf(_SC_LEVEL3_CACHE_LINESIZE) << " 字节\n";
#endif
#ifdef _SC_PAGE_SIZE
    std::cout << "  页面大小:        "
              << sysconf(_SC_PAGE_SIZE) << " 字节\n";
#endif
#ifdef _SC_NPROCESSORS_ONLN
    std::cout << "  在线 CPU 数:      "
              << sysconf(_SC_NPROCESSORS_ONLN) << "\n";
#endif

    std::cout << "\n  也可查看: lscpu | grep -E 'L1|L2|L3|Model name|Socket'";
    std::cout << "\n";
}
