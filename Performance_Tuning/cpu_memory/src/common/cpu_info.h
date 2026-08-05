#pragma once

// CPU / system information helpers.
// Reads cache topology from /sys/devices/system/cpu/cpu0/cache/,
// page size, huge page configuration, and NUMA topology.

#include <cstdint>
#include <string>
#include <vector>

namespace cpuinfo {

struct CacheInfo {
    int level = 0;
    std::string type;           // Data / Instruction / Unified
    long size_bytes = 0;
    int ways = 0;
    int line_size = 0;          // coherency_line_size in bytes
    int sets = 0;
    std::vector<int> shared_cpus;  // CPUs sharing this cache
};

// Detect the L1 data cache line size in bytes (64 on most x86-64).
long l1d_line_size();

// Cache line size for the given level (1/2/3), 0 if unknown.
long line_size_for_level(int level);

// List of all caches of CPU 0.
std::vector<CacheInfo> caches();

// Total L1d size in bytes (CPU 0), 0 if unknown.
long l1d_size_bytes();

// Total L2 size in bytes (CPU 0), 0 if unknown.
long l2_size_bytes();

// Total L3 size in bytes (CPU 0), 0 if unknown.
long l3_size_bytes();

// System page size in bytes.
long page_size();

// Huge page size in bytes (from /proc/meminfo Hugepagesize), 0 if unknown.
long huge_page_size();

// Number of pre-allocated huge pages (HugePages_Total), 0 if none.
long huge_pages_total();

// Whether THP is enabled (always/madvise/never).
std::string thp_enabled();

// Number of NUMA nodes; 1 means non-NUMA (or single-node).
int numa_nodes();

// Log a compact system summary to stdout.
void print_summary();

}  // namespace cpuinfo
