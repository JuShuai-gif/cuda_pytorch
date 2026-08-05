#include "cpu_info.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <unistd.h>

namespace cpuinfo {

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        return std::string();
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

long parse_long(const std::string& s) {
    if (s.empty())
        return 0;
    long v = std::strtol(s.c_str(), nullptr, 0);
    // Handle suffixes like "48K", "1M", "2G" used by cache size files.
    for (const char* p = s.c_str(); *p; ++p) {
        if (*p == 'K' || *p == 'k') v *= 1024;
        else if (*p == 'M' || *p == 'm') v *= 1024 * 1024;
        else if (*p == 'G' || *p == 'g') v *= 1024 * 1024 * 1024;
    }
    return v;
}

}  // namespace

long l1d_line_size() {
    long v = line_size_for_level(1);
    // Fallback: parse cache line size from any L1 cache dir.
    if (v == 0) {
        for (auto& c : caches()) {
            if (c.level == 1 && c.line_size > 0) {
                v = c.line_size;
                break;
            }
        }
    }
    return v;
}

long line_size_for_level(int level) {
    for (auto& c : caches()) {
        if (c.level == level && c.line_size > 0)
            return c.line_size;
    }
    return 0;
}

std::vector<CacheInfo> caches() {
    std::vector<CacheInfo> out;
    // Enumerate cache index directories for cpu0.
    for (int idx = 0; idx < 32; ++idx) {
        std::string base = "/sys/devices/system/cpu/cpu0/cache/index" +
                           std::to_string(idx) + "/";
        std::ifstream prober(base + "level");
        if (!prober)
            break;  // no more index dirs

        CacheInfo ci;
        ci.level = static_cast<int>(parse_long(read_file(base + "level")));
        ci.type = read_file(base + "type");
        if (!ci.type.empty() && !ci.type.empty() && ci.type.back() == '\n')
            ci.type.pop_back();
        ci.size_bytes = parse_long(read_file(base + "size"));
        ci.ways = static_cast<int>(parse_long(read_file(base + "ways_of_associativity")));
        ci.line_size = static_cast<int>(parse_long(read_file(base + "coherency_line_size")));
        ci.sets = static_cast<int>(parse_long(read_file(base + "number_of_sets")));

        // shared_cpu_list is like "0-3" or "0,2". Parse simple cases.
        std::string lst = read_file(base + "shared_cpu_list");
        if (lst.empty())
            lst = read_file(base + "shared_cpu_map");
        // crude parser: accept comma-separated and dash ranges of numbers
        std::istringstream iss(lst);
        std::string tok;
        while (std::getline(iss, tok, ',')) {
            tok.erase(std::remove(tok.begin(), tok.end(), '\n'), tok.end());
            auto dash = tok.find('-');
            if (dash != std::string::npos) {
                long lo = std::strtol(tok.c_str(), nullptr, 10);
                long hi = std::strtol(tok.c_str() + dash + 1, nullptr, 10);
                for (long c = lo; c <= hi; ++c)
                    ci.shared_cpus.push_back(static_cast<int>(c));
            } else if (!tok.empty()) {
                ci.shared_cpus.push_back(static_cast<int>(std::strtol(tok.c_str(), nullptr, 10)));
            }
        }
        out.push_back(ci);
    }
    return out;
}

long l1d_size_bytes() {
    for (auto& c : caches()) {
        if (c.level == 1 && (c.type == "Data" || c.type == "Unified"))
            return c.size_bytes;
    }
    return 0;
}

long l2_size_bytes() {
    for (auto& c : caches()) {
        if (c.level == 2)
            return c.size_bytes;
    }
    return 0;
}

long l3_size_bytes() {
    for (auto& c : caches()) {
        if (c.level == 3)
            return c.size_bytes;
    }
    return 0;
}

long page_size() {
    long v = ::sysconf(_SC_PAGESIZE);
    return v > 0 ? v : 4096;
}

long huge_page_size() {
    std::string m = read_file("/proc/meminfo");
    auto pos = m.find("Hugepagesize:");
    if (pos == std::string::npos)
        return 0;
    pos = m.find(':', pos);
    long kb = std::strtol(m.c_str() + pos + 1, nullptr, 10);
    return kb > 0 ? kb * 1024 : 0;
}

long huge_pages_total() {
    std::string m = read_file("/proc/meminfo");
    auto pos = m.find("HugePages_Total:");
    if (pos == std::string::npos)
        return 0;
    return std::strtol(m.c_str() + pos + std::strlen("HugePages_Total:"), nullptr, 10);
}

std::string thp_enabled() {
    return read_file("/sys/kernel/mm/transparent_hugepage/enabled");
}

int numa_nodes() {
    int count = 0;
    for (;;) {
        std::string path = "/sys/devices/system/node/node" + std::to_string(count);
        std::ifstream in(path + "/cpulist");
        if (!in)
            break;
        ++count;
    }
    return count;
}

void print_summary() {
    std::printf("=== System summary ===\n");
    std::printf("Page size: %ld bytes\n", page_size());
    std::printf("L1d cache line: %ld bytes\n", l1d_line_size());
    std::printf("L1d size: %ld bytes, L2: %ld bytes, L3: %ld bytes\n",
                l1d_size_bytes(), l2_size_bytes(), l3_size_bytes());
    std::printf("Huge page size: %ld bytes, preallocated: %ld\n",
                huge_page_size(), huge_pages_total());
    std::printf("THP enabled: %s", thp_enabled().c_str());
    std::printf("NUMA nodes: %d\n", numa_nodes());
    for (auto& c : caches()) {
        std::printf("  cache L%d %-12s size=%ld ways=%d line=%d\n",
                    c.level, c.type.c_str(), c.size_bytes, c.ways, c.line_size);
    }
}

}  // namespace cpuinfo
