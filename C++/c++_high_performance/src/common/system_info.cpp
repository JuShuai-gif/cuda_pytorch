#include "system_info.hpp"

#include <cstdio>
#include <thread>

#if defined(__linux__)
#include <fstream>
#include <string>
#endif

namespace chp {

namespace {

const char* os_name() {
#if defined(__linux__)
    return "Linux";
#elif defined(__APPLE__) && defined(__MACH__)
    return "macOS";
#elif defined(_WIN32)
    return "Windows";
#else
    return "Unknown";
#endif
}

const char* compiler_name() {
#if defined(__clang__)
    return "Clang";
#elif defined(__GNUC__)
    return "GCC";
#else
    return "Unknown";
#endif
}

// Reads "model name" from /proc/cpuinfo (Linux only).
void print_cpu_model() {
#if defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        const std::string prefix = "model name";
        if (line.compare(0, prefix.size(), prefix) == 0) {
            const std::size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::printf("%s", line.substr(pos + 2).c_str());
                return;
            }
        }
    }
#endif
    std::printf("(unavailable)");
}

}  // namespace

void print_system_info() {
    std::printf("[%s | %s", os_name(), compiler_name());
#if defined(__clang__)
    std::printf(" %d.%d.%d", __clang_major__, __clang_minor__,
                __clang_patchlevel__);
#elif defined(__GNUC__)
    std::printf(" %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#endif
    std::printf(" | %u threads | CPU: ", std::thread::hardware_concurrency());
    print_cpu_model();
    std::printf("]");
}

}  // namespace chp
