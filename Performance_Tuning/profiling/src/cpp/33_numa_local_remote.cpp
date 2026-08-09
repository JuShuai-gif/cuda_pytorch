// Purpose: 多NUMA节点local/remote memory带宽与延迟对照。
// Bad: CPU固定node A而内存分配在node B；Good: CPU和内存位于同一node。
// Recommended Profiler: numastat, perf, LIKWID, Intel PCM, VTune Memory Access.
#include "benchmark.hpp"
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void pin_to_node(int node) {
  bitmask* cpus = numa_allocate_cpumask();
  if (numa_node_to_cpus(node, cpus) != 0 || numa_sched_setaffinity(0, cpus) != 0) {
    numa_free_cpumask(cpus);
    throw std::runtime_error("无法绑定CPU node");
  }
  numa_free_cpumask(cpus);
}

double stream_read(const double* data, std::size_t count, int iterations,
                   double& checksum) {
  const auto begin = lab::Clock::now();
  double sum = 0.0;
  for (int r = 0; r < iterations; ++r) {
    for (std::size_t i = 0; i < count; ++i) sum += data[i];
  }
  checksum = sum;
  return std::chrono::duration<double>(lab::Clock::now() - begin).count();
}
}  // namespace

int main(int argc, char** argv) {
  if (numa_available() < 0) {
    std::cout << "SKIP: libnuma报告NUMA不可用\n";
    return 0;
  }
  const int nodes = numa_max_node() + 1;
  if (nodes < 2) {
    std::cout << "SKIP: 需要至少2个NUMA node，当前nodes=" << nodes << "\n";
    return 0;
  }
  const int cpu_node = argc > 1 ? std::stoi(argv[1]) : 0;
  const int memory_node = argc > 2 ? std::stoi(argv[2]) : cpu_node;
  const std::size_t mib = argc > 3 ? std::stoull(argv[3]) : 256;
  const int iterations = argc > 4 ? std::stoi(argv[4]) : 8;
  if (cpu_node < 0 || cpu_node >= nodes || memory_node < 0 || memory_node >= nodes)
    throw std::invalid_argument("node编号越界");
  pin_to_node(cpu_node);
  const std::size_t bytes = mib * 1024ULL * 1024ULL;
  auto* data = static_cast<double*>(numa_alloc_onnode(bytes, memory_node));
  if (!data) throw std::bad_alloc();
  const std::size_t count = bytes / sizeof(double);
  for (std::size_t i = 0; i < count; ++i) data[i] = 1.0 + (i & 7) * 0.01;
  double warm_checksum = 0.0, checksum = 0.0;
  stream_read(data, count, 2, warm_checksum);
  const double seconds = stream_read(data, count, iterations, checksum);
  const double expected = warm_checksum * iterations / 2.0;
  const double relative_error = std::abs(checksum - expected) / std::max(1.0, std::abs(expected));
  std::cout << "cpu_node=" << cpu_node << " memory_node=" << memory_node
            << " bytes=" << bytes << " iterations=" << iterations
            << " elapsed_s=" << seconds
            << " read_GBps=" << bytes * iterations / seconds / 1e9
            << " checksum=" << checksum
            << " relative_error=" << relative_error << '\n';
  numa_free(data, bytes);
  return relative_error < 1e-12 ? 0 : 2;
}
