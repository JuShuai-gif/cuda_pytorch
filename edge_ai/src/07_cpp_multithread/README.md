# 07_cpp_multithread - C++ Production Multithreading Demos

## Overview

This directory contains production-quality C++ multithreading implementations:

- **thread_pool.h**: Work-stealing thread pool with `std::future` task submission
- **lockfree_queue.h**: Bounded MPMC lock-free queue using atomics
- **benchmarks.h / benchmarks.cpp**: All benchmark and demo functions
- **main.cpp**: Thin entry point that runs all benchmarks

## File Structure

```
07_cpp_multithread/
  thread_pool.h    - ThreadPool class (work-stealing thread pool)
  lockfree_queue.h - LockFreeQueue<T, Capacity> (bounded MPMC lock-free queue)
  benchmarks.h     - Benchmark function declarations + Timer utility
  benchmarks.cpp   - Thread pool stress test, lock-free queue test,
                     queue comparison, priority inversion, memory ordering,
                     producer-consumer demos
  main.cpp         - Entry point calling all benchmarks
  CMakeLists.txt
  README.md
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./multithread_demos
```

## What It Tests

1. Thread pool stress test - submit N tasks, verify all complete
2. Lock-free queue stress test - MPMC producer-consumer
3. Throughput comparison: lock-free vs mutex-based
4. Priority inversion simulation (requires SCHED_FIFO, may need sudo)
5. Atomic memory ordering (relaxed vs seq_cst)
6. Producer-Consumer with Condition Variables

## Requirements

- C++17 compiler (GCC 9+, Clang 10+)
- CMake >= 3.14
- Linux (for pthread real-time scheduling features)

## Performance Notes

- Thread pool uses `std::thread` with condition variable
- Lock-free queue uses `std::atomic` with acquire-release ordering
- For best benchmarking, run with `sudo` to enable real-time scheduling
- Bind to isolated CPUs for consistent results: `taskset -c 2-5 ./multithread_demos`
