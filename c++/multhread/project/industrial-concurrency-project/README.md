# Industrial-Grade C++ Concurrency Project

## AI/ML Operator Inference Task Scheduling System

A comprehensive C++20 concurrent task scheduling system for AI/ML inference workloads.
This project implements every core concurrency concept from **"C++ Concurrency in Action (2nd Edition)"**
by Anthony Williams in a practical, industrial-grade codebase.

### Book Chapter Coverage

| Chapter | Topic | Module |
|---------|-------|--------|
| Ch2 | Thread Management | `main.cpp`, `thread_pool.cpp` |
| Ch3 | Sharing Data Between Threads | `task_queue.hpp`, `concurrent_cache.hpp`, `logger.hpp` |
| Ch4 | Synchronizing Concurrent Operations | `thread_pool.hpp`, `task_scheduler.hpp` |
| Ch5 | C++ Memory Model & Atomics | `spinlock.hpp`, `stop_token.hpp` |
| Ch6 | Lock-based Concurrent Data Structures | `task_queue.hpp`, `priority_task_queue.hpp` |
| Ch7 | Lock-free Data Structures | (design notes only) |
| Ch8 | Designing Concurrent Code | `task_scheduler.hpp`, examples |
| Ch9 | Advanced Thread Management | `thread_pool.hpp`, `stop_token.hpp` |
| Ch10 | Testing & Debugging Concurrent Code | `tests/` directory |
| Ch11 | Multi-threading Best Practices | `logger.hpp`, project patterns |

### Project Structure

```
industrial-concurrency-project/
├── include/task_scheduler/    # Header-only and template libraries
│   ├── task_scheduler.hpp      # Core scheduler (Ch8.5)
│   ├── thread_pool.hpp         # Fixed-size thread pool with work stealing (Ch9.1)
│   ├── task_queue.hpp          # MPMC lock-based queue (Ch6.2)
│   ├── priority_task_queue.hpp # Priority-based MPMC queue (Ch6.3)
│   ├── spinlock.hpp            # TTAS spinlock (Ch5.3)
│   ├── concurrent_cache.hpp    # LRU cache with shared_mutex (Ch3.3)
│   ├── stop_token.hpp          # Simplified stop mechanism (Ch9.2)
│   └── logger.hpp              # Thread-safe logger (Ch11)
├── src/                        # Non-template implementations
│   ├── main.cpp
│   ├── thread_pool.cpp
│   ├── task_scheduler.cpp
│   └── logger.cpp
├── tests/                      # Unit and stress tests
│   ├── test_thread_pool.cpp
│   ├── test_task_queue.cpp
│   ├── test_task_scheduler.cpp
│   └── test_stress.cpp
├── examples/                   # Usage examples
│   ├── example_basic.cpp
│   ├── example_pipeline.cpp
│   ├── example_inference.cpp
│   └── example_producer_consumer.cpp
└── docs/                       # Architecture and design documentation
    ├── architecture.md
    └── design_notes.md
```

### Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Run examples
./example_basic
./example_pipeline
./example_inference
./example_producer_consumer

# Run main demo
./main
```

### ThreadSanitizer Build

```bash
mkdir build-tsan && cd build-tsan
cmake .. -DCMAKE_BUILD_TYPE=Tsan
make -j$(nproc)
./test_stress
```

### Key Features

- **Work-Stealing Thread Pool**: Fixed-size pool with per-thread local queues (Ch9.1)
- **Priority Scheduling**: Multi-level priority queue for latency-sensitive tasks (Ch6.3)
- **Pipeline Execution**: Multi-stage task pipelines with future chaining (Ch8.3)
- **Concurrent LRU Cache**: Read-optimized cache with `std::shared_mutex` (Ch3.3)
- **TTAS Spinlock**: Test-Test-And-Set with exponential backoff (Ch5.3)
- **Graceful Shutdown**: Stop token mechanism for cooperative interruption (Ch9.2)
- **Thread-Safe Logging**: Timestamped, leveled logging with atomic fast-path (Ch11)
- **RAII Everywhere**: No raw `new`/`delete`, exception-safe resource management
- **TSan Ready**: Designed for ThreadSanitizer verification (Ch10)

### Requirements

- C++20 compiler (GCC 12+, Clang 16+)
- CMake 3.14+
- pthread (Linux/macOS)

### License

MIT - See LICENSE file.
