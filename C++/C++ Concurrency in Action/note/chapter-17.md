# 第17章：工程化实践

> 写出正确的并发代码是一回事，将其集成到工程化的工作流中是另一回事。本章涵盖单元测试、Sanitizer、CMake 优化和 CI/CD 最佳实践。

---

## 17.1 并发代码的单元测试

### 挑战

- **非确定性**：每次运行结果可能不同
- **时序依赖**：bug 只在特定交错下出现
- **死锁风险**：测试本身可能挂起
- **状态污染**：并发测试之间可能互相影响

### 测试金字塔

```
       ┌──────┐
       │ 压力  │  ← 长时间高负载运行
      ┌┴──────┴┐
      │ 集成   │  ← 多组件并发交互
     ┌┴────────┴┐
     │ 功能     │  ← 单组件并发正确性
    ┌┴──────────┴┐
    │ 单元      │  ← 单线程正确性
    └────────────┘
```

### 测试模式

1. **确定性测试**：固定线程数和数据，验证输出确定
2. **不变式检查**：验证并发操作不破坏数据结构不变式
3. **重复测试**：同一测试运行 N 次，增加时序覆盖
4. **超时机制**：为每个测试设置 deadlock 时间上限
5. **并发断言**：在临界区内验证状态一致性

### Google Test 集成

```cpp
#include <gtest/gtest.h>
#include <thread>
#include <atomic>

TEST(AtomicTest, ConcurrentIncrement) {
    std::atomic<int> counter{0};
    constexpr int kThreads = 4;
    constexpr int kIters = 1000;

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < kIters; ++j) {
                counter.fetch_add(1);
            }
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(counter.load(), kThreads * kIters);
}
```

---

## 17.2 Sanitizer 工具

### ThreadSanitizer (TSan)

检测数据竞争的最强工具：

```bash
# 编译时加上 sanitize flag
g++ -fsanitize=thread -g -O1 program.cpp -o program

# 运行即可自动检测
./program
# 输出: WARNING: ThreadSanitizer: data race ...
```

**TSan 能检测**：
- 数据竞争（无同步的并发读写）
- 锁顺序反转（可能导致死锁）
- 线程泄漏

**TSan 限制**：
- 2-5x 性能开销
- 内存开销 ~5-10x
- 只检测实际发生的竞争（非静态分析）

### AddressSanitizer (ASan)

检测内存错误：

```bash
g++ -fsanitize=address -g program.cpp -o program
```

### UndefinedBehaviorSanitizer (UBSan)

```bash
g++ -fsanitize=undefined -g program.cpp -o program
```

---

## 17.3 Google Benchmark

### 基础用法

```cpp
#include <benchmark/benchmark.h>

static void BM_AtomicFetchAdd(benchmark::State& state) {
    std::atomic<long long> counter{0};
    for (auto _ : state) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}
BENCHMARK(BM_AtomicFetchAdd)->Threads(1)->Threads(2)->Threads(4);

BENCHMARK_MAIN();
```

### 多线程基准测试

```cpp
static void BM_MutexVsAtomic(benchmark::State& state) {
    static std::mutex mtx;
    static long long counter = 0;
    for (auto _ : state) {
        std::lock_guard lock(mtx);
        ++counter;
    }
}
// 对比不同线程数下的性能
BENCHMARK(BM_MutexVsAtomic)
    ->ThreadRange(1, 8)
    ->UseRealTime();
```

---

## 17.4 CMake 工程优化

### 推荐的 CMake 结构

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_concurrent_app VERSION 1.0 LANGUAGES CXX)

# 全局设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=thread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native")

# 启用 LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# 依赖管理
find_package(Threads REQUIRED)
find_package(OpenMP QUIET)
find_package(GTest QUIET)
find_package(benchmark QUIET)

# 测试
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()

# 基准测试
if(benchmark_FOUND)
    add_subdirectory(benchmarks)
endif()
```

### 多配置构建

```bash
# Debug (带 TSan)
cmake -DCMAKE_BUILD_TYPE=Debug -B build/debug
cmake --build build/debug

# Release (优化全开)
cmake -DCMAKE_BUILD_TYPE=Release -B build/release
cmake --build build/release

# RelWithDebInfo (优化 + 调试符号，适合 perf)
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build/relwithdebinfo
cmake --build build/relwithdebinfo
```

---

## 17.5 CI/CD 中的并发测试

### GitHub Actions 示例

```yaml
concurrency-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Build with TSan
      run: |
        cmake -DCMAKE_BUILD_TYPE=Debug \
              -DCMAKE_CXX_FLAGS="-fsanitize=thread" \
              -B build
        cmake --build build -j$(nproc)
    - name: Run tests
      run: |
        cd build
        ctest --output-on-failure --timeout 60
    - name: Stress test
      run: |
        ./build/stress_test --duration 30
```

---

## 17.6 压力测试（Stress Testing）

```cpp
// 持续高负载运行，发现偶发并发 bug
void stress_test() {
    const auto kDuration = std::chrono::seconds(30);
    auto deadline = std::chrono::steady_clock::now() + kDuration;

    std::atomic<bool> stop{false};
    std::vector<std::jthread> threads;

    for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
        threads.emplace_back([&]() {
            while (!stop.load()) {
                // 随机混合 push/pop 操作
                if (rand() % 2) queue.push(rand());
                else queue.try_pop();
            }
        });
    }

    std::this_thread::sleep_until(deadline);
    stop.store(true);
}
```

---

## 17.7 调试并发代码的技巧

1. **std::osyncstream**：线程安全的流输出（C++20）
2. **并发日志**：带线程 ID 和时间戳的日志
3. **确定性重放**：rr（Record and Replay）调试器
4. **Helgrind**：Valgrind 的竞态检测工具
5. **`std::this_thread::get_id()`**：线程识别

---

## 17.8 知识体系交叉引用

| 本章主题 | 相关章节 |
|----------|----------|
| 单元测试 | 第11章 测试调试 |
| TSan | 第5章 数据竞争、第11章 |
| Benchmark | 第15章 性能分析 |
| CMake | 全部章节的构建系统 |

---

## 17.9 本章小结

工程化是并发编程的"最后一公里"：

1. **TSan 是你的安全网**——每次提交前运行
2. **单元测试 + 压力测试 = 双保险**——功能正确性 + 时序健壮性
3. **Benchmark 提供数据驱动的优化决策**——不要凭感觉优化
4. **CMake 多配置**让开发-调试-发布流程无缝
5. **CI/CD 自动化**让并发测试成为基础设施而非事后补救
