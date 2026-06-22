// 04_cmake_best_practices.cpp — CMake 最佳实践演示
// 这是一个文档型文件，展示并发项目的 CMake 推荐配置

#include <iostream>

/*
 * ================================================================
 * 并发 C++ 项目的 CMake 最佳实践
 * ================================================================
 *
 * === 1. 项目结构 ===
 *
 * project/
 * ├── CMakeLists.txt              # 顶层
 * ├── cmake/
 * │   ├── CompilerSettings.cmake  # 编译器设置
 * │   ├── Sanitizers.cmake        # Sanitizer 配置
 * │   └── FindXXX.cmake           # 自定义 Find 模块
 * ├── src/
 * │   ├── CMakeLists.txt
 * │   ├── lib1/
 * │   └── lib2/
 * ├── tests/
 * │   └── CMakeLists.txt
 * ├── benchmarks/
 * │   └── CMakeLists.txt
 * └── examples/
 *     └── CMakeLists.txt
 *
 * === 2. 推荐顶层 CMakeLists.txt ===
 *
 * cmake_minimum_required(VERSION 3.16)
 * project(my_concurrent_app VERSION 1.0 LANGUAGES CXX)
 *
 * # C++ 标准
 * set(CMAKE_CXX_STANDARD 20)
 * set(CMAKE_CXX_STANDARD_REQUIRED ON)
 * set(CMAKE_CXX_EXTENSIONS OFF)
 *
 * # 全局编译选项
 * add_compile_options(-Wall -Wextra -Wpedantic)
 *
 * # 构建类型
 * set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
 * set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
 * set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native")
 *
 * # Sanitizer 选项 (仅在 Debug)
 * option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
 * option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
 *
 * if(ENABLE_TSAN)
 *     add_compile_options(-fsanitize=thread)
 *     add_link_options(-fsanitize=thread)
 * endif()
 *
 * # LTO
 * set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
 *
 * # 依赖
 * find_package(Threads REQUIRED)
 * find_package(OpenMP QUIET)
 *
 * # 子目录
 * add_subdirectory(src)
 *
 * # 可选组件
 * if(GTest_FOUND OR BUILD_TESTING)
 *     enable_testing()
 *     add_subdirectory(tests)
 * endif()
 *
 * === 3. 并发测试的最佳实践 ===
 *
 * # tests/CMakeLists.txt
 * foreach(test_src ${TEST_SOURCES})
 *     add_executable(${test_name} ${test_src})
 *     target_link_libraries(${test_name}
 *         PRIVATE GTest::gtest GTest::gtest_main
 *         PRIVATE Threads::Threads)
 *     # 设置超时 (防止死锁挂起 CI)
 *     add_test(NAME ${test_name} COMMAND ${test_name})
 *     set_tests_properties(${test_name}
 *         PROPERTIES TIMEOUT 30)
 * endforeach()
 *
 * === 4. Benchmark 配置 ===
 *
 * find_package(benchmark QUIET)
 * if(benchmark_FOUND)
 *     add_subdirectory(benchmarks)
 * endif()
 *
 * === 5. 编译命令 ===
 *
 * # Debug + TSan
 * cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TSAN=ON -B build/debug
 * cmake --build build/debug -j$(nproc)
 *
 * # Release (最高性能)
 * cmake -DCMAKE_BUILD_TYPE=Release -B build/release
 * cmake --build build/release -j$(nproc)
 *
 * # RelWithDebInfo (perf 分析用)
 * cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build/perf
 * cmake --build build/perf -j$(nproc)
 *
 * === 6. 常见陷阱 ===
 *
 * 1. 忘记 find_package(Threads REQUIRED) — 链接错误
 * 2. TSan 和 ASan 不能同时启用
 * 3. TSan 需要 -O1 以上优化级别才能正常工作
 * 4. 不要在 Release 构建中启用 sanitizer
 * 5. 使用 -march=native 的二进制文件不要分发到不同 CPU
 */

int main() {
    std::cout << "并发 C++ 项目的 CMake 最佳实践\n";
    std::cout << "================================\n\n";

    std::cout << "关键 CMake 配置:\n";
    std::cout << "  1. 使用 C++20: "
              << "set(CMAKE_CXX_STANDARD 20)\n";
    std::cout << "  2. 多构建配置: Debug/Release/RelWithDebInfo\n";
    std::cout << "  3. TSan: cmake -DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug\n";
    std::cout << "  4. 测试超时: set_tests_properties(... TIMEOUT 30)\n";
    std::cout << "  5. LTO 优化: "
              << "CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE\n\n";

    std::cout << "推荐工具链:\n";
    std::cout << "  - 编译器: GCC 12+ / Clang 16+\n";
    std::cout << "  - 构建: CMake 3.20+\n";
    std::cout << "  - 测试: Google Test + CTest\n";
    std::cout << "  - 基准: Google Benchmark\n";
    std::cout << "  - 检测: ThreadSanitizer + AddressSanitizer\n";
    std::cout << "  - 分析: perf + FlameGraph\n";
    std::cout << "  - 格式化: clang-format\n";
    std::cout << "  - 静态分析: clang-tidy\n";

    return 0;
}
