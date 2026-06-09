#pragma once

#include <chrono>
#include <iostream>
#include <string>

// ============================================================================
// 计时器工具
// ============================================================================
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    double elapsed_ns() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::nano>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// 辅助函数：打印节标题
// ============================================================================
inline void print_header(const std::string &title) {
    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

// ============================================================================
// 反优化：防止编译器消除死代码
// ============================================================================
extern volatile long g_sink;
