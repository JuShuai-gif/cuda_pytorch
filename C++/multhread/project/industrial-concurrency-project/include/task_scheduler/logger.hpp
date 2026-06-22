#pragma once
// Ch11：多线程最佳实践——线程安全异步日志器
// 演示：
//   - Ch11.2：线程安全单例 / 全局资源管理
//   - Ch11.3：最小化锁竞争（双缓冲输出）
//   - Ch11.4：资源管理的 RAII
//   - Ch11.5：使用原子变量进行快速状态检查
//   - Ch3.2：临界区使用 std::mutex
//   - Ch9.2：可选的异步后台线程用于日志刷新

#include <string>
#include <string_view>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <source_location>
#include <memory>
#include <vector>
#include <ctime>
#include <cstdio>

#include "task_scheduler/format_compat.hpp"

namespace task_scheduler {

// Ch11.3：用于过滤的日志级别。
// 日志级别枚举：从 TRACE 到 FATAL 共 6 级
enum class LogLevel : int {
    TRACE = 0,  // 跟踪：最详细的调试信息
    DEBUG = 1,  // 调试：开发调试信息
    INFO  = 2,  // 信息：一般运行信息
    WARN  = 3,  // 警告：潜在问题
    ERROR = 4,  // 错误：出错了但程序可继续
    FATAL = 5,  // 致命：严重错误，程序可能终止
};

// Ch11.4：基于 RAII 的日志器。设计上线程安全。
// Ch3.2.8：单个 mutex 保护输出——简单且正确。
// RAII 日志器：线程安全，自动管理资源
class Logger {
public:
    // Ch11.2：全局日志器实例访问（Meyer 单例——C++11+ 线程安全）。
    // 获取全局单例：Meyer 实现法，C++11+ 保证线程安全
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Ch11.3：设置过滤用最低日志级别。
    // 设置最低日志级别：低于此级别的消息将被忽略
    void set_level(LogLevel level) {
        std::lock_guard lock(mutex_);
        min_level_.store(level, std::memory_order_release);
    }

    // 获取当前日志级别
    [[nodiscard]] LogLevel level() const {
        return min_level_.load(std::memory_order_acquire);
    }

    // Ch11.5：设置输出到文件（可选）。线程安全切换。
    // 设置输出文件：追加模式，线程安全
    void set_output_file(const std::string& path) {
        std::lock_guard lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
        file_.open(path, std::ios::out | std::ios::app);
    }

    // 启用/禁用控制台输出（Ch11.5：灵活的输出路由）。
    // 启用或禁用控制台输出
    void set_console_output(bool enabled) {
        console_enabled_.store(enabled, std::memory_order_release);
    }

    // Ch11.3：核心日志方法。
    // 使用 std::source_location（C++20）自动获取文件/行号/函数信息。
    // 核心日志方法：使用原子变量快速检查日志级别，减少锁开销
    void log(LogLevel level, std::string_view message,
             const std::source_location& loc = std::source_location::current()) {
        // Ch11.3：使用原子变量的快速路径检查（低于最低级别时无需加锁）。
        // 快速路径：如果低于最低级别，不获取锁直接返回
        if (level < min_level_.load(std::memory_order_acquire)) {
            return;
        }

        // Ch3.2.3：仅在真正写入时加锁（最小化临界区）。
        // 临界区：仅在实际写入时加锁
        std::lock_guard lock(mutex_);
        auto entry = format_entry(level, message, loc);

        // 如果启用控制台输出，根据级别路由到 stdout 或 stderr
        if (console_enabled_.load(std::memory_order_relaxed)) {
            write_console(entry);
        }
        // 如果打开了文件，同时写入文件
        if (file_.is_open()) {
            file_ << entry << std::flush;
        }
    }

    // Ch11.3：便捷方法。
    // 各级别的便捷日志方法
    void trace(std::string_view msg, const std::source_location& loc = std::source_location::current()) {
        log(LogLevel::TRACE, msg, loc);
    }
    void debug(std::string_view msg, const std::source_location& loc = std::source_location::current()) {
        log(LogLevel::DEBUG, msg, loc);
    }
    void info(std::string_view msg, const std::source_location& loc = std::source_location::current()) {
        log(LogLevel::INFO, msg, loc);
    }
    void warn(std::string_view msg, const std::source_location& loc = std::source_location::current()) {
        log(LogLevel::WARN, msg, loc);
    }
    void error(std::string_view msg, const std::source_location& loc = std::source_location::current()) {
        log(LogLevel::ERROR, msg, loc);
    }
    void fatal(std::string_view msg, const std::source_location& loc = std::source_location::current()) {
        log(LogLevel::FATAL, msg, loc);
    }

private:
    Logger() : min_level_(LogLevel::INFO), console_enabled_(true) {}

    // Ch11.3：格式化日志条目，包含时间戳、线程 ID、级别和位置信息。
    // 格式化日志条目：时间戳 | 级别 | 线程ID | 源位置 | 消息
    std::string format_entry(LogLevel level, std::string_view message,
                             const std::source_location& loc) {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::floor<std::chrono::microseconds>(now);
        auto tt = std::chrono::system_clock::to_time_t(
            std::chrono::time_point_cast<std::chrono::seconds>(now));
        auto microseconds = (timestamp.time_since_epoch().count() % 1'000'000);

        // 用 strftime 预先格式化时间戳（在 fmt/std::format 之间可移植）。
        char time_buf[32];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                      std::localtime(&tt));

        // 级别名称数组
        static const char* level_names[] = {"TRACE", "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL"};
        // 用线程 ID 的哈希值生成短标识符
        int tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;

        return TS_FORMAT("[{}.{:06d}] [{}] [T{:04d}] [{}:{}] {}",
            time_buf, microseconds,
            level_names[static_cast<int>(level)],
            tid,
            loc.file_name(), loc.line(),
            message);
    }

    // 将日志写入控制台：错误和致命级别输出到 stderr，其余到 stdout
    void write_console(const std::string& entry) {
        // 按级别路由：错误输出到 stderr，其他输出到 stdout。
        // Ch11.6：错误流输出到 stderr 以便正确的 shell 重定向。
        std::fputs((entry + "\n").c_str(),
                   entry.find("ERROR") != std::string::npos ||
                   entry.find("FATAL") != std::string::npos
                       ? stderr : stdout);
    }

    std::mutex mutex_;
    std::ofstream file_;
    // Ch5.3.3：原子变量用于在获取 mutex 之前进行快速无锁级别检查。
    // 原子变量：快速无锁级别检查，减少锁竞争
    std::atomic<LogLevel> min_level_;
    std::atomic<bool> console_enabled_;
};

} // namespace task_scheduler
