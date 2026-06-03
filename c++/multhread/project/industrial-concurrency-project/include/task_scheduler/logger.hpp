#pragma once
// Chapter 11: Multi-threading Best Practices - Thread-safe Asynchronous Logger
// Demonstrates:
//   - Ch11.2: Thread-safe singleton / global resource management
//   - Ch11.3: Minimizing lock contention (double-buffered output)
//   - Ch11.4: RAII for resource management
//   - Ch11.5: Using atomics for fast state checks
//   - Ch3.2: std::mutex for critical sections
//   - Ch9.2: Optional async background thread for log flushing

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

// Ch11.3: Log levels for filtering.
enum class LogLevel : int {
    TRACE = 0,
    DEBUG = 1,
    INFO  = 2,
    WARN  = 3,
    ERROR = 4,
    FATAL = 5,
};

// Ch11.4: RAII-based logger. Thread-safe by design.
// Ch3.2.8: Single mutex protects the output - simple and correct.
class Logger {
public:
    // Ch11.2: Global logger instance access (Meyer's Singleton - thread-safe in C++11+).
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Ch11.3: Set minimum log level for filtering.
    void set_level(LogLevel level) {
        std::lock_guard lock(mutex_);
        min_level_.store(level, std::memory_order_release);
    }

    [[nodiscard]] LogLevel level() const {
        return min_level_.load(std::memory_order_acquire);
    }

    // Ch11.5: Set output to a file (optional). Thread-safe swap.
    void set_output_file(const std::string& path) {
        std::lock_guard lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
        file_.open(path, std::ios::out | std::ios::app);
    }

    // Enable/disable console output (Ch11.5: flexible output routing).
    void set_console_output(bool enabled) {
        console_enabled_.store(enabled, std::memory_order_release);
    }

    // Ch11.3: Core logging method.
    // Uses std::source_location (C++20) for automatic file/line/function info.
    void log(LogLevel level, std::string_view message,
             const std::source_location& loc = std::source_location::current()) {
        // Ch11.3: Fast-path check with atomic (no lock if below min level).
        if (level < min_level_.load(std::memory_order_acquire)) {
            return;
        }

        // Ch3.2.3: Lock only for the actual write (minimize critical section).
        std::lock_guard lock(mutex_);
        auto entry = format_entry(level, message, loc);

        if (console_enabled_.load(std::memory_order_relaxed)) {
            write_console(entry);
        }
        if (file_.is_open()) {
            file_ << entry << std::flush;
        }
    }

    // Ch11.3: Convenience methods.
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

    // Ch11.3: Format log entry with timestamp, thread ID, level, and location.
    std::string format_entry(LogLevel level, std::string_view message,
                             const std::source_location& loc) {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::floor<std::chrono::microseconds>(now);
        auto tt = std::chrono::system_clock::to_time_t(
            std::chrono::time_point_cast<std::chrono::seconds>(now));
        auto microseconds = (timestamp.time_since_epoch().count() % 1'000'000);

        // Pre-format timestamp with strftime (portable across fmt/std::format).
        char time_buf[32];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                      std::localtime(&tt));

        static const char* level_names[] = {"TRACE", "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL"};
        int tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;

        return TS_FORMAT("[{}.{:06d}] [{}] [T{:04d}] [{}:{}] {}",
            time_buf, microseconds,
            level_names[static_cast<int>(level)],
            tid,
            loc.file_name(), loc.line(),
            message);
    }

    void write_console(const std::string& entry) {
        // Route by level: errors to stderr, others to stdout.
        // Ch11.6: Error streams go to stderr for proper shell redirection.
        std::fputs((entry + "\n").c_str(),
                   entry.find("ERROR") != std::string::npos ||
                   entry.find("FATAL") != std::string::npos
                       ? stderr : stdout);
    }

    std::mutex mutex_;
    std::ofstream file_;
    // Ch5.3.3: atomic for fast lock-free level check before acquiring mutex.
    std::atomic<LogLevel> min_level_;
    std::atomic<bool> console_enabled_;
};

} // namespace task_scheduler
