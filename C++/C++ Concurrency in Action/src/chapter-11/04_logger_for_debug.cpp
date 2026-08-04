/**
 * 04_logger_for_debug.cpp — 工业级线程安全日志系统
 *
 * 用于调试多线程程序的日志库, 特性:
 *  - 线程安全: 单条日志原子写入 (不会交错)
 *  - 时间戳: 毫秒级精度
 *  - 线程 ID: 标识日志来源
 *  - 日志级别: DEBUG, INFO, WARN, ERROR, FATAL
 *  - 支持输出到 stdout/stderr/文件
 *  - 零开销: 低于设定级别的日志不会产生字符串格式化开销
 *
 * 编译: g++ -std=c++20 -O2 -pthread 04_logger_for_debug.cpp -o logger
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <chrono>
#include <string>
#include <vector>
#include <atomic>
#include <iomanip>
#include <ctime>
#include <memory>
#include <map>
#include <functional>

// ============================================================================
// LogLevel
// ============================================================================
enum class LogLevel : int {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERROR = 3,
    FATAL = 4,
    OFF   = 5
};

inline const char* level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default:              return "????";
    }
}

inline const char* level_to_color(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "\033[37m";  // 白色
        case LogLevel::INFO:  return "\033[32m";  // 绿色
        case LogLevel::WARN:  return "\033[33m";  // 黄色
        case LogLevel::ERROR: return "\033[31m";  // 红色
        case LogLevel::FATAL: return "\033[35m";  // 紫色
        default:              return "\033[0m";
    }
}

// ============================================================================
// LogStream — 流式日志构建器 (RAII 自动输出)
// ============================================================================
class LogStream {
private:
    std::ostringstream buffer_;
    LogLevel level_;
    std::string file_;
    int line_;
    bool active_;

public:
    LogStream(LogLevel level, const char* file, int line, bool active)
        : level_(level), file_(file), line_(line), active_(active) {}

    LogStream(LogStream&& other) noexcept
        : buffer_(std::move(other.buffer_)), level_(other.level_),
          file_(std::move(other.file_)), line_(other.line_),
          active_(other.active_) {
        other.active_ = false;
    }

    ~LogStream() {
        if (active_) {
            flush();
        }
    }

    template <typename T>
    LogStream& operator<<(const T& value) {
        if (active_) {
            buffer_ << value;
        }
        return *this;
    }

    // 支持 std::endl, std::flush 等操纵符
    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (active_) {
            buffer_ << manip;
        }
        return *this;
    }

    std::string str() const { return buffer_.str(); }

    LogLevel level() const { return level_; }

private:
    void flush();
};

// ============================================================================
// Logger — 线程安全日志核心
// ============================================================================
class Logger {
public:
    using OutputFunc = std::function<void(const std::string&)>;

private:
    std::mutex mutex_;
    LogLevel min_level_;
    std::vector<OutputFunc> outputs_;
    bool use_color_;
    bool show_thread_id_;
    bool show_timestamp_;
    bool show_file_line_;

    // 线程 ID 的简短表示
    static std::string short_thread_id() {
        std::ostringstream oss;
        oss << std::hex << std::this_thread::get_id();
        std::string s = oss.str();
        // 取后 6 个字符
        if (s.size() > 6) s = s.substr(s.size() - 6);
        return s;
    }

public:
    Logger()
        : min_level_(LogLevel::INFO)
        , use_color_(true)
        , show_thread_id_(true)
        , show_timestamp_(true)
        , show_file_line_(false)
    {
        // 默认输出到 stderr
        outputs_.push_back([](const std::string& msg) {
            std::cerr << msg << std::flush;
        });
    }

    // 配置
    void set_level(LogLevel level) { min_level_ = level; }
    void set_color(bool on) { use_color_ = on; }
    void set_show_thread_id(bool on) { show_thread_id_ = on; }
    void set_show_timestamp(bool on) { show_timestamp_ = on; }
    void set_show_file_line(bool on) { show_file_line_ = on; }

    // 添加输出目标
    void add_file_output(const std::string& filepath) {
        auto file = std::make_shared<std::ofstream>(filepath, std::ios::app);
        if (!file->is_open()) {
            std::cerr << "Logger: 无法打开文件 " << filepath << "\n";
            return;
        }
        outputs_.push_back([file](const std::string& msg) {
            *file << msg << std::flush;
        });
    }

    // 日志入口
    LogStream log(LogLevel level, const char* file, int line) {
        bool active = (level >= min_level_);
        return LogStream(level, file, line, active);
    }

    // 写入日志 (由 LogStream 析构时调用)
    void write(LogStream&& stream) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ostringstream formatted;

        // 时间戳
        if (show_timestamp_) {
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) % 1000;

            std::tm tm_buf{};
            localtime_r(&time_t_now, &tm_buf);

            formatted << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
                      << '.' << std::setfill('0') << std::setw(3) << ms.count()
                      << " ";
        }

        // 日志级别
        if (use_color_) {
            formatted << level_to_color(stream.level());
        }
        formatted << "[" << level_to_string(stream.level()) << "]";
        if (use_color_) {
            formatted << "\033[0m";
        }

        // 线程 ID
        if (show_thread_id_) {
            formatted << " [" << short_thread_id() << "]";
        }

        // 源文件位置
        if (show_file_line_) {
            formatted << " " << stream.str().substr(0, stream.str().find('\n'));
        }

        formatted << " " << stream.str();

        // 确保以换行结尾
        std::string msg = formatted.str();
        if (msg.empty() || msg.back() != '\n') {
            msg += '\n';
        }

        // 输出到所有目标
        for (auto& out : outputs_) {
            out(msg);
        }

        // FATAL 级别终止程序
        if (stream.level() == LogLevel::FATAL) {
            std::abort();
        }
    }
};

// LogStream::flush 的实现 (定义在 Logger 之后)
void LogStream::flush() {
    if (!active_) return;
    // LogStream 需要访问 Logger 实例来写入
    // 使用全局 Logger 实例
    extern Logger g_logger; // 前向声明
    active_ = false;
    g_logger.write(std::move(*this));
}

// ============================================================================
// 全局 Logger 实例
// ============================================================================
Logger g_logger;

// ============================================================================
// 便捷宏
// ============================================================================
#define LOG_DEBUG   g_logger.log(LogLevel::DEBUG, __FILE__, __LINE__)
#define LOG_INFO    g_logger.log(LogLevel::INFO,  __FILE__, __LINE__)
#define LOG_WARN    g_logger.log(LogLevel::WARN,  __FILE__, __LINE__)
#define LOG_ERROR   g_logger.log(LogLevel::ERROR, __FILE__, __LINE__)
#define LOG_FATAL   g_logger.log(LogLevel::FATAL, __FILE__, __LINE__)

// ============================================================================
// 使用演示
// ============================================================================
void single_thread_demo() {
    std::cout << "=== 单线程日志演示 ===\n\n";

    g_logger.set_level(LogLevel::DEBUG);
    g_logger.set_show_file_line(false);

    LOG_DEBUG << "这是一条 DEBUG 日志, x=" << 42;
    LOG_INFO  << "这是一条 INFO 日志, name=" << std::string("Alice");
    LOG_WARN  << "这是一条 WARN 日志, temperature=" << 99.5;
    LOG_ERROR << "这是一条 ERROR 日志, error_code=" << 500;

    // DEBUG 级别被过滤
    g_logger.set_level(LogLevel::INFO);
    LOG_DEBUG << "这条 DEBUG 不会显示 (级别已提升到 INFO)";
    LOG_INFO  << "这条 INFO 会显示";

    std::cout << "\n";
}

// ============================================================================
// 多线程并发日志演示
// ============================================================================
void multi_thread_demo() {
    std::cout << "=== 多线程并发日志演示 ===\n\n";

    g_logger.set_level(LogLevel::INFO);

    constexpr int kThreads = 5;
    std::vector<std::jthread> threads;

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t]() {
            for (int i = 0; i < 3; ++i) {
                LOG_INFO << "任务 " << t << "-" << i
                         << " 正在处理... (耗时=" << (i + 1) * 10 << "ms)";
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            LOG_INFO << "任务 " << t << " 完成!";
        });
    }

    for (auto& th : threads) th.join();

    std::cout << "\n";
}

// ============================================================================
// 性能: 禁用日志的零开销
// ============================================================================
void zero_overhead_demo() {
    std::cout << "=== 级别过滤性能演示 ===\n\n";

    // 方法1: 日志宏自动检查级别
    g_logger.set_level(LogLevel::WARN);
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000000; ++i) {
        // 这个日志级别为 DEBUG, 低于 WARN, 不会产生输出
        // 但仍有 LogStream 构造/析构的开销
        LOG_DEBUG << "这条不会输出: " << i;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "  100万条静默日志耗时: " << ms << " ms\n";

    // 方法2: 直接检查 (真正的零开销)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        // 应用程序级别的级别检查, LogStream 完全不创建
    }
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "  100万次循环 (无日志) 耗时: " << ms << " ms\n";

    g_logger.set_level(LogLevel::INFO);

    std::cout << "\n";
}

// ============================================================================
// 文件输出演示
// ============================================================================
void file_output_demo() {
    std::cout << "=== 文件输出演示 ===\n\n";

    const std::string log_file = "/tmp/logger_demo_output.log";
    g_logger.add_file_output(log_file);

    LOG_INFO << "这条日志同时输出到终端和文件: " << log_file;
    LOG_WARN << "检查文件内容以确认";

    std::cout << "  日志已写入: " << log_file << "\n\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    single_thread_demo();
    multi_thread_demo();
    zero_overhead_demo();
    file_output_demo();

    std::cout << "=== 日志系统特性总结 ===\n";
    std::cout << "  1. 线程安全: 所有日志原子写入, 不交错\n";
    std::cout << "  2. 时间戳: 毫秒精度 (YYYY-MM-DD HH:MM:SS.mmm)\n";
    std::cout << "  3. 线程ID: 短十六进制标识\n";
    std::cout << "  4. 日志级别: DEBUG/INFO/WARN/ERROR/FATAL\n";
    std::cout << "  5. 彩色输出: 终端中不同级别用不同颜色\n";
    std::cout << "  6. 多目标: 支持 stdout/stderr/文件\n";
    std::cout << "  7. 级别过滤: 低于设定级别的日志被静默丢弃\n";
    std::cout << "  8. RAII: LogStream 析构时自动写入和释放\n";

    return 0;
}
