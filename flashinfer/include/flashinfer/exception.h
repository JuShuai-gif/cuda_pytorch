#ifndef FLASHINFER_EXCEPTION_H_
#define FLASHINFER_EXCEPTION_H_

// 异常相关基础设施：
// 1. 提供统一的错误抛出宏 FLASHINFER_ERROR / FLASHINFER_CHECK
// 2. 提供统一的警告输出宏 FLASHINFER_WARN
// 3. 将函数名、文件名、行号和用户消息拼接成可读的诊断信息

#include <exception>
#include <iostream>
#include <sstream>

// 抛出 flashinfer 自定义异常。
// 会自动携带当前函数名、源文件名和代码行号，便于定位问题。
#define FLASHINFER_ERROR(message) throw flashinfer::Error(__FUNCTION__,__FILE__,__LINE__,message)

// 递归展开可变参数时的终止重载。
// 当没有额外参数时不做任何事情。
inline void write_to_stream(std::ostringstream& oss){
    // No-op for empty arguments
}

template <typename T>
// 将单个值写入字符串流，用于拼接错误/警告信息。
void write_to_stream(std::ostringstream& oss,T&& val){
    oss << std::forward<T>(val);
}

template <typename T,typename... Args>
// 将多个参数依次写入字符串流，参数之间插入空格。
// 这样可以支持 FLASHINFER_CHECK(a, "x =", x, "y =", y) 这类写法。
void write_to_stream(std::ostringstream& oss,T&& val,Args&&... args){
  oss << std::forward<T>(val) << " ";
  write_to_stream(oss, std::forward<Args>(args)...);
}

// 辅助宏：条件不满足时，直接按给定 message 抛出异常。
// 这个宏本身没有被下面主宏直接使用，但保留后可作为简单检查接口。
#define FLASHINFER_CHECK_IMPL(condition, message) \
  if (!(condition)) {                             \
    FLASHINFER_ERROR(message);                    \
  }

// 主检查宏：
// 1. 当 condition 为假时，收集可变参数并拼成错误消息
// 2. 如果用户没传消息，则自动生成 "Check failed: 条件表达式"
// 3. 最终统一抛出 flashinfer::Error
#define FLASHINFER_CHECK(condition, ...)   \
  do {                                     \
    if (!(condition)) {                    \
      std::ostringstream oss;              \
      write_to_stream(oss, ##__VA_ARGS__); \
      std::string msg = oss.str();         \
      if (msg.empty()) {                   \
        msg = "Check failed: " #condition; \
      }                                    \
      FLASHINFER_ERROR(msg);               \
    }                                      \
  } while (0)

// 警告宏：
// 1. 组织警告消息
// 2. 如果未提供消息，则使用默认文本
// 3. 通过 flashinfer::Warning 输出到标准错误流，而不是抛异常
#define FLASHINFER_WARN(...)                                           \
  do {                                                                 \
    std::ostringstream oss;                                            \
    write_to_stream(oss, ##__VA_ARGS__);                               \
    std::string msg = oss.str();                                       \
    if (msg.empty()) {                                                 \
      msg = "Warning triggered";                                       \
    }                                                                  \
    flashinfer::Warning(__FUNCTION__, __FILE__, __LINE__, msg).emit(); \
  } while (0)

namespace flashinfer {
// 自定义异常类型，继承自 std::exception。
// 主要职责是把函数、文件、行号和用户消息拼成完整错误信息。
class Error : public std::exception {
 private:
  // 保存最终格式化后的错误字符串。
  std::string message_;

 public:
  Error(const std::string& func, const std::string& file, int line, const std::string& message) {
    std::ostringstream oss;
    oss << "Error in function '" << func << "' "
        << "at " << file << ":" << line << ": " << message;
    message_ = oss.str();
  }

  // 返回错误信息给 catch(...) 或日志系统使用。
  virtual const char* what() const noexcept override { return message_.c_str(); }
};

// 警告类型：
// 与 Error 类似，也会格式化上下文信息；
// 不同点是它不会抛出异常，而是由 emit() 主动打印。
class Warning {
 private:
  // 保存最终格式化后的警告字符串。
  std::string message_;

 public:
  Warning(const std::string& func, const std::string& file, int line, const std::string& message) {
    std::ostringstream oss;
    oss << "Warning in function '" << func << "' "
        << "at " << file << ":" << line << ": " << message;
    message_ = oss.str();
  }

  // 将警告输出到标准错误流。
  void emit() const { std::cerr << message_ << std::endl; }
};

}  // namespace flashinfer

#endif  // FLASHINFER_EXCEPTION_H_

























