#pragma once
// std::format（C++20）/ fmt::format（C++11+）兼容层。
// 格式化兼容层：在 std::format 和 {fmt} 库之间自动切换
//
// 使用 __cpp_lib_format 特性测试宏检测 std::format 是否实际可用
//（不仅仅是头文件，因为 GCC 11 的头文件是占位符 stub）。
//
// 回退到 {fmt} 库，它使用相同的格式字符串语法。
// 对运行时格式字符串使用 fmt::vformat（fmt 9+ 要求 fmt::format 使用编译时
// 格式字符串，但我们通过 TS_FORMAT 传递的是运行时字符串）。
//
// 用法：在整个代码库中使用 TS_FORMAT(...) 代替 std::format(...)。

#include <version>

#ifdef __cpp_lib_format
  #include <format>
  namespace task_scheduler {
  template <typename... Args>
  auto ts_format(std::format_string<Args...> fmt, Args&&... args) {
      return std::format(fmt, std::forward<Args>(args)...);
  }
  }
#else
  #include <fmt/format.h>
  namespace task_scheduler {
  template <typename... Args>
  auto ts_format(fmt::string_view fmt, Args&&... args) {
      return fmt::vformat(fmt, fmt::make_format_args(args...));
  }
  }
#endif

// 可移植格式化的主宏。
// 使用此宏替代直接调用 std::format 或 fmt::format
#define TS_FORMAT(...) ::task_scheduler::ts_format(__VA_ARGS__)
