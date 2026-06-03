#pragma once
// Compatibility layer for std::format (C++20) / fmt::format (C++11+).
//
// Uses the __cpp_lib_format feature test macro to detect whether std::format
// is actually available (not just the header, which GCC 11 has as a stub).
//
// Falls back to the {fmt} library which uses identical format string syntax.
// Uses fmt::vformat for runtime format strings (fmt 9+ requires compile-time
// format strings with fmt::format, but we pass runtime strings from TS_FORMAT).
//
// Usage: Use TS_FORMAT(...) instead of std::format(...) throughout the codebase.

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

// Main macro for portability.
#define TS_FORMAT(...) ::task_scheduler::ts_format(__VA_ARGS__)
