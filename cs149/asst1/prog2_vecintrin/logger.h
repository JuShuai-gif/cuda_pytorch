// 头文件保护宏，防止重复包含
#ifndef LOGGER_H_
#define LOGGER_H_

#include <stdio.h>
#include <vector>
#include <string.h>
using namespace std;

// 单条指令的最大长度（字符数）
#define MAX_INST_LEN 32

// 前向声明：向量掩码结构体
struct __cs149_mask;

// Log 结构体：记录一条向量指令的日志信息
struct Log {
  char instruction[MAX_INST_LEN];     // 指令名称字符串
  unsigned long long mask; // 向量掩码，支持最多 64 宽度的向量 (support vector width up to 64)
};

// Statistics 结构体：统计向量指令的执行情况
struct Statistics {
  unsigned long long utilized_lane;      // 实际被使用的向量通道数
  unsigned long long total_lane;         // 总的向量通道数
  unsigned long long total_instructions; // 总指令数
};

// Logger 类：日志记录器，用于追踪和统计向量指令
class Logger {
  private:
    vector<Log> log;     // 存储所有指令日志
    Statistics stats;    // 保存统计信息

  public:
    // 添加一条指令日志：记录指令名、掩码，可选参数 N 表示向量宽度
    void addLog(const char * instruction, __cs149_mask mask, int N = 0);
    // 打印统计信息（通道利用率等）
    void printStats();
    // 打印所有指令日志
    void printLog();
};

#endif
