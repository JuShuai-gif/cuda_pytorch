#include "logger.h"
#include "CS149intrin.h"

// 添加一条向量指令执行日志
// instruction: 指令名称, mask: 向量掩码(哪些通道是活跃的), N: 通道总数
void Logger::addLog(const char * instruction, __cs149_mask mask, int N) {
  Log newLog;
  strcpy(newLog.instruction, instruction);
  newLog.mask = 0;
  // 遍历所有通道，将活跃通道记录到位掩码中
  for (int i=0; i<N; i++) {
    if (mask.value[i]) {
      newLog.mask |= (((unsigned long long)1)<<i);  // 将第 i 位置为 1，表示该通道被使用
      stats.utilized_lane++;
    }
  }
  stats.total_lane += N;
  stats.total_instructions += (N>0);
  log.push_back(newLog);
}

// 打印向量单元统计信息
void Logger::printStats() {
  printf("****************** Printing Vector Unit Statistics *******************\n");
  printf("Vector Width:              %d\n", VECTOR_WIDTH);
  printf("Total Vector Instructions: %lld\n", stats.total_instructions);
  printf("Vector Utilization:        %.1f%%\n", (double)stats.utilized_lane/stats.total_lane*100);
  printf("Utilized Vector Lanes:     %lld\n", stats.utilized_lane);
  printf("Total Vector Lanes:        %lld\n", stats.total_lane);
}



// 打印向量单元执行日志
// 每行显示一条指令，以及每条向量通道的占用情况（* 表示活跃，_ 表示不活跃）
void Logger::printLog() {
  printf("***************** Printing Vector Unit Execution Log *****************\n");
  printf(" Instruction | Vector Lane Occupancy ('*' for active, '_' for inactive)\n");
  printf("------------- --------------------------------------------------------\n");
  for (int i=0; i<log.size(); i++) {
    printf("%12s | ", log[i].instruction);
    for (int j=0; j<VECTOR_WIDTH; j++) {
      if (log[i].mask & (((unsigned long long)1)<<j)) {
        printf("*");
      } else {
        printf("_");
      }
    }
    printf("\n");
  }
}
