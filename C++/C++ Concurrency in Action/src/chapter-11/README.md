# Chapter 11: Testing and Debugging Concurrent Programs (测试与调试并发程序)

## 文件说明

| 文件 | 内容 |
|------|------|
| `01_deadlock_demo.cpp` | 死锁演示与修复: `std::scoped_lock`, 固定锁序, 层级锁, try_lock 超时 |
| `02_race_detector_demo.cpp` | ThreadSanitizer 使用: 数据竞争示例, TSan 编译/运行指南 |
| `03_stress_test.cpp` | 并发压力测试: 对线程安全队列的正确性/唯一性/稳定性测试 |
| `04_logger_for_debug.cpp` | 线程安全日志: 时间戳, 线程ID, 级别过滤, 彩色输出, 文件输出 |

## 编译

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### 启用 ThreadSanitizer

```bash
# 单独编译 02_race_detector_demo.cpp 时添加 TSan 标志
g++ -std=c++20 -g -O1 -fsanitize=thread -pthread \
    ../02_race_detector_demo.cpp -o tsan_demo
./tsan_demo
```

## 关键技术点

- **死锁预防**: `std::lock` / `scoped_lock` 原子获取多锁, 固定锁序, 层级锁, 超时回退
- **TSan**: `-fsanitize=thread` 编译标志, 运行时动态检测数据竞争
- **压力测试**: 高并发 push/pop 混合操作, 数据唯一性验证, 长时间稳定性
- **调试日志**: 线程安全原子写入, 毫秒时间戳, 彩色级别输出, 多输出目标
