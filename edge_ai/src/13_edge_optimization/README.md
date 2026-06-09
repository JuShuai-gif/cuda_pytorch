# 13_edge_optimization - 边缘端性能优化实战

## 概述

本项目基于 RK3588 机器人视觉管线的真实优化案例，演示了边缘端 C++ 性能优化的关键技术：

1. **Uncached vs Cached 内存访问** — DMA buffer 的缓存一致性问题
2. **DMA_BUF_IOCTL_SYNC** — 缓存同步的正确使用姿势
3. **NEON SIMD** — FP16→FP32 / BGR→FP16 RGB 硬件加速转换
4. **Fail-Closed 错误处理** — 安全关键系统的错误处理模式

## 文件结构

```
13_edge_optimization/
  memory_bench.h        - 内存访问基准测试声明
  memory_bench.cpp      - uncached vs cached / DMA_SYNC 模拟 / 带宽争抢
  neon_convert.h        - NEON 转换函数声明
  neon_convert.cpp      - FP16→FP32 / BGR→FP16 RGB 的标量 & NEON 实现
  fail_closed.h         - Fail-closed 模式声明
  fail_closed.cpp       - 多步管线失败处理演示（含 fail-open 对比）
  main.cpp              - 入口：依次运行所有基准，输出 JSON 指标
  CMakeLists.txt
  README.md
```

## 推荐阅读顺序

1. **`memory_bench.h` + `memory_bench.cpp`** — 内存访问模式（uncached → cached + DMA_SYNC），边缘端性能的基础课题
2. **`neon_convert.h` + `neon_convert.cpp`** — NEON SIMD 转换（FP16↔FP32、BGR↔FP16 RGB），独立但同属边缘端优化主题
3. **`fail_closed.h` + `fail_closed.cpp`** — 安全关键系统的错误处理模式（fail-closed vs fail-open）
4. **`main.cpp`** — 最后阅读，按三个部分依次调用所有演示并输出 JSON 指标

## 构建 & 运行

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./edge_optimization
```

运行后会在当前目录生成 `edge_optimization_metrics.json`。

### ARM 交叉编译（RK3588）

```bash
mkdir build_arm && cd build_arm
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/aarch64-toolchain.cmake
make -j$(nproc)
# 将 edge_optimization 推送到设备
scp edge_optimization user@rk3588:/tmp/
```

**arm64 工具链文件示例** (`aarch64-toolchain.cmake`):

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

## 需求

- C++17 编译器（GCC 9+, Clang 10+）
- CMake >= 3.14
- Linux
- ARM aarch64 平台可获得完整 NEON 加速（x86 上以标量 fallback 运行）

## 关键数值参考（RK3588 实测）

| 优化项 | 优化前 | 优化后 | 提升 |
|--------|--------|--------|------|
| 6MB 帧读取 P50 | 15ms | 3.6ms | 4.2x |
| 6MB 帧读取 P99 | 45ms | 7ms | 6.4x |
| 延迟抖动 | ±20ms | ±2ms | 10x |
| FP16→FP32 转换(145万) | 9.68ms | 0.3ms | 32x |

## 注意事项

- 在 x86 上运行仅作算法验证，实际 NEON 性能数据需要在 aarch64 平台上获取
- DDR 带宽争抢模拟使用多线程随机访问，其行为和真实 NPU/RGA/Display 同时运行的场景定性一致但数值不可直接对比
- Fail-closed 模式中的概率参数模拟真实硬件故障率；实际生产中故障率更低但后果更严重
