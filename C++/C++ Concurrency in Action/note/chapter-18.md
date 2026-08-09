# 第18章：边缘端 VLA 部署的并发架构

> 目标不是“把 CPU/GPU 跑满”，而是在功耗、内存和热设计约束下，稳定满足控制周期与安全 deadline。

## 18.1 数据面与控制面

典型链路是 `camera → preprocess → H2D → vision/language/action inference → postprocess → actuator`。数据面处理张量，控制面负责模型热更新、健康检查、停止和降级。不要让模型加载、日志落盘或遥测阻塞实时数据面。

每个跨线程对象必须明确：所有权、最大存活时间、队列容量、满队列策略、停止协议和异常传播。优先使用值语义、RAII、`std::jthread` 和 `std::stop_token`，避免 detached thread。

## 18.2 背压是正确性要求

无界队列会把吞吐不足伪装成不断增长的延迟。对闭环控制，旧观测通常不再有价值，适合容量 1~2 的 latest-value mailbox；离线批处理才适合完整 FIFO。

| 策略 | 适用场景 | 风险 |
|---|---|---|
| block producer | 不允许丢数据 | 可能阻塞采集线程 |
| drop newest | 当前积压必须完成 | 输出逐渐陈旧 |
| drop oldest/latest-value | 控制与机器人感知 | 必须记录 drop rate |
| dynamic batching | 吞吐服务 | 增加排队延迟 |

容量应由预算推导：`capacity <= latency_budget / worst_case_service_time`，并在过载测试中验证，而不是凭经验设 1024。

## 18.3 CPU/GPU 异步边界

- CUDA kernel launch 是异步的；CPU 计时不能代表设备执行时间，设备段用 CUDA event。
- 每条长期流水线使用明确的非默认 stream；跨 stream 依赖用 event，少用全局同步。
- pinned host memory 能加速异步 H2D/D2H，但会占用不可分页内存，必须做池化和上限控制。
- 推理热路径禁止频繁 `cudaMalloc`、模型加载、隐式 tensor copy 和同步日志。
- TensorRT execution context 通常按并发执行单元独占；不要在未确认线程安全契约时共享可变 context。

## 18.4 实时性与调度

先定义 SLO：控制频率、端到端 P50/P95/P99、deadline miss rate、drop rate、RSS/GPU memory 和功耗。平均延迟不能描述尾延迟。

线程绑核、实时优先级、NUMA first-touch 只能在 profiling 证明调度抖动是瓶颈后使用。隔离相机/控制线程与繁重预处理线程，并为失败路径定义 hold-last-safe-action、零动作或安全停机。

## 18.5 测量方法

1. 预热模型，区分 cold start 与 steady state。
2. 端到端时间用 `steady_clock`，GPU stage 用 event。
3. 报告分位数、样本数、频率/功耗模式、线程亲和性与模型精度配置。
4. 压测必须覆盖输入突发、推理变慢、设备错误、取消、队列满和长时间 soak。
5. 使用 TSan 检 CPU 数据竞争；ASan/UBSan 单独构建，不能把 TSan 与 ASan 混在一个进程。

## 18.6 岗位面试检查表

- 能解释 acquire/release 建立的 happens-before，而不只是背诵内存序。
- 能说明 lock-free 不等于 wait-free，也不天然快；内存回收是无锁结构的核心难点。
- 能从 deadline 反推队列和并发度，并解释过载降级策略。
- 能区分 host latency、device latency、吞吐和端到端尾延迟。
- 能写出可取消、可 drain、有超时、可观测、TSan 可验证的流水线。

配套源码：`src/chapter-18/01_edge_vla_pipeline.cpp`。它用容量一邮箱模拟推理慢于相机时的丢旧帧策略，并输出 drop、deadline miss 与 P50/P95/P99。
