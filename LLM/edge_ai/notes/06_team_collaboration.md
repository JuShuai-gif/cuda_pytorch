# 跨团队协作与性能治理

## 1. 与感知团队协作

### 1.1 模型延迟预算

感知团队负责深度学习模型的推理。性能工程师需要与感知团队协商：

- **每帧延迟预算**：感知阶段允许的最大耗时（例如在 100ms 端到端预算中，感知占 50ms）
- **尾延迟约束**：不只看平均延迟，P99 延迟也必须满足预算（因为系统按最差情况设计）
- **批量大小约束**：大 batch 提升 GPU 利用率但增加延迟，需找到平衡点

**实践**：在 ONNX/TensorRT 模型导出时，附带性能元数据文件（profiling.json）记录不同 batch size 下的延迟分布。

### 1.2 输入/输出契约

感知模块的接口必须在性能和正确性之间建立契约：

```cpp
struct PerceptionContract {
    // Input: sensor data at defined resolution & rate
    int input_width;           // pixels
    int input_height;          // pixels
    float input_fps;           // frames per second
    PixelFormat input_format;  // RGB8 / YUV420 / RAW10

    // Output: detections with confidence
    int max_detections;        // upper bound for memory planning
    LatencyBudget budget;      // p50/p99 in microseconds
};
```

**常见冲突**：
- 感知团队想要更高分辨率 → 更多计算时间
- 性能团队要求降低分辨率 → 检测精度下降
- 解决：做精度-延迟的 Pareto 曲线分析，选择满足精度要求下的最快配置

## 2. 与控制团队协作

### 2.1 实时性保证

控制团队（规划/控制）需要确定性的低延迟和高可靠性的周期执行：
- **确定性**：每个控制周期必须在截止时间前完成，超时可能意味着安全事故
- **周期精确**：控制算法假设固定步长（如 10ms），任何抖动都会影响控制质量

### 2.2 抖动预算（Jitter Budget）

定义从传感器曝光到执行器响应的端到端时间窗口：

```
传感器曝光 → 数据传输 → 感知 → 融合 → 规划 → 控制 → 执行器
|<----------------- 传感器到执行器延迟 ------------------>|
|<-- 软实时约束：100ms -->|<-- 硬实时约束：10ms -->|
```

分配抖动预算的方法：

1. **自顶向下**：从系统级 SLA（如 100ms e2e）开始，按权重分配给各团队
2. **自底向上**：各团队测量本模块的延迟分布，汇总评估是否满足总预算
3. **迭代协商**：如果汇总超出总预算，从贡献最大（或优化空间最大）的模块开始压预算

### 2.3 优先级管理

控制相关线程应使用实时优先级（`SCHED_FIFO` 或 `SCHED_RR`），感知线程使用普通优先级。

```bash
# 示例：将关键线程设置为实时优先级
chrt -f -p 80 <tid>
```

**注意**：Linux 的 `SCHED_FIFO` 没有时间片概念，高优先级线程可以永久饿死低优先级线程。合理使用 `sched_yield()` 或 `SCHED_RR` 防止饥饿。

## 3. 与平台团队协作

### 3.1 操作系统配置

平台团队负责内核配置和系统调优。性能工程师需要提出的需求：

| 配置项 | 推荐值 | 影响 |
|--------|--------|------|
| `isolcpus` | 隔离 2-4 个核心给关键线程 | 消除调度抖动 |
| `nohz_full` | 对被隔离的核心关闭 tick | 消除周期中断抖动 |
| `rcu_nocbs` | 将被隔离核心从 RCU 回调中排除 | 消除 RCU 软中断 |
| `irqaffinity` | 将硬件中断导向非隔离核心 | 避免中断打扰实时核心 |
| `preempt` 内核 | `CONFIG_PREEMPT_RT` | 降低内核抢占延迟 |
| CPU governor | `performance` | 禁止降频 |
| 透明大页 | `madvise`（非 `always`） | 避免内存压缩暂停 |

### 3.2 内核版本与驱动

不同内核版本的调度行为差异显著：
- 5.4 之前：CFS 调度器对延迟敏感任务支持有限
- 5.15+：Core Scheduling（缓解 SMT 侧信道但可能影响吞吐）
- 6.1+：EEVDF 调度器替代 CFS

GPU 驱动版本也必须锁定：同一 CUDA 版本在不同驱动上性能差异可达 5-15%。

### 3.3 系统资源分配

使用 cgroup v2 控制 CPU、内存和 I/O 带宽的分配：

```bash
# 为机器人核心进程预留 CPU（保证 80% 的 CPU 时间）
echo "+cpu" > /sys/fs/cgroup/robot/cgroup.subtree_control
echo "80000 100000" > /sys/fs/cgroup/robot/cpu.max
```

## 4. 性能 SLA/SLO 的制定与执行

### 4.1 SLA vs SLO

- **SLA（Service Level Agreement）**：对外承诺（或系统级硬要求）。违反时有严重后果（如安全事故）。
- **SLO（Service Level Objective）**：内部目标，为满足 SLA 留有余量。违反时触发报警但非紧急。

**例子**：
- SLA：端到端延迟 P99 < 100ms
- SLO：端到端延迟 P99 < 80ms（留有 20% 缓冲）

### 4.2 性能契约文档

每个模块的输出应附带性能契约（Performance Contract）：

```protobuf
message ModuleContract {
  string module_name = 1;
  string version = 2;
  LatencyRequirement latency = 3;  // p50/p99/max in us
  ThroughputRequirement throughput = 4;  // fps
  JitterRequirement jitter = 5;  // max allowed jitter in us
  ResourceRequirement resource = 6;  // CPU%, GPU%, memory MB
}
```

将性能契约纳入 CI/CD：每次提交都对比实际测量值与契约，超出阈值则阻断合并。

## 5. 延迟预算谈判方法论

### 5.1 五步法

1. **设定系统级目标**：如"传感器到执行器延迟 P99 < 100ms"
2. **识别所有阶段**：绘制完整数据流图（从传感器驱动到执行器指令）
3. **测量各阶段基线**：在目标硬件上用代表性负载测试每个阶段
4. **按比例分配预算**：占总延迟 40% 的阶段得 40% 预算，但需给瓶颈阶段额外缓冲
5. **预留缓冲池（10-20%）**：预留给未预见的开销（序列化、内存分配、OS 抖动）

### 5.2 预算超支处理

当某团队确认无法满足预算时，按顺序尝试：

1. 该团队内部优化（选择更优算法/模型）
2. 借用缓冲池
3. 从富裕团队转移预算
4. 升级系统硬件
5. 降低系统级 SLA（最后手段，需产品经理和客户确认）

## 6. 共享性能看板

### 6.1 看板设计

Grafana dashboard 应包含：

| 面板 | 类型 | 受众 |
|------|------|------|
| 端到端延迟 P50/P99 时间序列 | 折线图 | 全体 |
| 各阶段延迟堆叠图 | 堆叠面积图 | 性能团队 |
| GPU 利用率 + 显存 | 双 Y 轴折线图 | 感知/平台团队 |
| 队列深度热力图 | 热力图（时间 x 阶段） | 性能团队 |
| 帧丢失计数 | 计数器 | 全体 |
| CPU 核心使用率 | 每条线一个核心 | 平台团队 |
| 违反 SLA 事件 | 时间线标注 | 全体 |

### 6.2 告警规则

- 端到端 P99 > SLO * 0.9 → Warning（接近阈值）
- 端到端 P99 > SLA → Critical
- 任何阶段延迟占比突变超过 20% → Warning（可能有回归）
- GPU 利用率 < 20% 且队列深度 > 3 → Warning（流水线阻塞）

## 7. CI/CD 中的性能回归检测

### 7.1 基准测试集成

```yaml
# CI 流水线中的性能测试 stage
perf_regression:
  stage: test
  script:
    - ./build_and_run_benchmark.sh
    - python compare_perf.py --baseline main --current HEAD
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
```

### 7.2 评判标准

- 延迟恶化 > 5%（P50）或 > 10%（P99）：**阻断合并**
- 内存增长 > 5%：**阻断合并**
- GPU 利用率下降 > 10%（同一模型）：**要求解释但不阻断**
- FPS 下降 > 3%：**阻断合并**

### 7.3 噪声处理

基准测试的干扰来源：CPU 频率波动、其他进程、NUMA 效应、thermal throttling。

**最佳实践**：
- 专用物理机运行基准测试（非虚拟机）
- 每个基准跑 10-30 次取中位数（非平均值，抗离群值）
- 禁用 CPU 频率调节（`cpupower frequency-set -g performance`）
- 用 `taskset` 固定基准进程的 CPU 亲和性
