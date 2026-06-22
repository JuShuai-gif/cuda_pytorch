# 实时系统基础

## 1. 什么是实时系统

实时系统（Real-Time System）的核心要求不是"快"，而是**"在截止时间前完成"**。一个计算结果即使完全正确，如果返回得太晚，它也是无价值的，甚至是危险的。

### 1.1 三类实时系统

| 类型 | 定义 | 错过期限的后果 | 示例 |
|------|------|----------------|------|
| **硬实时 (Hard RT)** | 任何一次错过期限都是系统失败 | 灾难性（人身伤害/系统损毁） | 飞行控制、汽车制动 |
| **固实时 (Firm RT)** | 偶尔错过期限可容忍，但结果作废 | 结果无效，需要重算或丢弃 | 视频流丢帧、交易系统 |
| **软实时 (Soft RT)** | 可以接受偶尔超时，结果价值随时间递减 | 服务质量下降 | 游戏渲染、语音助手 |

在自动驾驶中：**制动控制是硬实时，路径规划是固实时，日志上传是软实时**。

### 1.2 确定性 (Determinism)

实时系统的核心品质是**确定性**——相同的输入在相同的时间内产生相同的输出。这与平均性能优化的思路完全不同：

- 平均延迟 1ms，但 p99 是 50ms → **不是实时系统**
- 平均延迟 10ms，但最差也是 10.1ms → **是实时系统**

**抖动 (Jitter)** 是确定性的敌人。100μs 的抖动在 100Hz 控制循环中意味着 1% 的不确定性。

## 2. 确定性为什么重要

### 2.1 控制系统的视角

以自动驾驶车辆的横向控制为例：

- 控制周期 T = 10ms（100Hz）
- 如果实际执行间隔存在 ±2ms 的抖动
- 在 100km/h 速度下，2ms 对应 5.6cm 的位移误差
- 积分控制器会累积这种误差，导致轨迹振荡

### 2.2 抖动对控制质量的影响

```
等周期控制：x---x---x---x---x---x--- （采样均匀，控制稳定）
有抖动控制：x--x-----x--x------x--x  （采样不均，控制振荡）
```

研究表覍：当控制周期的抖动超过周期本身的 20% 时，控制质量开始显著下降。

### 2.3 抖动的系统级来源

| 来源 | 量级 | 缓解措施 |
|------|------|----------|
| 调度器抢占 | 1μs～10ms | CPU 隔离 + 实时调度 |
| 中断处理 | 10μs～100μs | IRQ affinity |
| TLB miss | ～50ns | 大页 (Huge Pages) |
| Cache miss | 10ns～100ns | Cache locking / prefetch |
| 电源管理 | 10μs～1ms | 固定频率 (performance governor) |
| SMI (System Management Interrupt) | ～100μs | BIOS 禁用 SMI |

## 3. 实时调度理论

### 3.1 任务模型

每个实时任务 τ_i 由三个参数定义：
- **C_i (Worst-Case Execution Time, WCET)**：最坏情况执行时间
- **T_i (Period)**：周期
- **D_i (Deadline)**：相对截止时间（通常 D_i ≤ T_i）

利用率：U = Σ(C_i / T_i)。对于 n 个任务：
- RMS 可调度的充分条件：U ≤ n(2^(1/n) - 1)

### 3.2 速率单调调度 (Rate Monotonic Scheduling, RMS)

**规则**：周期越短，优先级越高（静态优先级）。

- **Liu & Layland 定理 (1973)**：RMS 是最优的固定优先级调度算法
- **可调度性条件**：Σ(C_i / T_i) ≤ n(2^(1/n) - 1)
  - n=1: 100%
  - n=2: 82.8%
  - n→∞: 69.3% (ln(2))
- **适用场景**：任务周期已知且固定（如传感器采样）

**示例**：
```
任务 A: C=1ms, T=4ms  → 优先级高
任务 B: C=2ms, T=6ms  → 优先级中
任务 C: C=3ms, T=12ms → 优先级低

利用率: 1/4 + 2/6 + 3/12 = 0.25 + 0.33 + 0.25 = 0.833 > 0.78
→ 不能保证 RMS 可调度，需要 EDF 或降低负载
```

### 3.3 最早截止时间优先 (Earliest Deadline First, EDF)

**规则**：截止时间越近，优先级越高（动态优先级，每步重新计算）。

- **Dertouzos 定理**：EDF 是最优的动态优先级单处理器调度算法
- **可调度性条件**：Σ(C_i / T_i) ≤ 100%（必要且充分）
- **优势**：CPU 利用率可达 100%，优于 RMS
- **劣势**：动态优先级实现更复杂，过载时行为不可预测

### 3.4 RMS vs EDF 对比

| 特性 | RMS | EDF |
|------|-----|-----|
| 优先级类型 | 静态（固定） | 动态（每次决策） |
| 可调度利用率上限 | ~69% (n→∞) | 100% |
| 实现复杂度 | 低 | 中 |
| 过载行为 | 低优先级先丢 | 不可预测（domino 效应） |
| 实时系统的常见选择 | 嵌入式 RTOS | Linux SCHED_DEADLINE |

## 4. 优先级反转与解决方案

### 4.1 什么是优先级反转

经典的优先级反转三任务场景：

```
高优先级任务 H：等待锁 L
中优先级任务 M：不需要锁，正常执行
低优先级任务 L_owner：持有锁 L

问题：
1. H 被阻塞等待 L
2. M 抢占了 L_owner（因为 M 优先级 > L_owner）
3. M 持续执行，H 被无限期推迟
→ H 的等待时间不再取决于临界区长度，而取决于 M 的执行时间
```

**历史经典案例**：1997 年火星探路者 (Mars Pathfinder) 因优先级反转导致系统反复重启，NASA JPL 团队最终通过远程启用优先级继承修复。

### 4.2 优先级继承 (Priority Inheritance)

**机制**：当高优先级任务阻塞在锁上时，持有锁的低优先级任务**临时继承**高优先级。

```
初始: prio(H)=high, prio(M)=mid, prio(L)=low
1. L 获取锁
2. H 尝试获取锁 → L 继承 H 的优先级 → prio(L)=high
3. M 无法抢占 L（L 现在是高优先级）
4. L 释放锁，恢复原优先级
5. H 获取锁继续执行
```

**局限性**：可能发生死锁（需要结合锁排序等健壮设计）。

### 4.3 优先级天花板 (Priority Ceiling)

**机制**：每把锁分配一个"天花板优先级" = 所有可能使用该锁的任务的最高优先级。任务获取锁时，立即提升到天花板优先级。

| 特性 | 优先级继承 | 优先级天花板 |
|------|-----------|-------------|
| 阻塞次数 | ≤ 1 次（递归阻塞） | ≤ 1 次（入口阻塞） |
| 实现复杂度 | 中 | 高 |
| 适用锁类型 | 动态锁 | 静态已知全部使用者 |
| 是否防止死锁 | 否 | 是 |

## 5. Linux 实时支持

### 5.1 PREEMPT_RT 补丁

PREEMPT_RT 将 Linux 变成完全可抢占的内核：

- **原理**：将自旋锁替换为可睡眠的 rt_mutex，中断处理线程化
- **效果**：内核抢占延迟从毫秒级降至微秒级
- **代价**：吞吐量下降 10%～30%（因为锁开销增加）

```bash
# 检查是否启用 PREEMPT_RT
uname -v | grep PREEMPT
```

### 5.2 Linux 实时调度策略

| 调度策略 | 优先级范围 | 特点 |
|----------|-----------|------|
| `SCHED_FIFO` | 1～99 | 运行直到主动让出或更高优先级就绪 |
| `SCHED_RR` | 1～99 | 同优先级时间片轮转（timeslice=100ms） |
| `SCHED_DEADLINE` | 最高 | EDF + Constant Bandwidth Server |
| `SCHED_OTHER` | 0 (nice -20～19) | CFS 完全公平调度器（非实时） |

`SCHED_DEADLINE` (Linux 3.14+) 是 Linux 中最先进的实时调度：
```c
struct sched_attr attr = {
    .size = sizeof(attr),
    .sched_policy = SCHED_DEADLINE,
    .sched_runtime = 1 * 1000 * 1000,    // 1ms WCET
    .sched_deadline = 10 * 1000 * 1000,  // 10ms deadline
    .sched_period = 10 * 1000 * 1000,    // 10ms period
};
sched_setattr(0, &attr, 0);
```

### 5.3 内存锁定

实时线程必须锁定所有内存页面，防止缺页异常导致的不可预测延迟：

```c
mlockall(MCL_CURRENT | MCL_FUTURE);
```

## 6. 实时编程的常见陷阱

### 6.1 动态内存分配

`malloc`/`new` 永远不应出现在实时路径中。原因：
- 可能触发系统调用来扩展堆 (sbrk/mmap)
- 可能触发内存碎片整理
- 执行时间不可预测

**解决方案**：预分配内存池、使用栈分配、Placement new。

### 6.2 日志与 I/O

- `printf`/`cout` 可能触发缓冲区刷新和系统调用
- `fprintf(stderr, ...)` 通常是行缓冲的，不可预测
- 磁盘 I/O 可能触发文件系统日志和缓存写回

**解决方案**：无锁环形缓冲区 + 后台日志线程。

### 6.3 系统调用

以下系统调用即使在 `SCHED_FIFO` 下也可能导致阻塞：
- `open()` / `close()` 操作文件系统
- `read()` / `write()` 操作磁盘文件
- `ioctl()` 涉及硬件交互

### 6.4 优先级设置顺序

```c
// 正确顺序
mlockall(MCL_CURRENT | MCL_FUTURE);  // 1. 先锁内存
set_scheduler(SCHED_FIFO, 80);      // 2. 再提升优先级
// ... real-time code ...
```

如果先提优先级再 `mlockall`，缺页异常将以高优先级处理，可能使其他关键任务饿死。

## 7. 延迟测量工具

### 7.1 cyclictest

标准 Linux 实时延迟测量工具：

```bash
# 测量调度延迟：每秒报告一次，运行1小时
cyclictest --mlockall --priority=99 --interval=200 --distance=0 \
           --duration=1h --histogram=1000
```

输出含延迟分布直方图。理想的 PREEMPT_RT 系统上，最大延迟应 < 20μs。

### 7.2 oslat

操作系统的整体延迟基准测试：

```bash
oslat --runtime 3600 --rt-priority 99
```

### 7.3 自定义测量

在代码中测量运行时抖动：

```c
auto before = clock_gettime(CLOCK_MONOTONIC);
// ... real-time work ...
auto after = clock_gettime(CLOCK_MONOTONIC);
int64_t delta_ns = (after.tv_sec - before.tv_sec) * 1e9
                 + (after.tv_nsec - before.tv_nsec);
```

**注意**：`CLOCK_MONOTONIC` 不受系统时间调整影响，适用于延迟测量；`CLOCK_MONOTONIC_RAW`（Linux 2.6.28+）进一步避免 NTP 频率调整的影响。

## 8. 实时系统的实际部署建议

1. **内核**：使用 PREEMPT_RT 补丁的内核（如 Ubuntu 的 `linux-lowlatency`）
2. **CPU 隔离**：`isolcpus=2-7` 内核参数隔离核心，然后 `taskset` 绑定实时任务
3. **中断亲和性**：将不需要的中断导向隔离核心之外
4. **频率锁定**：`cpupower frequency-set -g performance` 禁用动态调频
5. **禁用 C-States**：`processor.max_cstate=0` 内核参数，禁止深度睡眠
6. **大页**：`hugepagesz=2M default_hugepagesz=2M hugepages=1024`
7. **禁用 SMI**：通过 BIOS/UEFI 设置禁用 System Management Interrupts
8. **RCU 隔离**：`rcu_nocbs=2-7 rcu_nocb_poll` 将 RCU 回调移出隔离核心

**验证**：在以上所有配置生效后，使用 `cyclictest` 验证最坏情况延迟 ≤ 20μs，方可声称系统具有实时能力。
