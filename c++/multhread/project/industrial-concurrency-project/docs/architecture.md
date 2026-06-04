# 架构文档

## AI/ML 算子推理任务调度系统

---

## 1. 系统架构

```mermaid
graph TB
    subgraph "客户端层"
        CLI[客户端应用 / REST API]
    end

    subgraph "任务调度器核心"
        TS[TaskScheduler<br/>Ch8.5：核心编排器]
        PQ[PriorityTaskQueue<br/>Ch6.3：优先级排序]
        TP[ThreadPool<br/>Ch9.1：工作线程管理]
    end

    subgraph "线程池内部结构"
        subgraph "工作线程"
            W1[Worker 0<br/>本地队列 + 工作循环]
            W2[Worker 1<br/>本地队列 + 工作循环]
            W3[Worker N<br/>本地队列 + 工作循环]
        end
        GQ[全局 TaskQueue<br/>Ch6.2：MPMC 队列]
        WS[工作窃取<br/>Ch8.4：负载均衡]
    end

    subgraph "并发原语"
        SL[Spinlock<br/>Ch5.3：TTAS 锁]
        CC[ConcurrentCache<br/>Ch3.3：shared_mutex LRU]
        ST[StopToken<br/>Ch9.2：协作式停止]
        LG[Logger<br/>Ch11：线程安全日志]
    end

    CLI --> TS
    TS --> PQ
    TS --> TP
    TS --> CC
    TS --> LG
    TP --> GQ
    TP --> W1 & W2 & W3
    W1 & W2 & W3 --> WS
    GQ --> SL
    W1 --> SL
```

## 2. 任务调度流程

```mermaid
sequenceDiagram
    participant 客户端
    participant 任务调度器
    participant 优先级队列
    participant 线程池
    participant 工作线程
    participant 缓存

    客户端->>任务调度器: submit(task, priority)
    任务调度器->>优先级队列: push(task, priority)
    任务调度器->>任务调度器: dispatch_pending()
    
    loop 直到队列为空或达到批量上限
        任务调度器->>优先级队列: try_pop(最高优先级)
        任务调度器->>线程池: submit_to_local(worker_idx, task)
    end

    工作线程->>工作线程: get_task()
    
    alt 本地队列有任务
        工作线程->>工作线程: try_pop(local_queue)
    else 从邻居窃取
        工作线程->>工作线程: steal_task(victim_worker)
    else 检查全局队列
        工作线程->>工作线程: try_pop(global_queue)
    else 无可用任务
        工作线程->>工作线程: 在 condition_variable 上等待
    end

    工作线程->>缓存: 检查结果缓存（shared_lock）
    工作线程->>工作线程: 执行任务
    工作线程->>缓存: 存储结果（unique_lock）
    工作线程->>任务调度器: 任务完成
```

## 3. 线程池工作流程

```mermaid
stateDiagram-v2
    [*] --> 空闲: 线程创建
    空闲 --> 检查本地队列: 被唤醒 / 新任务提交
    
    检查本地队列 --> 执行中: 本地队列有任务
    检查本地队列 --> 检查全局队列: 本地队列为空
    检查全局队列 --> 执行中: 全局队列有任务
    检查全局队列 --> 窃取中: 全局队列为空
    
    窃取中 --> 执行中: 从邻居窃取到任务
    窃取中 --> 等待中: 所有队列为空
    
    等待中 --> 检查本地队列: 收到新任务通知
    等待中 --> 退出中: 收到停止请求
    
    执行中 --> 检查本地队列: 任务完成
    退出中 --> [*]: 线程已 join
```

## 4. 模块与章节对应关系

| 模块 | Ch2 | Ch3 | Ch4 | Ch5 | Ch6 | Ch7 | Ch8 | Ch9 | Ch10 | Ch11 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|
| `spinlock.hpp` | | | | X | | | | | | |
| `stop_token.hpp` | | | | X | | | | X | | |
| `task_queue.hpp` | | X | X | | X | | | | | |
| `priority_task_queue.hpp` | | X | X | | X | | | | | |
| `concurrent_cache.hpp` | | X | | | | | | | | |
| `thread_pool.hpp` | | X | X | | | | X | X | | |
| `task_scheduler.hpp` | X | | X | | | | X | X | | |
| `logger.hpp` | | X | | X | | | | | | X |
| `main.cpp` | X | | | | | | | | | |
| `test_*.cpp` | | | | | | | | | X | |

## 5. 性能考量

### 5.1 锁竞争层级

| 竞争级别 | 组件 | 策略 |
|-----------------|-----------|----------|
| 极高 | Spinlock | 带指数退避的 TTAS（Ch5.3.3） |
| 高 | 任务队列（全局） | 单 mutex，批量操作（Ch6.2.5） |
| 中等 | 优先级队列 | 单 mutex，O(log n) 堆操作 |
| 低 | Logger | 原子变量快速路径检查（Ch11.3） |
| 极低 | 缓存（读操作） | shared_mutex 支持读并发（Ch3.3.2） |

### 5.2 设计权衡

1. **简洁性 vs. 性能**：选择基于锁的队列而非无锁方案（Ch7），以保证正确性和可维护性
2. **工作窃取开销**：随机选择受害线程（O(1)）vs. 顺序选择（O(n)）。随机化以较低的竞争换取更好的公平性
3. **优先级反转**：单 mutex 优先级队列避免了优先级反转，但出队操作会串行化。对于中等队列深度是可接受的
4. **缓存一致性**：TTAS 自旋锁先进行只读轮询（L1 缓存共享状态），仅在锁似乎可用时才尝试原子写入

## 6. 扩展方向

1. **无锁队列**（Ch7）：用 Michael-Scott 队列替换基于锁的队列
2. **NUMA 感知调度**：将工作线程绑定到 NUMA 节点（Ch11）
3. **GPU 任务卸载**：扩展 ThreadPool 以支持 CUDA 流管理
4. **分布式调度**：通过 gRPC 实现多节点任务分发
5. **动态批处理**：对小推理请求进行自动批处理合并
6. **性能分析集成**：纳秒级精度的每任务计时
7. **A/B 模型部署**：在推理流水线中热切换模型版本
