# Architecture Document

## AI/ML Operator Inference Task Scheduling System

---

## 1. System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[Client Applications / REST API]
    end

    subgraph "Task Scheduler Core"
        TS[TaskScheduler<br/>Ch8.5: Core Orchestrator]
        PQ[PriorityTaskQueue<br/>Ch6.3: Priority Ordering]
        TP[ThreadPool<br/>Ch9.1: Worker Management]
    end

    subgraph "Thread Pool Internals"
        subgraph "Worker Threads"
            W1[Worker 0<br/>Local Queue + Worker Loop]
            W2[Worker 1<br/>Local Queue + Worker Loop]
            W3[Worker N<br/>Local Queue + Worker Loop]
        end
        GQ[Global TaskQueue<br/>Ch6.2: MPMC Queue]
        WS[Work Stealing<br/>Ch8.4: Load Balancing]
    end

    subgraph "Concurrency Primitives"
        SL[Spinlock<br/>Ch5.3: TTAS Lock]
        CC[ConcurrentCache<br/>Ch3.3: shared_mutex LRU]
        ST[StopToken<br/>Ch9.2: Cooperative Stop]
        LG[Logger<br/>Ch11: Thread-Safe Logging]
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

## 2. Task Scheduling Flow

```mermaid
sequenceDiagram
    participant Client
    participant TaskScheduler
    participant PriorityQueue
    participant ThreadPool
    participant Worker
    participant Cache

    Client->>TaskScheduler: submit(task, priority)
    TaskScheduler->>PriorityQueue: push(task, priority)
    TaskScheduler->>TaskScheduler: dispatch_pending()
    
    loop Until queue empty or batch limit
        TaskScheduler->>PriorityQueue: try_pop(highest priority)
        TaskScheduler->>ThreadPool: submit_to_local(worker_idx, task)
    end

    Worker->>Worker: get_task()
    
    alt Local queue has task
        Worker->>Worker: try_pop(local_queue)
    else Steal from neighbor
        Worker->>Worker: steal_task(victim_worker)
    else Check global queue
        Worker->>Worker: try_pop(global_queue)
    else No tasks available
        Worker->>Worker: wait on condition_variable
    end

    Worker->>Cache: check result cache (shared_lock)
    Worker->>Worker: execute task
    Worker->>Cache: store result (unique_lock)
    Worker->>TaskScheduler: task complete
```

## 3. Thread Pool Workflow

```mermaid
stateDiagram-v2
    [*] --> Idle: Thread created
    Idle --> CheckingLocal: Woken up / New task submitted
    
    CheckingLocal --> Executing: Local queue has task
    CheckingLocal --> CheckingGlobal: Local queue empty
    CheckingGlobal --> Executing: Global queue has task
    CheckingGlobal --> Stealing: Global queue empty
    
    Stealing --> Executing: Stole from neighbor
    Stealing --> Waiting: All queues empty
    
    Waiting --> CheckingLocal: New task notification
    Waiting --> Exiting: Stop requested
    
    Executing --> CheckingLocal: Task done
    Exiting --> [*]: Thread joined
```

## 4. Module-to-Chapter Correspondence

| Module | Ch2 | Ch3 | Ch4 | Ch5 | Ch6 | Ch7 | Ch8 | Ch9 | Ch10 | Ch11 |
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

## 5. Performance Considerations

### 5.1 Lock Contention Hierarchy

| Contention Level | Component | Strategy |
|-----------------|-----------|----------|
| Very High | Spinlock | TTAS with exponential backoff (Ch5.3.3) |
| High | Task Queue (global) | Single mutex, bulk operations (Ch6.2.5) |
| Moderate | Priority Queue | Single mutex, O(log n) heap ops |
| Low | Logger | Atomic fast-path check (Ch11.3) |
| Very Low | Cache (reads) | shared_mutex for read concurrency (Ch3.3.2) |

### 5.2 Design Trade-offs

1. **Simplicity vs. Performance**: Lock-based queues chosen over lock-free (Ch7)
   for correctness guarantees and maintainability
2. **Work Stealing Overhead**: Random victim selection (O(1)) vs sequential
   (O(n)). Random trades fairness for lower contention
3. **Priority Inversion**: Single-mutex priority queue avoids priority inversion
   but serializes dequeue. Acceptable for moderate queue depths
4. **Cache Coherence**: TTAS spinlock polls read-only first (L1 cache shared state),
   only attempting atomic write when lock appears free

## 6. Extension Directions

1. **Lock-free Queues** (Ch7): Replace lock-based queues with Michael-Scott queue
2. **NUMA-Aware Scheduling**: Pin workers to NUMA nodes (Ch11)
3. **GPU Task Offloading**: Extend ThreadPool with CUDA stream management
4. **Distributed Scheduling**: Multi-node task distribution via gRPC
5. **Dynamic Batching**: Auto-batching of small inference requests
6. **Profiling Integration**: Per-task timing with nanosecond precision
7. **A/B Model Deployment**: Hot-swap model versions in inference pipeline
