# Lecture 5: Performance Optimization Part 1 - Work Distribution and Scheduling

**Source**: Stanford CS149, Fall 2025 - Lecture 5 PDF

---

## Core Concepts Summary

### 1. Programming for High Performance

**Key goals (often at odds):**
- Balance workload onto available execution resources
- Reduce communication (to avoid stalls)
- Reduce extra work (overhead) to increase parallelism, manage assignment, etc.

**TIP #1**: Always implement the simplest solution first, then measure performance to determine if you need to do better.

### 2. Static Assignment

**Definition**: Assignment of work to threads does NOT depend on dynamic behavior.
- Assignment may still depend on runtime parameters (input size, thread count)
- Simple, essentially zero runtime overhead

**When applicable**:
- Cost of work and amount of work is predictable
- All work has the same cost (simplest case)
- Work has unequal but known cost; statistics are predictable (same cost on average)

**Example from Programming Assignment 1**: Assign equal number of grid cells to each thread. Different static assignments (blocked, interleaved) of grid regions to threads.

### 3. "Semi-Static" Assignment

- Cost of work is predictable for near-term future
- Recent past is a good predictor of near future
- Application periodically profiles its execution and re-adjusts assignment
- Assignment is "static" for the interval between re-adjustments

**Use cases**:
- Particle simulation: redistribute particles as they move (slow motion = infrequent redistribution)
- Adaptive mesh: mesh changes slowly; color indicates processor assignment

### 4. Dynamic Assignment

**Definition**: Program determines assignment dynamically at runtime to ensure well-distributed load.
- Used when execution time or total number of tasks is unknown/unpredictable

**Implementation with shared counter**:
```cpp
int counter = 0;  // shared variable
while (1) {
    lock(counter_lock);
    i = counter++;
    unlock(counter_lock);
    if (i >= N) break;
    is_prime[i] = test_primality(x[i]);
}
```

### 5. Dynamic Assignment Using Work Queues

```
Sub-problems (tasks) → Shared work queue → Worker threads pull/push work
```

- Worker threads pull data from shared work queue
- Push new work to queue as it is created
- When queue is empty, thread goes idle

### 6. Task Granularity

**Fine granularity** (1 task = 1 element):
- Good workload balance (many small tasks)
- High synchronization cost (frequent critical section entry)
- High overhead

**Coarse granularity** (1 task = 10+ elements):
- Decreased synchronization cost
- Fewer critical section entries
- Potentially worse load balance

**Ideal granularity depends on**: workload characteristics, machine parameters.

**Rule of thumb**: Have many more tasks than processors (for good balance), but not so many that overhead dominates.

### 7. Smarter Task Scheduling

**Problem with simple queue**: Long task run last → load imbalance.

**Solutions**:
1. Divide work into larger number of smaller tasks
2. Schedule long tasks first (requires workload knowledge)
3. Distributed work queues with work stealing

### 8. Distributed Work Queues & Work Stealing

```
Set of work queues (one per worker thread)
Worker: pull from OWN queue, push to OWN queue
When local queue empty → STEAL from another worker's queue
```

**Benefits**:
- Avoid need for all workers to synchronize on single work queue
- Reduces contention
- Theft only occurs when a thread would be idle anyway

### 9. Fork-Join Parallelism (Cilk Plus)

**Core primitives**:
```cpp
cilk_spawn foo(args);  // "fork": caller may continue executing asynchronously with foo
cilk_sync;             // "join": returns when all spawned calls have completed
```

**Note**: Implicit `cilk_sync` at end of every function containing `cilk_spawn`.

### 10. Parallel Quicksort in Cilk Plus

```cpp
void quick_sort(int* begin, int* end) {
    if (begin >= end - PARALLEL_CUTOFF)
        std::sort(begin, end);           // sequential for small problems
    else {
        int* middle = partition(begin, end);
        cilk_spawn quick_sort(begin, middle);
        quick_sort(middle + 1, end);
    }
}
```

- Switch to sequential sort when problem size is small enough
- Overhead of spawn would trump benefits of parallelization for small problems

### 11. Cilk Work Scheduler Design

**Pool of worker threads**: Exactly as many threads as execution contexts.

**At spawn, choice**: Run child first or run continuation first?

| Strategy | Behavior | Queue Content |
|----------|----------|--------------|
| **Run child first** ("continuation stealing") | Thread executes foo(), enqueues continuation | Single continuation (represents all remaining iterations) |
| **Run continuation first** ("child stealing") | Thread enqueues child, continues with next spawn | O(N) items in queue |

**Cilk uses "run child first" (continuation stealing)**:
- Depth-first traversal of call graph
- Order of execution same as sequential program if no stealing
- Work queue storage bounded: at most T times stack storage of sequential execution

### 12. Dequeue Per Worker

Work queue implemented as a **dequeue** (double-ended queue):
- **Local thread**: pushes/pops from the **tail** (bottom) → LIFO
- **Remote threads**: steal from the **head** (top) → FIFO

**Why steal from head?**
- Steals largest amount of work (reduces number of steals)
- Maximum locality in work each thread performs
- Local thread and stealing thread don't contend for same elements
- Enables efficient lock-free implementations

### 13. Victim Selection

- Idle threads **randomly** choose a thread to attempt to steal from
- Random choice distributes stealing load

### 14. Sync Implementation

**Descriptor per sync block**:
- Tracks: number of outstanding spawns, number completed
- Created only when stealing occurs

**No-stealing case**: `cilk_sync` is a no-op (all work done by same thread).

**Stealing case**: Threads check descriptor; last thread to complete a spawn resumes continuation.

### 15. Greedy Join Scheduling

- All threads always attempt to steal if there is nothing to do
- Threads only go idle if there is **no work to steal** in the entire system
- Worker that initiated spawn may NOT be the thread that executes logic after `cilk_sync`

**Key design insight**: Overhead of bookkeeping steals and managing sync points only occurs when steals actually happen.

### 16. Parallel Programming Rules of Thumb

- Want at least as much work as parallel execution capability
- Want more independent work than execution capability for good load balance
- "Parallel slack" = ratio of independent work to machine's parallel execution capability (in practice: ~8 is good)
- But not too much: too much slack incurs overhead of managing fine-grained work

---

## Actionable Learning Points

1. **Start simple, measure first**: don't over-optimize prematurely.
2. **Static when predictable, dynamic when unpredictable**: choose assignment strategy based on workload knowledge.
3. **Task granularity is a tradeoff**: fine = good balance but high overhead; coarse = low overhead but potential imbalance.
4. **Work stealing is locality-aware**: continuation stealing ensures depth-first execution, good cache behavior.
5. **Distributed queues reduce contention**: one queue per worker, steal only when idle.
6. **Dequeue structure matters**: local LIFO, remote FIFO - maximizes locality and minimizes steals.
7. **Steal from head of queue**: takes largest piece of work, reducing total steal count.
8. **Parallel slack ~8**: practical rule for amount of over-decomposition.
9. **Overhead only paid when stealing occurs**: Cilk's design minimizes common-case cost.
10. **Greedy scheduling**: never wait at sync if there's work to steal elsewhere.

---

## C++ Source Files Reference

| Knowledge Point | C++ File |
|----------------|----------|
| Static vs dynamic assignment, task granularity | `../src/lecture5_part1.cpp` |
| Work stealing scheduler simulation with dequeues | `../src/lecture5_part2.cpp` |
| Fork-join parallelism, quicksort, Cilk-like simulation | `../src/lecture5_part3.cpp` |
