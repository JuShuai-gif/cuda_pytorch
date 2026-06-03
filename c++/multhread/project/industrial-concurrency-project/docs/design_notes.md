# Design Notes

## Why This Project Direction?

AI/ML inference scheduling is an ideal domain for demonstrating C++ concurrency
because it combines:

- **CPU-bound computation** (model inference, matrix operations)
- **I/O-bound operations** (data loading, result serialization)
- **Latency-sensitive tasks** (real-time inference requires priority scheduling)
- **Batch processing** (throughput optimization via batching)
- **Pipeline parallelism** (preprocessing -> inference -> post-processing)
- **Result caching** (avoid redundant computation)

This naturally exercises thread pools, priority queues, work stealing, shared
data structures, and graceful shutdown - essentially all core concepts from
"C++ Concurrency in Action."

---

## Knowledge Points by Module

### 1. `spinlock.hpp` - TTAS Spinlock (Ch5)

| Technique | Source |
|-----------|--------|
| `std::atomic_flag` (always lock-free) | Ch5.3.2 |
| `test_and_set()` with `memory_order_acquire` | Ch5.3.3 |
| `clear()` with `memory_order_release` | Ch5.3.3 |
| TTAS: Test before Test-And-Set | Ch5.3.4 (cache ping-pong) |
| Exponential backoff | Ch5.3.5 (contention management) |
| `_mm_pause()` / `yield` instruction | Ch5.3.5 (power efficiency) |
| `memory_order_relaxed` for polling | Ch5.3.1 (no synchronization needed) |
| RAII lock guard pattern | Ch3.2.1 (`std::lock_guard` design) |

### 2. `stop_token.hpp` - Cooperative Stop (Ch9.2)

| Technique | Source |
|-----------|--------|
| Stop source / stop token pattern | Ch9.2.1-9.2.2 |
| `std::atomic<bool>` for stop flag | Ch5.3.3 |
| `std::condition_variable` for blocking wait | Ch4.1.2 |
| `wait_for()` with timeout | Ch9.2.6 |
| StopRequestedException at interruption points | Ch9.2.7-9.2.8 |
| `std::shared_ptr` for shared state (ref counting) | Ch3.2.6 |
| Move-only stop_source, copyable stop_token | Ch9.2.3 |

### 3. `task_queue.hpp` - MPMC Queue (Ch6.2)

| Technique | Source |
|-----------|--------|
| `std::mutex` + `std::condition_variable` | Ch4.1.1 |
| `std::lock_guard` for exception safety | Ch3.2.3 |
| `std::unique_lock` for CV wait (deferred lock) | Ch3.2.6 |
| `condition_variable::wait()` with predicate | Ch4.1.2 (avoids spurious wake) |
| `wait_for()` with timeout | Ch4.1.2 |
| `notify_one()` outside lock | Ch4.1.1 (hurry-up-and-wait) |
| Thread-safe via single mutex | Ch6.2.1 (simple = correct) |
| `const` methods with `mutable` mutex | Ch3.2.8 |
| Move semantics for zero-copy transfer | Ch6.2.2 |
| Bulk pop operations | Ch6.2.5 |
| Deleted copy/move constructors | Ch6.2.1 (no data race on queue itself) |

### 4. `priority_task_queue.hpp` - Priority Queue (Ch6.3)

| Technique | Source |
|-----------|--------|
| `std::priority_queue` + mutex | Ch6.2 (extended pattern) |
| Monotonic sequence counter for FIFO | Ch6.2.7 (within priority band) |
| Single-mutex design for inversion avoidance | Ch3.2.7 (deadlock avoidance) |
| Priority enum for AI workloads | Ch6.3 (domain-specific design) |
| Batch pop by priority | Ch6.2.5 (bulk operations) |

### 5. `concurrent_cache.hpp` - LRU Cache (Ch3.3)

| Technique | Source |
|-----------|--------|
| `std::shared_mutex` (C++17) | Ch3.3.2 |
| `std::shared_lock` for reads | Ch3.3.2 (multiple concurrent readers) |
| `std::unique_lock` for writes | Ch3.3.1 (exclusive write) |
| LRU eviction policy with `std::list` | Ch6.2.7 (order maintenance) |
| `std::optional` for cache miss | Ch4.2.5 (monadic interface) |
| Read-heavy workload optimization | Ch3.3.2 (think readers-writer pattern) |

### 6. `thread_pool.hpp` + `thread_pool.cpp` - Thread Pool (Ch9.1)

| Technique | Source |
|-----------|--------|
| Fixed-size thread pool | Ch9.1.1 |
| `std::jthread` (C++20) | Ch2.1 + Ch9.2.3 |
| `submit()` returning `std::future` | Ch9.1.3 + Ch4.4.1 |
| `std::packaged_task` for callable wrapping | Ch4.4.1 |
| Type erasure via `std::function<void()>` | Ch4.4.2 |
| Per-thread local queues | Ch9.1.4 |
| Work stealing (random victim) | Ch8.4.2-8.4.4 |
| Global task queue fallback | Ch9.1.11 |
| `std::atomic<size_t>` for active task counter | Ch5.3.3 |
| `condition_variable` for worker wake-up | Ch4.1.2 |
| Exception catching in worker loop | Ch4.4.6 (future stores exception) |
| RAII shutdown in destructor | Ch2.1 (thread lifecycle) |
| `hardware_concurrency()` auto-sizing | Ch8.4.1 |

### 7. `task_scheduler.hpp` + `task_scheduler.cpp` - Core Scheduler (Ch8.5)

| Technique | Source |
|-----------|--------|
| Integration of pool + priority queue | Ch8.4 (composing concurrent structures) |
| Batch task submission | Ch8.2.2 (data decomposition) |
| Pipeline execution (continuation style) | Ch8.3.1-8.3.3 |
| `std::promise`/`std::future` for results | Ch4.2.1 |
| `std::shared_ptr<std::packaged_task>` | Ch4.4.3 (shared ownership) |
| Periodic task scheduling | Ch8.5.1 + Ch4.1.3 (timed wait) |
| Stop token for all subsystems | Ch9.2.1 |
| Round-robin distribution | Ch8.4.5 (load balancing) |
| Result caching integration | Ch3.3.2 |

### 8. `logger.hpp` - Thread-Safe Logger (Ch11)

| Technique | Source |
|-----------|--------|
| Singleton pattern (Meyer's) | Ch11.2 (thread-safe init in C++11+) |
| `std::atomic<LogLevel>` for fast check | Ch5.3.3 (no lock for common case) |
| Single mutex for output serialization | Ch3.2.1 |
| `std::format` (C++20) | Ch11.3 (modern string formatting) |
| `std::source_location` (C++20) | Ch11.3 (automatic caller info) |
| Timestamp + thread ID in output | Ch11.3 (debugging aid) |
| stderr for errors, stdout otherwise | Ch11.6 (proper stream routing) |

---

## Concurrency Design Challenges & Solutions

### Challenge 1: Work Stealing Contention

**Problem**: Multiple workers stealing from the same victim creates contention.
**Solution**: Random victim selection with `rand()` (Ch8.4.3). Each worker starts
at a random index and steals from the first non-empty queue found. This
distributes theft attempts evenly across all workers.

### Challenge 2: Priority Inversion

**Problem**: Low-priority task holds a lock needed by a high-priority task.
**Solution**: Single-mutex design (Ch3.2.7). The priority queue uses a single
`std::mutex` for both push and pop. This avoids nested locks and lock ordering
issues. All operations are O(log n) on the heap, so contention is bounded.

### Challenge 3: Graceful Shutdown

**Problem**: How to stop worker threads without losing in-flight tasks.
**Solution**: Two-phase stop (Ch9.2.3):
1. Request stop (atomic flag).
2. Workers check flag at safe points, drain remaining tasks, then exit.
3. `std::jthread` destructor joins automatically (RAII).

### Challenge 4: Exception Propagation

**Problem**: Tasks throw exceptions; workers must not crash.
**Solution**: `std::packaged_task` stores exceptions in the associated future
(Ch4.4.6). Worker catches all exceptions to prevent thread termination. The
caller retrieves the exception via `future::get()`.

### Challenge 5: Cache Line Contention (False Sharing)

**Problem**: Workers' atomic counters on same cache line cause ping-pong.
**Solution**: Each `Worker` struct has its `running` flag placed with other
frequently-used members. In production, `alignas(64)` would be added. Noted
as future optimization.

---

## Deadlock Avoidance Strategies

1. **Single Lock Per Component**: Each concurrent data structure uses exactly
   one mutex (Ch3.2.5 - avoid nested locks)
2. **Lock Outside Notify**: `condition_variable::notify_one()` called outside
   the mutex lock (Ch4.1.1 - avoid "hurry up and wait")
3. **No Lock Ordering**: Components never call into each other while holding
   their own lock (Ch3.2.5 - hierarchical locking avoided)
4. **RAII Lock Management**: `std::lock_guard`, `std::unique_lock`,
   `std::shared_lock` ensure locks are released on any code path (Ch3.2.3)
5. **Time-bounded Waits**: All blocking operations have timeout variants
   (Ch4.1.2) to prevent infinite blocking

---

## Performance Optimization Points

| Optimization | Technique | Benefit |
|-------------|-----------|---------|
| TTAS Spinlock | Read before test-and-set | Reduces cache coherence traffic (Ch5.3.4) |
| Bulk Queue Operations | `try_pop_bulk()` | Amortizes lock overhead (Ch6.2.5) |
| Atomic Fast-Path | Logger checks level before lock | Avoids lock for filtered messages (Ch11.3) |
| shared_mutex | Multiple readers in cache | Parallelizes read-heavy workloads (Ch3.3.2) |
| Local Queues | Per-worker task queues | Reduces global queue contention (Ch9.1.4) |
| Work Stealing | Random victim selection | Load balances without coordination (Ch8.4.3) |
| Move Semantics | `std::move` in queue | Avoids copies of large objects (Ch6.2.2) |

---

## Testing Strategy

### Unit Tests (Ch10.3)
- **Thread Pool**: Basic submit, concurrent submits, work stealing, shutdown,
  exception propagation, wait_for_tasks
- **Task Queue**: Push/pop basic, MPSC, MPMC try_pop, timeout wait, empty/bulk
- **Task Scheduler**: Priority execution, batch submission, mixed priorities,
  pipeline, periodic tasks, cache integration

### Stress Tests (Ch10.4)
- High-contention thread pool (5000 tasks, 8 threads)
- MPMC queue with multiple producers and consumers (10000 items)
- Priority queue under concurrent access (2000 items, 4 producers)
- Concurrent cache with mixed read/write load (50000 operations)
- Multi-component integration test (scheduler + cache simultaneously)
- Spinlock stress test (100000 increments, 4 threads)

### Tool Verification
- **ThreadSanitizer (TSan)**: `-fsanitize=thread` build mode detects data races
- **AddressSanitizer (ASan)**: Detects use-after-free, buffer overflows
- **Valgrind Helgrind**: Detects lock ordering violations and potential deadlocks

---

## Style Conventions

- C++20 standard throughout
- No raw `new`/`delete` - `std::make_unique`/`std::make_shared` only
- RAII for all resource management
- English comments explaining the "why" not the "what"
- Each comment references the relevant book chapter (ChX.Y)
- `const` correctness on all methods
- `[[nodiscard]]` on pure queries
- `noexcept` on guaranteed-no-throw functions
