# Lecture 16: Fine-Grained Synchronization & Lock-Free Programming

**PDF**: Lecture 16 - Implementing Locks, Fine-Grained Synchronization, and Lock-Free Programming
**Course**: Stanford CS149, Fall 2025

---

## Core Concepts Summary

### 1. Concurrency Terminology
- **Deadlock**: System has outstanding operations but none can make progress (mutual exclusion + hold-and-wait + no preemption + circular wait)
- **Livelock**: System is executing many operations, but no thread makes meaningful progress (e.g., transactions continually abort and retry)
- **Starvation**: System makes overall progress, but some processes make no progress (fairness issue, not correctness)

### 2. Lock Implementations

| Lock Type | Latency (low contention) | Interconnect Traffic | Scalability | Fairness |
|---|---|---|---|---|
| Test-and-set | Low | High (O(P²) invalidations) | Poor | None |
| Test-and-test-and-set | Slightly higher | Lower (O(P) per release) | Better | None |
| Ticket lock | Moderate | Lowest (1 invalidation/release) | Good | Yes (FIFO) |
| CAS-based lock | Low (shared read) | Moderate | Good | None |

### 3. Atomic Primitives
- **Test-and-set (T&S)**: atomically set flag to 1 and return old value
- **Compare-and-swap (CAS)**: `cmpxchg` - compare and exchange if equal
- **Load-linked / Store-conditional (LL/SC)**: two-instruction pair (ARM LDREX/STREX)
- **C++11 `atomic<T>`**: provides atomic read/write/read-modify-write with memory ordering semantics
- **CUDA atomic operations**: `atomicAdd`, `atomicCAS`, `atomicMin`, `atomicMax`, etc.

### 4. Fine-Grained Locking: Hand-over-Hand Traversal
- Strategy: lock current node, then lock next node, then release previous node's lock
- **Good**: enables parallelism on different parts of data structure
- **Bad**: overhead of taking lock per step; extra storage cost (lock per node); trickier correctness
- **Deadlock-free guarantee**: locks acquired in a fixed order (list traversal direction), so no circular wait

### 5. Lock-Free Data Structures
- **Blocking**: one thread can prevent others from completing indefinitely (e.g., lock holder swapped out)
- **Lock-free**: some thread is guaranteed to make systemwide progress
- Key mechanism: CAS-based retry loop (speculative update, retry if conflict)
- **ABA Problem**: a location changes from A→B→A, CAS succeeds but data structure is corrupted
- **Solutions**: double-wide CAS, counter (pop_count), hazard pointers for memory reclamation

### 6. Lock-Free Examples Covered
- **Single reader/writer bounded queue**: no locks, circular buffer, head/tail with modular arithmetic
- **Single reader/writer unbounded queue**: linked-list based, producer thread handles memory reclaim
- **Lock-free stack**: CAS on top pointer, ABA problem with counter, hazard pointers for safe deletion
- **Lock-free linked list insertion**: CAS on `prev->next`, simpler than deletion

---

## Knowledge Points → C++ File Mapping

| Knowledge Point | C++ File |
|---|---|
| Lock implementations (T&S, T&T&S, Ticket, CAS) | `lecture16_part1.cpp` |
| Fine-grained linked list with hand-over-hand locking | `lecture16_part2.cpp` |
| Lock-free stack with CAS + ABA problem | `lecture16_part3.cpp` |
| Lock-free bounded/unbounded queues | `lecture16_part4.cpp` |

---

## Actionable Learning Points

1. **Understand the four deadlock conditions** and verify that hand-over-hand locking satisfies none of them
2. **Implement ticket lock** - it's the simplest fair lock with minimal coherence traffic
3. **Build atomic fetch-and-op from CAS** - a classic interview question
4. **Recognize the ABA problem** in lock-free code and know the counter-based solution
5. **Understand when to use lock-free vs. fine-grained locking**: lock-free for OS/database threads (preemption-safe), fine-grained locks for HPC (simpler, fast when no preemption)

---

## Key Code Patterns

### Ticket Lock
```c
struct lock { int next_ticket; int now_serving; };
void Lock(lock* l) {
    int my_ticket = atomic_increment(&l->next_ticket);
    while (my_ticket != l->now_serving);
}
void unlock(lock* l) { l->now_serving++; }
```

### CAS-based Lock (optimized for contention)
```c
void lock(Lock* l) {
    while (1) {
        while (*l == 1);           // spin on read (low traffic)
        if (atomicCAS(l, 0, 1) == 0) return;  // try to acquire
    }
}
```

### Lock-Free Stack Push
```c
void push(Stack* s, Node* n) {
    while (1) {
        Node* old_top = s->top;
        n->next = old_top;
        if (compare_and_swap(&s->top, old_top, n) == old_top)
            return;
    }
}
```
