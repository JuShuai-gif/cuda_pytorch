# Lecture 17: Transactional Memory (Part I)

**PDF**: Lecture 17 - Transactional Memory
**Course**: Stanford CS149, Fall 2025

---

## Core Concepts Summary

### 1. Motivation: Raising Abstraction for Synchronization

| Approach | Abstraction Level | Programmer Effort |
|---|---|---|
| Atomic instructions (T&S, CAS, LL/SC) | Low (machine-level) | High - build locks/barriers |
| Locks (coarse/fine-grained) | Medium | Medium - manage lock order, deadlock risk |
| Lock-free data structures | Medium-High | High - CAS loops, ABA problem |
| **Transactional Memory** | **High (declarative)** | **Low - just declare `atomic { }`** |

### 2. Transaction Semantics
- **Atomicity**: All or nothing - on commit, all writes take effect at once; on abort, none appear
- **Isolation**: No other processor can observe writes before transaction commits
- **Serializability**: Transactions appear to commit in a single serial order
- Modeled after database transactions

### 3. Locks vs. Transactions - Code Comparison

```c
// Lock-based (imperative)
void deposit(Account a, int amount) {
    lock(a.lock);
    int tmp = bank.get(a);
    tmp += amount;
    bank.put(a, tmp);
    unlock(a.lock);
}

// Transaction-based (declarative)
void deposit(Account a, int amount) {
    atomic {
        int tmp = bank.get(a);
        tmp += amount;
        bank.put(a, tmp);
    }
}
```

### 4. Advantages of Transactional Memory

1. **Easy to use**: Declare atomicity, system implements it
2. **Performance**: Often as good as fine-grained locks (automatic read-read concurrency)
3. **Failure atomicity**: No lost locks when thread fails, abort + restart
4. **Composability**: `transfer(A, B, 100)` and `transfer(B, A, 200)` compose safely without global lock-ordering policies

### 5. `atomic { }` ≠ `lock() + unlock()`
- Atomic is a **declarative** high-level construct specifying what should be atomic
- Lock is a **low-level blocking primitive** that does not provide atomicity/isolation on its own
- Locks can implement atomic blocks, but also serve purposes beyond atomicity
- `atomic` eliminates data races but doesn't prevent atomicity violations (e.g., splitting logical atomic sequence into two `atomic` blocks)

### 6. TM Implementation Basics

Two key design decisions:

#### Data Versioning Policy
| Policy | Mechanism | Commit | Abort | Fault Tolerance |
|---|---|---|---|---|
| **Eager (undo-log)** | Write to memory immediately, keep undo log | Fast (data already in place) | Slow (must undo from log) | Poor (crash leaves partial writes) |
| **Lazy (write-buffer)** | Buffer writes, flush on commit | Slow (must flush buffer) | Fast (just clear buffer) | Good (no partial writes) |

#### Conflict Detection Policy
| Policy | When | Philosophy | Pros | Cons |
|---|---|---|---|---|
| **Pessimistic (eager)** | Check on each load/store | "Conflicts likely, detect early" | Early detection, some stalls instead of aborts | No forward progress guarantee, fine-grained overhead |
| **Optimistic (lazy)** | Check at commit time | "Hope for best, sort out at commit" | Forward progress guarantee, bulk detection | Late detection, can waste work |

### 7. Performance: Locks vs. Transactions
- TCC (hardware TM) matches or beats fine-grained locking on HashMap and Balanced Tree benchmarks
- Transactional approach provides automatic read-read concurrency without programmer effort

---

## Knowledge Points → C++ File Mapping

| Knowledge Point | C++ File |
|---|---|
| Lock vs Transaction: bank account transfer example | `lecture17_part1.cpp` |
| TM versioning: eager (undo-log) + lazy (write-buffer) simulation | `lecture17_part2.cpp` |
| TM conflict detection: pessimistic vs optimistic simulation | `lecture17_part3.cpp` |

---

## Actionable Learning Points

1. **Understand the semantic difference** between `atomic { }` and `lock()/unlock()` - they are NOT interchangeable
2. **Identify composability problems** with locks: `transfer(A,B)` + `transfer(B,A)` = deadlock without global lock ordering
3. **Know when optimistic detection beats pessimistic**: high contention → pessimistic (avoid wasted work); low contention → optimistic (avoid overhead)
4. **Recognize that TM is NOT a silver bullet**: still suffers from atomicity violations if programmer splits logical atomic operations
5. **Understand that TM can be implemented using locks**: the implementation choice is orthogonal to the programming model

---

## Key Design Space Matrix

|  | Eager Versioning (undo-log) | Lazy Versioning (write-buffer) |
|---|---|---|
| **Pessimistic Detection** | Intel STM, LogTM (HW) | MIT LTM, Intel VTM |
| **Optimistic Detection** | Intel McRT STM | Sun TL2, Stanford TCC |
