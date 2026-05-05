# Lecture 15: Memory Consistency

**PDF:** Lecture 15 — Memory Coherency and Consistency  
**Course:** Stanford CS149, Fall 2025 — Parallel Computing

---

## Core Concepts Summary

### 1. Coherence vs. Consistency (Critical Distinction)
| Concept | Scope | Question Answered |
|---------|-------|-------------------|
| **Coherence** | Single memory location | In what order do writes to X become visible? |
| **Consistency** | All memory locations | When do writes to X become visible *relative* to reads/writes to Y? |

- Coherence: all processors agree on the order of reads/writes to *the same address*
- Consistency: defines the allowed behavior of loads/stores to *different addresses*

### 2. Sequential Consistency (SC) — Lamport 1976
- All memory operations appear to execute in some **total sequential order**
- Each thread's operations appear in **program order**
- Maintains all four memory operation orderings:
  - W → R: write must commit before subsequent read
  - R → R: read must commit before subsequent read
  - R → W: read must commit before subsequent write
  - W → W: write must commit before subsequent write

**Switch metaphor:** Memory chooses a processor at random, performs one operation to completion, then chooses another.

### 3. Relaxed Consistency Models

| Model | Relaxes | Behavior |
|-------|---------|----------|
| **TSO** (Total Store Order) | W → R only | Processor can move its own reads ahead of its own writes (write buffer) |
| **PC** (Processor Consistency) | W → R only | *Any* processor can see a write before it's globally visible |
| **PSO** (Partial Store Order) | W → R, W → W | Writes can be reordered (e.g., flag visible before data) |
| **WO** (Weak Ordering) / **RC** (Release Consistency) | All four | Only ordering at synchronization points |

### 4. Write Buffers and TSO
- **Motivation**: Writes take 100s of cycles (cache coherence traffic); don't stall reads waiting for writes
- Write buffer allows processor to issue reads while writes drain
- Side effect: the classic Dekker-like pattern can produce `r1 = r2 = 0` (impossible under SC)
- **x86 uses a TSO-like model** — almost every modern processor

### 5. Memory Fences (Barriers)
- Prevent memory operation reordering at specified points
- x86 instructions:
  - `mfence`: all prior loads + stores complete before any subsequent load/store
  - `lfence`: all prior loads complete before any subsequent load
  - `sfence`: all prior stores complete before any subsequent store
- Expensive but necessary for correct synchronization

### 6. Data Races
- **Data race**: two accesses to the same location, at least one is a write, unordered by synchronization
- **Conflicting accesses**: same location + at least one write
- Unsynchronized programs containing data races have **non-deterministic** results
- **Happens-before graph**: used to reason about possible program outcomes; cycles = impossible outcomes

### 7. C++11 Memory Model: "SC for DRF"
- **Guarantee**: data-race-free programs behave as if executed under sequential consistency
- **No guarantees** for programs with data races
- **Implication**: use synchronization libraries (locks, barriers, atomics) — don't write ad-hoc shared variable accesses
- Most real-world programs ARE synchronized, so reordering is invisible to the programmer

### 8. Language-Level Memory Models
- Compilers can also reorder memory operations (optimizations visible to programmers in concurrent code)
- C++11, C11, Java 5+ all provide "SC for DRF" guarantees
- Compilers automatically insert necessary fences for the target hardware

---

## Knowledge Points → Corresponding C++ Files

| Knowledge Point | C++ File |
|-----------------|----------|
| SC vs TSO examples (Dekker's pattern) | `lecture15_part1.cpp` |
| C++11 atomics: relaxed, acquire-release, SC | `lecture15_part2.cpp` |
| Memory fences (mfence), data race detection | `lecture15_part3.cpp` |

---

## Actionable Learning Points
1. **Coherence ≠ Consistency** — coherence is per-address; consistency is cross-address
2. **SC is intuitive but expensive** — prevents nearly all hardware memory optimizations
3. **All real processors are relaxed** — x86 ≈ TSO, ARM is very relaxed
4. **Data-race-free programs are sequentially consistent** — use synchronization
5. **Write buffers explain why `r1=r2=0` is possible** even on x86
6. **Memory fences are the escape hatch** — use them sparingly, only when you understand why
7. **C++ `std::atomic` with `memory_order_seq_cst`** gives SC guarantees
8. **Happens-before analysis** is the tool for reasoning about concurrent program outcomes
