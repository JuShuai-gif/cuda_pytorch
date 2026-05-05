# CS149 Lecture 1: Why Parallelism? Why Efficiency?

**PDF**: `01_efficiency_hyF1AJq.pdf`

**Instructors**: Prof. Kayvon Fatahalian, Prof. Kunle Olukotun

**University**: Stanford CS149, Fall 2025

---

## Core Concepts Summary

### 1. Speedup & Parallelism Fundamentals

- **Speedup formula**: `speedup(P) = T(1) / T(P)` (execution time with 1 processor / execution time with P processors)
- **Communication overhead** limits real speedup (partial sums must be communicated between processors)
- **Work imbalance** limits speedup (idle processors waiting for others to finish)
- **Amdahl's Law**: $S(P) = \frac{1}{(1 - f_{par}) + f_{par} / P}$, where $f_{par}$ is the parallelizable fraction
  - Even with infinite processors, speedup is bounded by $1/(1 - f_{par})$

> **C++ Demo**: `lecture1_part1.cpp` — Parallel sum with timing, Amdahl's Law table, balanced vs unbalanced work distribution

### 2. What Is a Program? (Processor's Perspective)

- A program is a **list of processor instructions** (compiled from C/C++ source)
- Instructions modify processor state: registers and memory
- Simple processor: Fetch/Decode → ALU (Execute) → Registers → Memory
- Each instruction: load data, compute, store result

> **C++ Demo**: `lecture1_part2.cpp` — Instruction dependency graph, superscalar scheduling simulation

### 3. Instruction-Level Parallelism (ILP) & Superscalar Execution

- **Superscalar processor**: automatically finds independent instructions and executes them in parallel on multiple execution units
- Example: `a = x*x + y*y + z*z` → 3 independent multiplies can execute simultaneously
- **ILP is limited** by instruction dependencies (critical path length)
- **Diminishing returns**: most available ILP exploited by ~4-wide superscalar (beyond 4-issue width, negligible benefit)

> **C++ Demo**: `lecture1_part2.cpp` — Schedule 5-instruction program on 1/2/3/4-wide superscalar, compute IPC

### 4. The Power Wall & End of Frequency Scaling

- **Power ∝ Capacitance × Voltage² × Frequency**
- Increasing frequency requires increasing voltage → quadratic power growth
- **Dynamic power** (switching) + **Static power** (leakage) both increase with transistor count
- ~2005: clock frequency stopped increasing (power/thermal limits)
- **ILP tapped out** + **frequency scaling ended** → shift to multi-core architectures
- "No more free lunch for software developers" — must write parallel code to see performance gains

> **C++ Demo**: `lecture1_part2.cpp` (Part 3) — Power scaling calculation for different frequencies

### 5. Multi-Core CPUs, GPUs & Specialized Hardware

- **Multi-core CPUs**: use transistor budget to add more cores instead of more sophisticated single-core logic
  - Intel i9-10900K: 10 cores
  - AMD Threadripper 3990X: 64 cores
- **GPUs**: massive parallelism (NVIDIA RTX 4090: 18,432 fp32 multipliers)
- **Specialized hardware**: Apple Neural Engine, Google TPU, etc. for domain-specific efficiency
- **Mobile**: power constraints → heterogeneous designs (big.LITTLE: 2 big + 4 small cores)

### 6. Memory Hierarchy

- **Memory** = byte-addressable array
- **Load instruction**: `ld R0 ← mem[R2]` — access memory at address in R2, store in R0
- **Memory access latency**: DRAM access ~100s of cycles → causes processor stalls
- **Cache hierarchy**: L1 (32KB, ~4 cycles) → L2 (256KB, ~12 cycles) → L3 (8-20MB, ~38 cycles) → DRAM (~248 cycles)
- **Cache**: on-chip copy of a subset of memory, operates at cache line granularity
- **LRU replacement policy**: evict least recently used line when cache is full

> **C++ Demo**: `lecture1_part3.cpp` — LRU cache simulator, temporal/spatial locality

### 7. Cache Locality & Performance

- **Temporal locality**: repeated accesses to the same memory address → cache hits
- **Spatial locality**: loading a cache line preloads nearby addresses → subsequent hits
- **Miss types**:
  - **Cold miss**: first access to data (cache is initially empty)
  - **Capacity miss**: working set exceeds cache size (eviction occurs)
  - **Conflict miss**: set-associativity issues (not covered in detail)
- **Caches reduce memory access latency AND provide higher bandwidth**

### 8. Data Movement Energy Costs

| Operation | Energy (ballpark) |
|-----------|-------------------|
| Integer op | ~1 pJ |
| Floating point op | ~20 pJ |
| Read 64b from on-chip SRAM (1mm away) | ~26 pJ |
| Read 64b from mobile DRAM (LPDDR) | ~1200 pJ |

- **Implication**: reading from DRAM costs ~1200x more than an integer operation
- Reading 10 GB/sec from memory: ~1.6 watts
- Mobile GPU power budget: ~1 watt total
- **Exploiting locality matters** — it's not just about performance, it's about battery life

---

## Actionable Learning Points

1. **Always measure speedup, not just speed**: 2x speedup on 10 processors = 20% efficiency
2. **Think about Amdahl's Law**: identify and minimize serial bottlenecks
3. **Write code with cache locality in mind**: sequential access > strided > random
4. **Understand the memory hierarchy**: L1 is fast but tiny; DRAM is huge but slow
5. **Consider power**: efficient programs extend battery life on mobile devices

---

## C++ Source Files

| File | Topic | Key Demonstration |
|------|-------|-------------------|
| `lecture1_part1.cpp` | Speedup & Amdahl's Law | Parallel sum timing, work imbalance, Amdahl table |
| `lecture1_part2.cpp` | ILP & Superscalar | Dependency graph, issue width scheduling, power wall |
| `lecture1_part3.cpp` | Cache Simulation | LRU cache, temporal/spatial locality, energy costs |

### Compilation

```bash
g++ -std=c++17 -O2 -pthread lecture1_part1.cpp -o lecture1_part1 && ./lecture1_part1
g++ -std=c++17 -O2 lecture1_part2.cpp -o lecture1_part2 && ./lecture1_part2
g++ -std=c++17 -O2 lecture1_part3.cpp -o lecture1_part3 && ./lecture1_part3
```

---

## Course Meta-Information

- 5 programming assignments (56% of grade): ISPC, task scheduling, CUDA, DNN, CUDA optimization
- 4 written assignments in teams of 3 (12%)
- Per-lecture participation (4%)
- Midterm + Final exam (28%)
- 8 late days for programming assignments 1-4
- No textbook — use course website and internet resources
