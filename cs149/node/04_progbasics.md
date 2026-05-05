# Lecture 4: Parallelizing Code - The Programming Thought Process

**Source**: Stanford CS149, Fall 2025 - Lecture 4 PDF

---

## Core Concepts Summary

### 1. ISPC Semantics Deeper Dive

**SPMD Programming Model** (Single Program, Multiple Data):
- Call to ISPC function spawns a "gang" of ISPC "program instances"
- All instances run ISPC code concurrently
- Each instance has its own copy of local variables (varying)
- Upon return, all instances have completed

**Key ISPC Keywords:**
- `programCount`: number of simultaneously executing instances in the gang (uniform value, e.g., 8)
- `programIndex`: id of the current instance in the gang (0..programCount-1, varying)
- `uniform`: a type modifier - all instances have the same value; purely an optimization
- `varying`: each instance has its own copy (default for local variables)

**Abstraction vs. Implementation:**
- **Programming abstraction**: SPMD - programmer thinks of spawning `programCount` logical instruction streams
- **Implementation**: SIMD - ISPC compiler emits vector instructions (AVX2, ARM NEON) that carry out gang logic
- ISPC handles mapping of conditional control flow to vector instructions (masking vector lanes)

### 2. Interleaved vs Blocked Assignment

**Interleaved Assignment**:
- `idx = i + programIndex` (i increments by programCount)
- Consecutive elements processed by different instances in the same iteration
- Load pattern: contiguous memory → efficient packed vector load (`vmovaps`)

**Blocked Assignment**:
- `start = programIndex * count; idx = start + i`
- Each instance processes a contiguous block
- Load pattern: non-contiguous → requires gather instruction (`vgatherdps`), more costly

### 3. foreach Abstraction

- Declares parallel loop iterations: `foreach (i = 0 ... N)`
- Programmer says "these are the iterations the entire gang must perform"
- ISPC runtime handles assignment of iterations to program instances
- In simple cases, allows expressing program almost as sequential code
- Possible implementations: interleave, block, dynamic assignment via atomic counter

### 4. Cross-Instance Operations

```c
uniform int64 reduce_add(int32 x);     // sum across all instances
uniform int32 reduce_min(int32 a);     // min across all instances
int32 broadcast(int32 value, uniform int index);  // broadcast from one instance
int32 rotate(int32 value, uniform int offset);    // rotate values between instances
```

- `reduce_add` is critical for parallel reductions (e.g., summing array elements)
- Each instance accumulates private partial sum, then `reduce_add` combines them

### 5. ISPC Tasks (Multi-Core)

- Gang abstraction uses SIMD on one core
- ISPC "tasks" achieve multi-core execution
- `launch[N] my_task(...)` creates N tasks
- ISPC runtime assigns tasks to worker threads in a thread pool

### 6. Amdahl's Law

$$
\text{speedup} \leq \frac{1}{S}
$$

Where S = fraction of execution that is inherently sequential.

**Key insight**: A small serial region can severely limit speedup on a large parallel machine.
- S = 0.01 (1% serial) → max speedup = 100x
- S = 0.001 (0.1% serial) on 148M ALU machine → max speedup ≈ 1000x

### 7. Creating a Parallel Program: Decomposition → Assignment → Orchestration

```
Problem → Decomposition → Subproblems (tasks)
Subproblems → Assignment → Parallel Threads (workers)
Parallel Threads → Orchestration → Parallel Program (communicating threads)
Parallel Program → Mapping → Execution on parallel machine
```

**Decomposition**: Break up problem into independent tasks; identify dependencies.

**Assignment**: Assign tasks to workers.
- Static: determined before execution (simple, zero runtime overhead)
- Dynamic: determined at runtime (handles unpredictable workloads)
- ISPC `foreach`: system-managed assignment

**Orchestration**: Structure communication, add synchronization, organize data, schedule tasks.

### 8. Case Study: 2D Grid Solver (Gauss-Seidel)

**Problem**: Solve PDE on (N+2) × (N+2) grid iteratively:
```
A[i,j] = 0.2 * (A[i,j] + A[i,j-1] + A[i-1,j] + A[i,j+1] + A[i+1,j])
```

**Dependencies**: Each row element depends on left neighbor; each row depends on previous row.

**Red-Black Coloring** (parallelism-enabling algorithm change):
- Update all RED cells in parallel
- When done, update all BLACK cells in parallel (respecting dependency on red cells)
- Repeat until convergence

### 9. Two Programming Models for Grid Solver

**Data-Parallel Expression**:
- `for_all` over red/black cells
- Synchronization: implicit barrier at end of for_all block
- Communication: implicit in loads/stores
- Built-in primitives: `reduceAdd`

**Shared Address Space (SPMD) Expression**:
- Multiple threads, shared variables
- Synchronization: locks (mutual exclusion), barriers (phase dependencies)
- Communication: implicit in reads/writes to shared address space
- Programmer manages synchronization explicitly

### 10. Synchronization Primitives

- **Locks**: mutual exclusion - only one thread in critical section at a time
- **Barriers**: `barrier(num_threads)` - all threads must reach barrier before any proceeds
- **Atomic operations**: `atomicAdd`, hardware-supported read-modify-write

### 11. Performance Optimization in Shared Address Space Solver

- Accumulate partial sum locally per worker, then do global reduction once per iteration
- Lock only once per thread per iteration (not once per grid cell!)
- Remove unnecessary barriers by using multiple diff variables (space-for-dependencies tradeoff)

---

## Actionable Learning Points

1. **Think in terms of abstraction vs. implementation**: SPMD is the abstraction; SIMD is one possible implementation.
2. **Data layout matters for SIMD**: interleaved assignment enables contiguous memory access (packed loads); blocked assignment requires gathers.
3. **Use foreach when possible**: let the system manage assignment; focus on expressing parallelism.
4. **Amdahl's Law is unforgiving**: minimize serial portions aggressively.
5. **Decomposition first, then assignment, then orchestration**: follow the structured thought process.
6. **Algorithm changes can unlock parallelism**: red-black coloring is an example of changing algorithm to expose more parallelism.
7. **Lock granularity matters**: accumulate locally, then lock once for global update.
8. **Space-for-dependencies tradeoff**: use extra storage to remove synchronization dependencies.

---

## C++ Source Files Reference

| Knowledge Point | C++ File |
|----------------|----------|
| SPMD abstraction, interleaved vs blocked assignment, foreach, reduce_add | `../src/lecture4_part1.cpp` |
| Amdahl's Law simulation and speedup calculation | `../src/lecture4_part2.cpp` |
| Grid solver: data-parallel with red-black coloring | `../src/lecture4_part3.cpp` |
| Grid solver: shared address space SPMD with barriers/locks | `../src/lecture4_part4.cpp` |
