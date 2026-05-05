# CS149 Lecture 3: Multi-Core Architecture Part II + ISPC Parallel Abstractions

**PDF**: `03_multicore2-ispc_WueDBzT.pdf`

**University**: Stanford CS149, Fall 2025

---

## Core Concepts Summary

### 1. Throughput Computing Hardware (Review)

Three key ideas for throughput-oriented hardware:
- **Multi-core execution**: multiple independent cores on one chip
- **SIMD execution**: single instruction operates on multiple data elements
- **Hardware multi-threading**: interleaving multiple threads on a core to hide latency

### 2. Latency vs. Bandwidth

- **Latency**: time to complete a single task (e.g., 0.5 hours driving SF→Stanford)
- **Bandwidth** (Throughput): rate of completing tasks (e.g., 2 cars/hour)
- **Improving throughput**:
  - Speed up each unit (drive faster → 200 km/hr → 4 cars/hr)
  - Add more resources (build more lanes → 8 cars/hr with 4 lanes)
  - Use resources more efficiently (pack cars tighter → 400 cars/hr)
- **Memory bandwidth**: rate at which memory system provides data (e.g., 20 GB/s)
- **Memory latency**: time to retrieve one item from memory (~100s of cycles)

> **C++ Demo**: `lecture3_part1.cpp` — Latency vs bandwidth simulation (car pipeline, laundry pipeline, memory bandwidth-bound computation)

### 3. The Bandwidth Wall

- **Element-wise vector multiplication**: 3 memory ops (12 bytes) per MUL
- NVIDIA V100: 5120 fp32 ALUs @ 1.6 GHz → needs ~98 TB/sec bandwidth to keep ALUs busy
- Actually has only 900 GB/s → <1% ALU utilization
- **Key insight**: modern workloads are often bandwidth-limited, not compute-limited
- **Solution**: organize computation to fetch data less often (temporal locality), share data across threads

> **C++ Demo**: `lecture3_part1.cpp` — Bandwidth-bound computation simulation with utilization analysis

### 4. Instruction Pipelining

- **4-stage pipeline**: IF (Instruction Fetch) → D (Decode) → EX (Execute) → WB (Write Back)
- Pipelining increases **throughput** (1 instruction/clock) while **latency** remains 4 cycles
- Deeper pipelines: ~20 stages in modern CPUs
- **Key distinction**: IPC (Instructions Per Clock) = throughput, NOT latency

### 5. Abstraction vs. Implementation

- **Semantics (Abstraction)**: what operations mean; what answer a program computes
- **Implementation (Scheduling)**: how answer is computed on parallel hardware
- **Goal**: trace through what each part of the parallel computer is doing during each step

### 6. ISPC (Intel SPMD Program Compiler)

**SPMD**: Single Program, Multiple Data — define one function, run multiple instances in parallel.

> **C++ Demo**: `lecture3_part2.cpp` — ISPC-style SPMD simulation, foreach abstraction, interleaved vs blocked assignment

#### 6.1 ISPC Keywords

```c
export void ispc_sinx(
    uniform int N,        // same value for all instances
    uniform int terms,
    uniform float* x,
    uniform float* result)
{
    for (uniform int i=0; i<N; i+=programCount)
    {
        int idx = i + programIndex;  // unique per instance
        float value = x[idx];        // "varying" - different per instance
        // ...
    }
}
```

- **`programCount`**: number of simultaneously executing instances in a gang (uniform)
- **`programIndex`**: ID of current instance (0 to programCount-1) (varying)
- **`uniform`**: type modifier — all instances have the same value (optimization, not for correctness)
- **`varying`**: default type — each instance has its own copy

#### 6.2 Gang Execution Model

```
main() [sequential C code]
  ↓ call ispc_sinx()
  ispc_sinx(): [0][1][2][3][4][5][6][7]  ← 8 program instances run concurrently
  ↓ return
main() [sequential C code resumes]
```

- Call to ISPC function spawns a "gang" of program instances
- All instances run ISPC code concurrently
- Each instance has its own copy of local variables
- Upon return, all instances have completed

#### 6.3 Interleaved vs. Blocked Assignment

**Interleaved** (stride = programCount):
```c
for (uniform int i=0; i<N; i+=programCount) {
    int idx = i + programIndex;
    // instance 0: indices 0, 8, 16, ...
    // instance 1: indices 1, 9, 17, ...
}
```
Advantage: contiguous memory access → efficient **packed vector load** (`vmovaps`)

**Blocked** (contiguous chunks):
```c
uniform int count = N / programCount;
int start = programIndex * count;
for (uniform int i=0; i<count; i++) {
    int idx = start + i;
    // instance 0: indices 0, 1, 2, ...
    // instance 1: indices 8, 9, 10, ...
}
```
Requires **gather instruction** (`vgatherdps`) — non-contiguous memory access, more costly

#### 6.4 The `foreach` Abstraction

```c
export void ispc_function(uniform int N, uniform float* x, uniform float* y) {
    foreach (i = 0 ... N) {
        float val = x[i];
        float result = /* compute from val */;
        y[i] = result;
    }
}
```

- Declares parallel loop iterations — programmer writes sequential-like code
- ISPC runtime assigns iterations to program instances
- Possible implementations: sequential, interleaved, blocked, dynamic (work stealing)

#### 6.5 Cross-Instance Operations

> **C++ Demo**: `lecture3_part3.cpp` — Cross-instance operations simulation (reduce_add, broadcast, rotate, parallel reduction tree)

| Operation | Description |
|---|---|
| `reduce_add(x)` | Sum of x across all instances in gang |
| `reduce_min(x)` | Minimum of x across all instances |
| `broadcast(x, idx)` | Send value from instance idx to all |
| `rotate(x, offset)` | Pass x to instance (i + offset) % programCount |

**Example**: Parallel reduction (8 elements in log₂(8) = 3 steps):
```c
float val1 = x[programIndex];
float val2 = shift(val1, 1);
if (programIndex % 2 == 0) val1 = val1 * val2;
val2 = shift(val1, 2);
if (programIndex % 4 == 0) val1 = val1 * val2;
val2 = shift(val1, 4);
if (programIndex % 8 == 0) *result = val1 * val2;
```

### 7. ISPC Tasks (Multi-core)

- **Gang abstraction**: implemented by SIMD instructions on **one** CPU core
- **Task abstraction**: used for multi-core execution (multiple gangs on multiple cores)
- Covered in Assignment 1

### 8. From Low-Level ISPC to High-Level Abstractions

**Level 1 (ISPC with programIndex)**: explicit instance ID, manual assignment
**Level 2 (foreach)**: declare parallel iterations, compiler assigns work
**Level 3 (no array indexing)**: `map(doWork, collection)` — purely functional data-parallel
**Level 4 (NumPy-style)**: `Z = X + Y` — operations on whole collections

---

## Actionable Learning Points

| # | Concept | C++ File |
|---|---|---|
| 1 | Latency vs bandwidth with car/laundry analogies | `lecture3_part1.cpp` |
| 2 | Bandwidth-bound computation & ALU utilization | `lecture3_part1.cpp` |
| 3 | SPMD programming model & ISPC gang simulation | `lecture3_part2.cpp` |
| 4 | `foreach` abstraction, interleaved vs blocked assignment | `lecture3_part2.cpp` |
| 5 | Cross-instance reduce/broadcast/rotate operations | `lecture3_part3.cpp` |
| 6 | Parallel reduction tree (log₂N steps) | `lecture3_part3.cpp` |
| 7 | SIMD gather vs packed load performance | `lecture3_part2.cpp` |

---

## Key Takeaways

1. **Bandwidth is the critical resource** in modern throughput-optimized systems — organize computation to fetch data less often
2. **Abstraction vs. implementation**: always think about what a program *means* vs. how it *executes*
3. **ISPC SPMD model**: think of multiple program instances running same code on different data
4. **`foreach` is preferred** for simple data-parallel operations (hides assignment details)
5. **Cross-instance operations** enable communication within a gang without breaking the SPMD model
6. **Interleaved assignment** enables efficient SIMD packed loads; blocked assignment may need costly gathers
