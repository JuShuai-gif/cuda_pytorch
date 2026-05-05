# Lecture 6: Performance Optimization Part II - Locality, Communication, and Contention

**Source**: Stanford CS149, Fall 2025 - Lecture 6 PDF

---

## Core Concepts Summary

### 1. The Reality of Shared Address Space Hardware

The abstraction of a single shared address space is implemented by a complex hierarchy:
```
L1 cache (32 KB) → L2 cache (256 KB) → L3 cache (20 MB) → DRAM (32 GB)
```
Each core has its own L1/L2; L3 is shared.

### 2. Hardware Interconnects

**Intel Ring Interconnect** (Sandy Bridge+):
- Four rings for different message types: request, snoop, ack, data (32 bytes)
- Six interconnect nodes: four L3 cache slices + system agent + graphics
- Each L3 bank connected to ring bus twice
- Peak BW ~435 GB/sec at 3.4 GHz (when each core accesses its local slice)

**SUN Niagara 2 (UltraSPARC T2)**: Crossbar interconnect
- All cores connected directly to all others
- Crossbar area ≈ area of one core

### 3. NUMA (Non-Uniform Memory Access)

**Definition**: Latency of accessing a memory location differs from different processing cores.

**Example**: Modern multi-socket systems:
- Each socket has its own memory controller and local memory
- Accessing remote memory (other socket) has higher latency/lower bandwidth
- NUMA behavior even on single-socket systems (different cache slices at different distances)

**Implication**: Shared address space model requires reasoning about locality for performance.

### 4. Message Passing Model (Alternative to Shared Address Space)

**Abstraction**:
- Threads operate within their own **private** address spaces
- Communication only via explicit `send()` and `recv()` messages
- `send(X, recipient, msg_id)`: send contents of local variable X to thread recipient
- `recv(Y, sender, msg_id)`: receive message from sender into local variable Y

**Implementation**:
- Hardware need not implement a single shared address space
- Can connect commodity systems via network (Infiniband)
- Programming model for clusters and supercomputers

**Grid solver in message passing**:
- Each thread has its own private array (partition of grid)
- "Ghost cells": grid cells replicated from remote address space
- Threads send/receive ghost rows to/from neighbors before computation

**Synchronous send/recv**:
- `send()`: returns when acknowledgement received (data in receiver's address space)
- `recv()`: returns when data copied and ack sent

**Deadlock problem**: If all threads try to send first → deadlock. Solution: even threads send then recv; odd threads recv then send.

**Non-blocking asynchronous send/recv**:
- `send()`: returns immediately (buffer cannot be modified until send complete)
- `recv()`: posts intent, returns immediately
- `checksend()`, `checkrecv()`: poll for completion

### 5. Extended Memory Hierarchy - "Communication" Generalized

Think of "communication" at ALL levels:
- Processor ↔ its cache
- Processor ↔ memory (same machine)
- Processor ↔ remote memory (other node in cluster)

```
Reg → Local L1 → Local L2 → L2 from another core → L3 → Local memory → Remote memory (1 hop) → Remote memory (N hops)
Lower latency, higher BW, smaller capacity                                                     Higher latency, lower BW, larger capacity
```

### 6. Arithmetic Intensity

$$
\text{Arithmetic Intensity} = \frac{\text{amount of computation (e.g., instructions)}}{\text{amount of communication (e.g., bytes)}}
$$

- 1 / Arithmetic Intensity = communication-to-computation ratio
- High arithmetic intensity = low communication-to-computation ratio (desirable)
- Required to efficiently utilize modern parallel processors

### 7. Inherent vs. Artifactual Communication

**Inherent communication**: Communication that fundamentally MUST occur given the algorithm and assignment.
- Example: sending ghost rows in message-passing grid solver

**Artifactual communication**: All other communication resulting from practical implementation details.
- Example: loading entire cache line when only one float is needed (minimum granularity)
- Example: capacity misses (cache too small to retain data between accesses)
- Example: unnecessary loads (loading cache line only to overwrite it entirely)

### 8. Reducing Inherent Communication via Assignment

**Grid solver example**:

| Assignment | Elements computed/proc | Elements communicated/proc | Arithmetic Intensity |
|------------|----------------------|---------------------------|---------------------|
| 1D blocked | N²/P | 2N | N/(2P) |
| 1D interleaved | N²/P | ~N²/2 | 2 |
| 2D blocked | N²/P | ∝ N/√P | N/√P |

**2D blocked assignment captures 2D locality**: communication costs increase sub-linearly with P.

### 9. Artifactual Communication from Cache Behavior

**Row-major grid traversal problem**:
- Cache line = 4 grid elements, cache capacity = 24 elements (6 lines)
- By the time we return to access elements from previous rows, they've been evicted
- Result: 3 cache line loads for every 4 output elements (instead of 1)

**Solution: Blocked iteration**:
- Process grid in blocks that fit in cache
- Now: 2 cache line loads for every 6 output elements

### 10. Loop Fusion

**Before** (separate loops):
```cpp
void add(float* A, float* B, float* C, int n);
void mul(float* A, float* B, float* C, int n);
// E = D + ((A + B) * C) → 3 separate loops
// Arithmetic intensity = 1/3
```

**After** (fused loop):
```cpp
void fused(float* A, float* B, float* C, float* D, float* E, int n) {
    for (int i = 0; i < n; i++)
        E[i] = D[i] + (A[i] + B[i]) * C[i];
}
// Arithmetic intensity = 3/5 (4 loads, 1 store per 3 math ops)
```

**Tradeoff**: Modularity vs. performance. NumPy-style array operations are modular but lower arithmetic intensity.

### 11. Contention

**Definition**: Many requests to a resource within a small window of time → "hot spot".

**Examples**:
- Multiple threads updating a shared variable
- Flat communication (high contention, low latency without contention)
- Tree-structured communication (reduces contention, higher latency without contention)

**Solutions**:
- Replicate contended resources (local copies, fine-grained locks, distributed work queues)
- Stagger access to contended resources

### 12. False Sharing

When multiple threads modify different variables that happen to reside on the same cache line, the cache coherence protocol causes unnecessary invalidations.

### 13. Performance Analysis Strategy

**Three questions to diagnose bottlenecks**:
1. Is performance limited by **computation**?
2. Is performance limited by **memory bandwidth** (or memory latency)?
3. Is performance limited by **synchronization**?

### 14. Roofline Model

```
        |  Compute-limited region (horizontal)
GFLOP/s |  ................
        | / Memory BW-limited region (diagonal)
        |/
        +-------------------------------
              Arithmetic Intensity
```

- Diagonal region: memory bandwidth limited execution
- Horizontal region: compute limited execution
- Each point = a program with different arithmetic intensity
- Maximum obtainable throughput for given arithmetic intensity

### 15. Establishing High Watermarks

**Techniques to identify bottlenecks**:
1. **Add math**: If execution time increases linearly with operation count → compute-limited
2. **Remove math but keep loads**: If execution time doesn't decrease much → memory bottleneck
3. **Change all accesses to `A[0]`**: Upper bound on locality improvement benefit
4. **Remove all atomics/locks**: Upper bound on sync overhead reduction benefit

### 16. Performance Monitoring Tools

- Intel Performance Counter Monitor (PCM)
- Intel VTune
- PAPI (Performance API)
- oprofile

Modern processors have performance counters: instructions completed, clock ticks, cache hits/misses, bytes read from memory controller, etc.

### 17. Scaling Issues

**Fixed problem size speedup pitfalls**:
- Too small: parallelism overheads dominate (may even slow down)
- Too large for small machine: working set may not fit in cache/memory (thrashing)
- Super-linear speedup: working set fits in cache on large machine but not on single core

**Weak scaling vs. strong scaling**:
- Scale problem size with machine size (buy bigger machine to compute more, not just faster)

---

## Actionable Learning Points

1. **Communication happens at ALL levels**: not just network messages; cache-memory communication dominates.
2. **Arithmetic intensity is key**: maximize computation per byte of data transferred.
3. **Assignment dramatically affects communication**: 2D blocked assignment captures 2D locality.
4. **Cache behavior creates artifactual communication**: tune traversal order to keep data in cache.
5. **Loop fusion improves arithmetic intensity**: combine operations on same data into one pass.
6. **Contention destroys scalability**: distributed queues, fine-grained locks, staggered access.
7. **False sharing is invisible but expensive**: align/pad data to avoid cache line sharing.
8. **Roofline model guides optimization**: know if you're compute-bound or memory-bound.
9. **High watermarks establish upper bounds**: remove components to isolate bottlenecks.
10. **Problem size matters for scaling**: fixed problem size speedup can be misleading.

---

## C++ Source Files Reference

| Knowledge Point | C++ File |
|----------------|----------|
| Memory locality: blocked traversal, loop fusion, arithmetic intensity | `../src/lecture6_part1.cpp` |
| Cache coherency simulation, false sharing demonstration | `../src/lecture6_part2.cpp` |
| Contention, NUMA, message passing simulation | `../src/lecture6_part3.cpp` |
| Roofline model, high watermarks, performance counters | `../src/lecture6_part4.cpp` |
