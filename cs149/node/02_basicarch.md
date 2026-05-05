# CS149 Lecture 2: A Modern Multi-Core Processor (Part I)

**PDF**: `02_basicarch.pdf`

**Instructor**: Prof. Kayvon Fatahalian

**University**: Stanford CS149, Fall 2025

---

## Core Concepts Summary

### Review from Lecture 1

- A program = list of processor instructions
- Superscalar processors exploit ILP within a single instruction stream
- Memory hierarchy: cache reduces latency, data locality matters
- Power wall ended frequency scaling (~2005)

### Today's Focus

Three ideas in throughput computing hardware:
1. **Multi-core execution** — multiple independent instruction streams
2. **SIMD execution** — multiple ALUs controlled by one instruction
3. **Hardware multi-threading** — hide memory stalls by interleaving threads

---

## 1. Three Forms of Parallel Execution

| Form | Mechanism | Who Controls? | Key Property |
|------|-----------|---------------|--------------|
| **Superscalar** | Exploit ILP within one instruction stream | Hardware (automatic) | Within a single core |
| **SIMD** | Multiple ALUs, same instruction, different data | Compiler (explicit) or HW (implicit) | Amortizes control cost over many ALUs |
| **Multi-core** | Multiple independent instruction streams | Programmer (creates threads) | Each core can run different code |

### Multi-Core Rationale

- **Pre multi-core era**: transistors used for fancy branch predictors, OoO logic, large caches — all to accelerate a single instruction stream
- **Multi-core era**: use increasing transistor count to add more cores (simpler cores, but more of them)
- Trade-off: each core may be ~25% slower, but 2 cores × 0.75 = 1.5x potential speedup

> **C++ Demo**: `lecture2_part1.cpp` — Sequential vs multi-core vs SIMD vs combined sin(x) computation

---

## 2. The sin(x) Example Program

```c
void sinx(int N, int terms, float* x, float* y) {
    for (int i = 0; i < N; i++) {
        float value = x[i];
        float numer = x[i] * x[i] * x[i];
        int denom = 6;               // 3!
        int sign = -1;
        for (int j = 1; j <= terms; j++) {
            value += sign * numer / denom;
            numer *= x[i] * x[i];
            denom *= (2*j+2) * (2*j+3);
            sign *= -1;
        }
        y[i] = value;
    }
}
```

This sequential code runs on **one core** with **one instruction stream**. To use multiple cores, we need to express parallelism.

### Expressing Parallelism

1. **C++ threads** (explicit): manually split work across threads
2. **Data-parallel "forall"** (abstraction): declare loop iterations as independent — compiler can auto-generate threaded code
3. **ISPC/OpenMP**: higher-level parallel abstractions

The `forall` construct is Kayvon's fictitious parallel language feature:
```c
forall (int i from 0 to N) {
    // iterations are independent — compiler can parallelize
}
```

> **C++ Demo**: `lecture2_part1.cpp` — Thread-based parallel sinx, forall simulation

---

## 3. SIMD Execution (Single Instruction, Multiple Data)

### Key Idea
**Amortize the cost/complexity of managing an instruction stream across many ALUs.**

Instead of one fetch/decode per ALU, use one fetch/decode for N ALUs that all perform the same operation on different data.

### AVX Intrinsics Example (Vector sinx)

```c
#include <immintrin.h>
for (int i = 0; i < N; i += 8) {
    __m256 origx = _mm256_load_ps(&x[i]);    // load 8 floats
    __m256 value = origx;
    __m256 numer = _mm256_mul_ps(origx, _mm256_mul_ps(origx, origx));
    // ... compute sin(x) for 8 elements simultaneously ...
    _mm256_store_ps(&y[i], value);           // store 8 results
}
```

- `__m256` = vector of eight 32-bit floats (256 bits)
- `_mm256_load_ps` = vector load (vmovaps)
- `_mm256_mul_ps` = vector multiply (vmulps)
- Each instruction operates on **8 data elements simultaneously**

### Conditional Execution in SIMD

When different lanes need different paths:
```
if (t > 0.0) {
    t = t * t;        // only lanes where condition is true
} else {
    t = t + 30.0;     // only lanes where condition is false
}
```

- **Mask bits**: each ALU lane has a mask bit (1 = active, 0 = masked off)
- Masked lanes still execute but results are **discarded**
- **Worst case**: only 1/WIDTH lanes active → 1/WIDTH peak performance
- This is why **coherent execution** is critical for SIMD efficiency

> **C++ Demo**: `lecture2_part2.cpp` — SIMD vector class, mask simulation, conditional execution

### Coherent vs. Divergent Execution

| Property | Coherent | Divergent |
|----------|----------|-----------|
| Instruction stream | Same for all data | Differs per data element |
| SIMD efficiency | 100% ALU utilization | Reduced (masked lanes) |
| Multi-core impact | Not necessary | No problem (independent fetch/decode) |

### SIMD on Real Hardware

| Instruction Set | Width | Data Types |
|----------------|-------|------------|
| Intel AVX2 | 256-bit | 8×32b or 4×64b |
| Intel AVX512 | 512-bit | 16×32b or 8×64b |
| ARM Neon | 128-bit | 4×32b or 2×64b |

- **Explicit SIMD** (CPU): compiler generates vector instructions, programmer uses intrinsics or auto-vectorization
- **Implicit SIMD** (GPU): compiler generates scalar instructions, hardware runs N instances in lockstep on SIMD ALUs

---

## 4. Hardware Multi-Threading

### Problem: Memory Stalls

- DRAM access: ~248 cycles at 4 GHz
- During a memory load, the processor **stalls** (cannot execute dependent instructions)
- This wastes ALU resources

### Solution: Interleave Threads

When one thread stalls on a memory load, **execute instructions from another thread** on the same core's ALUs.

**Key idea**: potentially increase time to complete one thread, but increase overall system throughput.

### Types of Multi-Threading

| Type | Description | Example |
|------|-------------|---------|
| **Interleaved** (temporal) | Each clock, core chooses one thread to run | GPU-style |
| **Simultaneous** (SMT) | Each clock, core runs instructions from multiple threads | Intel Hyper-Threading (2 threads/core) |

### Thread Count vs. Utilization

- More arithmetic per memory access → **fewer threads needed** for 100% utilization
- `threads_needed = ceil(1 + latency / math_per_load)`
- Example: 3 math + 1 load (12-cycle latency) → need 5 threads for 100%

### Context Storage Trade-off

| Design | Pros | Cons |
|--------|------|------|
| Many small contexts | Excellent latency hiding | Small per-thread working set, more cache pressure |
| Few large contexts | Large per-thread working set, better cache locality | Less latency hiding ability |

> **C++ Demo**: `lecture2_part3.cpp` — Multi-threaded core simulator, latency hiding analysis

---

## 5. Real Processor Examples

### Intel i7-7700K (Kaby Lake, 4 cores)

- 4 cores × 8-wide AVX2 × 3 ALUs × 4.2 GHz ≈ 400 GFLOPs
- 2-way SMT (Hyper-Threading): 2 threads per core
- L1: 32KB, L2: 256KB, L3: 8MB

### NVIDIA V100 GPU

- 80 SMs (Streaming Multiprocessors)
- Per SM: 64 fp32 ALUs, 64 warp execution contexts
- 64 warps × 32 SIMD width = 2048 concurrent data items per SM
- 80 × 2048 = **163,840 concurrent data items** for maximal latency hiding
- 16 GB HBM2 at 900 GB/s

---

## 6. Review: How It All Fits Together

The lecture builds up from the simplest processor to a full modern design:

1. **Simple processor**: 1 core, 1 thread, 1 scalar instruction/clock
2. **Superscalar**: 1 core, 1 thread, up to N independent instructions/clock
3. **SIMD**: 1 core, 1 thread, 1 vector instruction operates on W data elements
4. **Heterogeneous superscalar**: 1 core, scalar + vector ALUs
5. **Multi-threaded**: 1 core, M threads (interleaved execution)
6. **Multi-threaded superscalar**: 1 core, M threads, N instructions/clock
7. **Multi-core**: C cores, each with M threads and N-issue superscalar + SIMD
8. **GPU SIMT**: many cores, each with many warps, implicit SIMD execution

---

## Actionable Learning Points

1. **Know your processor's capabilities**: cores × SIMD width × ALUs × frequency = peak throughput
2. **Write coherent code for SIMD**: avoid divergent branches in data-parallel regions
3. **Use enough threads**: to hide memory latency (more threads ≠ better if already 100%)
4. **Cache matters**: L1 hits are 60x faster than DRAM (4 vs 248 cycles)
5. **Amortize control**: SIMD shares one fetch/decode across many ALUs

---

## C++ Source Files

| File | Topic | Key Demonstration |
|------|-------|-------------------|
| `lecture2_part1.cpp` | Multi-core & SIMD sin(x) | Sequential vs threads vs SIMD vs combined |
| `lecture2_part2.cpp` | SIMD conditional execution | Mask simulation, coherent vs divergent |
| `lecture2_part3.cpp` | Hardware multi-threading | Latency hiding, thread count vs. utilization |

### Compilation

```bash
g++ -std=c++17 -O2 -pthread lecture2_part1.cpp -o lecture2_part1 && ./lecture2_part1
g++ -std=c++17 -O2 lecture2_part2.cpp -o lecture2_part2 && ./lecture2_part2
g++ -std=c++17 -O2 -pthread lecture2_part3.cpp -o lecture2_part3 && ./lecture2_part3
```

---

## Terminology to Know

- **Instruction stream**: sequence of instructions for one logical thread
- **Multi-core processor**: multiple independent cores on one chip
- **SIMD execution**: single instruction broadcast to multiple ALUs
- **Coherent control flow**: all lanes follow the same execution path
- **Divergent execution**: different lanes take different paths (bad for SIMD)
- **Hardware multi-threading**: interleaving/simultaneous execution of multiple HW threads
- **SMT**: simultaneous multi-threading (Intel Hyper-Threading)
- **Prefetching**: hardware guesses future memory accesses to pre-load into cache
