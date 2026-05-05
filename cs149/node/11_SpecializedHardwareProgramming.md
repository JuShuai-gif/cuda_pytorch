# Lecture 11: Programming Specialized Hardware for AI

**PDF**: Lecture 11 - Programming Specialized Hardware for AI (Stanford CS149, Fall 2025)

---

## Core Concepts

### 1. Synchronous vs Asynchronous Execution

**Synchronous (blocking)**:
```
LD0 → AO0 → ST0 → LD1 → AO1 → ST1 → LD2 → AO2 → ST2
```
Each operation waits for the previous to complete.

**Asynchronous (non-blocking)**:
```
LD0 → AO0 → ST0
      LD1 → AO1 → ST1
            LD2 → AO2 → ST2
```
Overlap: later operations start before earlier ones finish.
- Requires: async instructions, synchronization primitives, or hardware out-of-order execution

### 2. Three Hardware Platforms Studied

| Platform | Architecture | Programming Model |
|---|---|---|
| **Google TPU** | Systolic array | Simple instructions: read_weights, matrix_multiply, activate |
| **NVIDIA H100/B100** | Tensor cores + TMA | CUDA → PTX → tcgen05; DSLs like ThunderKittens |
| **SambaNova SN40L** | Reconfigurable dataflow | Metapipelining with data-parallel patterns |

### 3. Systolic Array Dataflow Types

| Type | What stays in PE | What streams through | Goal |
|---|---|---|---|
| **Weight-Stationary (WS)** | Weight values | Inputs + partial sums | Minimize weight reload |
| **Output-Stationary (OS)** | Partial sums | Inputs + weights | Minimize output movement |
| **Input-Stationary (IS)** | Input activations | Weights + partial sums | Minimize input reload |

The TPU uses a weight-stationary design.

### 4. B100 Tensor Core Programming (Not Your Father's CUDA)

Traditional CUDA: warps, thread blocks, SIMD lanes.
B100:
- **Single threads** execute MMA (no more warps!)
- TMEM allocation: `tcgen05.alloc`
- Async prefetch: `cp.async.bulk.tensor` + `mbarrier`
- Async MMAs: `tcgen05.mma` batch + `tcgen05.commit`
- Order & retire: `tcgen05.fence`

### 5. Tensor Memory Accelerator (TMA)

- Special-purpose instructions for data movement
- Async load/store tensor regions (global → shared memory)
- Copy descriptor describes the region
- Single thread issues TMA; hardware handles address gen + data movement
- Signals barrier when complete
- **Eliminates 1000s of instructions** and unnecessary data movement through L1/registers

### 6. ThunderKittens DSL

**Design Principles**:
1. **16×16 tile as primitive data type** (matches tensor core compute)
2. **Asynchrony everywhere**: producer-consumer pipeline
3. **High-level GPU coordination patterns**

**Embedded CUDA DSL** with templated types:
- Register tiles (2D tensors on register file)
- Shared memory tiles
- Register/shared vectors
- Operations: init, unary, binary, row/column ops

**TK Matmul pipeline**:
```
Producer (TMA loads tiles) → Shared Memory → Consumer (MMA compute) → Store
```
Uses warp groups, mbarrier synchronization, 4-stage pipeline depth.

### 7. Metapipelining (SambaNova)

**Definition**: Hierarchical coarse-grained pipeline - a "pipeline of pipelines"
- Converts parallel loops into streaming pipelines
- Insert pipe stages in loop body; stages execute in parallel
- Overlap execution of multiple loop iterations
- Double buffering for intermediate data
- Works with tiling and fusion

**Matmul Metapipeline example**:
```
METAPIPE(M/MM) {
    a_tile = LOAD_TILE(A)
    METAPIPE(N/NN) {
        b_tile = LOAD_TILE(B)
        c = MAT_MUL(a_tile, b_tile)
        STORE_TILE(C, c)
    }
}
```

### 8. Dataflow Programming Model

- Composable compute primitives: MM, Map, Zip, Reduce, Gather, Scatter
- Flexible scheduling in space and time → **spatial execution**
- Tiling → Parallelization → Metapipelining → Place & Route → Codegen
- No lock-based synchronization: **token-controlled dataflow execution**

### 9. GPU vs RDU Kernel Fusion

**GPU (H100)**: TensorRT-LLM — limited kernel fusion
- Llama3.1 8B: ~800 kernel calls per token
- Separate kernels for Q/K/V GEMM, attention, output GEMM, etc.
- High launch and synchronization overheads

**RDU (SN40L)**: Aggressive kernel fusion
- Entire decoder fused into **one kernel call**
- ~3 calls per token (100x fewer!)
- 5x SRAM advantage (520MB vs 100MB)
- Dataflow fusion eliminates GBs of off-chip intermediate result traffic

### 10. Kernel Loop Pattern

Instead of launching one kernel per decoder (32 decoders → 32 launches):
- Launch **one kernel** that loops over all decoders internally
- Completely overlap weight load with compute
- Keep HBM busy all the time
- Zero extra launch overheads

### 11. Asynchrony → Overlap

Goal: overlap compute, memory access, and chip-to-chip communication
- AllReduce fully overlapped with weight load and compute
- AllReduce does not consume HBM capacity or bandwidth

---

## Knowledge → C++ File Mapping

| Concept | C++ File |
|---|---|
| Asynchronous pipeline execution simulation | `lecture11_part1.cpp` |
| Metapipelining / hierarchical pipeline simulation | `lecture11_part2.cpp` |

---

## Actionable Learning Points

1. **Model async execution**: Simulate a producer-consumer pipeline with TMA-like async loads
2. **Implement metapipelining**: Build a hierarchical pipeline with double buffering
3. **Compare synchronous vs async throughput**: Measure the performance gap
4. **Study ThunderKittens design**: Why 16×16 tiles? Why warp groups?
5. **Trace a FlashAttention dataflow**: Map attention computation to spatial hardware
6. **Quantify kernel launch overhead**: How many kernel calls does your favorite model use?
