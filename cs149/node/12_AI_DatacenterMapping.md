# Lecture 12: Mapping AI Applications to the AI Datacenter

**PDF**: Lecture 12 - Mapping AI Applications to the AI Datacenter (Stanford CS149, Fall 2025)

---

## Core Concepts

### 1. Memory Systems: CPU vs GPU

| | CPU | GPU |
|---|---|---|
| Memory type | DRAM | HBM (High-Bandwidth Memory) |
| Bus width | 64-bit | 1024-bit (per stack) |
| Access pattern | Latency-optimized | Bandwidth-optimized |

**HBM3 on H100**: 6 stacks × 1024-bit = 6144-bit interface, 3.2 TB/s peak BW, 80 GB

### 2. 3D Stacking Technology

- Through-Silicon Vias (TSVs): vertical connections through DRAM chips
- Logic layer (base): memory controller
- Silicon interposer: high-bandwidth interconnect between stack and processor
- **Benefits**: higher bandwidth, lower power, smaller form factor

**HBM evolution**:
- HBM (2015, AMD Fury): 4 stacks, 512 GB/s
- HBM2 (2016, NVIDIA P100): 4 stacks, 720 GB/s
- HBM3 (2022, NVIDIA H100): 6 stacks, 3.2 TB/s
- HBM4: Custom logic die (SRAM cache, KV cache compression, compute)

### 3. DRAM Internals

**Structure**:
- Array of 1 transistor + 1 capacitor per bit
- Row buffer (e.g., 2 Kbits)
- Data pins (e.g., 8 bits per chip)

**Operation (byte load)**:
1. Precharge (ready bit lines): ~10 ns
2. Row activation (transfer row to buffer): ~10 ns
3. Column selection: ~10 ns
4. Data transfer onto bus

**Access latency**:
- Best case (row hit): CAS only
- Worst case (row miss): PRE + RAS + CAS
- Burst mode: amortizes latency over larger transfers

### 4. DRAM Banks & Pipelining

Multiple banks share data pins; only one transfer at a time.
- Pipeline: precharge/activate one bank while transferring from another
- Goal: maximize data pin utilization

**DIMM organization**: Multiple DRAM chips → wider interface (e.g., 8 chips × 8-bit = 64-bit bus)
- Physical addresses interleaved across chips at byte granularity
- 64-byte cache line serviced in parallel bursts across all chips

### 5. Memory Controller Scheduling

- Receives load/store requests from LLC
- Conflicting goals: max throughput, min latency, min energy
- **FR-FCFS policy** (First-Ready, First-Come-First-Serve):
  1. Service requests to currently **open row** first (maximize row locality)
  2. Service other requests in FIFO order
- May coalesce small requests into large contiguous requests (burst mode)

### 6. Energy Cost of Data Movement (Revisited)

| Operation | Energy (pJ) |
|---|---|
| FP math op (32-bit) | ~0.9 |
| Local SRAM access | ~5 |
| Read 32b from LPDDR | ~640 |

**Implications**:
- Recomputing values can be cheaper than storing+reloading!
- Exploiting locality is critical for energy efficiency
- Reading 10 GB/s from memory ≈ 1.6 watts (entire mobile GPU budget ~1W)

### 7. Message Passing Primitives

| Primitive | Description |
|---|---|
| **AllReduce** | Sum values across all ranks, result on all ranks |
| **ReduceScatter** | Sum values, then scatter result chunks across ranks |
| **AllGather** | Collect chunks from all ranks into full tensor on all ranks |
| **All-to-All** | Transpose data distribution across ranks |

**AllReduce = ReduceScatter + AllGather** (ring algorithm)

### 8. Types of Parallelism in AI Training

| Parallelism Type | Split Dimension | Communication Primitive |
|---|---|---|
| **Data Parallel (DP)** | Batch dimension | Reduce-Scatter + All-Gather |
| **Tensor Parallel (TP)** | Hidden dimension (weights) | Reduce-Scatter + All-Gather |
| **Pipeline Parallel (PP)** | Layer dimension | Send-Recv (P2P) |
| **Expert Parallel (EP)** | MoE experts | All-to-All |
| **Sequence Parallel (SP)** | Sequence length | Reduce-Scatter |
| **Context Parallel (CP)** | Context tokens | All-Reduce |

### 9. Distributed Matrix-Multiply

Partition K dimension across S ranks:
```
A[M×K] × B[K×N] → each rank computes [M×K/S] × [K/S×N] = [M×N] partial
Reduce-Scatter: combine S partial results → final [M×N] distributed across ranks
```

### 10. Compute-Communication Overlap

**Without overlap**: FLOPS utilization drops as node count increases
- 32 sockets: theoretical peak 52% utilization without overlap
- With overlap (RDU): sustained **70-79% utilization** at 32 sockets

Key insight: AllReduce fully overlapped with weight load and compute.
Communication time doesn't consume HBM bandwidth.

### 11. Fine-grained Pipeline Parallelism

- Divide mini-batch into micro-batches
- Pipeline forward and backward computations across micro-batches
- Reduces idle time (pipeline bubbles)
- Trade-off: more micro-batches → smaller bubbles, but more communication

### 12. DGX SuperPOD Architecture

- 140 DGX A100 nodes (1,120 GPUs)
- Each node: 2× AMD EPYC + 8× A100 GPUs, NVLink 3.0
- Mellanox HDR 200Gb/s InfiniBand, full fat-tree topology
- Separate networks for compute vs storage
- Key: modular scaling with high-bandwidth interconnect

### 13. HBM4 Evolution

Future HBM4: custom logic die at the base of the stack
- SRAM cache
- KV cache compression
- I/O interfaces (Ethernet, PCIe)
- Possibly near-memory compute

### 14. Scaling Laws & Utilization

Larger models → higher utilization (more compute per communication):
- 1.7B params: 44% peak FLOPS
- 530B params: 49% peak FLOPS
- 1T params: 49% peak FLOPS

---

## Knowledge → C++ File Mapping

| Concept | C++ File |
|---|---|
| Distributed GEMM + AllReduce/ReduceScatter simulation | `lecture12_part1.cpp` |
| DRAM simulator (banks, burst, memory controller) | `lecture12_part2.cpp` |

---

## Actionable Learning Points

1. **Implement distributed matrix multiply**: Partition K across ranks, simulate Reduce-Scatter
2. **Build a DRAM simulator**: Model banks, row buffer hits/misses, burst mode, FR-FCFS scheduling
3. **Calculate arithmetic intensity**: For a given layer, compute FLOPs/byte ratio
4. **Analyze communication scaling**: How does AllReduce time grow with node count?
5. **Model pipeline parallelism**: Simulate micro-batch pipelining with bubble analysis
6. **Study HBM advantages**: Quantify bandwidth and energy benefits of 3D stacking
