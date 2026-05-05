# Lecture 10: Hardware Specialization

**PDF**: Lecture 10 - Hardware Specialization (Stanford CS149, Fall 2025)

---

## Core Concepts

### 1. Why Specialize? Energy-Constrained Computing

General-purpose CPUs spend enormous energy on non-compute overhead:
- Instruction fetch/decode, pipeline control, register file access, data movement
- Even SIMD-optimized H.264 encoding: ~90% energy goes to non-functional-unit overhead

**Rules of thumb** (vs high-quality C code on CPU):
| Architecture | Perf/Watt Improvement |
|---|---|
| GPU cores | ~10x (data-parallel, compute-bound) |
| Fixed-function ASIC | ~100-1000x (non-FP, compute-bound) |

### 2. The Efficiency-Programmability Spectrum

```
CPU (easiest) → GPU → DSP → Domain-specific Accelerator → FPGA → ASIC (100-1000x eff.)
```

- **GPU**: 10x efficiency, wide SIMD + tensor cores
- **DSP** (e.g., Qualcomm Hexagon): VLIW; 29 RISC ops/cycle in FFT inner loop
- **FPGA**: LUT-based reconfigurable logic; middle ground
- **ASIC**: Fixed function, $10-100M to design; e.g., Google TPU, Anton for molecular dynamics

### 3. Why GPUs are Sub-optimal for AI

GPUs are still general-purpose processors:
- Most TFLOPS (94-98% in recent generations) reside in **Tensor Cores**, not SIMD units
- SIMD units → diminishing fraction of total compute
- Complex instruction stream overhead limits efficiency

### 4. Systolic Arrays (Google TPU)

**Key insight**: Data-driven computation, not instruction-driven.
- Weights loaded into processing elements (PEs) in a grid
- Inputs stream through; partial sums accumulate
- **No instruction fetch/decode** → extreme efficiency

**SIMD vs Systolic Array**:
| Feature | SIMD | Systolic Array |
|---|---|---|
| Dataflow | Control-driven | Data-driven (wavefront) |
| Locality | Limited | Temporal + spatial |
| Communication | Global (register/memory) | Local (neighbor PEs) |
| Control | Centralized | Distributed |
| Efficiency | Medium | Very high |

### 5. TPU Architecture

- Arithmetic units: ~30% of chip area (very high!)
- Low control area footprint
- Key instructions: read_host_memory, write_host_memory, read_weights, matrix_multiply, activate
- Performance/watt: ~30-80x vs CPU+GPU alternatives (incremental TPU cost only)

### 6. Modern GPU Specialization (NVIDIA)

**A100**: 432 tensor cores, 312 TFLOPS (fp16/32 mixed)
**H100**: 4th-gen tensor cores, TMA (Tensor Memory Accelerator), HBM3 80GB
**B100**: Single-thread MMA execution, tcgen05 instructions, fp4 support

Tensor Core operations: `A[8x4] * B[4x8] + D[8x8]` in one instruction

### 7. Dataflow Architecture (Plasticine / SambaNova)

- **No instructions** → no instruction fetch/decode overhead
- **Extreme asynchrony**: compute, memory, and communication all overlap
- Spatial mapping of dataflow graph onto hardware
- PCU (Pattern Compute Unit) + PMU (Pattern Memory Unit) + Switch mesh

### 8. Arithmetic Intensity & Roofline Model

- **Arithmetic Intensity** = FLOPs / Bytes accessed
- Compute-bound: high arithmetic intensity, limited by peak FLOPs
- Memory-bound: low arithmetic intensity, limited by memory bandwidth

### 9. Amortizing Instruction Overhead

| Instruction Type | Overhead |
|---|---|
| Half-precision FMA | 2000% |
| Half-precision DP4 (vec4 dot) | 500% |
| Half-precision 4x4 MMA | 27% |

Complex instructions amortize control overhead across many operations.

### 10. Numerical Formats for AI

- **BF16**: 1 sign + 8 exponent + 7 mantissa (same range as FP32, lower precision)
- **BF8 E4M3**: 1+4+3, range 0-448
- **BF8 E5M2**: 1+5+2, range 0-57344
- **FP4**: emerging in B100

### 11. Hardware Lottery (Sara Hooker)

When a research idea wins because it is suited to available hardware, not because it is universally superior. Example: dense matrix multiply on TPUs → Transformer dominance.

### 12. Data Movement Energy Cost

| Operation | Energy (pJ) |
|---|---|
| Integer op | ~1 |
| FP op | ~20 |
| Read 64b from SRAM (1mm) | ~26 |
| Read 64b from LPDDR | ~1200 |

---

## Knowledge → C++ File Mapping

| Concept | C++ File |
|---|---|
| Systolic array GEMM simulation | `lecture10_part1.cpp` |
| Roofline model & arithmetic intensity | `lecture10_part2.cpp` |

---

## Actionable Learning Points

1. **Write a systolic array simulator**: Track dataflow through a 2D PE grid, measure throughput
2. **Implement roofline analysis**: Given compute and memory specs, determine if a kernel is compute-bound or memory-bound
3. **Compare energy models**: Quantify the energy advantage of systolic vs SIMD for matrix multiply
4. **Study TPU instruction set**: What minimal instructions achieve maximal efficiency?
5. **Understand dataflow graphs**: Map simple neural network layers to spatial hardware
