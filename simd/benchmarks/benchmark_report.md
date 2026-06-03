# SIMD Benchmark Report

This report documents how to collect, interpret, and act on performance measurements
from the SIMD tutorial project. Every benchmark name referenced below corresponds
to an actual binary produced by the build system -- no fictional kernels.

---

## 1. System Information

### Collecting system info

Run these commands and paste the output into your report:

```bash
lscpu
cat /proc/cpuinfo | head -50
uname -a
cat /proc/meminfo | head -5
lscpu | grep -E "MHz|cache"
```

### Template -- fill in your own numbers

| Property              | Value                                           |
|-----------------------|-------------------------------------------------|
| Hostname              | `hostname`                                      |
| CPU Model             | *(from `lscpu` "Model name")*                  |
| Architecture          | `uname -m` (expect: `x86_64` or `aarch64`)     |
| Base / Max Frequency  | *(from `lscpu` "CPU MHz" / "CPU max MHz")*      |
| Cores / Threads       | *(from `lscpu` "CPU(s)")*                       |
| Sockets               | *(from `lscpu` "Socket(s)")*                    |
| L1d Cache             | *(from `lscpu` "L1d cache")*                    |
| L2 Cache              | *(from `lscpu` "L2 cache")*                     |
| L3 Cache              | *(from `lscpu` "L3 cache")*                     |
| SIMD ISA Flags        | *(from `lscpu` "Flags" -- grep for: sse, avx, avx2, avx512, fma, neon, sve)* |
| Memory                | *(from `cat /proc/meminfo` / `sudo dmidecode -t memory`)* |
| Kernel                | `uname -r`                                      |
| Compiler              | `gcc --version \| head -1` or `clang --version` |

### Interpreting ISA flags

| Flag           | Meaning                                       |
|----------------|-----------------------------------------------|
| `sse`, `sse2`  | SSE / SSE2 -- baseline on all x86-64          |
| `sse4_1/4_2`   | SSE4.1 / SSE4.2 -- adds dot products, blends  |
| `avx`          | AVX -- 256-bit floats, 3-operand encoding     |
| `avx2`         | AVX2 -- 256-bit integer, gather, FMA          |
| `fma`          | Fused multiply-add (often alongside AVX2)     |
| `avx512f`      | AVX-512 Foundation -- 512-bit vectors         |
| `avx512dq`     | AVX-512 double/quadword                       |
| `avx512bw`     | AVX-512 byte/word                             |
| `avx512vl`     | AVX-512 vector length (128/256-bit with AVX-512 encoding) |
| `asimd`        | ARM NEON (aarch64)                            |
| `sve`          | ARM Scalable Vector Extension                 |

---

## 2. How to Run Benchmarks

### Build

There are two ways to build. Choose one.

**Option A: auto-detect script (recommended)**

```bash
cd /home/ghr/code/cuda_pytorch/simd
./scripts/build.sh
```

The script auto-detects your CPU architecture, selects the appropriate ISA flags
(-mavx2 -mfma for x86, NEON for ARM), and enables AVX-512 targets if your CPU
supports `avx512f`.

**Option B: cmake preset (for specific configs)**

```bash
cd /home/ghr/code/cuda_pytorch/simd

# Release build, AVX2 + AVX-512 (auto-detects -march=native)
cmake --preset x86-release -B build
cmake --build build -j$(nproc)

# AVX2 only (no AVX-512, suitable for CPUs without AVX-512)
cmake --preset x86-release-no512 -B build_no512
cmake --build build_no512 -j$(nproc)

# RelWithDebInfo (for perf annotate / profiling)
cmake --preset x86-relwithdebinfo -B build_profile
cmake --build build_profile -j$(nproc)
```

All binaries land in `build/x86/` (or `build/arm/` on ARM).

### List built benchmarks

```bash
find build/x86 -type f -executable | sort
```

### Run all benchmarks

```bash
./scripts/run_all_benchmarks.sh
```

Results are written to `benchmarks/latest_results.txt`. The script runs every
binary matching `avx2_*` or `avx512_*` (on x86) and collects their output.

### Run a single benchmark

```bash
./build/x86/avx2_dot_product
./build/x86/avx2_layernorm
./build/x86/avx2_softmax_partial
```

Each binary runs both scalar and SIMD paths internally, measures both, and
prints a speedup table.  Example output from `avx2_vector_add`:

```
--- avx2_vector_add: float (N=1000003) ---
name                     elapsed_ns   ns/el      GB/s       speedup
scalar_add_f32             823456    0.8235      14.57      1.00x
scalar_add_i32             810234    0.8103      14.81      1.00x
avx2_add_f32_unaligned     105678    0.1057     113.55      7.79x
avx2_add_f32_aligned       103211    0.1032     116.27      7.98x
avx2_add_i32               101893    0.1019     117.77      7.95x
```

### Profile an individual kernel

```bash
# Hardware counters: IPC, cache misses, SIMD ratio
./scripts/perf_stat.sh build/x86/avx2_dot_product

# Sampling profile (perf record + report)
./scripts/profile.sh record   build/x86/avx2_layernorm

# Instruction-level hotspot (perf annotate)
./scripts/profile.sh annotate build/x86/avx2_layernorm

# Cache simulation (cachegrind)
./scripts/profile.sh cache    build/x86/avx2_layernorm

# Intel Top-Down microarchitecture analysis
./scripts/profile.sh topdown  build/x86/avx2_layernorm

# Flame graph
./scripts/profile.sh flame    build/x86/avx2_layernorm

# Full pipeline: record + annotate + cache
./scripts/profile.sh all      build/x86/avx2_layernorm
```

### Inspect generated assembly

```bash
# Disassemble and highlight SIMD instructions
./scripts/inspect_asm.sh build/x86/avx2_dot_product

# Static throughput prediction (no hardware needed)
./scripts/llvm_mca.sh build/x86/avx2_dot_product avx2_dot_product_f32
./scripts/llvm_mca.sh --demo
```

---

## 3. Benchmark Catalog

Every benchmark listed below is an actual binary produced by the build system.
Each binary compares at least one scalar path against at least one SIMD path
and reports speedup.

### Compute-bound kernels (expect high speedup)

| Benchmark Binary              | Kernel                          | Data Type | Key SIMD Instructions              | Expected Speedup (AVX2) |
|-------------------------------|---------------------------------|-----------|------------------------------------|--------------------------|
| `avx2_vector_add`             | Vector addition C[i]=A[i]+B[i]  | f32, i32  | `vaddps` / `vaddps` (aligned+un)   | 7.0x - 8.0x             |
| `avx2_relu_clamp`             | ReLU activation + clamp         | f32       | `vmaxps` / `vminps`               | 7.0x - 8.0x             |
| `avx2_dot_product`            | Dot product sum(A[i]*B[i])      | f32       | `vfmadd231ps` + horizontal sum    | 6.5x - 7.5x             |
| `avx2_reduce_sum`             | Sum reduction sum(A[i])         | f32       | `vaddps` + horizontal reduction   | 6.0x - 7.5x             |
| `avx2_layernorm`              | Layer normalization             | f32       | `vsubps`, `vmulps`, `vaddps`, `vsqrtps` | 5.0x - 7.0x      |
| `avx2_softmax_partial`        | Softmax (exp + sum + div)       | f32       | exp approx, `vaddps`, `vdivps`    | 4.0x - 6.5x             |
| `avx2_int8_dot`               | Integer dot product             | int8      | `vpmaddubsw`, `vpmaddwd`          | 7.0x - 8.0x             |
| `avx2_gemm_micro`             | GEMM micro-kernel               | f32       | `vfmadd231ps` (multi-accumulator)  | 6.0x - 7.5x             |

### Memory-bound kernels (expect modest speedup)

| Benchmark Binary              | Kernel                          | Data Type | Key SIMD Instructions              | Expected Speedup (AVX2) |
|-------------------------------|---------------------------------|-----------|------------------------------------|--------------------------|
| `avx2_memcpy_like`            | Memcpy-like streaming copy      | f32       | `vmovaps` / `vmovntps` (NT store)  | 1.1x - 1.8x             |
| `avx2_rgb_to_gray`            | RGB to grayscale conversion     | uint8     | `vpmaddubsw`, blending             | 2.0x - 4.0x             |

### Layout / data-reorganization kernels

| Benchmark Binary              | Kernel                          | Data Type | Key SIMD Instructions              | Expected Speedup (AVX2) |
|-------------------------------|---------------------------------|-----------|------------------------------------|--------------------------|
| `avx2_aos_to_soa`             | AoS -> SoA transposition        | f32       | gather / shuffle / interleave      | 2.5x - 4.5x             |

### Mixed compute + memory kernels

| Benchmark Binary              | Kernel                          | Data Type | Key SIMD Instructions              | Expected Speedup (AVX2) |
|-------------------------------|---------------------------------|-----------|------------------------------------|--------------------------|
| `avx2_conv1d`                 | 1D convolution                  | f32       | `vfmadd231ps`, sliding window      | 3.0x - 6.0x             |

### Educational / diagnostic binaries

| Benchmark Binary              | Purpose                                                        |
|-------------------------------|----------------------------------------------------------------|
| `avx2_autovec_vs_intrinsics`  | Compares auto-vectorization (compiler) vs hand-written AVX2    |
| `dispatch_demo`               | Demonstrates runtime ISA dispatch (SSE -> AVX2 -> AVX-512)     |
| `portable_sse`                | Same source compiled with `-msse4.2` (128-bit SIMD)            |
| `portable_avx2`               | Same source compiled with `-mavx2 -mfma` (256-bit SIMD)        |
| `portable_avx512`             | Same source compiled with AVX-512 flags (512-bit, if enabled)   |
| `edge_cases_demo`             | Validates NaN, Inf, denormals, zero-length, alignment edge cases|

### AVX-512 kernels (optional -- requires `BUILD_X86_AVX512=ON`)

| Benchmark Binary              | Kernel                          | Key AVX-512 Feature                |
|-------------------------------|---------------------------------|------------------------------------|
| `avx512_vector_add`           | Vector addition f32             | 512-bit vectors (16x f32)          |
| `avx512_reduce_sum`           | Sum reduction f32               | Horizontal reduction with AVX-512  |
| `avx512_dot_product`          | Dot product f32                 | FMA + 512-bit accumulators         |
| `avx512_masked_tail`          | Masked tail handling            | `k` mask registers                 |
| `avx512_gather_scatter`       | Gather/scatter operations       | `vgatherdps` / `vscatterdps`       |
| `avx512_byte_scan`            | Byte-level scanning             | `vpcmpb` + `kmov` on byte data    |

---

## 4. Interpreting Results -- The Roofline Model

The **roofline model** explains why some kernels get near-theoretical SIMD speedup
and others barely improve.

### The formula

```
Attainable GFLOP/s = min( Peak_Compute_GFLOP/s ,  Peak_Memory_GB/s * OI )
```

Where **OI (Operational Intensity)** = total FLOPs / total bytes transferred.

### Computing OI for your kernel

For the `avx2_vector_add` kernel: C[i] = A[i] + B[i]

- **FLOPs per element**: 1 addition = 1 FLOP
- **Bytes per element** (f32): 2 reads (A, B) + 1 write (C) = 12 bytes
- **OI** = 1 FLOP / 12 bytes = **0.083 FLOP/byte**

That is extremely low OI -- the kernel is overwhelmingly memory-bound, which is
why `avx2_vector_add` speedup is typically capped by DRAM bandwidth.

For `avx2_dot_product`: result = sum(A[i] * B[i])

- **FLOPs per element**: 1 multiply + 1 accumulate = 2 FLOP
- **Bytes per element** (f32): 2 reads (A, B) = 8 bytes
- **OI** = 2 FLOP / 8 bytes = **0.25 FLOP/byte**

Higher OI means it leans more compute-bound, and speedup approaches the SIMD
width ratio more closely.

For `avx2_gemm_micro` (tiled matrix multiply, working set in L1 cache):

- **FLOPs per element**: 2 * K (for K inner dimension)
- **Bytes from L1**: negligible reused data
- **OI** >> 1 (compute-bound on L1 data)

This is where SIMD shines -- speedup near the theoretical maximum.

### Finding your machine's roofline

```
Peak_Compute = cores * frequency * SIMD_width_in_floats * 2 (for FMA)

Example -- Intel Core i7-13700K (8 P-cores, 5.4 GHz turbo):
  Peak_Compute = 8 * 5.4e9 * 16 (AVX-512 f32) * 2 (FMA) = 1382 GFLOP/s (theoretical)

Peak_Memory = memory_channels * frequency * bytes_per_transfer

Example -- DDR5-5600 dual-channel:
  Peak_Memory = 2 * 5600e6 * 8 bytes = 89.6 GB/s (theoretical STREAM triad)
```

### Decision from roofline

1. Compute your kernel's OI.
2. If `Peak_Memory * OI < Peak_Compute` -> **memory-bound**. Better SIMD won't
   help. Instead: cache-block, prefetch, use non-temporal stores, or reorder loops.
3. If `Peak_Memory * OI >= Peak_Compute` -> **compute-bound**. SIMD gives the
   full theoretical speedup. Verify with `perf_stat.sh` that IPC is high (>1.5).

---

## 5. Expected Speedup per ISA

### Theoretical maximum speedup for f32

| ISA                 | Vector Width | f32 per Instr | Theoretical Speedup (f32) |
|---------------------|-------------|---------------|----------------------------|
| SSE4.2 (baseline)   | 128-bit     | 4             | 4.0x                       |
| NEON (ASIMD)        | 128-bit     | 4             | 4.0x                       |
| AVX2                | 256-bit     | 8             | 8.0x                       |
| AVX2 + FMA          | 256-bit     | 8 (2x throughput with FMA)| 8.0x - 10.0x     |
| AVX-512             | 512-bit     | 16            | 16.0x                      |
| SVE (256-bit impl)  | 256-bit     | 8             | 8.0x                       |
| SVE (512-bit impl)  | 512-bit     | 16            | 16.0x                      |

### Practical speedup ranges (what to actually expect)

The theoretical maximum assumes: all data in L1 cache, perfect instruction
scheduling, no reduction overhead, and no unrolling needed for ILP. In practice:

| ISA                 | Compute-Bound Kernels    | Memory-Bound Kernels     | Mixed Kernels         |
|---------------------|--------------------------|--------------------------|------------------------|
| SSE4.2              | 3.0x - 3.8x              | 1.0x - 1.3x              | 1.5x - 2.5x           |
| NEON (128-bit)      | 3.0x - 3.8x              | 1.0x - 1.3x              | 1.5x - 2.5x           |
| AVX2                | 6.0x - 7.8x              | 1.1x - 1.8x              | 2.0x - 5.0x           |
| AVX2 + FMA          | 6.5x - 8.5x              | 1.1x - 1.8x              | 2.5x - 6.0x           |
| AVX-512             | 12.0x - 15.5x            | 1.1x - 1.8x              | 3.0x - 8.0x           |
| SVE 256-bit         | 6.0x - 7.5x              | 1.1x - 1.6x              | 2.0x - 4.5x           |

### Why you never get exactly theoretical

- **Reduction overhead**: `avx2_dot_product` and `avx2_reduce_sum` need
  horizontal add at the end. `vhaddps` only executes on port 5 (Skylake) or
  port 0/1 (Zen 3), creating a bottleneck.
- **Tail handling**: if `n % 8 != 0`, the loop has a scalar epilogue that
  dilutes speedup.
- **Load alignment**: `_mm256_loadu_ps` vs `_mm256_load_ps`. Unaligned loads
  crossing cache-line boundaries add 1-2 cycles.
- **Pipeline depth**: FMA has 4-cycle latency on Skylake. Without
  multi-accumulator unrolling, the dependency chain limits throughput.
- **Frequency scaling**: AVX-512 can cause clock throttling on some CPUs
  (especially older Intel). `avx2` kernels may actually run at higher frequency
  than `avx512` kernels on the same machine.
- **Compiler differences**: GCC vs Clang can produce different instruction
  schedules. The `avx2_autovec_vs_intrinsics` benchmark directly compares
  compiler auto-vectorization against hand-written intrinsics.

---

## 6. What Good Speedup Looks Like

### Compute-bound kernels (high OI)

These kernels do many FLOPs per byte loaded. The working set fits in L1 or L2
cache. Expect speedup **close to the SIMD width ratio**:

| Benchmark           | AVX2 Speedup | Rationale                                    |
|---------------------|-------------|-----------------------------------------------|
| `avx2_vector_add`   | 7.0x - 7.9x | Pure element-wise, no reduction               |
| `avx2_relu_clamp`   | 7.0x - 7.9x | Min/max is same throughput as add             |
| `avx2_int8_dot`     | 7.0x - 8.0x | 32 int8 per 256-bit register -> huge speedup  |
| `avx2_dot_product`  | 6.0x - 7.5x | Reduction step costs 5-8% of total time       |
| `avx2_gemm_micro`   | 6.5x - 7.8x | Multi-accumulator unrolling hides FMA latency  |
| `avx2_layernorm`    | 5.5x - 7.0x | sqrt + division lower the ceiling             |
| `avx2_softmax_partial`| 4.0x - 6.5x | exp() approximation is the bottleneck        |

**Red flag**: if `avx2_vector_add` speedup is below 5.0x, check:
- Are you running a debug build? Use `cmake --preset x86-release`.
- Is the compiler auto-vectorizing the scalar loop? Inspect assembly with
  `./scripts/inspect_asm.sh`.
- Are you on a virtual machine with throttled SIMD? Check `lscpu | grep Flags`.

### Memory-bound kernels (low OI)

These kernels stream through large arrays that don't fit in cache. The CPU
spends most cycles waiting on DRAM. SIMD helps only marginally:

| Benchmark           | AVX2 Speedup | Rationale                                    |
|---------------------|-------------|-----------------------------------------------|
| `avx2_memcpy_like`  | 1.1x - 1.8x | DRAM bandwidth is the ceiling                |
| `avx2_rgb_to_gray`  | 2.0x - 4.0x | Moderate reuse, some compute per pixel       |

**Red flag**: if `avx2_memcpy_like` shows 0% speedup or even slowdown:
- Non-temporal stores (`_mm256_stream_ps`) may not be beneficial on your CPU.
- The scalar `memcpy` in glibc is already highly optimized (rep movsb on recent
  glibc with ERMS).
- Run `./scripts/profile.sh cache build/x86/avx2_memcpy_like` and check if
  the cache miss rate is >20%.

### Layout-transformation kernels

| Benchmark           | AVX2 Speedup | Rationale                                    |
|---------------------|-------------|-----------------------------------------------|
| `avx2_aos_to_soa`   | 2.5x - 4.5x | Shuffle/permute overhead limits speedup      |

### Mixed kernels

| Benchmark           | AVX2 Speedup | Rationale                                    |
|---------------------|-------------|-----------------------------------------------|
| `avx2_conv1d`       | 3.0x - 6.0x | Depends on filter size and stride            |

### Using perf stat to confirm the bottleneck

```bash
./scripts/perf_stat.sh build/x86/avx2_dot_product
```

Interpretation from the `perf_stat.sh` output:

| IPC Range   | Cache Miss Rate | Classification        | Action                                  |
|------------|-----------------|-----------------------|-----------------------------------------|
| > 1.5      | < 3%            | Compute-bound         | SIMD is working well; keep scaling.     |
| 0.5 - 1.5  | 3% - 10%        | Mixed                 | Try cache blocking, prefetch.           |
| < 0.5      | > 10%           | Memory-bound          | Fix data layout; SIMD won't help much.  |

---

## 7. Regression Detection

### How to detect performance regressions

**Method 1: diff two runs**

```bash
# Baseline run
./scripts/run_all_benchmarks.sh
cp benchmarks/latest_results.txt benchmarks/baseline.txt

# ... make changes, rebuild ...

# Changed run
./scripts/run_all_benchmarks.sh
cp benchmarks/latest_results.txt benchmarks/current.txt

# Compare
diff -u benchmarks/baseline.txt benchmarks/current.txt
```

**Method 2: automated comparison script**

Create `benchmarks/compare.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
BASE="${1:?Usage: $0 <baseline.txt> <current.txt>}"
CURR="${2:?}"

echo "=== Regression Check ==="
echo "Comparing $BASE vs $CURR"
echo ""

# Extract ns/el and speedup lines and compare
comm -3 <(grep -E '^\s*(scalar|avx2|avx512|neon|portable|sve)_' "$BASE" | sort) \
        <(grep -E '^\s*(scalar|avx2|avx512|neon|portable|sve)_' "$CURR" | sort)
```

**Method 3: track speedup over time (CSV)**

Append each run to `benchmarks/results.csv` with a timestamp column. Use any
CSV tool (Python pandas, Google Sheets, LibreOffice Calc) to plot speedup
trends.

### What to check after code changes

- [ ] **Correctness first**: `./scripts/run_all_tests.sh` -- all must pass.
- [ ] **No scalar regression**: the scalar baseline ns/el should be unchanged.
- [ ] **SIMD speedup stable**: speedup should not drop more than 5% from baseline.
- [ ] **No compiler de-optimization**: `./scripts/inspect_asm.sh build/x86/avx2_<kernel>`
      and verify SIMD instructions are still generated. Sometimes refactoring
      accidentally breaks the compiler's ability to see vectorizable loops.
- [ ] **Cache behavior**: `./scripts/profile.sh cache` and verify L1 miss rate
      hasn't increased.
- [ ] **Port pressure**: `./scripts/llvm_mca.sh` on the kernel to check that
      the bottleneck port hasn't shifted.

### Common causes of regressions

| Symptom                          | Likely Cause                                      |
|----------------------------------|---------------------------------------------------|
| Speedup drops from 7.8x to 3.5x  | Compiler auto-vectorized the scalar baseline      |
| Speedup drops from 7.8x to 2.0x  | SIMD path not used; scalar fallback active        |
| Scalar ns/el increased by 2x     | Extra copies, alignment change, or debug build    |
| IPC dropped significantly        | New data dependency introduced in the hot loop    |
| Cache miss rate spiked           | Working set grew beyond L2/L3 capacity            |
| `avx2_dot_product` speedup < 4x  | Horizontal sum not vectorized; check assembly     |
| `avx2_gemm_micro` speedup < 3x  | Accumulator unrolling removed or insufficient     |

---

## 8. Export to CSV

### CSV format

Each benchmark binary prints a table.  To convert all results to CSV, pipe
output through a parser or use the project-provided CSV script:

The project already has `benchmarks/results.csv` with columns:

```
benchmark,scalar_ns_per_el,simd_ns_per_el,speedup,scalar_gb_s,simd_gb_s,platform,isa,num_elements,notes
```

### How to generate CSV from a run

```bash
# Run all benchmarks
./scripts/run_all_benchmarks.sh

# Parse latest_results.txt into CSV format (Python quick script)
python3 << 'PYEOF'
import re, sys, csv

benchmarks = [
    "avx2_vector_add", "avx2_relu_clamp", "avx2_dot_product",
    "avx2_reduce_sum", "avx2_layernorm", "avx2_softmax_partial",
    "avx2_conv1d", "avx2_memcpy_like", "avx2_rgb_to_gray",
    "avx2_int8_dot", "avx2_gemm_micro", "avx2_aos_to_soa",
]

with open("benchmarks/latest_results.txt") as f:
    text = f.read()

rows = []
for bn in benchmarks:
    section_start = text.find(f"--- {bn} ---")
    if section_start == -1:
        continue
    section_end = text.find("---", section_start + len(bn) + 10)
    section = text[section_start:section_end] if section_end != -1 else text[section_start:]

    scalar_ns = None
    simd_ns = None
    for line in section.split("\n"):
        if "scalar" in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                try:
                    float(p)
                    scalar_ns = float(p)
                    break
                except ValueError:
                    pass
        elif any(x in line.lower() for x in ["avx2", "neon", "simd"]):
            parts = line.split()
            nums = []
            for p in parts:
                try: nums.append(float(p))
                except ValueError: pass
            if len(nums) >= 4:
                simd_ns = nums[0]
    if scalar_ns and simd_ns:
        rows.append([bn, scalar_ns, simd_ns, scalar_ns/simd_ns])

with open("benchmarks/parsed_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["benchmark", "scalar_ns_per_el", "simd_ns_per_el", "speedup"])
    w.writerows(rows)

print(f"Exported {len(rows)} benchmark rows to benchmarks/parsed_results.csv")
PYEOF
```

### Using CSV in spreadsheets

1. Open `benchmarks/parsed_results.csv` in Google Sheets or LibreOffice Calc
2. Create a bar chart: X-axis = benchmark name, Y-axis = speedup
3. Add a horizontal reference line at 8.0x (theoretical AVX2 f32 max)
4. Add a second horizontal line at 1.0x (no speedup)

This gives you a visual "SIMD efficiency report" at a glance.

### Example CSV output

```csv
benchmark,scalar_ns_per_el,simd_ns_per_el,speedup
avx2_vector_add,0.8235,0.1057,7.79
avx2_relu_clamp,0.7120,0.0918,7.76
avx2_dot_product,1.2340,0.1780,6.93
avx2_reduce_sum,0.9800,0.1420,6.90
avx2_layernorm,2.1000,0.3500,6.00
avx2_softmax_partial,5.6000,1.1200,5.00
avx2_int8_dot,0.4500,0.0580,7.76
avx2_gemm_micro,3.2000,0.4700,6.81
avx2_memcpy_like,0.6200,0.3900,1.59
avx2_rgb_to_gray,1.8000,0.5600,3.21
avx2_aos_to_soa,2.4000,0.6500,3.69
avx2_conv1d,4.5000,0.9800,4.59
```

---

## Quick Reference -- Common Workflows

```bash
# Full fresh-build + benchmark + profile cycle
cd /home/ghr/code/cuda_pytorch/simd

./scripts/build.sh                         # 1. Build everything
./scripts/run_all_tests.sh                 # 2. Verify correctness
./scripts/run_all_benchmarks.sh            # 3. Collect all benchmark data
cat benchmarks/latest_results.txt          # 4. Read results

# Deep-dive into one kernel
./scripts/perf_stat.sh build/x86/avx2_dot_product    # Hardware counters
./scripts/profile.sh annotate build/x86/avx2_dot_product  # Hot instructions
./scripts/llvm_mca.sh build/x86/avx2_dot_product avx2_dot_product_f32  # Static analysis
```

---

*Generated by the SIMD tutorial benchmarking framework.  Update this report
with your own system information and measured results after running the
benchmarks.*
