# SVE Programming Guide: Vector-Length Agnostic SIMD for Production

## 1. SVE Concepts (Deep Dive)

### 1.1 The Problem SVE Solves

NEON (ARM's original SIMD) has a fundamental limitation: **the vector width is fixed at 128 bits**. This creates three severe problems for production software:

1. **Code is not future-proof**: Code optimized for 128-bit NEON leaves half the throughput unused on 256-bit hardware (Graviton3) and 75% unused on 512-bit hardware (A64FX).
2. **Multiple code paths required**: Each vector width demands its own unroll factor, tail loop, and tuning parameters.
3. **Tail loops are bug-prone**: The scalar tail loop is the single most common source of SIMD bugs -- off-by-one errors, uninitialized reads, and buffer overflows.

SVE eliminates all three problems via **Vector Length Agnosticism (VLA)**.

### 1.2 What VLA Means in Practice

A single SVE binary runs identically -- without recompilation -- on implementations ranging from 128 bits (minimum) to 2048 bits (maximum). The same loop handles all vector widths and all iteration counts:

```
NEON (fixed 128-bit):
  for (i = 0; i <= n - 4; i += 4) { ... }   // main loop: 4 floats
  for (; i < n; i++) { ... }                  // tail: scalar (bug-prone)

SVE (VLA):
  uint64_t i = 0;
  while (i < n) {
      svbool_t pg = svwhilelt_b32(i, n);     // which lanes are valid?
      ...                                      // main loop + tail = ONE loop
      i += svcntw();                           // actual lane count on this CPU
  }
```

### 1.3 How SVE Achieves VLA: Predicate Registers

The key insight: **predicate registers control per-lane execution**. A predicate register holds 1 bit per vector lane. When a lane's bit is 1, the operation executes on that lane; when 0, it is skipped (with behavior depending on the `_m`, `_z`, or `_x` suffix).

For a 256-bit SVE implementation working on `float32` data:

```
Z0 (256-bit):  [ lane7 | lane6 | lane5 | lane4 | lane3 | lane2 | lane1 | lane0 ]
P0 (predicate, 1 bit per byte of Z0):  each float lane consumes 1 predicate bit
   bit:          [   0   |   1   |   1   |   0   |   1   |   1   |   1   |   1   ]
                    ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑
                  skip    keep    keep    skip    keep    keep    keep    keep
```

The predicate logic means:
- The loop counter `i` can be any value; `svwhilelt_b32(i, n)` generates exactly the right predicate for the remaining elements.
- No scalar tail loop is ever needed.
- The same binary automatically takes advantage of wider vectors on newer hardware.

### 1.4 Current SVE Hardware Landscape

| Implementation | Vector Length | Device Type | SVE2? | Notes |
|---------------|--------------|-------------|-------|-------|
| **AWS Graviton3** | 256-bit (SVE 256) | Cloud server (Neoverse V1) | No (SVE1) | Widest deployment for SVE cloud workloads |
| **Fujitsu A64FX** | 512-bit | HPC (Fugaku supercomputer) | No (SVE1) | Largest vector width; strong HBM2 bandwidth |
| **NVIDIA Grace** | 256-bit | Cloud/HPC (Neoverse V2) | SVE2 | ARMv9, inside Grace-Hopper and Grace-Blackwell |
| **Apple M4** | 128-bit (SME) | Mobile/laptop | No | Apple implements SME (Streaming SVE), not standard SVE |
| **AWS Graviton4** | 256-bit | Cloud server (Neoverse V2) | SVE2 | ARMv9 with SVE2 + NEON |
| **QEMU user-mode** | Configurable (128-2048) | Emulation | Optional | Export `SVE_VECTOR_LENGTH=n` to test different VLs |
| **Android flagships** | N/A (NEON only) | Mobile | No | A76/A78/A715/X3: NEON only; SVE expected in Cortex-X5+ |

Key takeaway: as of 2024-2025, SVE is primarily a **server-side technology**. Mobile adoption will come with SVE2-mandatory ARMv9 cores.

### 1.5 SVE vs Fixed-Width SIMD: Fundamental Tradeoffs

| Aspect | NEON (fixed 128-bit) | SVE (VLA) |
|--------|---------------------|-----------|
| **Code portability** | Requires per-width variants | Single binary, all widths |
| **Performance predictability** | Exact (known unroll factor) | Approximate (unknown VL at compile time) |
| **Tail handling** | Manual scalar tail loop | Automatic via predicates |
| **Unroll tuning** | Deterministic | Requires runtime VL detection |
| **Debugging** | Mature tools | Fewer tools, but improving |
| **Compiler auto-vec** | Mature in GCC/Clang | Improving but not yet equal |
| **Gather/Scatter** | Limited (`LD1` + `TBL`) | First-class via `svld1_gather` |
| **Fault tolerance** | None | Fault-suppressing loads (`svld1`) |

---

## 2. Predicate Programming (The Core of SVE)

Predicates are the fundamental abstraction that makes VLA possible. This section covers every aspect of predicate programming with complete, compilable examples.

### 2.1 The `svbool_t` Type

```c
#include <arm_sve.h>

// svbool_t is an opaque type representing VL/8 bits of predicate information.
// Each byte of the vector register has one corresponding predicate bit.
// For 256-bit SVE: svbool_t contains 32 bits (one per byte of Z0).

svbool_t pg;  // an opaque predicate -- never access its bits directly
```

### 2.2 Predicate Creation Patterns

```c
#include <arm_sve.h>

// --- Pattern 1: Loop counter predicate (MOST IMPORTANT) ---
// svwhilelt_b32(i, n): For lane k, pg[k] = 1 if (i + k) < n, else 0.
// This is the canonical SVE loop predicate. It handles the main body AND
// the tail in one expression -- no separate scalar tail loop.
uint64_t i = 0;
uint64_t n = 100;
svbool_t pg = svwhilelt_b32(i, n);  // generates: [1,1,1,...,0,0] depending on VL

// --- Pattern 2: All-true predicate ---
svbool_t pg_all = svptrue_b32();           // all lanes enabled
// Equivalent to:
svbool_t pg_all = svptrue_pat_b32(SV_ALL);

// --- Pattern 3: Pattern-based predicates (known multiples) ---
// When you KNOW n is a multiple of the vector width, avoid whilelt overhead:
svbool_t pg_vl1  = svptrue_pat_b32(SV_VL1);   // only lane 0 enabled
svbool_t pg_vl2  = svptrue_pat_b32(SV_VL2);   // lanes 0-1
svbool_t pg_vl3  = svptrue_pat_b32(SV_VL3);   // lanes 0-2
svbool_t pg_vl4  = svptrue_pat_b32(SV_VL4);   // lanes 0-3  (== 128-bit for float32)
svbool_t pg_vl5  = svptrue_pat_b32(SV_VL5);
svbool_t pg_vl6  = svptrue_pat_b32(SV_VL6);
svbool_t pg_vl7  = svptrue_pat_b32(SV_VL7);
svbool_t pg_vl8  = svptrue_pat_b32(SV_VL8);   // lanes 0-7  (== 256-bit for float32)

// --- Pattern 4: Element-type specific ptrue ---
svbool_t pg_b8  = svptrue_b8();   // all lanes for byte elements
svbool_t pg_b16 = svptrue_b16();  // all lanes for halfword elements
svbool_t pg_b32 = svptrue_b32();  // all lanes for word elements
svbool_t pg_b64 = svptrue_b64();  // all lanes for doubleword elements
```

### 2.3 The `_m`, `_z`, `_x` Suffixes: Controlling Inactive Lanes

This is the most critical design decision in every SVE instruction. Choosing the wrong suffix silently introduces bugs:

```c
#include <arm_sve.h>

// Consider: svfloat32_t result = svadd_f32_?(pg, op1, op2);
//
// For active lanes   (pg[k] == 1): result[k] = op1[k] + op2[k]    (always)
// For inactive lanes (pg[k] == 0): behavior depends on the suffix

// _m (merge): inactive lanes get op1's value -- SAFE for accumulators
svfloat32_t result_m = svadd_f32_m(pg, op1, op2);
// Lane 0 (active):   result[0] = op1[0] + op2[0]
// Lane 1 (inactive): result[1] = op1[1]            <-- preserves op1

// _z (zero): inactive lanes get zero -- SAFE for fresh results
svfloat32_t result_z = svadd_f32_z(pg, op1, op2);
// Lane 0 (active):   result[0] = op1[0] + op2[0]
// Lane 1 (inactive): result[1] = 0.0f               <-- zeroed

// _x (don't care): inactive lanes are undefined -- FASTEST, but risky
svfloat32_t result_x = svadd_f32_x(pg, op1, op2);
// Lane 0 (active):   result[0] = op1[0] + op2[0]
// Lane 1 (inactive): result[1] = ???                <-- ANY value, including NaN
```

**Rule of thumb:**
- Use `_m` when the first operand is an accumulator you must preserve (e.g., `acc = svmla_f32_m(pg, acc, a, b)`).
- Use `_z` when creating a new masked result for storage.
- Use `_x` only when inactive lane values are immediately overwritten by a subsequent predicated operation.

### 2.4 Loop Predicate: `svwhilelt_b32` in Depth

```c
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>

// The canonical SVE loop pattern: ONE loop, no scalar tail.
// This works correctly for ANY n (including n=0, n=1, n < VL/32).
void saxpy_sve(const float *x, const float *y, float *out,
               uint64_t n, float alpha) {
    uint64_t i = 0;
    svfloat32_t valpha = svdup_f32(alpha);

    while (i < n) {
        // whilelt generates a predicate: for lane k, pg[k] = (i + k) < n
        // On the last iteration with n=10 and VL=8:
        //   pg = [1,1,0,0,0,0,0,0]  (only lanes 0-1 active)
        svbool_t pg = svwhilelt_b32(i, n);

        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svfloat32_t vy = svld1_f32(pg, &y[i]);

        // mul uses _z: inactive lanes produce 0, since they'll be added to vy
        // add uses _m: inactive lanes preserve vy (the first operand)
        svfloat32_t vtmp = svmul_f32_z(pg, vx, valpha);
        svfloat32_t vout = svadd_f32_m(pg, vy, vtmp);

        svst1_f32(pg, &out[i], vout);

        i += svcntw();  // advance by the actual number of float lanes
    }
}
```

### 2.5 Comparison Predicates

```c
#include <arm_sve.h>

// All comparison intrinsics follow the pattern:
//   svbool_t svcmpXX_f32(svbool_t pg, svfloat32_t a, svfloat32_t b);
//
// The first argument (pg) is the governing predicate:
//   - If pg[k] == 0, the comparison result for lane k is forced to 0.
//   - If pg[k] == 1, the comparison is performed normally.

void compare_examples(svbool_t pg, svfloat32_t a, svfloat32_t b) {
    svbool_t gt  = svcmpgt_f32(pg, a, b);   // a > b
    svbool_t ge  = svcmpge_f32(pg, a, b);   // a >= b
    svbool_t eq  = svcmpeq_f32(pg, a, b);   // a == b
    svbool_t lt  = svcmplt_f32(pg, a, b);   // a < b
    svbool_t le  = svcmple_f32(pg, a, b);   // a <= b
    svbool_t ne  = svcmpne_f32(pg, a, b);   // a != b
    (void)gt; (void)ge; (void)eq; (void)lt; (void)le; (void)ne;
}

// Using comparison predicates for conditional selection (svsel):
svfloat32_t relu_sve(svbool_t pg, svfloat32_t x) {
    svfloat32_t zero = svdup_f32(0.0f);
    svbool_t positive = svcmpge_f32(pg, x, zero);   // x >= 0 ?
    return svsel_f32(positive, x, zero);              // if true: x, else: 0
}

// Clipping values to a range [lo, hi]:
void clip_sve(const float *src, float *dst, uint64_t n,
              float lo, float hi) {
    uint64_t i = 0;
    svfloat32_t vlo = svdup_f32(lo);
    svfloat32_t vhi = svdup_f32(hi);

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &src[i]);

        svbool_t too_low  = svcmplt_f32(pg, v, vlo);
        svbool_t too_high = svcmpgt_f32(pg, v, vhi);

        v = svsel_f32(too_low,  vlo, v);    // clamp low
        v = svsel_f32(too_high, vhi, v);    // clamp high

        svst1_f32(pg, &dst[i], v);
        i += svcntw();
    }
}
```

### 2.6 Predicate Logical Operations

```c
#include <arm_sve.h>

// SVE provides bitwise logical operations on predicates.
// These operate at byte-granularity (hence the _b_z suffix).

void predicate_logic_example(svbool_t p1, svbool_t p2, svbool_t p3) {
    // AND: result = p1 & p2, with p3 as governing predicate
    svbool_t p_and = svand_b_z(p3, p1, p2);
    // For each bit: if p3[bit] == 1 → p_and[bit] = p1[bit] & p2[bit]
    //               if p3[bit] == 0 → p_and[bit] = 0

    // OR: result = p1 | p2
    svbool_t p_or  = svorr_b_z(p3, p1, p2);

    // XOR: result = p1 ^ p2
    svbool_t p_xor = sveor_b_z(p3, p1, p2);

    // NOT: result = ~p2
    svbool_t p_not = svnot_b_z(p3, p2);

    // NAND: result = ~(p1 & p2)
    svbool_t p_nand = svnand_b_z(p3, p1, p2);

    // NOR: result = ~(p1 | p2)
    svbool_t p_nor = svnor_b_z(p3, p1, p2);

    // ORN: result = p1 | ~p2
    svbool_t p_orn = svorn_b_z(p3, p1, p2);

    (void)p_and; (void)p_or; (void)p_xor; (void)p_not;
    (void)p_nand; (void)p_nor; (void)p_orn;
}

// Practical example: combining two conditions to select elements
// Select elements where (x > 0) AND (y < 100)
void select_combined_sve(const float *x, const float *y, float *dst,
                         uint64_t n) {
    uint64_t i = 0;
    svfloat32_t zero = svdup_f32(0.0f);
    svfloat32_t hundred = svdup_f32(100.0f);

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svfloat32_t vy = svld1_f32(pg, &y[i]);

        svbool_t gt_zero = svcmpgt_f32(pg, vx, zero);
        svbool_t lt_hundred = svcmplt_f32(pg, vy, hundred);

        // Combine: both conditions must be true
        svbool_t combined = svand_b_z(pg, gt_zero, lt_hundred);

        svst1_f32(combined, &dst[i], vx);   // store x[i] where condition holds
        i += svcntw();
    }
}
```

---

## 3. Vector-Length Agnostic Programming Patterns

### 3.1 Count-Up Loop (Standard Pattern)

```c
#include <arm_sve.h>

// Standard count-up with whilelt: simple, readable, works for all n and VL.
void vec_add_countup(const float *a, const float *b, float *c, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        svfloat32_t vc = svadd_f32_m(pg, va, vb);
        svst1_f32(pg, &c[i], vc);
        i += svcntw();
    }
}
```

### 3.2 Count-Down Loop (Often Faster)

```c
#include <arm_sve.h>

// Count-down loop pattern:
// Advantages:
//   1. Loop termination uses `subs + b.ne` (single instruction, zero flag)
//      vs. count-up's `cmp + b.lt` (two instructions).
//   2. Better for out-of-order CPUs: the decrement is on the critical path
//      of address calculation, reducing register pressure.
//   3. No need for an extra induction variable comparison per iteration.

void vec_add_countdown(const float *a, const float *b, float *c, int64_t n) {
    int64_t i = n;  // start at n, count down to 0

    do {
        svbool_t pg = svwhilelt_b32(0, i);   // pg[k] = 1 if k < i
        i -= svcntw();                         // decrement first for correct indexing

        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        svfloat32_t vc = svadd_f32_m(pg, va, vb);
        svst1_f32(pg, &c[i], vc);
    } while (i > 0);
}
```

### 3.3 Runtime Vector Length Detection

```c
#include <arm_sve.h>
#include <stdio.h>

// svcntb(), svcnth(), svcntw(), svcntd() return the number of
// elements per vector at runtime for each element width.

void report_vector_length(void) {
    uint64_t bytes      = svcntb();   // VL / 8
    uint64_t halfwords  = svcnth();   // VL / 16
    uint64_t words      = svcntw();   // VL / 32
    uint64_t doublewords = svcntd();  // VL / 64

    printf("SVE vector length: %lu bits\n", bytes * 8);
    printf("  %lu bytes per vector\n", bytes);
    printf("  %lu int32 per vector\n", words);
    printf("  %lu double per vector\n", doublewords);
}

// svlen() returns the total vector length in bits.
// Equivalent to svcntb() * 8 but available as a single intrinsic.
uint64_t vl_bits(void) {
    return svlen();  // returns VL in bits (128, 256, 512, etc.)
}

// The inline assembly equivalent of svcntb():
//   rdvl x0, #1   -- read vector length in bytes, multiplied by the immediate
static inline uint64_t rdvl_bytes(void) {
    uint64_t vl;
    __asm__ volatile("rdvl %0, #1" : "=r"(vl));
    return vl;
}
```

### 3.4 Writing Code Optimal Across All SVE Widths

```c
#include <arm_sve.h>
#include <stdlib.h>

// Problem: the optimal unroll factor depends on VL.
// Solution: detect VL at runtime and select an unroll strategy.

// Strategy 1: Use svcntw() to naturally adapt to any VL.
// The loop body naturally processes VL/32 elements per iteration.
void adaptive_vec_add(const float *a, const float *b, float *c, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        svst1_f32(pg, &c[i], svadd_f32_m(pg, va, vb));
        i += svcntw();
    }
}

// Strategy 2: Manual unrolling tuned to the VL.
// After runtime detection, pick a specialization.
void vec_add_tuned(const float *a, const float *b, float *c, uint64_t n) {
    uint64_t words = svcntw();  // VL / 32

    if (words == 4) {
        // VL=128: NEON-equivalent, unroll by 2 or 4
        uint64_t i = 0;
        for (; i + 7 < n; i += 8) {
            svbool_t pg0 = svwhilelt_b32(i,     n);
            svbool_t pg1 = svwhilelt_b32(i + 4, n);
            svfloat32_t va0 = svld1_f32(pg0, &a[i]);
            svfloat32_t va1 = svld1_f32(pg1, &a[i + 4]);
            svfloat32_t vb0 = svld1_f32(pg0, &b[i]);
            svfloat32_t vb1 = svld1_f32(pg1, &b[i + 4]);
            svst1_f32(pg0, &c[i],     svadd_f32_m(pg0, va0, vb0));
            svst1_f32(pg1, &c[i + 4], svadd_f32_m(pg1, va1, vb1));
        }
        for (; i < n; i++) {
            c[i] = a[i] + b[i];
        }
        return;
    }

    if (words == 8) {
        // VL=256: Graviton3, unroll by 2
        uint64_t i = 0;
        for (; i + 15 < n; i += 16) {
            svbool_t pg0 = svwhilelt_b32(i,     n);
            svbool_t pg1 = svwhilelt_b32(i + 8, n);
            svfloat32_t va0 = svld1_f32(pg0, &a[i]);
            svfloat32_t va1 = svld1_f32(pg1, &a[i + 8]);
            svfloat32_t vb0 = svld1_f32(pg0, &b[i]);
            svfloat32_t vb1 = svld1_f32(pg1, &b[i + 8]);
            svst1_f32(pg0, &c[i],     svadd_f32_m(pg0, va0, vb0));
            svst1_f32(pg1, &c[i + 8], svadd_f32_m(pg1, va1, vb1));
        }
        for (; i < n; i++) {
            c[i] = a[i] + b[i];
        }
        return;
    }

    // VL=512 or larger, or unknown: use generic VLA loop
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        svst1_f32(pg, &c[i], svadd_f32_m(pg, va, vb));
        i += svcntw();
    }
}
```

### 3.5 Element Count vs Indexing Gotchas

```c
#include <arm_sve.h>

// CRITICAL: svcntw() returns an element count (number of float32 lanes).
// When using it to advance a pointer of type float*, no scaling needed.
// When using it to advance a pointer of type uint8_t*, you must scale manually.

void element_scaling_example(const float *src, float *dst, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        // For float*: i increments by svcntw() (elements of float32)
        // ptr + i is automatically scaled by sizeof(float) by the compiler
        svfloat32_t v = svld1_f32(pg, &src[i]);
        svst1_f32(pg, &dst[i], v);
        i += svcntw();  // correct: i counts float32 elements
    }
}

// When working with bytes, you need svcntb() and byte-wide whilelt:
void memset_like_sve(uint8_t *dst, uint8_t value, uint64_t n) {
    uint64_t i = 0;
    svuint8_t fill = svdup_u8(value);
    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);  // b8 variant for byte counting
        svst1_u8(pg, &dst[i], fill);
        i += svcntb();  // bytes per vector, matches uint8_t* indexing
    }
}
```

---

## 4. Predicated Load/Store

### 4.1 Basic Predicated Load (`svld1`)

```c
#include <arm_sve.h>

// svld1(pg, ptr): loads from memory using a predicate.
//   Active lanes (pg[k]==1): loaded from *(&ptr[k]).
//   Inactive lanes (pg[k]==0): UNSPECIFIED behavior -- may load, may not.
//
// CRITICAL SAFETY FEATURE: svld1 is FAULT-SUPPRESSING.
// The CPU guarantees it will NOT raise a segmentation fault for
// addresses in inactive lanes. This is what makes the tail loop
// pattern work: inactive lanes can point beyond the buffer boundary.

void load_store_basics(const float *src, float *dst, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        // Safe: inactive lanes may point past the buffer without faulting
        svfloat32_t v = svld1_f32(pg, &src[i]);

        // Only active lanes are stored; inactive lane stores are suppressed
        svst1_f32(pg, &dst[i], v);

        i += svcntw();
    }
}
```

### 4.2 First-Fault Load (`svldff1`) for String Processing

```c
#include <arm_sve.h>
#include <stddef.h>

// svldff1: First-Faulting Load.
// Loads elements sequentially. If any element would cause a page fault,
// the load STOPS at that element. The predicate is updated to show which
// lanes were successfully loaded.
//
// This is perfect for:
//   - strlen() / strcpy() (null-terminated strings)
//   - Reading from mmap'd regions that might be partially mapped
//   - Speculative loads ahead of a processing loop

// SVE strlen: find the terminating null byte
size_t sve_strlen(const char *str) {
    const char *ptr = str;
    svbool_t pg = svptrue_b8();   // start with all lanes active

    while (1) {
        // First-fault load: loads VL bytes, updating predicate on fault
        svuint8_t v = svldff1_u8(pg, (const uint8_t *)ptr);

        // Find the first zero byte (null terminator)
        svbool_t nulls = svcmpeq_u8(pg, v, svdup_u8(0));

        // Check if any lane had a match
        if (svptest_any(pg, nulls)) {
            // Found: report the position using svbrkb (bit reversal)
            // svbrkb finds the first true bit from LSB
            svbool_t first_zero = svbrkb_b_z(pg, pg, nulls);
            // Count the number of bytes before the first zero
            uint64_t len = svcntp_b8(pg, svnot_b_z(pg, first_zero));
            return ptr - str + len;
        }

        ptr += svcntb();
    }
}

// SVE memchr: find first occurrence of byte c in the first n bytes
const void *sve_memchr(const void *s, int c, size_t n) {
    const uint8_t *ptr = (const uint8_t *)s;
    svuint8_t target = svdup_u8((uint8_t)c);
    uint64_t i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);
        svuint8_t v = svld1_u8(pg, &ptr[i]);  // fault-suppressing for tail

        svbool_t matches = svcmpeq_u8(pg, v, target);
        if (svptest_any(pg, matches)) {
            svbool_t first_hit = svbrkb_b_z(pg, pg, matches);
            uint64_t offset = svcntp_b8(pg, svnot_b_z(pg, first_hit));
            return &ptr[i + offset];
        }
        i += svcntb();
    }
    return NULL;
}
```

### 4.3 Non-Temporal Store Hints

```c
#include <arm_sve.h>

// Non-temporal stores bypass the cache hierarchy for write-once data.
// Use when: dst[] is large and will NOT be read again soon.
// Example: memset-like initialization, output buffers in streaming pipelines.

void nt_memcpy(const float *src, float *dst, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &src[i]);

        // Non-temporal store: does not pollute cache
        svstnt1_f32(pg, &dst[i], v);

        i += svcntw();
    }
}

// Gather with non-temporal store: useful for scatter patterns
void nt_scatter(const float *src, const int32_t *indices, float *dst,
                uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svint32_t vidx = svld1_s32(pg, &indices[i]);
        svfloat32_t v = svld1_gather_s32index_f32(pg, src, vidx);

        // Non-temporal scatter store
        svstnt1_scatter_s32index_f32(pg, dst, vidx, v);

        i += svcntw();
    }
}
```

### 4.4 Contiguous vs Gather/Scatter Loads

```c
#include <arm_sve.h>

// svld1: contiguous load -- fastest, best cache behavior
void contiguous_load(float *dst, const float *src, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &src[i]);   // simple, fast
        svst1_f32(pg, &dst[i], v);
        i += svcntw();
    }
}

// svld1_gather: indexed gather -- flexible, but slower
// Useful for: sparse matrix operations, permutation, table lookups
void gather_load_example(float *dst, const float *table,
                         const int32_t *indices, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svint32_t vidx = svld1_s32(pg, &indices[i]);

        // Gather from table[*indices] -- non-contiguous access
        svfloat32_t v = svld1_gather_s32index_f32(pg, table, vidx);

        svst1_f32(pg, &dst[i], v);
        i += svcntw();
    }
}
```

---

## 5. Reduction with SVE

### 5.1 Horizontal Reduction Primitives

```c
#include <arm_sve.h>

// svaddv, svmaxv, svminv: horizontal reductions across all active lanes.
// These reduce a vector to a single scalar value.

float sve_sum_reduce(svfloat32_t vec) {
    svbool_t pg = svptrue_b32();
    return svaddv_f32(pg, vec);  // sum of all lanes in vec
}

float sve_max_reduce(svfloat32_t vec) {
    svbool_t pg = svptrue_b32();
    return svmaxv_f32(pg, vec);  // maximum of all lanes in vec
}

float sve_min_reduce(svfloat32_t vec) {
    svbool_t pg = svptrue_b32();
    return svminv_f32(pg, vec);  // minimum of all lanes in vec
}
```

### 5.2 Complete Vector Sum (Single Accumulator)

```c
#include <arm_sve.h>

// Simple SVE sum reduction with scalar tail handling via predicates.
// Works for any n and any VL.
float sve_sum(const float *data, uint64_t n) {
    svfloat32_t acc = svdup_f32(0.0f);
    uint64_t i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &data[i]);

        // _m preserves accumulator for inactive lanes
        acc = svadd_f32_m(pg, acc, v);

        i += svcntw();
    }

    // Horizontal reduction: sum all lanes of acc
    return svaddv_f32(svptrue_b32(), acc);
}
```

### 5.3 Tree Reduction for Better ILP

```c
#include <arm_sve.h>

// Problem with single-accumulator sum: it's a dependency chain.
//   acc = acc + v[0]; acc = acc + v[1]; ... (sequential)
//
// Tree reduction uses multiple accumulators and reduces them
// at the end, breaking the dependency chain for better ILP.
// Typically 2x to 4x accumulators provide good results.

float sve_sum_tree4(const float *data, uint64_t n) {
    // 4 accumulators for instruction-level parallelism
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);

    uint64_t i = 0;
    uint64_t step = svcntw();

    // Process 4 vectors per iteration
    for (; i + 4 * step <= n; i += 4 * step) {
        svbool_t pg = svptrue_b32();  // known multiple: all lanes active
        acc0 = svadd_f32_m(pg, acc0, svld1_f32(pg, &data[i]));
        acc1 = svadd_f32_m(pg, acc1, svld1_f32(pg, &data[i + step]));
        acc2 = svadd_f32_m(pg, acc2, svld1_f32(pg, &data[i + 2 * step]));
        acc3 = svadd_f32_m(pg, acc3, svld1_f32(pg, &data[i + 3 * step]));
    }

    // Handle remaining full vectors
    for (; i + step <= n; i += step) {
        svbool_t pg = svptrue_b32();
        acc0 = svadd_f32_m(pg, acc0, svld1_f32(pg, &data[i]));
    }

    // Tail elements
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        acc0 = svadd_f32_m(pg, acc0, svld1_f32(pg, &data[i]));
    }

    // Merge accumulators
    svbool_t pg = svptrue_b32();
    acc0 = svadd_f32_m(pg, acc0, acc1);
    acc0 = svadd_f32_m(pg, acc0, acc2);
    acc0 = svadd_f32_m(pg, acc0, acc3);

    return svaddv_f32(pg, acc0);
}
```

### 5.4 Folding Reduction: `svadda`

```c
#include <arm_sve.h>

// svadda: stream-based folding reduction.
// Reduces the source vector into a scalar, and returns a predicate
// indicating which lanes still need to be processed.
//
// This is useful for "early exit" reductions and combining
// partial results across loop iterations.

void folding_reduce_example(void) {
    float scalar = 0.0f;
    svfloat32_t data = svdup_f32(1.0f);
    svbool_t pg = svptrue_b32();

    // svadda_f32: accumulate into scalar, return remaining predicate
    // On VL=256 (8 floats): first call reduces some lanes,
    // remaining predicate shows which lanes are left to process.
    svbool_t remaining = svadda_f32(pg, &scalar, data);

    // scalar now contains partial sum; remaining shows unprocessed lanes
    if (svptest_any(svptrue_b32(), remaining)) {
        // more work to do -- call svadda again with the remaining predicate
        svadda_f32(remaining, &scalar, data);
    }

    (void)scalar;
}
```

### 5.5 Complete Dot Product Example

```c
#include <arm_sve.h>

// Dot product: sum(a[i] * b[i]) for i in [0, n)
// Uses tree reduction for ILP. Handles any n and any VL.

float sve_dot_product(const float *a, const float *b, uint64_t n) {
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);

    uint64_t i = 0;
    uint64_t step = svcntw();

    // 4-way unrolled main body
    for (; i + 4 * step <= n; i += 4 * step) {
        svbool_t pg = svptrue_b32();

        svfloat32_t va0 = svld1_f32(pg, &a[i]);
        svfloat32_t vb0 = svld1_f32(pg, &b[i]);
        svfloat32_t va1 = svld1_f32(pg, &a[i + step]);
        svfloat32_t vb1 = svld1_f32(pg, &b[i + step]);
        svfloat32_t va2 = svld1_f32(pg, &a[i + 2 * step]);
        svfloat32_t vb2 = svld1_f32(pg, &b[i + 2 * step]);
        svfloat32_t va3 = svld1_f32(pg, &a[i + 3 * step]);
        svfloat32_t vb3 = svld1_f32(pg, &b[i + 3 * step]);

        // FMA: acc = acc + va * vb
        acc0 = svmla_f32_m(pg, acc0, va0, vb0);
        acc1 = svmla_f32_m(pg, acc1, va1, vb1);
        acc2 = svmla_f32_m(pg, acc2, va2, vb2);
        acc3 = svmla_f32_m(pg, acc3, va3, vb3);
    }

    // Remaining full vectors
    for (; i + step <= n; i += step) {
        svbool_t pg = svptrue_b32();
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        acc0 = svmla_f32_m(pg, acc0, va, vb);
    }

    // Tail
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        acc0 = svmla_f32_m(pg, acc0, va, vb);
    }

    // Merge accumulators
    svbool_t pg = svptrue_b32();
    acc0 = svadd_f32_m(pg, acc0, acc1);
    acc0 = svadd_f32_m(pg, acc0, acc2);
    acc0 = svadd_f32_m(pg, acc0, acc3);

    return svaddv_f32(pg, acc0);
}

// Integer dot product using SVE's native svdot instruction.
// Computes int8 * int8 -> int32 accumulation, 4x throughput vs float.
// This is the SVE equivalent of NEON's vdotq_s32.
int32_t sve_dot_product_i8(const int8_t *a, const int8_t *b, uint64_t n) {
    svint32_t acc = svdup_s32(0);
    uint64_t i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);
        svint8_t va = svld1_s8(pg, &a[i]);
        svint8_t vb = svld1_s8(pg, &b[i]);

        // svdot: int8*int8 + int32 -> int32
        // Works on groups of 4 bytes per 32-bit lane
        acc = svdot_s32(acc, va, vb);

        i += svcntb();
    }

    return svaddv_s32(svptrue_b32(), acc);
}
```

---

## 6. SVE2 Enhancements

SVE2 (ARMv9 mandatory) significantly expands the instruction set, adding support for DSP workloads, cryptography, and complex integer arithmetic.

### 6.1 Complex Integer Multiply (DSP)

```c
#include <arm_sve.h>

// SVE2 introduces complex integer multiply instructions.
// These are essential for software-defined radio (SDR), radar signal
// processing, and any workload involving complex arithmetic.

// Complex multiply: (a_re + j*a_im) * (b_re + j*b_im)
//   = (a_re*b_re - a_im*b_im) + j*(a_re*b_im + a_im*b_re)
// SVE2's svcmla computes this in one instruction per component.

void complex_multiply_sve2(const int16_t *a_re, const int16_t *a_im,
                            const int16_t *b_re, const int16_t *b_im,
                            int16_t *c_re, int16_t *c_im, uint64_t n) {
    svint16_t acc_re = svdup_s16(0);
    svint16_t acc_im = svdup_s16(0);
    uint64_t i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b16(i, n);
        svint16_t va_re = svld1_s16(pg, &a_re[i]);
        svint16_t va_im = svld1_s16(pg, &a_im[i]);
        svint16_t vb_re = svld1_s16(pg, &b_re[i]);
        svint16_t vb_im = svld1_s16(pg, &b_im[i]);

        // svcmla: complex multiply-add
        // Rotation = 0: real part of result
        // Rotation = 90: imaginary part of result
        acc_re = svcmla_s16(acc_re, va_re, vb_re, 0);   // real
        acc_re = svcmla_s16(acc_re, va_im, vb_im, 90);  // subtract im*im
        acc_im = svcmla_s16(acc_im, va_re, vb_im, 0);   // im = re*im
        acc_im = svcmla_s16(acc_im, va_im, vb_re, 90);  //     + im*re

        svst1_s16(pg, &c_re[i], acc_re);
        svst1_s16(pg, &c_im[i], acc_im);
        i += svcnth();
    }
}
```

### 6.2 Multi-Vector Operations

```c
#include <arm_sve.h>

// SVE2 adds multi-vector loads, which load 2, 3, or 4 vectors
// in a single instruction. This is the primary way to increase
// memory throughput in SVE2.

// svld2: load 2 interleaved vectors (deinterleave on load)
void deinterleave_complex_sve2(const float *interleaved,
                                float *re, float *im, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        // Load 2 interleaved vectors: {re0, im0, re1, im1, ...}
        svfloat32x2_t pair = svld2_f32(pg, &interleaved[2 * i]);

        // Extract the two vectors
        svfloat32_t vre = svget2_f32(pair, 0);
        svfloat32_t vim = svget2_f32(pair, 1);

        svst1_f32(pg, &re[i], vre);
        svst1_f32(pg, &im[i], vim);

        i += svcntw();
    }
}

// svmul_lane: multiply by a specific lane of another vector
// Useful for applying scalar factors from a vector of coefficients.
void apply_channel_gains_sve2(const float *src, float *dst,
                               const float *gains, uint64_t n,
                               uint64_t num_channels) {
    // Load channel gains into a vector
    svbool_t pg_gain = svwhilelt_b32(0, num_channels);
    svfloat32_t vgains = svld1_f32(pg_gain, gains);

    uint64_t i = 0;
    uint64_t channel = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &src[i]);

        // Multiply by the gain for this channel (lane 'channel' of vgains)
        svfloat32_t result = svmul_lane_f32(v, vgains, channel);

        svst1_f32(pg, &dst[i], result);
        i += svcntw();
        channel = (channel + 1) % num_channels;
    }
}
```

### 6.3 Character Match and String Operations

```c
#include <arm_sve.h>

// svmatch: find characters in a set (like strspn/strcspn).
// Compares each byte in the source against a set of characters.

// Check if all characters in str[0:n] are in the allowed set.
bool sve_strspn_check(const uint8_t *str, const uint8_t *charset,
                      size_t str_len, size_t set_len) {
    // Load the character set into a vector (up to VL bytes)
    svbool_t pg_set = svwhilelt_b8(0, set_len);
    svuint8_t vset = svld1_u8(pg_set, charset);

    size_t i = 0;
    while (i < str_len) {
        svbool_t pg = svwhilelt_b8(i, str_len);
        svuint8_t vstr = svld1_u8(pg, &str[i]);

        // svmatch: for each byte in vstr, returns true if it matches
        // any byte in vset. Returns false if any byte does NOT match.
        svbool_t matched = svmatch_u8(pg, vstr, vset);

        // If not all bytes matched, return false
        if (!svptest_any(pg, svnot_b_z(pg, matched))) {
            // svptest_any returns false: meaning NOT(any unmatched) => all matched
            // This is a double negative; let's check more directly:
        }
        if (svptest_any(pg, svnot_b_z(pg, matched))) {
            return false;  // at least one character not in the set
        }

        i += svcntb();
    }
    return true;
}
```

### 6.4 Histogram Instructions

```c
#include <arm_sve.h>

// SVE2 introduces svhistcnt and svhistseg for building histograms.
// These are critical for image processing (histogram equalization),
// color correction, and any bucketing-based algorithm.

// Build a 256-bin histogram of uint8_t values.
// Returns the count of elements in each bin.
void sve_histogram(const uint8_t *data, uint64_t n,
                   uint32_t *histogram /* 256 entries */) {
    // Initialize histogram to zero
    for (int i = 0; i < 256; i++) histogram[i] = 0;

    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);
        svuint8_t v = svld1_u8(pg, &data[i]);

        // svhistcnt: count occurrences of each value in v.
        // Each call increments histogram[value] for all active lanes.
        // On some implementations, this uses a 4x loop over the bins.
        for (int bin = 0; bin < 256; bin += 32) {
            // Process bins in chunks: the exact API depends on the
            // compiler's intrinsic definition.
            // svhistcnt_u8(pg, &histogram[bin], v);
        }

        i += svcntb();
    }
}
```

### 6.5 Bitwise Predicate Operations (SVE2)

```c
#include <arm_sve.h>

// SVE2 adds predicate-as-data operations:
//   - Treat a predicate register as a bitmask
//   - Perform bit-level AND/OR/XOR between predicates
//   - Convert predicate to/from integer registers

// Count population (number of 1 bits) in a predicate:
uint64_t popcount_predicate(svbool_t pg) {
    // svcntp_b8 counts the number of active elements (bits set to 1)
    // in the predicate pg. Each byte-sized lane has one predicate bit.
    return svcntp_b8(svptrue_b8(), pg);
}

// Extract the first set bit position from a predicate:
uint64_t first_set_bit(svbool_t pg) {
    svbool_t all_true = svptrue_b8();

    // svbrkb: break before first true, from the left (LSB)
    // Returns a predicate with 1s before the first 1 in pg, 0s after.
    svbool_t before_first = svbrkb_b_z(all_true, all_true, pg);

    // Count the number of 1s before the first set bit = position
    return svcntp_b8(all_true, before_first);
}

// Practical: use predicate bit counting for sparse array processing
void compact_nonzero(const float *src, float *dst, uint64_t n,
                     uint64_t *out_count) {
    uint64_t i = 0, written = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &src[i]);

        // Which lanes are non-zero?
        svfloat32_t zero = svdup_f32(0.0f);
        svbool_t nonzero = svcmpne_f32(pg, v, zero);

        // Extract non-zero elements to contiguous output
        svfloat32_t compacted = svcompact_f32(nonzero, v);

        // How many elements were compacted?
        uint64_t count = svcntp_b32(pg, nonzero);

        // Store only the compacted elements
        svbool_t pg_out = svwhilelt_b32(0, count);
        svst1_f32(pg_out, &dst[written], compacted);

        written += count;
        i += svcntw();
    }

    *out_count = written;
}
```

---

## 7. SVE + NEON Interoperability

### 7.1 Register Overlap

```
SVE and NEON share the same physical register file:

  Z0 [VL-1:0]    ←→   V0 [127:0]   (NEON Q0 register)
  Z0 [63:0]      ←→   D0 [63:0]    (NEON D0 register)

  Z1 [VL-1:0]    ←→   V1 [127:0]
  ...

  P0-P15          →   No NEON equivalent (new predicate registers)
  Z0[VL-1:128]   →   No NEON equivalent (bits beyond 128)

Key consequence: writing to Z0 modifies V0, and vice versa.
The high bits of Z0 (above 128) are invisible to NEON code.
```

### 7.2 Calling Conventions When Mixing SVE and NEON

```c
#include <arm_sve.h>
#include <arm_neon.h>

// AArch64 Procedure Call Standard (AAPCS) for SVE:
//
//   SVE registers: Z0-Z7, P0-P3 are caller-saved
//                  Z8-Z23, P4-P15 are callee-saved
//   NEON registers: V0-V7 are caller-saved
//                   V8-V15 are callee-saved
//
// Since Z0-Z7 share the low 128 bits with V0-V7:
//   - A function that uses V0 corrupts Z0 (and vice versa)
//   - Save/restore is needed when mixing in the same function

// SAFE: separate functions, each uses one ISA exclusively
float32x4_t neon_add(float32x4_t a, float32x4_t b) {
    return vaddq_f32(a, b);   // pure NEON, no SVE interference
}

svfloat32_t sve_add(svfloat32_t a, svfloat32_t b, svbool_t pg) {
    return svadd_f32_m(pg, a, b);  // pure SVE, no NEON interference
}

// DON'T DO THIS: mixing NEON and SVE in the same function
// is fragile and often incorrect without explicit save/restore.
//
// void bad_mixed(svfloat32_t sve_vec, float32x4_t neon_vec) {
//     // neon_vec is in V0, sve_vec is in Z0 -- these OVERLAP!
//     // Writing to one corrupts the other.
// }
```

### 7.3 When to Use SVE vs NEON in the Same Codebase

```c
#include <arm_sve.h>

// Rule of thumb: NEVER mix SVE and NEON in a single function.
// Instead, provide separate code paths and dispatch at runtime.

// Decision matrix:
//
// | Scenario                           | Use    | Reason                     |
// |------------------------------------|--------|----------------------------|
// | Server-side (Graviton3, Grace)     | SVE    | 2x width vs NEON           |
// | Mobile (A76, A78, X1)              | NEON   | No SVE hardware             |
// | Unknown hardware, need portability | Both   | Dispatch at runtime         |
// | Fixed-width performance critical   | NEON   | Predictable unroll factor   |
// | Scatter/gather heavy               | SVE    | First-class gather support  |
// | String processing                  | SVE    | First-fault load (svldff1)  |
// | Broad library code                 | Both   | Feature detection + fallback|

// Runtime dispatch example: selects SVE or NEON path at startup
typedef void (*saxpy_fn)(const float *x, const float *y, float *out,
                         uint64_t n, float alpha);

#ifdef __ARM_FEATURE_SVE
void saxpy_sve_impl(const float *x, const float *y, float *out,
                    uint64_t n, float alpha) {
    uint64_t i = 0;
    svfloat32_t valpha = svdup_f32(alpha);
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svfloat32_t vy = svld1_f32(pg, &y[i]);
        svfloat32_t vout = svmla_f32_m(pg, vy, vx, valpha);
        svst1_f32(pg, &out[i], vout);
        i += svcntw();
    }
}
#endif

#include <arm_neon.h>
void saxpy_neon_impl(const float *x, const float *y, float *out,
                     uint64_t n, float alpha) {
    float32x4_t valpha = vdupq_n_f32(alpha);
    uint64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        float32x4_t vy = vld1q_f32(&y[i]);
        float32x4_t vout = vmlaq_f32(vy, vx, valpha);
        vst1q_f32(&out[i], vout);
    }
    for (; i < n; i++) {
        out[i] = y[i] + x[i] * alpha;
    }
}

// Dispatch function: called once to select the best implementation
#include <sys/auxv.h>
#include <asm/hwcap.h>

saxpy_fn select_saxpy(void) {
#ifdef __ARM_FEATURE_SVE
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if (hwcap2 & HWCAP2_SVE) {
        return saxpy_sve_impl;
    }
#endif
    return saxpy_neon_impl;
}

// Usage:
//   saxpy_fn saxpy = select_saxpy();
//   saxpy(x, y, out, n, alpha);
```

---

## 8. Runtime Detection and Deployment

### 8.1 Detecting SVE Support at Runtime

```c
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <stdio.h>
#include <stdint.h>

// Use getauxval(AT_HWCAP) and getauxval(AT_HWCAP2) to query CPU features.
// These are available on Linux. For other OSes, use platform-specific APIs.

typedef enum {
    SIMD_NONE         = 0,
    SIMD_NEON         = 1,   // ARMv8 baseline
    SIMD_NEON_DOTPROD = 2,   // ARMv8.2: int8 dot product (vdotq_s32)
    SIMD_NEON_FP16    = 3,   // ARMv8.2: float16 support
    SIMD_NEON_I8MM    = 4,   // ARMv8.6: int8 matrix multiply
    SIMD_SVE          = 5,   // ARMv8.2+: Scalable Vector Extension
    SIMD_SVE2         = 6,   // ARMv9: SVE version 2
    SIMD_SVE_AES      = 7,   // SVE + AES crypto
    SIMD_SVE_BITPERM  = 8,   // SVE bit permutation
} simd_level_t;

// Comprehensive SIMD capability detection
simd_level_t detect_simd_capabilities(void) {
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    // SVE2 (ARMv9) implies all prior capabilities
    if (hwcap2 & HWCAP2_SVE2)      return SIMD_SVE2;
    if (hwcap2 & HWCAP2_SVE_AES)   return SIMD_SVE_AES;
    if (hwcap2 & HWCAP2_SVE_BITPERM) return SIMD_SVE_BITPERM;

    // SVE (ARMv8.2+)
    if (hwcap2 & HWCAP2_SVE)       return SIMD_SVE;

    // NEON with optional extensions
    if (hwcap & HWCAP2_I8MM)       return SIMD_NEON_I8MM;
    if (hwcap & HWCAP_ASIMDHP)     return SIMD_NEON_FP16;  // float16
    if (hwcap & HWCAP_ASIMDDP)     return SIMD_NEON_DOTPROD;

    // Baseline NEON (guaranteed on ARMv8-A)
    if (hwcap & HWCAP_ASIMD)       return SIMD_NEON;

    return SIMD_NONE;  // should never happen on ARMv8+
}

// Print the detected SIMD level for diagnostics
void report_simd_capabilities(void) {
    const char *names[] = {
        [SIMD_NONE]         = "None",
        [SIMD_NEON]         = "NEON (ARMv8)",
        [SIMD_NEON_DOTPROD] = "NEON + DotProd (ARMv8.2)",
        [SIMD_NEON_FP16]    = "NEON + FP16 (ARMv8.2)",
        [SIMD_NEON_I8MM]    = "NEON + I8MM (ARMv8.6)",
        [SIMD_SVE]          = "SVE (ARMv8.2+)",
        [SIMD_SVE2]         = "SVE2 (ARMv9)",
        [SIMD_SVE_AES]      = "SVE + AES",
        [SIMD_SVE_BITPERM]  = "SVE + BitPerm",
    };
    simd_level_t level = detect_simd_capabilities();
    printf("Detected SIMD level: %s\n", names[level]);
}
```

### 8.2 Detecting Vector Length at Runtime

```c
#include <arm_sve.h>
#include <stdio.h>
#include <stdint.h>

// Three ways to detect the vector length:

// Method 1: svcntb() intrinsic (recommended)
uint64_t get_vl_bytes(void) {
    return svcntb();  // VL/8
}

// Method 2: svlen() intrinsic
uint64_t get_vl_bits(void) {
    return svlen();  // VL in bits
}

// Method 3: rdvl instruction (inline assembly)
// rdvl Xd, #imm: Xd = VL_in_bytes * imm
static inline uint64_t rdvl(uint64_t multiplier) {
    uint64_t result;
    __asm__ volatile("rdvl %0, %1" : "=r"(result) : "I"(multiplier));
    return result;
}

// Report detailed SVE configuration
void report_sve_config(void) {
    if (!(getauxval(AT_HWCAP2) & HWCAP2_SVE)) {
        printf("SVE not supported on this CPU\n");
        return;
    }

    uint64_t vl_bytes = get_vl_bytes();
    printf("SVE vector length: %lu bits (%lu bytes)\n",
           vl_bytes * 8, vl_bytes);
    printf("  int8  per vector: %lu\n", (uint64_t)svcntb());
    printf("  int16 per vector: %lu\n", (uint64_t)svcnth());
    printf("  int32 per vector: %lu\n", (uint64_t)svcntw());
    printf("  int64 per vector: %lu\n", (uint64_t)svcntd());
    printf("  float32 per vector: %lu\n", (uint64_t)svcntw());
    printf("  float64 per vector: %lu\n", (uint64_t)svcntd());
}
```

### 8.3 Shipping SVE Binaries

```c
// Strategy 1: Compile-time feature selection (single binary)
//
// Compile with: gcc -march=armv8.2-a+sve -O3 -o prog prog.c
// This produces a binary that requires SVE at runtime.
// On non-SVE CPUs, the binary will crash with SIGILL.
//
// PRO: Simple, no runtime dispatch overhead.
// CON: Requires SVE hardware.

// Strategy 2: Runtime dispatch with SVE+NEON fallback (single binary)
//
// Compile with: gcc -march=armv8-a+simd -O3 -o prog prog.c
// The SVE path must be in a separate translation unit compiled with -march=armv8.2-a+sve.
//
// Or use function multi-versioning (FMV) with GCC/Clang attributes:

#if defined(__GNUC__) && __GNUC__ >= 10
// GCC 10+ supports target_clones for ARM
__attribute__((target_clones("default", "sve")))
void my_kernel(const float *a, const float *b, float *c, uint64_t n) {
    // Default (NEON) implementation -- compiled for baseline ARMv8
    uint64_t i = 0;
#ifdef __ARM_NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&c[i], vaddq_f32(va, vb));
    }
#endif
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
// GCC generates two versions: my_kernel.default and my_kernel.sve
// and automatically dispatches based on CPU features.
#endif

// Strategy 3: Separate shared libraries
//
// Build two .so files:
//   libkernel_neon.so  (compiled for ARMv8-A)
//   libkernel_sve.so   (compiled for ARMv8.2-A+SVE)
//
// At startup, dlopen the appropriate one based on AT_HWCAP.

// Strategy 4: ifunc (GNU indirect function)
// The dynamic linker calls a resolver at load time to select the implementation.

#ifdef __ARM_FEATURE_SVE
static void kernel_sve_impl(const float *a, const float *b, float *c, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        svst1_f32(pg, &c[i], svadd_f32_m(pg, va, vb));
        i += svcntw();
    }
}
#endif

static void kernel_neon_impl(const float *a, const float *b, float *c, uint64_t n) {
    uint64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&c[i], vaddq_f32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

// ifunc resolver: called by ld.so at load time
typedef void (*kernel_func)(const float *, const float *, float *, uint64_t);
kernel_func resolve_kernel(void) {
#ifdef __ARM_FEATURE_SVE
    if (getauxval(AT_HWCAP2) & HWCAP2_SVE) {
        return kernel_sve_impl;
    }
#endif
    return kernel_neon_impl;
}

// Tell the linker that `kernel` is resolved via `resolve_kernel`
// (Syntax depends on compiler; this is the GCC/clang form)
void kernel(const float *a, const float *b, float *c, uint64_t n)
    __attribute__((ifunc("resolve_kernel")));
```

### 8.4 Testing SVE Code Without SVE Hardware

```bash
# QEMU user-mode emulation: simulate any SVE vector length.
# This is ESSENTIAL for testing VLA correctness across VLs.

# Test at 128-bit VL (minimum):
qemu-aarch64 -cpu max,sve=on,sve128=on \
  -E SVE_VECTOR_LENGTH=128 ./sve_program

# Test at 256-bit VL (Graviton3 equivalent):
qemu-aarch64 -cpu max,sve=on,sve256=on \
  -E SVE_VECTOR_LENGTH=256 ./sve_program

# Test at 512-bit VL (A64FX equivalent):
qemu-aarch64 -cpu max,sve=on,sve512=on \
  -E SVE_VECTOR_LENGTH=512 ./sve_program

# Test at 1024-bit VL (future hardware):
qemu-aarch64 -cpu max,sve=on,sve1024=on \
  -E SVE_VECTOR_LENGTH=1024 ./sve_program

# Run your test suite across all VLs in a script:
# for vl in 128 256 512 1024; do
#     echo "Testing VL=$vl..."
#     qemu-aarch64 -cpu max,sve=on,sve${vl}=on \
#       -E SVE_VECTOR_LENGTH=$vl ./sve_tests || exit 1
# done

# Check the actual VL the hardware reports:
cat /sys/devices/system/cpu/sve/vl  # real hardware (e.g., Graviton3)
```

### 8.5 Compilation Flags Reference

```bash
# Compile SVE code (ARMv8.2-A + SVE):
gcc -march=armv8.2-a+sve    -O3 -moutline-atomics -o prog prog.c

# Compile SVE2 code (ARMv9-A):
gcc -march=armv9-a          -O3 -o prog prog.c

# Compile NEON-only code (compatible with everything ARMv8+):
gcc -march=armv8-a+simd     -O3 -o prog prog.c

# Compile with SVE auto-vectorization hints:
gcc -march=armv8.2-a+sve -O3 -ftree-vectorize \
    -fopt-info-vec-missed -fopt-info-vec  -o prog prog.c

# Enable SVE-specific compiler checks:
gcc -march=armv8.2-a+sve -O3 -Wall -Wextra \
    -Wvector-operation-performance -o prog prog.c

# Clang (same flags, better SVE auto-vec in some cases):
clang -march=armv8.2-a+sve -O3 -o prog prog.c
```

### 8.6 Future Outlook: ARMv9 and SVE2

ARMv9-A mandates SVE2 as part of the baseline architecture. This means:

- **Every ARMv9 CPU** (Cortex-X4, Cortex-A720, Neoverse V2, etc.) supports SVE2.
- The question shifts from "does this CPU have SVE?" to "what vector length does this CPU support?"
- Mobile chips will gradually gain SVE2 as ARMv9 cores replace ARMv8 cores.
- Libraries should prepare SVE2 code paths now for when SVE2 becomes ubiquitous.

```
Timeline (approximate):
  2020: A64FX (SVE), Graviton3 (SVE)        -- server only
  2022: Neoverse V2 (SVE2)                   -- ARMv9 server
  2023: Cortex-X4 (SVE2, 128-bit VL)         -- first mobile SVE2
  2024: Cortex-X925 (SVE2)                   -- flagship mobile
  2025: Mid-range ARMv9 (SVE2)               -- broader mobile adoption
  2026+: ARMv9 everywhere                     -- SVE2 baseline for new designs
```

---

## Appendix A: Complete Working Examples

### A.1 SAXPY (Scalar Alpha X Plus Y)

```c
#include <arm_sve.h>
#include <stdint.h>

// y[i] = alpha * x[i] + y[i]
void saxpy_sve(const float *x, float *y, uint64_t n, float alpha) {
    uint64_t i = 0;
    svfloat32_t valpha = svdup_f32(alpha);

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svfloat32_t vy = svld1_f32(pg, &y[i]);

        // FMA with merge: inactive lanes preserve vy
        svfloat32_t result = svmla_f32_m(pg, vy, vx, valpha);
        svst1_f32(pg, &y[i], result);

        i += svcntw();
    }
}
```

### A.2 Matrix-Vector Multiply (y = A * x)

```c
#include <arm_sve.h>
#include <stdint.h>

// y = A * x, where A is MxN (row-major), x is Nx1, y is Mx1
void sve_gemv(const float *A, const float *x, float *y,
              uint64_t M, uint64_t N) {
    for (uint64_t row = 0; row < M; row++) {
        svfloat32_t acc = svdup_f32(0.0f);
        uint64_t j = 0;

        while (j < N) {
            svbool_t pg = svwhilelt_b32(j, N);
            svfloat32_t va = svld1_f32(pg, &A[row * N + j]);
            svfloat32_t vx = svld1_f32(pg, &x[j]);

            // Dot product accumulation
            acc = svmla_f32_m(pg, acc, va, vx);

            j += svcntw();
        }

        y[row] = svaddv_f32(svptrue_b32(), acc);
    }
}
```

### A.3 Find Index of Maximum Element (argmax)

```c
#include <arm_sve.h>
#include <stdint.h>
#include <float.h>

// Returns the index of the maximum element in data[0..n-1]
uint64_t sve_argmax(const float *data, uint64_t n) {
    if (n == 0) return 0;

    svfloat32_t best_val = svdup_f32(-FLT_MAX);
    svuint32_t  best_idx = svindex_u32(0, 1);  // {0, 1, 2, ..., VL/32-1}
    svuint32_t  vidx_inc = svdup_u32((uint32_t)svcntw());
    svuint32_t  indices  = svindex_u32(0, 1);

    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &data[i]);

        // Compare: is v > current best?
        svbool_t better = svcmpgt_f32(pg, v, best_val);

        // Update best values and indices where condition is true
        best_val = svsel_f32(better, v,       best_val);
        best_idx = svsel_u32(better, indices, best_idx);

        indices  = svadd_u32_m(pg, indices, vidx_inc);
        i += svcntw();
    }

    // Find the maximum across all lanes of best_val
    svfloat32_t global_max = svmaxv_f32(svptrue_b32(), best_val);

    // Find which lane holds the global maximum
    svbool_t max_lanes = svcmpeq_f32(svptrue_b32(), best_val,
                                     svdup_f32(global_max));

    // Get the index of the first lane with the maximum value
    uint64_t lane = first_set_bit(max_lanes);
    uint64_t result;
    // Read index from lane 'lane' of best_idx
    result = best_idx[lane];  // compiler-dependent lane access

    return result;
}
```

### A.4 Softmax (Numerically Stable)

```c
#include <arm_sve.h>
#include <stdint.h>
#include <math.h>

// Numerically stable softmax using SVE:
//   softmax(x[i]) = exp(x[i] - max(x)) / sum(exp(x - max(x)))
void sve_softmax(float *x, uint64_t n) {
    if (n == 0) return;

    // Pass 1: find max(x) for numerical stability
    float max_val = -INFINITY;
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &x[i]);
        float lane_max = svmaxv_f32(pg, v);
        if (lane_max > max_val) max_val = lane_max;
        i += svcntw();
    }

    // Pass 2: compute exp(x[i] - max) and accumulate sum
    svfloat32_t sum_vec = svdup_f32(0.0f);
    i = 0;
    svfloat32_t vmax = svdup_f32(max_val);
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &x[i]);
        v = svsub_f32_m(pg, v, vmax);   // x - max

        // Compute exp using a polynomial approximation or library call
        // For production, use sv_expf32 or a tuned polynomial.
        // Here we use scalar exp; in practice you'd use a vectorized exp.
        float tmp[svcntw()];
        svst1_f32(pg, tmp, v);
        for (uint64_t k = 0; k < svcntw() && (i + k) < n; k++) {
            tmp[k] = expf(tmp[k]);
        }
        v = svld1_f32(pg, tmp);

        svst1_f32(pg, &x[i], v);        // store exp(x - max)
        sum_vec = svadd_f32_m(pg, sum_vec, v);  // accumulate sum
        i += svcntw();
    }

    // Reduce sum
    float sum = svaddv_f32(svptrue_b32(), sum_vec);
    svfloat32_t vsum = svdup_f32(sum);

    // Pass 3: normalize by sum
    i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &x[i]);
        v = svdiv_f32_m(pg, v, vsum);
        svst1_f32(pg, &x[i], v);
        i += svcntw();
    }
}
```

---

## Appendix B: Quick Reference Card

### B.1 Predicate Operations

| Operation | Intrinsic | Description |
|-----------|-----------|-------------|
| Loop predicate | `svwhilelt_b32(i, n)` | Lane k active if i+k < n |
| All true | `svptrue_b32()` | All lanes enabled |
| Pattern VL1 | `svptrue_pat_b32(SV_VL1)` | Only lane 0 enabled |
| Comparisons | `svcmpgt_f32(pg, a, b)` | a > b on active lanes |
| Selection | `svsel_f32(pg, a, b)` | pg=1: a, pg=0: b |
| Predicate AND | `svand_b_z(pg, p1, p2)` | p1 & p2 |
| Predicate OR | `svorr_b_z(pg, p1, p2)` | p1 \| p2 |
| Predicate NOT | `svnot_b_z(pg, p1)` | ~p1 |
| First true | `svbrkb_b_z(pg, pg, p)` | Lanes before first 1 in p |
| Popcount | `svcntp_b32(pg, p)` | Count of 1 bits in p |
| Test any | `svptest_any(pg, p)` | True if any lane is set |
| Compact | `svcompact_f32(p, v)` | Pack active lanes to LSB |

### B.2 Operation Suffixes

| Suffix | Inactive Lane Behavior | Use Case |
|--------|----------------------|----------|
| `_m` (merge) | Preserve first operand | Accumulators (acc = acc + x) |
| `_z` (zero) | Zero | Fresh results for storage |
| `_x` (don't care) | Undefined | Temporary values, overwritten immediately |

### B.3 Runtime Information

| Intrinsic / API | Returns |
|----------------|---------|
| `svcntb()` | Number of bytes per vector (VL/8) |
| `svcnth()` | Number of halfwords per vector (VL/16) |
| `svcntw()` | Number of words per vector (VL/32) |
| `svcntd()` | Number of doublewords per vector (VL/64) |
| `svlen()` | Vector length in bits |
| `getauxval(AT_HWCAP2)` | CPU feature flags |
| `rdvl x0, #1` | VL in bytes (assembly) |

### B.4 Compilation Flags

```bash
# SVE (ARMv8.2+)
gcc -march=armv8.2-a+sve    -O3 prog.c

# SVE2 (ARMv9)
gcc -march=armv9-a          -O3 prog.c

# SVE2 with specific features
gcc -march=armv9-a+sve2+sve2-bitperm -O3 prog.c

# QEMU testing
qemu-aarch64 -cpu max,sve=on,sve256=on ./prog
```

---

**Next: Applying SVE and NEON to 7 real-world industrial scenarios: image/audio processing, ML inference, data compression, and network packet processing.**
