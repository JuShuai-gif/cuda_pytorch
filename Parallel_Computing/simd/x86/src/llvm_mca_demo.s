# =============================================================================
# llvm_mca_demo.s -- Annotated x86-64 assembly for llvm-mca analysis
#
# This file demonstrates how to use llvm-mca to statically analyze SIMD loops.
#
# llvm-mca reads the assembly file and simulates execution on a target CPU
# without running any code. It predicts:
#   - IPC (instructions per cycle)
#   - Resource (port) pressure per iteration
#   - Bottleneck classification
#   - Block Reciprocal Throughput
#
# Regions are delimited by:
#   # LLVM-MCA-BEGIN <name>
#   ...instructions...
#   # LLVM-MCA-END
#
# Everything outside these markers is treated as "glue" code (uncounted).
#
# Run this file with:
#   llvm-mca --mcpu=skylake --iterations=100 --timeline --bottleneck-analysis llvm_mca_demo.s
#
# Or use the wrapper script:
#   ../../scripts/llvm_mca.sh --demo
#   ../../scripts/llvm_mca.sh --asm llvm_mca_demo.s fast_dot
# =============================================================================

    .intel_syntax noprefix
    .text

# ===========================================================================
# Glue code -- not analyzed by llvm-mca
#
# These instructions set up registers and are counted as "past" context.
# llvm-mca uses them to understand initial register values and memory
# dependencies but does NOT include them in the throughput/latency analysis.
# ===========================================================================

setup:
    # Initialize pointers and counters for all three loops
    # These are "glue" -- outside any LLVM-MCA-BEGIN/END block
    lea     rax, [rip + a]           # base address for array a
    lea     rbx, [rip + b]           # base address for array b
    lea     rcx, [rip + c]           # base address for array c (vector_add)
    mov     rdx, 256                  # loop count (256 elements = 32 AVX vectors)
    vxorps  xmm2, xmm2, xmm2         # clear accumulators for fast_dot
    vxorps  xmm3, xmm3, xmm3
    vxorps  xmm4, xmm4, xmm4
    vxorps  xmm5, xmm5, xmm5
    vxorps  xmm8, xmm8, xmm8         # clear accumulator for slow_dot


# ===========================================================================
# REGION 1: fast_dot -- Well-optimized 4-way unrolled AVX2 FMA dot product
#
# Key optimization decisions:
#   1. FOUR independent accumulator chains (ymm2, ymm3, ymm4, ymm5)
#      This hides the 4-cycle FMA latency on Skylake/Ice Lake.
#      Without unrolling, each vfmadd231ps must wait for the previous one.
#      With 4x unrolling, 4 FMAs can be in flight simultaneously.
#
#   2. Loads are interleaved with FMAs to spread across ports:
#      - vmovups: port 2 or port 3 (load AGU)
#      - vfmadd231ps: port 0 or port 1 (FMA unit)
#      The mix of load+FMA pairs means ports are well-utilized.
#
#   3. Integer arithmetic (add/sub/cmp) goes to ports 0/1/5/6,
#      which don't conflict with the vector loads and FMAs.
#
# Expected result (Skylake):
#   - IPC close to 4.0 (dispatch width = 4)
#   - Port 0: ~2.5 pressure, Port 1: ~2.5 pressure (FMAs)
#   - Port 2/3: ~1.0 each (loads evenly spread)
#   - Port 5/6: < 0.5 (integer loop control)
#   - RThroughput: ~2.0 cycles/iter (dictated by FMAs)
# ===========================================================================

# LLVM-MCA-BEGIN fast_dot
.Lfast_loop:
    # --- Unrolled block 1: elements [0..7] ---
    vmovups     ymm0, ymmword ptr [rax]          # load a[i+0..i+7]
    vmovups     ymm1, ymmword ptr [rbx]          # load b[i+0..i+7]
    vfmadd231ps ymm2, ymm0, ymm1                 # acc0 += a * b

    # --- Unrolled block 2: elements [8..15] ---
    vmovups     ymm0, ymmword ptr [rax + 32]     # load a[i+8..i+15]
    vmovups     ymm1, ymmword ptr [rbx + 32]     # load b[i+8..i+15]
    vfmadd231ps ymm3, ymm0, ymm1                 # acc1 += a * b

    # --- Unrolled block 3: elements [16..23] ---
    vmovups     ymm0, ymmword ptr [rax + 64]     # load a[i+16..i+23]
    vmovups     ymm1, ymmword ptr [rbx + 64]     # load b[i+16..i+23]
    vfmadd231ps ymm4, ymm0, ymm1                 # acc2 += a * b

    # --- Unrolled block 4: elements [24..31] ---
    vmovups     ymm0, ymmword ptr [rax + 96]     # load a[i+24..i+31]
    vmovups     ymm1, ymmword ptr [rbx + 96]     # load b[i+24..i+31]
    vfmadd231ps ymm5, ymm0, ymm1                 # acc3 += a * b

    # --- Advance pointers, decrement counter ---
    add         rax, 128                          # advance a by 4*32 = 128 bytes
    add         rbx, 128                          # advance b by 4*32 = 128 bytes
    sub         rdx, 32                           # processed 32 floats
    jnz         .Lfast_loop
# LLVM-MCA-END

    # Glue: reset loop counter and pointers for the next region
    # (rdx/rax/rbx were consumed by fast_dot's loop)
    lea     rax, [rip + a]
    lea     rbx, [rip + b]
    mov     rdx, 256

# ===========================================================================
# REGION 2: slow_dot -- Reduction every iteration, Port 5 bottleneck
#
# Why this is slow:
#   This version does a full horizontal reduction EVERY iteration.
#   On Skylake, vhaddps and all permute/shuffle instructions execute
#   exclusively on Port 5. vmulps also uses port 0 or 1.
#
#   The inner loop body contains:
#   - 2x vmovups (Port 2/3)
#   - 1x vmulps  (Port 0/1)
#   - 1x vhaddps (Port 5 ONLY)          <-- BOTTLENECK
#   - 1x vperm2f128 (Port 5 ONLY)       <-- BOTTLENECK
#   - 3x vpermilps shuffle (Port 5)     <-- BOTTLENECK
#   - 2-3x vaddss/vaddps (Port 0/1, optional on Port 5)
#
#   Total uOps on Port 5: ~4-5 per 8-element iteration.
#   Port 5 can handle 1 uOp/cycle on Skylake.
#   => Minimum 4-5 cycles per 8 elements, just for shuffles!
#
# Expected result (Skylake):
#   - IPC: ~1.5-2.0 (far below dispatch width of 4)
#   - Port 5: ~4.0-5.0 pressure (saturated!)
#   - Port 0/1: ~1.0-1.5 (idle most of the time)
#   - RThroughput: ~4.0-5.0 cycles (dictated by Port 5)
#
# Compare with fast_dot where the same amount of work happens with
# 2.0 cycles/32-elements of throughput instead of 5.0 cycles/8-elements.
# Effective throughput difference: (32/2) / (8/5) = 16/1.6 = 10x worse!
# ===========================================================================

# LLVM-MCA-BEGIN slow_dot
.Lslow_loop:
    # --- Load 8 elements from each array ---
    vmovups     ymm0, ymmword ptr [rax]          # load a[i..i+7]
    vmovups     ymm1, ymmword ptr [rbx]          # load b[i..i+7]

    # --- Element-wise multiply ---
    vmulps      ymm0, ymm0, ymm1                 # ymm0 = a * b

    # --- Horizontal reduction from 8 floats to 1 float ---
    # Step 1: hadd [a,b,c,d,e,f,g,h] -> [a+b,c+d,e+f,g+h, a+b,c+d,e+f,g+h]
    # Port 5 only on Skylake!
    vhaddps     ymm0, ymm0, ymm0

    # Step 2: swap high/low 128-bit lanes
    # Port 5 only!
    vperm2f128  ymm1, ymm0, ymm0, 0x01

    # Step 3: add lanes together -> [a+b+e+f, c+d+g+h, ...]
    # vaddps on Port 0/1
    vaddps      ymm0, ymm0, ymm1

    # Step 4: shuffle [x, y, _, _] -> [y, x, _, _] (swap first two elements)
    # vpermilps on Port 5!
    vpermilps   ymm1, ymm0, 0b00001110

    # Step 5: add first two elements -> x+y
    vaddss      xmm0, xmm0, xmm1

    # Step 6: shuffle again to extract the scalar
    # Port 5!
    vpermilps   ymm1, ymm0, 0b00000101

    # Step 7: final accumulation into running sum
    vaddss      xmm8, xmm8, xmm1

    # --- Advance pointers ---
    add         rax, 32                           # next 8 floats (32 bytes)
    add         rbx, 32
    sub         rdx, 8                            # processed 8 floats
    jnz         .Lslow_loop
# LLVM-MCA-END

    # Glue: reset loop counter and accumulator for the next region
    lea     rax, [rip + a]
    lea     rbx, [rip + b]
    lea     rcx, [rip + c]
    mov     rdx, 256

# ===========================================================================
# REGION 3: vector_add -- Simple memory-bound loop
#
# This is the classic streaming memory-bound pattern:
#   c[i] = a[i] + b[i]
#
# On Skylake:
#   - 2 loads (Port 2 + Port 3) per 1 store (Port 4 address, Port 7 data)
#   - Only 1 ALU uOp (vaddps on Port 0/1) for every 2 load uOps
#   - The CPU can issue 4 uOps/cycle, but L1 cache bandwidth is the limit
#
# The memory system can sustain:
#   - 2 x 32-byte loads per cycle from L1 -> 64 bytes/cycle
#   - 1 x 32-byte store per cycle from L1 -> 32 bytes/cycle
#
# Expected result (Skylake):
#   - IPC: ~2.0-3.0 (below dispatch width, but not terrible)
#   - Port 2: ~1.0 pressure (load)
#   - Port 3: ~1.0 pressure (load)
#   - Port 4/7: ~1.5 (store address + data)
#   - Port 0/1: ~0.5 (vaddps is cheap, only 1 uOp per 8 elements)
#   - This is MEMORY-BOUND: increasing compute won't help.
#     The fix is cache-blocking to reuse data in registers, not more SIMD.
# ===========================================================================

# LLVM-MCA-BEGIN vector_add
.Lva_loop:
    vmovups     ymm0, ymmword ptr [rax]           # load a[i..i+7]     Port 2 or 3
    vaddps      ymm0, ymm0, ymmword ptr [rbx]     # a + b (fused load)  Port 0/1 + Port 2/3
    vmovups     ymmword ptr [rcx], ymm0           # store c[i..i+7]    Port 4 (data) + Port 7 (addr)

    add         rax, 32                            # advance pointers
    add         rbx, 32
    add         rcx, 32
    sub         rdx, 8                             # processed 8 floats
    jnz         .Lva_loop
# LLVM-MCA-END

    # Epilogue -- after all analysis regions
    ret


# ===========================================================================
# DATA SECTION -- dummy initialized arrays for llvm-mca memory dependencies
#
# llvm-mca doesn't actually read these, but they tell it that the memory
# operands in the loops reference separate regions (no aliasing). This
# gives more accurate results because llvm-mca assumes worst-case alias
# behavior when loads/stores don't have known base addresses.
# ===========================================================================

    .data
    .balign 64
a:
    .rept 1024
    .float  1.0
    .endr

    .balign 64
b:
    .rept 1024
    .float  2.0
    .endr

    .balign 64
c:
    .rept 1024
    .float  0.0
    .endr
