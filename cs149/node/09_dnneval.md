# Lecture 9: Efficiently Evaluating DNNs

**Course:** Stanford CS149 - Parallel Computing, Fall 2025
**PDF:** Lecture 9

---

## Core Concepts

### 1. Deep Neural Networks as Circuits

- A neuron is a simple computation unit: `output = f(Σ x_i * w_i + b)`
- Common activation `f`: ReLU `max(0, x)`, sigmoid, softmax
- **Fully connected layer** = matrix-vector product + element-wise activation
- **Convolutional layer** = locally connected + shared weights across spatial positions
- **Pooling**: max-pool (2x2 region → maximum value) reduces spatial dimensions
- **Softmax**: normalizes outputs to probability distribution

### 2. Roofline Model & Arithmetic Intensity

**Arithmetic Intensity = FLOPs / Bytes Transferred**

| Regime | Behavior |
|---|---|
| Low AI (BW-bound) | Performance limited by memory bandwidth |
| High AI (Compute-bound) | Performance limited by ALU throughput |

**Key insight:** Faster hardware (more compute) makes programs MORE likely to be bandwidth-bound. Increasing arithmetic intensity (program change) makes programs more likely to be compute-bound.

#### Loop Fusion to Increase AI
```
// Program 1: AI = 1/3 (2 loads, 1 store per op)
add(A,B,tmp1); mul(tmp1,C,tmp2); add(tmp2,D,E);

// Program 2: AI = 3/5 (4 loads, 1 store per 3 ops)
for (i): E[i] = D[i] + (A[i]+B[i])*C[i];
```

### 3. Matrix Multiplication (GEMM) — Core of DNN Inference

#### Naive (Low AI)
```c
for (int j=0; j<M; j++)
    for (int i=0; i<N; i++)
        for (int k=0; k<K; k++)
            C[j][i] += A[j][k] * B[k][i];
```

#### Blocked (Increased AI)
```c
for (int jblock=0; jblock<M; jblock+=BS_J)
    for (int iblock=0; iblock<N; iblock+=BS_I)
        for (int kblock=0; kblock<K; kblock+=BS_K)
            // Compute partial result for C sub-block
            // while A, B sub-blocks stay in cache
```

#### Hierarchical Blocked
- Multiple blocking levels for L1, L2, L3 caches
- Final level: register blocking (micro-kernel)
- `BLOCKSIZE` chosen so sub-blocks fit in each cache level

#### SIMD Vectorization Variants
1. **Broadcast A, vector-load B** — good for large i-dimension
2. **Pre-transpose B, use SIMD dot product** — good for small i-dimension
3. **Pre-transpose A and C, outer product approach** — two pre-transpositions

### 4. Convolution Implementation Strategies

#### Direct Implementation (7-level nested loop)
```
for img, j, i, f (output):
    for kk (input channels):
        for jj, ii (spatial conv):
            output += weights[f][jj][ii][kk] * input[img][j+jj][i+ii][kk]
```
**Key data reuse:** filter weights reused across spatial positions; input values reused across filters.

#### Convolution as GEMM (im2col)
- Reshape input image into a "convolution matrix" where each row contains a filter-sized patch
- Then perform dense matrix multiplication: `C = W × X_reshaped`
- **Cost:** O(R×S) storage overhead; extra DRAM traffic

#### Implicit GEMM (CUTLASS approach)
- Materialize only sub-blocks of the convolution matrix in on-chip shared memory
- No additional off-chip storage required
- Use well-tuned shared-memory GEMM for sub-block computation

### 5. Operation Fusion for DNN Layers

#### Why Fuse?
- Without fusion: Conv (1GB output) → write to DRAM → read for Scale/Bias → write → read for MaxPool → write
- **With fusion:** compute conv output element → immediately scale/bias → feed into pool — no DRAM traffic for intermediates!

#### Examples of Fusion
1. **Conv + Scale/Bias + ReLU**: compute, then `tmp * scale + bias`, then `max(0, tmp)`
2. **Conv + Scale/Bias + MaxPool**: after computing 2x2 block of conv outputs, immediately compute max and write to pool output
3. **Softmax on rows**: load entire row → compute all steps in register → store final result (reads MN, writes MN vs. reads 5MN+2M, writes 3MN+2M)

### 6. Flash Attention — Fusion for Transformers

**Problem:** Naive attention computes `S = QK^T` (N×N matrix), then `P = softmax(S)` (N×N), then `O = PV`. Storing intermediate N×N matrix is expensive for long sequences.

**Solution (Flash Attention):**
```python
for each j:                    # outer loop over K,V blocks
    for each i:                # inner loop over Q blocks
        Load block Qi, KTj, Vj, Oi
        Compute Sij = Qi @ KTj
        Compute softmax components (Mij, Pij, lij) row-wise
        Multiply Pij @ Vj and accumulate into Oi (with rescaling)
```
- **Never materializes N×N matrix** — saves memory
- **High arithmetic intensity** — reads 3 blocks, computes 2 MMs + softmax, accumulates
- Softmax can be computed in chunks by tracking max and sum per chunk

### 7. Key Libraries and Tools

| Tool | Description |
|---|---|
| **cuDNN** | NVIDIA's library of high-performance DNN kernels; supports many algorithms for each layer |
| **CUTLASS** | Template library for writing custom high-performance GEMM/kernel code |
| **Triton** | Language for writing GPU kernels with block-level abstractions |
| **Thunderkittens** | CUDA library of tile-based programming primitives |
| **torch.compile** | Automatic kernel fusion and scheduling for PyTorch |

### 8. Optimization Techniques Summary

1. **Loop blocking/tiling**: increase temporal locality, improve arithmetic intensity
2. **Loop fusion**: eliminate intermediate DRAM traffic
3. **Implicit GEMM**: avoid materializing full convolution matrix
4. **Multi-level blocking**: exploit L1, L2, register hierarchy
5. **Low-precision values**: FP16, INT8, INT4 — more ops per byte
6. **Better algorithms**: depthwise separable convolutions (MobileNet), efficient topologies

### 9. Why GPU for DNN?

**Advantages:**
- High arithmetic intensity of GEMM matches flop-rich GPU architecture
- Massive data-parallelism in convolutions and matrix multiplies
- High memory bandwidth (900+ GB/sec) for weight/activation streaming
- Highly-optimized libraries exist (cuDNN, CUTLASS)
- SIMD execution efficiently handles vectorized matrix operations

**Disadvantages:**
- General-purpose processor may be overkill — specialized accelerators (TPU, NPU, Apple Neural Engine) can be more efficient
- Data movement costs energy
- Chip resources used for on-chip storage = resources not used for compute

---

## Knowledge Points → Corresponding C++ Files

| Knowledge Point | C++ File |
|---|---|
| Matrix multiplication: naive, blocked, tiled with SIMD simulation | `lecture9_part1.cpp` |
| Convolution: direct implementation and im2col | `lecture9_part2.cpp` |
| Loop fusion and arithmetic intensity optimization | `lecture9_part3.cpp` |

---

## Actionable Learning Points

1. **Arithmetic intensity determines the performance ceiling** — know your roofline
2. **Blocking is the primary technique** for increasing AI in matrix operations
3. **Convolution = GEMM**: reshape inputs, then use optimized matrix multiply
4. **Implicit is better than explicit**: don't materialize huge intermediate matrices
5. **Fuse everything you can**: avoid round-trips to DRAM for intermediate results
6. **Different layers need different strategies** — dimensions vary widely in a single DNN
7. **Flash Attention is a masterclass in fusion** — saves O(N²) memory and bandwidth
8. **Modern frameworks automate scheduling** (torch.compile, XLA) — but understanding the fundamentals is essential
