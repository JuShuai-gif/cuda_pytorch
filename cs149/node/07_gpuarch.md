# Lecture 7: GPU Architecture & CUDA Programming

**Course:** Stanford CS149 - Parallel Computing, Fall 2025
**PDF:** Lecture 7

---

## Core Concepts

### 1. GPU Evolution: From Graphics to Compute

- **Original purpose (pre-2007):** 3D rendering pipeline
  - Input: 3D triangle meshes, materials, lights, camera
  - Output: rendered image pixels
  - Shader programs run per-vertex/per-fragment in data-parallel fashion
- **Key observation (2001-2003):** GPUs are very fast at performing the SAME computation on LARGE collections of data — this is data-parallelism!
- **Hack era (GPGPU 2002-2003):** Scientists used graphics API to perform computation (e.g., rendering 2 triangles to cover a 512x512 output to run a shader on every element)
- **Brook (2004):** Stanford stream programming language that abstracted GPU as data-parallel processor; compiled to graphics commands
- **NVIDIA Tesla architecture (2007):** First non-graphics "compute mode" interface — `launch(kernel, N)` instead of `drawPrimitives()`

### 2. CUDA Programming Model

#### Thread Hierarchy
- **Grid**: collection of thread blocks
- **Thread Block**: group of threads (up to 1024 on modern GPUs)
- **Thread**: single execution unit with unique ID
- Thread IDs can be 1D, 2D, or 3D (convenient for N-D problems)
- Kernel launch syntax: `kernel<<<numBlocks, threadsPerBlock>>>(args)`

#### Memory Model (3 distinct device address spaces)
| Memory Type | Scope | Access | Speed |
|---|---|---|---|
| Global memory | All threads | Read/Write | Slow (DRAM) |
| Shared memory | Per-block | Read/Write | Fast (on-chip) |
| Per-thread private | Per-thread | Read/Write | Fastest (registers) |

#### Key Abstractions
- **`__global__`**: kernel function (runs on GPU, called from CPU)
- **`__device__`**: device function (called from GPU)
- **`__shared__`**: per-block shared memory variable
- **`__syncthreads()`**: barrier — all threads in block must arrive
- **`cudaMemcpy()`**: copy between host and device address spaces
- **Atomic operations**: `atomicAdd`, `atomicInc`, etc.

#### Host/Device Separation
- Host (CPU) and Device (GPU) have **distinct address spaces**
- Programmer explicitly manages data movement: `cudaMalloc`, `cudaMemcpy`
- Kernel launch returns when all threads terminate (implicit barrier)

### 3. GPU Architecture Deep Dive

#### NVIDIA V100 Specifications
- **80 Streaming Multiprocessors (SMs)**
- **5120 fp32 mul-add ALUs** = 12.7 TFLOPS
- **Up to 163,840 CUDA threads** concurrently
- **900 GB/sec** memory bandwidth (HBM, 16 GB)
- **6 MB L2 Cache**
- 1.245 GHz clock

#### SM Architecture (per SM)
- **4 sub-cores**, each with:
  - 16-wide SIMD fp32 units (32-wide every 2 clocks)
  - 16-wide SIMD int units
  - 8-wide SIMD fp64 units (every 4 clocks)
  - Tensor core units
  - Load/store units
- **64 KB registers per sub-core** (256 KB total per SM)
- **128 KB shared memory + L1 cache**
- Up to **64 warps** per SM

#### Warp and SIMT Execution
- **Warp** = group of 32 threads within a block
- **SIMT** (Single Instruction Multiple Thread): warp threads execute same instruction simultaneously
- Each sub-core selects one runnable warp per clock
- 16 ALUs → each instruction takes 2 clocks for full 32-thread warp
- **Divergent execution**: when threads in a warp take different paths → masked execution → performance loss

#### Thread Block Scheduling
- Thread blocks are assigned to SMs by a hardware scheduler
- Blocks can run on any SM, in any order (no inter-block dependencies assumed)
- Scheduler respects resource requirements (thread count, shared memory, registers)
- Multiple blocks can co-reside on an SM if resources allow
- Prevents deadlock: if a thread in a block is runnable, it will eventually run

### 4. CUDA vs. Other Parallel Models

| Concept | CUDA Analog |
|---|---|
| ISPC gang of instances | CUDA warp (but warp is implementation detail, not programming model) |
| ISPC tasks | CUDA thread blocks (no dependencies, scheduled dynamically) |
| pthreads | CUDA threads (but implementation is very different — lightweight HW threads) |
| Shared address space | Within a thread block (shared memory) |
| Message passing | Between host and device (memcpy), between blocks (global memory + atomics) |

### 5. Code Patterns

#### 1D Convolution — Version 1 (Naive, Global Memory)
```cuda
__global__ void convolve(int N, float* input, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i=0; i<3; i++)
        result += input[index + i];
    output[index] = result / 3.f;
}
```

#### 1D Convolution — Version 2 (Shared Memory Optimized)
```cuda
__global__ void convolve(int N, float* input, float* output) {
    __shared__ float support[THREADS_PER_BLK+2];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    support[threadIdx.x] = input[index];
    if (threadIdx.x < 2)
        support[THREADS_PER_BLK + threadIdx.x] = input[index+THREADS_PER_BLK];
    __syncthreads();
    float result = 0.0f;
    for (int i=0; i<3; i++)
        result += support[threadIdx.x + i];
    output[index] = result / 3.f;
}
```

#### Persistent Thread Pattern (Bonus)
- Programmer launches exactly as many blocks as will fill the GPU
- Blocks use atomic operations to grab work (work-stealing style)
- Assumes knowledge of GPU implementation — **not portable!**

---

## Knowledge Points → Corresponding C++ Files

| Knowledge Point | C++ File |
|---|---|
| Thread hierarchy (grid/blocks/threads) simulation | `lecture7_part1.cpp` |
| 1D convolution: naive vs. shared-memory | `lecture7_part2.cpp` |
| GPU memory hierarchy simulation | `lecture7_part3.cpp` |

---

## Actionable Learning Points

1. **Understand the 3-level thread hierarchy** and how it maps to GPU hardware
2. **Know when to use shared memory** — when threads in a block reuse the same data
3. **Barriers are your friend**: `__syncthreads()` enables cooperative data loading
4. **Avoid warp divergence**: branches in kernel code serialize execution
5. **Resource constraints matter**: shared memory and register limits determine occupancy
6. **CUDA's design philosophy**: low abstraction distance — abstractions closely match hardware
7. **Data-parallel model**: CUDA thread blocks are like forall loops; independent, run anywhere
8. **Inter-block dependencies are not guaranteed** — use atomics on global memory if needed
