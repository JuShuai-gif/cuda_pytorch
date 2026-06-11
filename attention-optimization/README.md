# Attention Optimization: From Theory to Industrial Kernel

-- A systematic learning path for Transformer Attention optimization at the GPU kernel level

## Target

- Read and understand FlashAttention source code
- Read and understand xFormers source code
- Read and understand TensorRT-LLM Attention Kernel
- Read and understand vLLM PagedAttention
- Write your own CUDA Attention Kernel
- Profile and optimize Attention kernels

## Structure

```
attention-optimization/
├── notes/                   # Chinese Markdown notes per chapter
├── src/
│   ├── chapter_01/          # Attention Basics
│   ├── chapter_02/          # GPU Attention Implementation
│   ├── chapter_03/          # Attention Profiling
│   ├── chapter_04/          # FlashAttention V1
│   ├── chapter_05/          # FlashAttention V2
│   ├── chapter_06/          # FlashAttention V3
│   ├── chapter_07/          # KV Cache
│   ├── chapter_08/          # PagedAttention
│   ├── chapter_09/          # MQA / GQA
│   ├── chapter_10/          # Sliding Window Attention
│   ├── chapter_11/          # Sparse Attention
│   ├── chapter_12/          # Linear Attention
│   ├── chapter_13/          # Quantized Attention
│   ├── chapter_14/          # TensorRT-LLM Attention
│   ├── chapter_15/          # xFormers Source Analysis
│   └── chapter_16/          # vLLM Source Analysis
├── benchmark/               # Shared benchmarking tools
├── profiler/                # Profiling scripts (Nsight, torch.profiler)
├── final_project/           # Mini TensorRT-LLM Attention Engine
├── docs/                    # Additional documentation
└── README.md
```

## Learning Roadmap

### Phase 1: Foundations (Ch01-03)

| Chapter | Topic | Key Takeaway |
|---------|-------|--------------|
| 01 | Attention Basics | O(N^2) complexity, memory bottleneck |
| 02 | GPU Attention | GEMM mapping, Tensor Cores, Shared Memory |
| 03 | Profiling | Nsight, torch.profiler, bottleneck identification |

### Phase 2: FlashAttention Family (Ch04-06)

| Chapter | Topic | Key Takeaway |
|---------|-------|--------------|
| 04 | FlashAttention V1 | Online Softmax, Tiling, Kernel Fusion |
| 05 | FlashAttention V2 | Work Partition, Warp Specialization |
| 06 | FlashAttention V3 | Hopper TMA, WGMMA, Tensor Memory Accelerator |

### Phase 3: Inference Optimization (Ch07-09)

| Chapter | Topic | Key Takeaway |
|---------|-------|--------------|
| 07 | KV Cache | Autoregressive inference, Prefill/Decode |
| 08 | PagedAttention | vLLM block table, memory fragmentation |
| 09 | MQA / GQA | Multi/Grouped Query Attention |

### Phase 4: Advanced Patterns (Ch10-12)

| Chapter | Topic | Key Takeaway |
|---------|-------|--------------|
| 10 | Sliding Window | Mistral, Long Context |
| 11 | Sparse Attention | Longformer, Block Sparse |
| 12 | Linear Attention | Performer, Kernel Trick |

### Phase 5: Production Systems (Ch13-16)

| Chapter | Topic | Key Takeaway |
|---------|-------|--------------|
| 13 | Quantized Attention | INT8/INT4/FP8, KV Cache Quantization |
| 14 | TensorRT-LLM | Fused Attention, Plugin System |
| 15 | xFormers | Memory Efficient Attention, Dispatch |
| 16 | vLLM | PagedAttention, Continuous Batching |

### Final Project

Mini TensorRT-LLM Attention Engine with:
- FlashAttention integration
- KV Cache management
- PagedAttention
- MQA / GQA support
- Continuous Batching
- Llama-style inference

## Prerequisites

- C++ and CUDA C++
- Basic PyTorch usage
- Linear algebra fundamentals
- NVIDIA GPU with CUDA support (compute capability >= 8.0 recommended)

## Build & Run

```bash
cd attention-optimization

# Build all chapters
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run specific chapter
./chapters/chapter_01/naive_attention

# Run Python benchmarks
python src/chapter_01/benchmark.py
```

## Coding Conventions

- Code comments in English
- Notes and documentation in Chinese
- Google C++ style (2 spaces indent)
- Python: PEP 8 (4 spaces indent)
