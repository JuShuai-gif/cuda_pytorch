# CS336 中文版 LLM System 学习路线

> **课程哲学**：`accuracy = efficiency × resources`
> 在大模型时代，系统能力决定你能走多远。本课程从 tokenization 到 distributed inference，带你系统掌握 LLM 全栈技术。

---

## 目录结构

```
cs336-llm-system/
│
├── README.md                  ← 你在这里
├── requirements.txt            ← Python 依赖
├── setup.py                    ← 项目安装配置
│
├── note/                       ← 17 篇课程笔记 (lecture-01 ~ lecture-17)
│   └── lecture-XX.md
│
├── papers/                     ← 8 篇核心论文精读
│   └── paper-XX.md
│
├── src/                        ← 17 个 lecture 对应的源码
│   ├── lecture-01/             ←  BPE tokenization
│   ├── lecture-02/             ←  FLOPs, memory, mixed precision
│   ├── lecture-03/             ←  Transformer 实现
│   ├── lecture-04/             ←  Training loop, optimizer, scheduler
│   ├── lecture-05/             ←  GPU specs, bandwidth, occupancy
│   ├── lecture-06/             ←  Triton basics: GELU, softmax, matmul
│   ├── lecture-07/             ←  DDP, FSDP, TP, PP
│   ├── lecture-08/             ←  Attention deep dive, FlashAttention
│   ├── lecture-09/             ←  Normalization (LayerNorm, RMSNorm), Activation
│   ├── lecture-10/             ←  Inference basics, KV cache
│   ├── lecture-11/             ←  Advanced inference: PagedAttention, continuous batching
│   ├── lecture-12/             ←  Evaluation, perplexity, benchmarks
│   ├── lecture-13/             ←  Data sources
│   ├── lecture-14/             ←  Data pipeline: filtering, dedup, mixing
│   ├── lecture-15/             ←  Alignment: RLHF, DPO, reward model
│   ├── lecture-16/             ←  Scaling laws: Kaplan, Chinchilla, muP
│   └── lecture-17/             ←  Multimodal: CLIP, ViT, fusion
│
├── labs/                       ← 5 个动手实验
│   ├── lab-01/                 ←  Tokenization & Training Basics
│   ├── lab-02/                 ←  Resource Accounting & GPU
│   ├── lab-03/                 ←  Systems: Kernels & Parallelism
│   ├── lab-04/                 ←  Scaling Laws
│   └── lab-05/                 ←  Data Pipeline & Alignment
│
├── project/
│   └── mini_llm_system/        ← 核心项目：mini LLM 全栈实现
│       ├── transformer/        ←  Transformer 模型 (attention, layers, RoPE, RMSNorm)
│       ├── training/           ←  训练框架 (dataset, optimizer, LR scheduler)
│       ├── inference/          ←  推理引擎 (generation, KV cache)
│       ├── distributed/        ←  分布式训练 (DDP, FSDP)
│       ├── tokenizer/          ←  BPE tokenizer
│       ├── profiling/          ←  性能分析工具
│       ├── benchmark/          ←  Benchmark suite
│       └── diagrams/           ←  架构图
│
├── benchmarks/                 ← (待建设) Benchmark 数据
├── cuda/                       ← (待建设) CUDA kernel 实现
├── diagrams/                   ← (待建设) 图表素材
├── distributed/                ← (待建设) 分布式训练进阶
├── inference/                  ← (待建设) 推理系统进阶
├── tokenizer/                  ← (待建设) Tokenizer 进阶
└── transformer/                ← (待建设) Transformer 变体
```

---

## 学习路线（10 周计划）

### 第 1 周：Tokenization + 资源基础

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| BPE tokenization 原理   | `src/lecture-01/`, `note/lecture-01.md` |
| FLOPs 计算              | `src/lecture-02/`            |
| Mixed precision 训练    | `src/lecture-02/`            |
| **动手实验**            | `labs/lab-01/`               |

**学习目标**：理解 BPE 算法，能手算 Transformer FLOPs，理解 cross-entropy 和 perplexity。

---

### 第 2 周：Transformer + Training

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| Transformer 架构        | `src/lecture-03/`, `project/mini_llm_system/transformer/` |
| Training loop           | `src/lecture-04/`            |
| Optimizer (AdamW)       | `src/lecture-04/optimizer.py` |
| LR Scheduler (cosine)   | `src/lecture-04/lr_scheduler.py` |

**学习目标**：完整实现一个 Transformer 模型，理解训练循环的每个组件。

---

### 第 3 周：GPU Architecture + Triton Kernels

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| GPU memory hierarchy    | `src/lecture-05/`            |
| Roofline model          | `src/lecture-05/`            |
| Triton 编程基础         | `src/lecture-06/`            |
| Fused GELU / Softmax    | `src/lecture-06/`            |
| **动手实验**            | `labs/lab-02/`, `labs/lab-03/` |

**学习目标**：理解 GPU 计算模型，能用 Triton 写 fused kernel。

---

### 第 4 周：Distributed Training

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| DDP (Distributed Data Parallel) | `src/lecture-07/ddp_train.py` |
| FSDP (Fully Sharded)    | `src/lecture-07/fsdp_simple.py` |
| Tensor Parallelism      | `src/lecture-07/tensor_parallel.py` |
| Pipeline Parallelism    | `src/lecture-07/pipeline_parallel.py` |
| 通信原语 (AllReduce)    | `src/lecture-07/collective_ops_demo.py` |

**学习目标**：理解 4 种并行策略的原理和适用场景，能动手配置 DDP/FSDP 训练。

---

### 第 5 周：Attention Deep Dive

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| Multi-Head Attention    | `src/lecture-08/attention.py` |
| GQA / MQA / MLA         | `papers/`                    |
| FlashAttention (原理)   | `src/lecture-08/flash_attention_simple.py` |
| Memory-efficient attention | `src/lecture-08/`         |

**学习目标**：深入理解 attention 的计算图、io-awareness、以及各种变体。

---

### 第 6 周：Normalization & Activation

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| LayerNorm vs RMSNorm    | `src/lecture-09/normalization.py` |
| GELU / SwiGLU / ReLU²   | `src/lecture-09/activation.py` |
| Pre-norm vs Post-norm   | `note/lecture-09.md`         |

**学习目标**：理解各种 normalization 和 activation 的数值特性与性能差异。

---

### 第 7 周：Inference System

| 内容                    | 材料                             |
| ----------------------- | -------------------------------- |
| KV Cache                | `src/lecture-10/kv_cache.py`, `project/mini_llm_system/inference/kv_cache.py` |
| Prefill vs Decode       | `src/lecture-10/inference_loop.py` |
| PagedAttention          | `src/lecture-11/paged_attention.py` |
| Continuous Batching     | `src/lecture-11/continuous_batching.py` |
| Speculative Decoding    | `src/lecture-11/speculative_decoding.py` |

**学习目标**：理解 LLM 推理的全流程，掌握 KV cache 优化和 PagedAttention。

---

### 第 8 周：Evaluation + Scaling Laws

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| Perplexity 评估         | `src/lecture-12/perplexity.py` |
| MMLU / HumanEval 等 benchmark | `src/lecture-12/`       |
| Kaplan Scaling Law      | `src/lecture-16/scaling_law_fit.py` |
| Chinchilla Scaling Law  | `src/lecture-16/compute_optimal.py` |
| muP (Maximal Update Param.) | `src/lecture-16/mup_demo.py` |
| **动手实验**            | `labs/lab-04/`               |

**学习目标**：理解 scaling laws 并动手拟合，理解 evaluation 的工程实现。

---

### 第 9 周：Data Pipeline

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| 数据来源分析            | `src/lecture-13/`            |
| 数据清洗与过滤          | `src/lecture-14/filtering.py` |
| 去重 (MinHash)          | `src/lecture-14/`            |
| 数据混合                | `src/lecture-14/data_mixing.py` |
| **动手实验**            | `labs/lab-05/`               |

**学习目标**：搭建完整的训练数据 pipeline，理解数据质量对模型的影响。

---

### 第 10 周：Alignment

| 内容                    | 材料                         |
| ----------------------- | ---------------------------- |
| RLHF (SFT + RM + PPO)   | `src/lecture-15/`            |
| DPO (Direct Preference) | `src/lecture-15/dpo_loss.py` |
| GRPO (DeepSeek)         | `src/lecture-15/grpo_simple.py` |
| Reward Modeling         | `src/lecture-15/reward_model.py` |
| Multimodal (CLIP/ViT)   | `src/lecture-17/`            |

**学习目标**：理解 alignment 的核心算法，能实现 DPO 和 GRPO loss。

---

## 如何阅读源码

### Llama 系列 (Meta)

```bash
git clone https://github.com/meta-llama/llama.git
```

| 关键文件               | 关注点                        |
| ---------------------- | ----------------------------- |
| `llama/model.py`       | RMSNorm, RoPE, SwiGLU, GQA    |
| `llama/generation.py`  | KV Cache, sampling strategies |
| `llama/tokenizer.py`   | SentencePiece BPE             |

**阅读顺序**：`model.py` → `generation.py` → `tokenizer.py`

### Megatron-LM (NVIDIA)

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
```

| 关键文件                        | 关注点                     |
| ------------------------------- | -------------------------- |
| `megatron/model/transformer.py` | TP (tensor parallelism)    |
| `megatron/core/pipeline_parallel/` | PP (pipeline parallelism) |
| `megatron/core/distributed/`    | FSDP, DDP, distributed init |

**阅读顺序**：理解 TP → 理解 PP → 理解混合并行

### vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
```

| 关键文件                       | 关注点                     |
| ------------------------------ | -------------------------- |
| `vllm/worker/model_runner.py`  | KV Cache 管理              |
| `vllm/core/block_manager.py`   | PagedAttention block table |
| `vllm/core/scheduler.py`       | Continuous batching        |

**阅读顺序**：理解 KV Cache → 理解 PagedAttention → 理解 scheduler

### FlashAttention (Dao-AILab)

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
```

| 关键文件                              | 关注点                 |
| ------------------------------------- | ---------------------- |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | Forward CUDA kernel |
| `csrc/flash_attn/src/flash_bwd_kernel.h` | Backward CUDA kernel |

**阅读建议**：先读 [FlashAttention 论文](https://arxiv.org/abs/2205.14135) 理解 tiling 和 recomputation 的算法原理，再看 Triton 版本（`flash_attn_triton.py`），最后看 CUDA 实现。

---

## 如何学习 CUDA for LLM

### 学习路径：从 Triton 到 CUDA

```
第 1 步：Triton 入门 (2 周)
├── Triton 官方教程 (Fused Softmax, Matrix Multiplication)
├── 本项目 src/lecture-06/ 下的 GELU / Softmax / MatMul kernel
├── labs/lab-03/ 的 Fused RMSNorm kernel
└── 目标：理解 block-level parallelism 和 memory hierarchy

第 2 步：CUDA 基础 (2 周)
├── CUDA C++ Programming Guide (前 3 章)
├── 理解 thread / block / grid hierarchy
├── 理解 shared memory, bank conflicts
├── 写一个简单的 vector_add kernel
└── 目标：能用 CUDA C++ 写简单 kernel

第 3 步：CUDA 进阶 (2 周)
├── Cooperative Groups
├── Warp-level primitives (__shfl_sync, etc.)
├── Async copy (cp.async)
├── Tensor Core programming (mma.sync)
└── 目标：理解 FlashAttention CUDA 实现的关键技术

第 4 步：实战 (持续)
├── 读 FlashAttention 源码
├── 读 CUTLASS 源码
├── 写自己的 fused attention kernel
└── 用 Nsight Compute 做 profiling
```

### 关键资源

| 资源                       | 链接                                                            |
| -------------------------- | --------------------------------------------------------------- |
| CUDA C++ Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/          |
| Triton 官方教程            | https://triton-lang.org/main/getting-started/tutorials/         |
| GPU Mode 课程              | https://github.com/gpu-mode/lectures                            |
| CUTLASS                    | https://github.com/NVIDIA/cutlass                               |
| PMPP (Programming Massively Parallel Processors) | 推荐书籍                        |

---

## 如何学习分布式训练

### 学习路径：从 DDP 到 Megatron-LM

```
第 1 步：DDP (2 天)
├── 运行 src/lecture-07/ddp_train.py
├── 理解 gradient AllReduce 的时机
├── 理解 gradient bucketing
└── 实验：单机 2 卡 vs 4 卡的 scaling

第 2 步：FSDP (3 天)
├── 运行 src/lecture-07/fsdp_simple.py
├── 理解 AllGather + ReduceScatter
├── 理解 activation checkpointing 的组合
└── 实验：不同 sharding strategy 的内存/速度 tradeoff

第 3 步：Tensor Parallelism (3 天)
├── 运行 src/lecture-07/tensor_parallel.py
├── 理解 column-wise vs row-wise parallelism
├── 理解 TP 的通信量分析
└── 实验：TP degree 对 throughput 的影响

第 4 步：Pipeline Parallelism (2 天)
├── 运行 src/lecture-07/pipeline_parallel.py
├── 理解 1F1B schedule
├── 理解 bubble ratio 公式
└── 实验：不同 micro-batch 数的效果

第 5 步：混合并行 (1 周)
├── 阅读 Megatron-LM 源码
├── 理解 TP + PP + DP 的 3D parallelism
├── 理解 ZeRO-1/2/3 与 FSDP 的关系
└── 设计你自己的 distributed strategy
```

### 关键概念速查

| 策略 | 切分什么             | 通信量                | 通信类型          |
| ---- | -------------------- | --------------------- | ----------------- |
| DDP  | 数据 (batch)         | AllReduce gradients   | collective        |
| FSDP | 参数 + 优化器状态    | AllGather + ReduceScatter | collective    |
| TP   | 层内权重 (列/行切分) | AllReduce per layer   | collective        |
| PP   | 层 (按层切分)        | P2P activations/grads | point-to-point    |

---

## 核心项目：mini_llm_system

`project/mini_llm_system/` 是本课程的核心项目，实现了一个完整的 mini LLM 系统。

### 项目架构

```
mini_llm_system/
├── transformer/          ← 模型实现
│   ├── config.py         ←  Model configuration (LLaMA-style)
│   ├── attention.py      ←  Multi-head attention + GQA
│   ├── rotary_embedding.py ← RoPE (Rotary Position Embedding)
│   ├── normalization.py  ←  RMSNorm (with Triton fused kernel support)
│   ├── layers.py         ←  TransformerBlock
│   └── test_model.py     ←  Model correctness tests
│
├── training/             ← 训练框架
│   ├── dataset.py        ←  Data loading + batching
│   ├── loss.py           ←  Cross-entropy + auxiliary losses
│   ├── optimizer.py      ←  AdamW with weight decay
│   ├── lr_scheduler.py   ←  Cosine warmup scheduling
│   ├── trainer.py        ←  Training loop + checkpointing
│   └── train.py          ←  入口脚本
│
├── inference/            ← 推理引擎
│   ├── inference_engine.py ← Generator class
│   ├── generation.py     ←  Sampling strategies (greedy, top-k, top-p)
│   └── kv_cache.py       ←  KV Cache implementation
│
├── distributed/          ← 分布式
│   ├── ddp_trainer.py    ←  DDP wrapper
│   └── fsdp_demo.py      ←  FSDP example
│
├── tokenizer/            ← Tokenizer
│   ├── bpe_tokenizer.py  ←  BPE implementation
│   └── test_tokenizer.py ←  Tokenizer tests
│
├── profiling/            ← 性能分析
│   ├── torch_profiler.py ←  PyTorch profiler wrapper
│   ├── memory_profiler.py ← Memory tracking
│   └── cuda_timing.py    ←  CUDA event timing
│
├── benchmark/            ← 基准测试
│   ├── benchmark_attention.py  ← Attention kernel benchmarks
│   ├── benchmark_inference.py  ← Inference throughput
│   └── benchmark_training.py   ← Training throughput
│
└── diagrams/             ← 架构图
    ├── architecture.mermaid      ← 系统架构
    ├── training_pipeline.mermaid ← 训练 pipeline
    ├── inference_pipeline.mermaid ← 推理 pipeline
    └── data_flow.mermaid         ← 数据流
```

### 运行指南

```bash
# 安装依赖
pip install -e .

# 运行测试
cd project/mini_llm_system
python transformer/test_model.py

# 训练一个小模型（单卡）
python training/train.py \
    --hidden_dim 512 \
    --num_layers 8 \
    --batch_size 32 \
    --seq_len 512 \
    --max_steps 1000

# DDP 训练（4 卡）
torchrun --nproc_per_node=4 training/train.py \
    --hidden_dim 512 \
    --num_layers 8 \
    --batch_size 128 \
    --seq_len 512

# 推理示例
python -c "
from inference.inference_engine import InferenceEngine
engine = InferenceEngine.from_pretrained('./checkpoints/model.pt')
output = engine.generate('Once upon a time', max_new_tokens=100)
print(output)
"
```

---

## 参考资源

### 核心论文

| 论文                                                           | 主题                    | 年份 |
| -------------------------------------------------------------- | ----------------------- | ---- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  | Transformer 架构        | 2017 |
| [Scaling Laws (Kaplan et al.)](https://arxiv.org/abs/2001.08361) | 缩放定律               | 2020 |
| [Megatron-LM](https://arxiv.org/abs/1909.08053)                | 模型并行                | 2019 |
| [FlashAttention](https://arxiv.org/abs/2205.14135)             | IO-aware attention      | 2022 |
| [Chinchilla](https://arxiv.org/abs/2203.15556)                 | Compute-optimal 训练    | 2022 |
| [LLaMA](https://arxiv.org/abs/2302.13971)                      | 高效 LLM 架构           | 2023 |
| [DPO](https://arxiv.org/abs/2305.18290)                        | Direct Preference Opt.  | 2023 |
| [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)      | 高效推理                | 2023 |
| [GQA](https://arxiv.org/abs/2305.13245)                        | Grouped Query Attention | 2023 |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691)           | 更快 attention          | 2023 |
| [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)        | Group Relative PPO      | 2024 |
| [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434)          | Multi-head Latent Attn  | 2024 |

### 推荐博客

| 博客                                                           | 内容                            |
| -------------------------------------------------------------- | ------------------------------- |
| [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/) | 推理 FLOPs/Memory 分析         |
| [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) | 训练 FLOPs 推导               |
| [How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/) | 分布式训练综述             |
| [A Guide to CUDA for AI](https://www.hpcaitech.com/blog/cuda-for-ai/) | CUDA 入门指南          |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Transformer 图解           |

### 推荐视频

| 视频                                                           | 内容                            |
| -------------------------------------------------------------- | ------------------------------- |
| [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 从零实现 GPT   |
| [CUDA Mode 系列](https://www.youtube.com/@CUDAMode)            | GPU 编程讲座系列                |
| [Keller Jordan - FlashAttention](https://www.youtube.com/watch?v=gMOAud4hZg4) | FlashAttention 详解          |
| [DeepLearning.AI - Efficient Training](https://www.deeplearning.ai/courses/) | 高效训练专项                |

### 推荐代码库

| 仓库                                                           | 用途                  |
| -------------------------------------------------------------- | --------------------- |
| [nanoGPT](https://github.com/karpathy/nanoGPT)                 | 最小 GPT 实现         |
| [lit-gpt](https://github.com/Lightning-AI/litgpt)              | 模块化 GPT 实现       |
| [gpt-neox](https://github.com/EleutherAI/gpt-neox)             | 大规模训练框架        |
| [Triton Kernels](https://github.com/unslothai/unsloth)         | 高效 Triton kernel    |
| [LLM Foundry](https://github.com/mosaicml/llm-foundry)         | MosaicML 训练框架     |

### 环境搭建

```bash
# 推荐使用 CUDA 12.1+ 和 PyTorch 2.3+
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

---

## 如何进入 LLM Infra 岗位

### 技能栈

```
基础层 (必须)
├── Python / C++ / CUDA C++
├── PyTorch (autograd, DDP, FSDP, torch.compile)
├── Linux 系统 (进程、内存、网络)
└── Git / CI/CD

系统层 (核心)
├── GPU Architecture (SM, memory hierarchy, Tensor Core)
├── NCCL / MPI 通信
├── CUDA kernel 编程 (Shared memory, bank conflicts, occupancy)
├── 性能分析工具 (Nsight Systems, Nsight Compute, PyTorch Profiler)
└── 数值计算基础 (FP16/BF16, mixed precision, gradient scaling)

模型层 (加分)
├── Transformer 架构深入理解 (attention variants, normalization)
├── Scaling laws (compute-optimal training)
├── Alignment 算法 (RLHF, DPO)
└── Inference 优化 (quantization, speculative decoding)

工程层 (加分)
├── 大规模集群管理
├── 分布式文件系统
├── 容错和 checkpointing
└── 资源调度 (Slurm / K8s)
```

### 学习建议

1. **先理解再动手**：读论文理解算法原理 → 看源码理解工程实现 → 自己写简化版
2. **从 nanoGPT 到 Megatron**：先跑通一个 100M 参数的模型在单卡上，再加入分布式
3. **profiling 是最好的老师**：使用 PyTorch Profiler、Nsight 分析性能瓶颈
4. **写 kernel**：从 Triton 开始（更友好），然后过渡到 CUDA
5. **读 vLLM 源码**：vLLM 是当前 LLM 推理系统的 SOTA 实现
6. **参与开源**：给 PyTorch、vLLM、Triton 等项目提交 PR，是最好的学习方式

### 常见面试方向

| 方向               | 考察点                                                       |
| ------------------ | ------------------------------------------------------------ |
| **分布式训练**     | DDP/FSDP/TP/PP 原理，AllReduce 通信分析，ZeRO optimizer      |
| **CUDA kernel**    | Memory hierarchy, bank conflict, occupancy, warp divergence   |
| **推理优化**       | KV Cache, PagedAttention, continuous batching, quantization   |
| **Scaling laws**   | Kaplan vs Chinchilla, compute-optimal ratio, muP              |
| **系统设计**       | 设计一个可训练 100B 模型的系统，瓶颈分析和方案设计            |

---

## 贡献指南

本课程持续更新中。欢迎提交 PR 改进代码、笔记或实验内容。

## License

MIT
