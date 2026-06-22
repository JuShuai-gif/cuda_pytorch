# MIT 6.5940 Efficient Deep Learning -- Source Code

本目录保存课程每讲的可运行代码。后续补齐工作以 ../MISSING_AND_EXPANSION.md 为准，目标是让每个 `note/lecture-XX.md` 都能对应一个可运行实验、明确输出指标和工业验收问题。

当前代码分两类：

- `lecture-XX/main.py`：讲义级概念实验，适合快速理解算法。
- `model_compression/`：更接近工业实战的统一 benchmark，覆盖剪枝、量化、ONNX Runtime、TensorRT 可选路径和报告生成。

推荐先跑统一压缩 benchmark，再回到单讲代码：

```bash
cd /home/hpc/ghr_code/cuda_pytorch/mit6.5940/mit6s5940-efficient-dl
python src/model_compression/benchmark_compression.py --runs 3 --warmup 1 --train-steps 1
```

所有代码默认优先 CPU 可运行；GPU/TensorRT/ONNX Runtime 能用时再开启对应路径。

## Directory Structure

```
src/
  lecture-01/   Efficient DL overview: efficiency metrics and trade-offs
  lecture-02/   FLOPs, MACs, memory wall, arithmetic intensity, roofline
  lecture-03/   Fine-grained pruning and sensitivity analysis
  lecture-04/   Structured/channel pruning and fine-tuning
  lecture-05/   Quantization basics: scale, zero point, calibration
  lecture-06/   PTQ/QAT, mixed precision, quantization error analysis
  lecture-07/   NAS basics: search space and search strategy
  lecture-08/   Hardware-aware NAS and Once-for-All style search
  lecture-09/   Knowledge distillation and teacher-student training
  lecture-10/   MCUNet: model-system co-design for MCU
  lecture-11/   TinyEngine: code generation, tiling, memory planning
  lecture-12/   Transformers and efficient attention
  lecture-13/   LLM deployment: KV cache, batching, quantization
  lecture-14/   LLM post-training: SFT, RLHF, DPO, LoRA/QLoRA
  lecture-15/   Long-context LLM: sparse/ring/streaming attention
  lecture-16/   Efficient ViT: token pruning/merging and attention cost
  lecture-17/   Efficient GAN/video/point-cloud models
  lecture-18/   Diffusion acceleration: fewer steps, distillation, quantization
  lecture-19/   Distributed training I: DP/DDP/FSDP/ZeRO basics
  lecture-20/   Distributed training II: compression and hybrid parallelism
  lecture-21/   On-device training, federated learning, TinyTL
  lecture-22/   Course summary and end-to-end compression pipeline
  lecture-23/   Quantum ML overview and limits
  model_compression/  Industrial-style compression benchmark and reports
```

每个 lecture 目录后续应满足：

- `README.md` 说明对应 note、运行命令和输出指标。
- `main.py` 可直接运行，优先 CPU smoke test。
- 输出至少一个可解释指标，例如 sparsity、latency、MSE、memory 或 tokens/s。

## Running

Each lecture is self-contained.  Run from the project root:

```bash
python mit6.5940/mit6s5940-efficient-dl/src/lecture-17/main.py
```

Or from any directory:

```bash
python /path/to/src/lecture-XX/main.py
```

## Dependencies

- Python 3.9+
- PyTorch >= 2.0
- torchvision
- numpy

Install with:

```bash
pip install torch torchvision numpy
```

## Lecture Summaries

| Lec | Topic | Practical Output |
| --- | --- | --- |
| 01 | Efficient DL overview | Understand accuracy/latency/memory trade-off |
| 02 | Metrics and roofline | FLOPs/MACs/arithmetic intensity report |
| 03 | Fine-grained pruning | Sparsity and sensitivity curves |
| 04 | Structured pruning | Channel-pruned model and latency comparison |
| 05 | Quantization basics | Scale/zp and quantization error demo |
| 06 | PTQ/QAT | Calibration and mixed-precision comparison |
| 07 | NAS I | Search space and sampled architectures |
| 08 | NAS II | Hardware-aware Pareto frontier |
| 09 | Knowledge distillation | Teacher/student loss and accuracy delta |
| 10 | MCUNet | Tiny model and memory budget |
| 11 | TinyEngine | Tiling/codegen/memory planning demo |
| 12 | Transformer efficiency | Attention memory and compute comparison |
| 13 | LLM deployment | KV-cache and decode benchmark |
| 14 | LLM post-training | LoRA/QLoRA/RLHF-style simulation |
| 15 | Long context | Sparse/ring attention memory comparison |
| 16 | Efficient ViT | Token merging/pruning cost comparison |
| 17 | GAN/video/point cloud | Structured compression demo |
| 18 | Diffusion efficiency | Step reduction and quantization trade-off |
| 19 | Distributed training I | Communication/memory model |
| 20 | Distributed training II | Gradient compression/parallelism model |
| 21 | On-device training | FedAvg/TinyTL activation-memory demo |
| 22 | Course summary | End-to-end compression report |
| 23 | Quantum ML | PQC simulator and limitation analysis |

## Industrial Baseline

For practical work, prefer `src/model_compression/benchmark_compression.py` as the first executable baseline. Single-lecture scripts teach concepts; the benchmark script is where metrics should converge.
