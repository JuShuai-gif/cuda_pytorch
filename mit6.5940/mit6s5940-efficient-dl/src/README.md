# MIT 6.5940 Efficient Deep Learning -- Source Code

Production-quality PyTorch implementations for MIT 6.5940 lectures 9-23.
All code runs entirely on CPU -- no GPU required.

## Directory Structure

```
src/
  lecture-09/   NAS: ENAS weight-sharing simulation
  lecture-10/   MCUNet: tiny model + patch-based inference
  lecture-11/   TinyEngine: code generation demo
  lecture-12/   Transformers & LLM: efficient attention patterns
  lecture-13/   LLM Deployment: KV-cache, quantization, speculative decoding
  lecture-14/   LLM Post-training: instruction tuning + RLHF simulation
  lecture-15/   Long-context LLM: ring attention, sparse attention
  lecture-16/   Vision Transformers: efficient ViT with token merging
  lecture-17/   Efficient GAN: channel pruning, TSM concept, FID simulation
  lecture-18/   Diffusion Model Efficiency: DDPM, DDIM, INT8 quantization
  lecture-19/   Distributed Training I: Data Parallel, ZeRO, comm overhead
  lecture-20/   Distributed Training II: DGC, 1-Bit SGD, hybrid parallelism
  lecture-21/   On-Device Training: FedAvg, TinyTL, activation memory
  lecture-22/   Course Summary: end-to-end compression pipeline + report
  lecture-23/   Quantum ML: PQC simulator, entanglement, expressivity
```

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

| Lec | Topic                                   | Key Concepts                                           |
| --- | --------------------------------------- | ------------------------------------------------------ |
| 09  | Neural Architecture Search              | ENAS weight-sharing, controller sampling               |
| 10  | MCUNet: Tiny Deep Learning on MCU       | Patch-based inference, memory optimization             |
| 11  | TinyEngine: Efficient Inference Engine  | Code generation, loop unrolling, tiling patterns       |
| 12  | Transformers and LLM Efficiency         | Sparse attention, linear attention, grouped-query       |
| 13  | LLM Deployment and Serving              | KV-cache, quantization, speculative decoding, batching  |
| 14  | LLM Post-training                       | SFT, RLHF, DPO, alignment techniques                   |
| 15  | Long-Context LLM                        | Ring attention, sparse attention, memory-efficient      |
| 16  | Vision Transformers (ViT)               | Token merging, efficient self-attention                |
| 17  | Efficient GAN / Video Optimization      | Channel pruning, TSM, latency benchmarking             |
| 18  | Diffusion Model Efficiency              | DDPM, DDIM, INT8 quantization, step reduction          |
| 19  | Distributed Training I                  | Data Parallel, ZeRO stages, communication overhead     |
| 20  | Distributed Training II                 | DGC, 1-Bit SGD, hybrid DP/PP/TP memory                 |
| 21  | On-Device Training & Transfer Learning  | Federated Learning, FedAvg, TinyTL                     |
| 22  | Course Summary                          | Full compression pipeline, pruning, quantization       |
| 23  | Quantum ML                              | PQC simulation, entanglement, limitations discussion   |
