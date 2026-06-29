# ml-systems-notes

个人 ML 系统工程笔记集，涵盖分布式计算、并行化、量化和 PyTorch 内部机制。

> 这里的所有内容都是进行中的工作。我在做实验和项目的过程中持续补充笔记。

## 目录

- [分布式技术](./distributed-techniques/) - 分布式训练基础：NCCL 集合通信（gather、all-gather、reduce、all-reduce、scatter、reduce-scatter）、混合专家模型（MoE）、并行策略（DP、DDP、ZeRO、张量/流水线并行），以及 torch.distributed 基础知识。

- [量化](./quantization/) - 从第一性原理理解模型量化：对称/非对称量化、LLM.int8()、AWQ、SmoothQuant、GPTQ/OBS/OBQ 和 QuIP。

- [torch 笔记](./torch-notes/) - PyTorch 内部机制

- [JAX 扩展手册](./jax-scaling-book/) - 在 JAX/TPU 环境下对矩阵乘法的 Roofline 分析练习。
