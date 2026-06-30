# PyTorch 笔记

个人 ML 系统工程笔记集，涵盖分布式计算、并行化、量化和 PyTorch 内部机制。

> 这里的所有内容都是进行中的工作。我在做实验和项目的过程中持续补充笔记。

## 目录

- [torch.compile](./torch.compile/) - torch.compile 编译栈详解（Dynamo、Inductor、FX Graph）
- [dynamo](./dynamo/) - TorchDynamo 内部机制：字节码拦截、guard、graph break
- [fx_graphs](./fx_graphs/) - FX 图追踪与变换
- [inductor](./inductor/) - TorchInductor 代码生成与优化
- [autograd](./autograd/) - 自动求导机制
- [dataloader](./dataloader/) - DataLoader 源码分析与 mini 实现
- [amp](./amp/) - 混合精度（AMP）与 Conv-BN 融合源码分析
- [分布式技术](./distributed_techniques/) - 分布式训练基础：NCCL 集合通信、MoE、并行策略（DP/DDP/ZeRO/TP/PP）、torch.distributed
- [量化](./quantization/) - 从第一性原理理解模型量化：对称/非对称量化、LLM.int8()、AWQ、SmoothQuant、GPTQ/OBS/OBQ、QuIP
- [JAX 扩展手册](./jax-scaling-book/) - 在 JAX/TPU 环境下对矩阵乘法的 Roofline 分析练习
- [PyTorch 源码阅读指南](./pytorch_source_guide.md) - PyTorch 四层架构概览
