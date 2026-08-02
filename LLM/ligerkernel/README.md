# LLM Training Kernel Lab

Liger-Kernel operator reimplementation with Triton/TileLang comparison.

Educational reimplementation inspired by
[LinkedIn Liger-Kernel](https://github.com/linkedin/Liger-Kernel).
It independently implements selected operators using PyTorch, Triton, and
TileLang for correctness and performance comparison.

The original Liger-Kernel project is licensed under the BSD 2-Clause License.

See [学习指南.md](学习指南.md) for methodology and [学习顺序.md](学习顺序.md) for the reading order.

## Planned operators

| Operator              | Reference | Triton | TileLang |
| --------------------- | :-------: | :----: | :------: |
| RMSNorm               |   todo    |  todo  |   todo   |
| LayerNorm             |   todo    |  todo  |   todo   |
| SwiGLU                |   todo    |  todo  |   todo   |
| GEGLU                 |   todo    |  todo  |    -     |
| RoPE                  |   todo    |  todo  |   todo   |
| CrossEntropy          |   todo    |  todo  |   todo   |
| Fused Linear CrossEntropy |  todo  |  todo  |   todo   |
