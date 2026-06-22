# Lecture 14 - PEFT (Parameter Efficient Fine-Tuning)

> MIT 6.5940: Efficient Deep Learning Computing  
> Topic: PEFT / LoRA — 低秩适配微调

## Overview

Parameter Efficient Fine-Tuning (PEFT) adapts large pretrained models to new tasks by training only a tiny fraction of parameters. LoRA (Low-Rank Adaptation) achieves this by injecting trainable low-rank matrices into frozen pretrained layers.

## Key Concepts

- **LoRA**: For a pretrained weight matrix `W ∈ R^(d×k)`, LoRA learns `A ∈ R^(r×k)` and `B ∈ R^(d×r)` where r << min(d, k). Output: `y = xW^T + (x A^T B^T) * (α/r)`
- **Rank r**: Controls capacity-efficiency tradeoff. Small r → fewer params but potentially lower accuracy.
- **Alpha scaling**: Controls the magnitude of LoRA updates relative to the base model.
- **Merging**: At inference time, LoRA weights can be merged back: `W_merged = W + (α/r) * B A`

## Implementation

| Component      | Description                                              |
| -------------- | -------------------------------------------------------- |
| LoRALinear     | Custom nn.Module with frozen W, trainable A and B        |
| Pretrained MLP | 3-layer MLP (784→256→128→10) pretrained on MNIST         |
| LoRA Fine-tune | Apply LoRA to hidden layers, train on small MNIST subset |
| Rank Sweep     | Compare r ∈ [2, 4, 8, 16] for accuracy vs params         |
| Merge/Unmerge  | Demonstrate weight merging and roundtrip fidelity         |

## Usage

```bash
cd src/lecture-14
python main.py
```

## Expected Output

```
============================================================
LoRA Fine-Tuning Results - MNIST (2048 samples)
============================================================
Method         Trainable Params    Accuracy    % of Full FT
------------------------------------------------------------
Full FT (all)      235,146          97.08%      100.00%
LoRA (r=2)           4,138          97.75%        1.76%
LoRA (r=4)           6,986          97.63%        2.97%
LoRA (r=8)          12,682          97.84%        5.39%
LoRA (r=16)         24,074          97.62%       10.24%
============================================================
```

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
- MIT 6.5940 Lecture 14 Slides
