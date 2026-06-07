# Lecture 09 - Knowledge Distillation

> MIT 6.5940: Efficient Deep Learning Computing  
> Topic: Knowledge Distillation (KD) — 大模型教小模型

## Overview

Knowledge Distillation transfers knowledge from a large "teacher" model to a small "student" model by matching softened logit distributions. This enables deploying compact models that retain most of the teacher's accuracy.

## Key Concepts

- **KD Loss**: `L = α * L_CE + (1-α) * T² * KL(softmax(z_t/T) || softmax(z_s/T))`
- **Temperature T**: Controls softness of probability distribution. Higher T → softer labels → more information transfer.
- **Teacher-Student Gap**: Larger gap requires careful temperature tuning.

## Implementation

| Component    | Description                                      |
| ------------ | ------------------------------------------------ |
| TeacherCNN   | 4 conv blocks + 2 FC layers (~2.8M params)       |
| StudentCNN   | 3 conv blocks + 1 FC layer (~0.2M params)        |
| KD Loss      | α=0.5 weighted combination of CE + KL divergence |
| Temperatures | T ∈ [1, 2, 4, 8, 16]                             |

## Usage

```bash
cd src/lecture-09
python main.py
```

The script automatically downloads CIFAR-10 (~170MB), trains teacher and student models on CPU, and prints a comparison table across all temperatures.

## Expected Output

```
============================================================
Knowledge Distillation Results - CIFAR-10
============================================================
Method              Temperature    Accuracy
------------------------------------------------------------
Student (no KD)         -           65.23%
Student + KD            T=1         67.12%
Student + KD            T=2         68.45%
Student + KD            T=4         69.87%
Student + KD            T=8         68.91%
Student + KD            T=16        67.34%
Teacher                  -          78.56%
============================================================
```

## References

- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- MIT 6.5940 Lecture 09 Slides
