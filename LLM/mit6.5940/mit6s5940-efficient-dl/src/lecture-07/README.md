# Lecture 07: Simplified NAS Random Search on CIFAR-10

## Overview

This code accompanies **MIT 6.5940 Lecture 07: Neural Architecture Search (NAS)**.
It implements a **random-search NAS** algorithm that explores a CNN search
space and visualizes the trade-off between model accuracy and computational
cost (MACs).

The search space is defined by three dimensions:

| Dimension     | Choices                | Description                          |
|---------------|------------------------|--------------------------------------|
| Kernel size   | `[3, 5, 7]`            | Per-layer Conv2d kernel size         |
| Channels      | `[16, 32, 64, 128]`    | Per-layer output channel count       |
| Depth         | `[1, 2, 3, 4]`         | Number of convolutional layers       |

This yields **22,620 possible architectures**. The random search samples
a subset (default 20) and trains each candidate for a few proxy epochs on
CIFAR-10. Results are visualized as an **accuracy vs MACs scatter plot**,
with a simple **Pareto-frontier** analysis to identify non-dominated
architectures.

## Prerequisites

```bash
pip install torch torchvision matplotlib
```

An internet connection is needed for the first run to download CIFAR-10
(~170 MB). Subsequent runs use the cached dataset.

## Usage

```bash
cd src/lecture-07
python main.py
```

The script runs entirely on CPU and produces:

1. **Search space summary** -- dimensionality, total configurations, sampling budget.
2. **CIFAR-10 loading** -- a fixed subset (5K train, 2K val) for fast proxy evaluation.
3. **Random search** -- 20 architectures sampled, each trained for 3 epochs.
4. **Results summary** -- accuracy range, MACs range, best architecture, Pareto frontier.
5. **Accuracy vs MACs plot** -- saved as `nas_accuracy_vs_macs.png`.

Typical runtime on a modern CPU: **2--3 minutes** for 20 architectures.

## Key Functions

| Function | Description |
|----------|-------------|
| `random_sample_architecture(kernel_choices, channel_choices, depth_choices)` | Randomly sample depth, per-layer kernel sizes, and per-layer channel counts |
| `NasCNN(spec)` | Build a VGG-style CNN from an `ArchSpec` (Conv2d->BN->ReLU->MaxPool each layer, then GAP+Linear) |
| `train_and_evaluate(spec, train_loader, val_loader)` | Build model, train for `NAS_EPOCHS`, return accuracy and wall time |
| `count_macs(model, input_shape)` | Estimate total Conv2d MACs via forward hooks |
| `estimate_macs_conv2d(in_c, out_c, k, h, w)` | Per-layer MACs formula |
| `plot_accuracy_vs_macs(results, save_path)` | Scatter plot with annotated architecture labels |
| `arch_summary(spec)` | Compact string representation of an architecture |

## Concepts

### Neural Architecture Search (NAS)

NAS automates the design of neural network architectures. Instead of manually
tuning hyper-parameters (depth, width, kernel size), a **search algorithm**
explores the space and selects architectures that optimize a trade-off
(e.g., accuracy vs FLOPs).

### Random Search Baseline

Random search is the simplest NAS strategy: uniformly sample architectures
from the search space, train each independently, and pick the best. Despite
its simplicity, random search often performs surprisingly well and serves as
an important baseline against which more sophisticated methods (evolutionary
search, reinforcement learning, differentiable NAS) are compared.

### Proxy Training

Training every candidate from scratch on the full dataset would be
prohibitively expensive. **Proxy training** uses a shortcut:
- A **subset** of the training data (5K CIFAR-10 samples)
- A **small number of epochs** (3 instead of 200+)
- The resulting accuracy **ranking** is usually well-correlated with
  the ranking from full training, even if the absolute numbers are lower.

### MACs (Multiply-Accumulate Operations)

MACs measure the computational cost of a forward pass:

```
MACs(Conv2d) = C_out * H_out * W_out * C_in * K_h * K_w
```

For a VGG-style CNN with MaxPool2d(2) after every layer, the feature map
size halves each layer: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2.

### Pareto Frontier

An architecture **A dominates B** if A has **both** higher accuracy **and**
lower MACs than B. The **Pareto frontier** is the set of architectures
that are not dominated by any other candidate. These represent the
best possible trade-offs achievable within the sampled architectures.

### Typical Results

With 20 random samples and 3 proxy-training epochs:

| Metric           | Value          |
|------------------|----------------|
| Best accuracy    | ~44%           |
| Lowest MACs      | ~0.88M         |
| Pareto frontier  | ~4 candidates  |
| Search time      | ~2--3 min (CPU)|

The scatter plot typically shows a **positive correlation** between MACs
and accuracy: larger/deeper models achieve higher accuracy but at higher
computational cost. The Pareto frontier highlights the "sweet spot"
architectures.

## References

- Zoph, B., & Le, Q. V. "Neural Architecture Search with Reinforcement
  Learning." ICLR 2017.
- Li, L., & Talwalkar, A. "Random Search and Reproducibility for Neural
  Architecture Search." UAI 2019.
- Bergstra, J., & Bengio, Y. "Random Search for Hyper-Parameter
  Optimization." JMLR 2012.
- Hutter, F., Kotthoff, L., & Vanschoren, J. (Eds.). "Automated Machine
  Learning: Methods, Systems, Challenges." Springer 2019.
- MIT 6.5940 Lecture 07: Neural Architecture Search --
  https://hanlab.mit.edu/courses/2025-fall-65940
