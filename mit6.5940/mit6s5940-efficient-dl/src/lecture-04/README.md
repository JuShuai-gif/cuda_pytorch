# Lecture 04: Channel Pruning with Frobenius Norm

## Overview

This code accompanies **MIT 6.5940 Lecture 04: Pruning & Sparsity (Part II)**.
It implements **structured channel pruning** -- removing entire output channels
(filters) from Conv2d layers -- using the **Frobenius norm** as the importance
criterion.

Unlike fine-grained (unstructured) pruning (Lecture 03), channel pruning yields
a smaller, dense model that runs faster on **any hardware** without requiring
special sparse-matrix libraries.

The pipeline includes:
1. Training a small CNN from scratch on synthetic data
2. Ranking Conv2d output channels by Frobenius (L2) norm
3. Removing the least-important channels and building a smaller model
4. Fine-tuning the pruned model (5 epochs) to recover accuracy
5. Comparing accuracy, parameter count, MACs, and inference latency
6. Saving the pruned model to disk

## Prerequisites

```bash
pip install torch
```

All computations run on CPU -- no GPU required.

## Usage

```bash
cd src/lecture-04
python main.py
```

Example output:

```
======================================================================
  LECTURE 04: Channel Pruning with Frobenius Norm
======================================================================

[1] Generating synthetic dataset ...
  Train: torch.Size([2000, 3, 32, 32]), Test: torch.Size([500, 3, 32, 32])

[2] Building DemoCNN and training for 10 epochs ...
  Epoch  1  loss=2.3152
  Epoch 10  loss=1.4254
  Original accuracy: 0.1340

[3] Frobenius norm importance ranking (sanity check) ...
  block1.conv weight shape: (64, 3, 3, 3)
  Importance scores (first 8 of 64): [0.606, 0.668, 0.627, ...]
  Top-5 channel indices: [6, 43, 23, 1, 53]

[4] Pruning 30% of channels per layer ...
  block1: 64 -> 44 output channels (20 pruned)
  block2: 128 -> 89 output channels (39 pruned)
  block3: 256 -> 179 output channels (77 pruned)
  block4: 256 -> 179 output channels (77 pruned)

[5] Fine-tuning pruned model (5 epochs) ...
    Epoch  1/5  loss=1.9766  acc=0.1080
    Epoch  5/5  loss=0.7302  acc=0.1020

[6] Comparison: Original vs Pruned (+ fine-tuned)
  Original              acc=0.1340  params=  964,170  MACs= 133,890,048
  Pruned + FT           acc=0.1020  params=  470,962  MACs=  65,399,616

[7] Saving pruned model to 'pruned_model.pth' ...
```

## Key Functions

| Function | Purpose |
|---|---|
| `frobenius_importance(weight)` | Compute per-output-channel importance via Frobenius (L2) norm |
| `select_top_channels(importance, ratio)` | Select top-k channel indices to keep |
| `channel_prune(model, prune_ratio)` | Build a smaller model by removing least-important channels |
| `fine_tune(model, ...)` | Retrain the pruned model for recovery (5 epochs) |
| `estimate_macs(model, input_shape)` | Estimate Conv2d MACs via forward hooks |
| `measure_latency(model, input_shape)` | Measure average CPU inference latency |
| `count_params(model)` | Count total parameters |

## What You Learn

### 1. Frobenius Norm as Importance Criterion

For a Conv2d weight tensor `W` of shape `[C_out, C_in, K, K]`, the Frobenius
norm of output channel `i` is:

```
importance[i] = ||W[i, :, :, :]||_F = sqrt( sum( W[i, :, :, :]^2 ) )
```

Channels with smaller Frobenius norm are considered less important because
they produce smaller-magnitude outputs.  By removing them, we discard filters
that contribute the least to the network's forward pass.

**Why Frobenius norm?**
- **Computationally cheap**: a single L2-norm per channel
- **Data-free**: no need for calibration data or gradients
- **Empirically effective**: correlates well with channel importance in practice

### 2. Structured vs Unstructured Pruning

| Aspect | Unstructured (Lecture 03) | Structured (Lecture 04) |
|---|---|---|
| **Granularity** | Individual scalar weights | Entire output channels |
| **Result** | Sparse weight matrices | Smaller dense matrices |
| **Speedup** | Needs sparse BLAS / HW | Works on any hardware |
| **Accuracy** | Better at same sparsity | Slightly worse |
| **Implementation** | Zero-masking | Model reconstruction |

Structured pruning produces a physically smaller model: the weight tensors
shrink, so both memory and compute decrease proportionally.

### 3. Cascading Channel Removal

Channel pruning is **not** independent per layer.  When we prune output
channels of layer `i`, we must also prune the **corresponding input channels**
of layer `i+1`, because the channel dimensions must match:

```
Block i output channels = Block i+1 input channels
```

Our implementation handles this automatically:
1. Compute per-channel importance for each block independently
2. Select kept output indices for block `i`
3. Use those indices as the kept **input** indices for block `i+1`
4. Also prune BatchNorm parameters (weight, bias, running stats) for removed channels
5. Also prune the Linear classifier's input dimension

### 4. Architecture After Pruning

```
Original DemoCNN:
  Conv(3, 64) -> Conv(64, 128) -> Conv(128, 256) -> Conv(256, 256) -> FC(256, 10)
  964,170 params, 133.9M MACs

After 30% channel pruning:
  Conv(3, 44) -> Conv(44, 89) -> Conv(89, 179) -> Conv(179, 179) -> FC(179, 10)
  470,962 params, 65.4M MACs  (51.2% reduction)
```

### 5. The Importance of Fine-Tuning

Channel pruning removes information from the network, causing an accuracy drop.
**Fine-tuning** (retraining for a few epochs) allows the remaining channels to
compensate for the removed ones, recovering most of the lost accuracy.

Without fine-tuning, the pruned model typically performs worse than with it.
Our pipeline automatically runs 5 fine-tuning epochs after pruning.

### 6. Model Saving

The pruned model's `state_dict` is saved to `pruned_model.pth`.  You can load
it later with:

```python
from main import PrunedCNN

# Recreate the pruned architecture
channels = [(3, 44, 1), (44, 89, 2), (89, 179, 1), (179, 179, 2)]
model = PrunedCNN(channels, num_classes=10)
model.load_state_dict(torch.load("pruned_model.pth", weights_only=True))
```

> **Note**: The architecture configuration must match what was produced during
> pruning.  The channel counts depend on `PRUNE_RATIO`.

## References

- MIT 6.5940 Lecture 04: [EfficientML.ai](https://efficientml.ai)
- HAN Lab: [https://hanlab.mit.edu](https://hanlab.mit.edu)
- Li et al., "Pruning Filters for Efficient ConvNets" (ICLR 2017)
- He et al., "Channel Pruning for Accelerating Very Deep Neural Networks" (ICCV 2017)
- Molchanov et al., "Pruning Convolutional Neural Networks for Resource Efficient Inference" (ICLR 2017)
