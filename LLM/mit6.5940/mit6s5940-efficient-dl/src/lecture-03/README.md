# Lecture 03: Fine-Grained Magnitude-Based Pruning

## Overview

This code accompanies **MIT 6.5940 Lecture 03: Pruning & Sparsity**.
It implements magnitude-based weight pruning and layer-wise sensitivity
analysis to understand which parts of a network can be compressed the most.

We use a simple 4-conv-layer CNN trained on synthetic data so the entire
pipeline runs on CPU in under a minute.

## Prerequisites

```bash
pip install torch matplotlib
```

## Usage

```bash
cd src/lecture-03
python main.py
```

The script runs entirely on CPU (no GPU required) and produces:

1. **Magnitude pruning** of individual weights
2. **Uniform global pruning** of the full model at 50% sparsity
3. **Layer-wise sensitivity scan** at sparsity levels [0.1, 0.3, 0.5, 0.7, 0.9]
4. **sensitivity_curves.png** -- a matplotlib plot saved to disk

Example output:

```
======================================================================
  LECTURE 03: Fine-Grained Magnitude-Based Pruning
======================================================================

[1] Generating synthetic dataset ...
  Train: torch.Size([2000, 3, 32, 32]), Test: torch.Size([500, 3, 32, 32])

[2] Building SimpleCNN and training ...
  Parameters: 93,634 total, 93,634 trainable
  Epoch  1  loss=2.3034
  ...
  Baseline test accuracy: 0.9460

[3] Testing magnitude_prune on a sample tensor ...
  Original: [0.5, -0.1, 0.8, -0.3, 0.02, -0.9, 0.0, 0.15]
  Pruned (50%): [0.5, 0.0, 0.8, -0.3, 0.0, -0.9, 0.0, 0.0]
  Zeros: 4 / 8

[4] Applying uniform pruning (sparsity=0.5) to entire model ...

[5] Sensitivity scan: pruning each layer independently ...

[6] Plotting sensitivity curves ...
Sensitivity curves saved to: sensitivity_curves.png
```

## Key Functions

| Function | Purpose |
|---|---|
| `magnitude_prune(weight, sparsity)` | Zero out smallest-magnitude weights to reach target sparsity |
| `apply_pruning_to_model(model, sparsity)` | Uniformly prune all Conv2d/Linear layers |
| `sensitivity_scan(model, test_images, test_labels)` | Prune each layer independently at multiple sparsity levels |
| `plot_sensitivity(results, baseline_acc)` | Plot accuracy vs sparsity per layer (matplotlib) |
| `count_sparsity(model)` | Measure achieved sparsity across prunable layers |
| `evaluate_accuracy(model, images, labels)` | Top-1 accuracy on test set |

## What You Learn

### 1. Magnitude-Based Weight Pruning

```
threshold = kth_smallest_abs_value
mask = |weight| >= threshold
pruned_weight = weight * mask
```

Given a weight tensor and a target sparsity (e.g. 0.5 = 50%), the function
finds the magnitude threshold such that exactly the target fraction of
weights falls below it, then zeros those weights out.  This is the simplest
and most widely used pruning criterion because:

- **Computationally cheap**: single O(N log N) sort or O(N) kthvalue
- **Empirically effective**: small-magnitude weights contribute little
  to the output
- **Orthogonal to other techniques**: can be combined with quantization,
  distillation, and architecture search

### 2. Fine-Grained vs Structured Pruning

This lecture covers **fine-grained** (unstructured) pruning -- zeroing
individual scalar weights.  The resulting weight matrix is sparse but
irregular, which requires special hardware or sparse BLAS libraries to
translate sparsity into actual speedup.

Lecture 04 covers **structured pruning** (removing entire channels,
filters, or neurons), which yields dense sub-matrices and immediate
speedup on any hardware.

| Aspect | Fine-Grained | Structured |
|---|---|---|
| Granularity | Individual weights | Channels / neurons |
| Sparsity pattern | Irregular | Regular (dense sub-blocks) |
| Hardware speedup | Needs sparse accelerators | Works on any hardware |
| Accuracy retention | Better (more degrees of freedom) | Worse at same sparsity |

### 3. Sensitivity Analysis

Not all layers are equally sensitive to pruning.  The **sensitivity
scan** prunes each layer independently and measures the accuracy
drop:

```
for each layer L:
    for each sparsity s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        original = save(L.weight)
        L.weight = magnitude_prune(original, s)
        accuracy[s] = evaluate(model)
        L.weight = original  # restore for next iteration

plot(sensitivity[L] vs sparsity)
```

Layers whose accuracy curves drop steeply (e.g., the first or last layer)
are **bottlenecks** and should be pruned lightly or not at all.  Layers
with flat curves can tolerate high sparsity.

### 4. Global vs Layer-wise Sparsity

- **Uniform pruning** (`apply_pruning_to_model`): every layer gets the
  same sparsity ratio.  Simple but suboptimal because it treats all
  layers equally.

- **Layer-adaptive pruning**: use the sensitivity curves to assign
  different sparsity levels to each layer.  More robust layers get
  higher sparsity; bottleneck layers get lower sparsity.  This is
  implemented in Lecture 04.

## The Sensitivity Plot

The generated `sensitivity_curves.png` shows:

- **X-axis**: sparsity percentage (0% to 100%)
- **Y-axis**: test accuracy
- **Each curve**: one prunable layer
- **Dashed line**: baseline accuracy (no pruning)

Typical observations:
- **First layer** (conv1): often sensitive because it processes raw input
- **Intermediate layers**: can usually tolerate 50-70% sparsity
- **Classifier layer** (fc): often sensitive because it directly affects
  logits. However, in small models, the classifier may be relatively robust
  because it has few parameters with high individual importance per weight.

Use this plot to guide layer-adaptive pruning strategies: prune aggressive
layers aggressively, preserve sensitive layers.

## References

- MIT 6.5940 Lecture 03: [EfficientML.ai](https://efficientml.ai)
- HAN Lab: [https://hanlab.mit.edu](https://hanlab.mit.edu)
- Han et al., "Learning both Weights and Connections for Efficient Neural
  Networks" (NeurIPS 2015)
- Frankle & Carbin, "The Lottery Ticket Hypothesis" (ICLR 2019)
