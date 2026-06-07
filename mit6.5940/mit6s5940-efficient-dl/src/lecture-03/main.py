"""
Fine-Grained Magnitude-Based Pruning (Lecture 03)
==================================================
Implements magnitude pruning and layer-wise sensitivity analysis for
understanding which layers can tolerate the most sparsity.

Key concepts:
  - magnitude_prune: zeros out the smallest-magnitude weights
  - apply_pruning_to_model: prunes all Conv2d/Linear layers uniformly
  - sensitivity_scan: prunes each layer independently at multiple sparsity
    levels and measures the accuracy impact
  - plot_sensitivity: visualises the sensitivity curves with matplotlib

All computations run on CPU; no GPU required.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Use a non-interactive backend so plots can be saved without a display
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARSITY_LEVELS: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
NUM_CLASSES: int = 10
INPUT_CHANNELS: int = 3
IMAGE_SIZE: int = 32
BATCH_SIZE: int = 64
NUM_SAMPLES: int = 2000  # synthetic training set size
NUM_TEST: int = 500  # synthetic test set size
SEED: int = 42


# ===========================================================================
# Magnitude Pruning
# ===========================================================================


def magnitude_prune(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Zero out the smallest-magnitude weights to achieve the target sparsity.

    The function computes a magnitude threshold at the given sparsity
    percentile and zeros out all weights whose absolute value falls
    below that threshold.

    Args:
        weight:   A 2-D (Linear) or 4-D (Conv2d) weight tensor.
        sparsity: Target sparsity ratio in (0, 1).  0.5 means 50% of
                  weights are set to zero.

    Returns:
        A new tensor with the same shape as `weight`, where the smallest
        `sparsity * weight.numel()` values by absolute magnitude are
        replaced with 0.

    Raises:
        ValueError: If sparsity is not in [0, 1].
    """
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"sparsity must be in [0, 1]; got {sparsity}")

    if sparsity == 0.0:
        return weight.clone()

    flat = weight.abs().flatten()
    k = max(1, int(sparsity * flat.numel()))

    # k-th smallest absolute value = the magnitude below which we prune
    threshold = flat.kthvalue(k).values.item()

    mask = weight.abs() >= threshold
    return weight * mask.float()


def _get_prunable_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return (name, module) pairs for all Conv2d and Linear layers.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        List of (name, module) tuples for prunable layers.
    """
    prunable: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable.append((name, module))
    return prunable


def apply_pruning_to_model(model: nn.Module, sparsity: float) -> nn.Module:
    """Apply uniform magnitude pruning to all Conv2d and Linear layers.

    Each layer's weight tensor is replaced with a pruned version in-place.
    This is a **global-uniform** pruning strategy: every prunable layer
    receives the same sparsity level.

    Args:
        model:    A PyTorch nn.Module.
        sparsity: Target sparsity ratio in (0, 1).

    Returns:
        The same model instance (modified in-place).
    """
    for _name, module in _get_prunable_modules(model):
        pruned = magnitude_prune(module.weight.data, sparsity)
        module.weight.data.copy_(pruned)

    return model


def count_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    """Count zero weights across all prunable layers.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        A tuple of (total_params, zero_params, sparsity_ratio).
    """
    total = 0
    zeros = 0
    for _name, module in _get_prunable_modules(model):
        w = module.weight.data
        total += w.numel()
        zeros += (w == 0).sum().item()
    sparsity = zeros / total if total > 0 else 0.0
    return total, zeros, sparsity


# ===========================================================================
# Simple CNN for Demonstration
# ===========================================================================


class SimpleCNN(nn.Module):
    """A compact 4-conv-layer CNN suitable for quick pruning experiments.

    Architecture:
        Conv2d(3, 16, 3, padding=1) -> BN -> ReLU
        Conv2d(16, 32, 3, stride=2, padding=1) -> BN -> ReLU
        Conv2d(32, 64, 3, padding=1) -> BN -> ReLU
        Conv2d(64, 128, 3, stride=2, padding=1) -> BN -> ReLU
        AdaptiveAvgPool2d(1) -> Flatten -> Linear(128, 10)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===========================================================================
# Data Utilities
# ===========================================================================


def _create_synthetic_dataset(
    num_samples: int,
    num_classes: int,
    channels: int,
    size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic images and random labels.

    Images are drawn from a normal distribution and labels are uniform
    random across num_classes.  This avoids external downloads and runs
    quickly on CPU.

    Args:
        num_samples: Number of samples to generate.
        num_classes: Number of label classes.
        channels:    Number of image channels.
        size:        Spatial size (square).

    Returns:
        Tuple of (images, labels).
    """
    images = torch.randn(num_samples, channels, size, size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels


# ===========================================================================
# Training & Evaluation
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    lr: float = 0.01,
) -> float:
    """Train the model for one epoch on the given data.

    Args:
        model:      A PyTorch nn.Module.
        images:     Training images tensor (N, C, H, W).
        labels:     Training labels tensor (N,).
        batch_size: Batch size.
        lr:         Learning rate.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    n = images.size(0)
    perm = torch.randperm(n)
    total_loss = 0.0
    num_batches = 0

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = images[idx], labels[idx]

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> float:
    """Evaluate top-1 accuracy on the given dataset.

    Args:
        model:      A PyTorch nn.Module.
        images:     Image tensor (N, C, H, W).
        labels:     Label tensor (N,).
        batch_size: Batch size for evaluation.

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    model.eval()
    n = images.size(0)
    correct = 0

    for i in range(0, n, batch_size):
        xb = images[i : i + batch_size]
        yb = labels[i : i + batch_size]
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()

    return correct / n


# ===========================================================================
# Sensitivity Analysis
# ===========================================================================


def sensitivity_scan(
    model: nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    sparsity_levels: List[float] | None = None,
) -> Dict[str, Dict[float, float]]:
    """Run layer-wise sensitivity analysis via iterative pruning.

    For **each** prunable layer and **each** sparsity level, the function:
      1. Saves a copy of the original weights.
      2. Prunes *only that layer* at the target sparsity.
      3. Evaluates accuracy on the test set.
      4. Restores the original weights.

    This reveals which layers are most sensitive to pruning: layers whose
    accuracy drops sharply even at low sparsity are "bottleneck" layers.

    Args:
        model:           A trained PyTorch nn.Module.
        test_images:     Test images tensor (N, C, H, W).
        test_labels:     Test labels tensor (N,).
        sparsity_levels: List of sparsity ratios to try.
                         Defaults to [0.1, 0.3, 0.5, 0.7, 0.9].

    Returns:
        Nested dict: {layer_name: {sparsity: accuracy, ...}, ...}
    """
    if sparsity_levels is None:
        sparsity_levels = SPARSITY_LEVELS

    model.eval()

    # Baseline accuracy (unpruned)
    baseline_acc = evaluate_accuracy(model, test_images, test_labels)
    print(f"\n  Baseline accuracy: {baseline_acc:.4f}")

    results: Dict[str, Dict[float, float]] = {}

    prunable = _get_prunable_modules(model)
    print(
        f"  Found {len(prunable)} prunable layers ({SPARSITY_LEVELS} sparsity levels each)"
    )
    print(f"  Total evaluations: {len(prunable) * len(SPARSITY_LEVELS)}\n")

    for layer_name, module in prunable:
        results[layer_name] = {}
        original_weight = module.weight.data.clone()

        for sp in sparsity_levels:
            # Prune only this layer
            pruned_w = magnitude_prune(original_weight, sp)
            module.weight.data.copy_(pruned_w)

            acc = evaluate_accuracy(model, test_images, test_labels)
            results[layer_name][sp] = acc

            print(f"  {layer_name:<30s}  sp={sp:.1f}  acc={acc:.4f}")

            # Restore original weight for the next iteration
            module.weight.data.copy_(original_weight)

    return results


# ===========================================================================
# Plotting
# ===========================================================================


def plot_sensitivity(
    results: Dict[str, Dict[float, float]],
    baseline_acc: float,
    save_path: str = "sensitivity_curves.png",
) -> None:
    """Plot sensitivity curves: accuracy vs sparsity for each layer.

    Each curve shows how pruning a single layer at increasing sparsity
    levels affects overall model accuracy.  Layers whose curves drop
    steeply are the most sensitive to pruning.

    Args:
        results:      Nested dict from sensitivity_scan().
        baseline_acc: Accuracy of the unpruned model.
        save_path:    File path to save the plot (PNG).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer_name, acc_dict in results.items():
        sparsities = sorted(acc_dict.keys())
        accuracies = [acc_dict[sp] for sp in sparsities]
        ax.plot(
            [s * 100 for s in sparsities],
            [a * 100 for a in accuracies],
            marker="o",
            linewidth=2,
            markersize=6,
            label=layer_name,
        )

    # Baseline horizontal line
    ax.axhline(
        y=baseline_acc * 100,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"baseline ({baseline_acc * 100:.1f}%)",
    )

    ax.set_xlabel("Sparsity (%)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Layer-wise Sensitivity to Magnitude Pruning", fontsize=14)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSensitivity curves saved to: {save_path}")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Run the full pruning pipeline: train, prune, sensitivity scan, plot."""
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  LECTURE 03: Fine-Grained Magnitude-Based Pruning")
    print("=" * 70)

    # ---- 1. Create synthetic data ------------------------------------------
    print("\n[1] Generating synthetic dataset ...")
    train_images, train_labels = _create_synthetic_dataset(
        NUM_SAMPLES, NUM_CLASSES, INPUT_CHANNELS, IMAGE_SIZE
    )
    test_images, test_labels = _create_synthetic_dataset(
        NUM_TEST, NUM_CLASSES, INPUT_CHANNELS, IMAGE_SIZE
    )
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")

    # ---- 2. Build and train model ------------------------------------------
    print("\n[2] Building SimpleCNN and training ...")
    model = SimpleCNN(num_classes=NUM_CLASSES)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Quick training on synthetic data
    for epoch in range(1, 11):
        loss = train_one_epoch(model, train_images, train_labels, BATCH_SIZE)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2d}  loss={loss:.4f}")

    baseline_acc = evaluate_accuracy(model, test_images, test_labels)
    print(f"\n  Baseline test accuracy: {baseline_acc:.4f}")

    # ---- 3. Sanity check: magnitude_prune on a single tensor ---------------
    print("\n[3] Testing magnitude_prune on a sample tensor ...")
    sample_w = torch.tensor([0.5, -0.1, 0.8, -0.3, 0.02, -0.9, 0.0, 0.15])
    pruned_w = magnitude_prune(sample_w, sparsity=0.5)
    num_zeros = (pruned_w == 0).sum().item()
    print(f"  Original: {sample_w.tolist()}")
    print(f"  Pruned (50%): {pruned_w.tolist()}")
    print(f"  Zeros: {num_zeros} / {sample_w.numel()}")

    # ---- 4. Apply global uniform pruning -----------------------------------
    print("\n[4] Applying uniform pruning (sparsity=0.5) to entire model ...")
    model.eval()
    apply_pruning_to_model(model, sparsity=0.5)
    total_p, zero_p, achieved_sp = count_sparsity(model)
    pruned_acc = evaluate_accuracy(model, test_images, test_labels)
    print(
        f"  Prunable weights: {total_p:,}  |  zeros: {zero_p:,}  "
        f"|  achieved sparsity: {achieved_sp:.4f}"
    )
    print(f"  Accuracy after 50% uniform pruning: {pruned_acc:.4f}")

    # ---- 5. Sensitivity scan (requires fresh model) ------------------------
    print("\n[5] Sensitivity scan: pruning each layer independently ...")
    model2 = SimpleCNN(num_classes=NUM_CLASSES)
    model2.eval()
    # Train the fresh model
    for epoch in range(1, 11):
        train_one_epoch(model2, train_images, train_labels, BATCH_SIZE)
    baseline_acc2 = evaluate_accuracy(model2, test_images, test_labels)
    print(f"  Fresh model baseline accuracy: {baseline_acc2:.4f}")

    sensitivity_results = sensitivity_scan(
        model2, test_images, test_labels, SPARSITY_LEVELS
    )

    # ---- 6. Plot sensitivity curves ----------------------------------------
    print("\n[6] Plotting sensitivity curves ...")
    plot_sensitivity(
        sensitivity_results,
        baseline_acc2,
        save_path="sensitivity_curves.png",
    )

    # ---- 7. Summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model: SimpleCNN ({total_params:,} parameters)")
    print(f"  Synthetic data: {NUM_SAMPLES} train / {NUM_TEST} test samples")
    print(f"  Baseline accuracy: {baseline_acc2:.4f}")
    print(f"  Uniform pruning (50%): accuracy = {pruned_acc:.4f}")
    print(
        f"  Sensitivity scan: {len(prunable_layers := _get_prunable_modules(model2))} layers "
        f"x {len(SPARSITY_LEVELS)} levels"
    )
    print("=" * 70)

    print("\nLecture 03 complete.")


if __name__ == "__main__":
    main()
