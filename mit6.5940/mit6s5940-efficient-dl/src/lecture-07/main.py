"""
Simplified NAS Random Search on CIFAR-10 (Lecture 07)
======================================================
Implements a random-search Neural Architecture Search (NAS) over a CNN
search space defined by:

    - Kernel sizes: [3, 5, 7]
    - Output channels: [16, 32, 64, 128]
    - Network depths: [1, 2, 3, 4]

For each randomly sampled architecture we train on CIFAR-10 for a few
epochs, evaluate validation accuracy, and estimate MACs.  The resulting
accuracy-vs-MACs scatter plot reveals the trade-off between model cost
and predictive performance.

Key concepts:
  - NAS search space definition
  - Random architecture sampling
  - Proxy-task training (short training for quick evaluation)
  - MACs estimation via forward hooks
  - Accuracy vs efficiency Pareto frontier

All computations run on CPU; no GPU required.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Use a non-interactive backend so plots can be saved without a display
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Search space
KERNEL_SIZES: List[int] = [3, 5, 7]
CHANNEL_CHOICES: List[int] = [16, 32, 64, 128]
DEPTHS: List[int] = [1, 2, 3, 4]

# NAS experiment
NUM_SAMPLES: int = 20  # number of random architectures to evaluate
NAS_EPOCHS: int = 3  # quick proxy-training epochs per architecture
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.01

# Data
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
TRAIN_SUBSET: int = 5000  # use a subset of CIFAR-10 for faster search
VAL_SUBSET: int = 2000  # fixed validation subset for consistent evaluation

# Reproducibility
SEED: int = 42

# Output
DEVICE = torch.device("cpu")
OUTPUT_PLOT: str = "nas_accuracy_vs_macs.png"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ArchSpec:
    """Specification of a single CNN architecture.

    Attributes:
        depth: Number of convolutional layers (1--4).
        kernel_sizes: Per-layer kernel sizes, length == depth.
        out_channels: Per-layer output channel counts, length == depth.
    """

    depth: int
    kernel_sizes: List[int]
    out_channels: List[int]

    def __post_init__(self) -> None:
        if len(self.kernel_sizes) != self.depth:
            raise ValueError(
                f"kernel_sizes length {len(self.kernel_sizes)} != depth {self.depth}"
            )
        if len(self.out_channels) != self.depth:
            raise ValueError(
                f"out_channels length {len(self.out_channels)} != depth {self.depth}"
            )


@dataclass
class EvalResult:
    """Evaluation result for one architecture.

    Attributes:
        arch: The architecture specification.
        accuracy: Validation top-1 accuracy in (0, 1).
        macs: Total Conv2d MACs (multiply-accumulate operations).
        train_time_s: Training wall-clock time in seconds.
    """

    arch: ArchSpec
    accuracy: float
    macs: int
    train_time_s: float


# ---------------------------------------------------------------------------
# Search Space: Random Sampler
# ---------------------------------------------------------------------------


def random_sample_architecture(
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    depth_choices: Sequence[int] = DEPTHS,
    rng: random.Random | None = None,
) -> ArchSpec:
    """Randomly sample an architecture from the search space.

    Args:
        kernel_choices:  Allowed kernel sizes (default [3, 5, 7]).
        channel_choices: Allowed output channel counts (default [16, 32, 64, 128]).
        depth_choices:   Allowed network depths (default [1, 2, 3, 4]).
        rng:             Optional seeded random.Random instance for reproducibility.

    Returns:
        An ArchSpec with randomly chosen depth, kernel sizes, and channels.
    """
    if rng is None:
        rng = random.Random()

    depth = rng.choice(list(depth_choices))
    kernel_sizes = [rng.choice(list(kernel_choices)) for _ in range(depth)]
    out_channels = [rng.choice(list(channel_choices)) for _ in range(depth)]

    return ArchSpec(depth=depth, kernel_sizes=kernel_sizes, out_channels=out_channels)


# ---------------------------------------------------------------------------
# CNN Builder
# ---------------------------------------------------------------------------


class NasCNN(nn.Module):
    """A VGG-style CNN built from an ArchSpec.

    Each layer consists of:
        Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2)

    After the convolutional backbone, features are reduced via
    AdaptiveAvgPool2d(1) and classified by a single Linear layer.

    Args:
        spec:        Architecture specification (depth, kernels, channels).
        in_channels: Number of input image channels (3 for CIFAR-10).
        num_classes: Number of output classes (10 for CIFAR-10).
    """

    def __init__(
        self,
        spec: ArchSpec,
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_ch = in_channels

        for i in range(spec.depth):
            out_ch = spec.out_channels[i]
            k = spec.kernel_sizes[i]
            layers.append(nn.Conv2d(in_ch, out_ch, k, padding=k // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# MACs Estimation via Forward Hooks
# ---------------------------------------------------------------------------


def estimate_macs_conv2d(
    in_c: int,
    out_c: int,
    k: int,
    h: int,
    w: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """Estimate MACs (multiply-accumulate) for a single Conv2d layer.

    Args:
        in_c:    Input channels.
        out_c:   Output channels.
        k:       Square kernel size.
        h:       Input height.
        w:       Input width.
        stride:  Stride.
        padding: Padding.

    Returns:
        MACs count for one forward pass (single sample).
    """
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    return out_c * h_out * w_out * in_c * k * k


def count_macs(model: nn.Module, input_shape: Tuple[int, int, int]) -> int:
    """Count total Conv2d MACs by tracing a forward pass with hooks.

    Args:
        model:       A PyTorch nn.Module.
        input_shape: (C, H, W) of the input tensor (no batch dim).

    Returns:
        Total Conv2d MACs.
    """
    model.eval()
    total_macs: int = 0
    dummy = torch.randn(1, *input_shape)

    def _hook(
        module: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        _out: torch.Tensor,
    ) -> None:
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            x = inp[0]
            total_macs += estimate_macs_conv2d(
                in_c=x.shape[1],
                out_c=module.out_channels,
                k=module.kernel_size[0],
                h=x.shape[2],
                w=x.shape[3],
                stride=module.stride[0],
                padding=module.padding[0],
            )

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(dummy)

    for h in handles:
        h.remove()

    return total_macs


# ---------------------------------------------------------------------------
# CIFAR-10 Data
# ---------------------------------------------------------------------------


def get_cifar10_subset(
    num_train: int = TRAIN_SUBSET,
    num_val: int = VAL_SUBSET,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 and create fixed training and validation subsets.

    Using smaller subsets keeps NAS search fast on CPU while still
    providing a meaningful accuracy signal for architecture ranking.

    Args:
        num_train: Number of training samples.
        num_val:   Number of validation samples (fixed for all architectures).
        seed:      Random seed for subset selection reproducibility.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    val_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_val,
    )

    # Fixed validation subset (deterministic for fair comparison)
    rng = np.random.RandomState(seed)
    val_indices = rng.choice(
        len(val_dataset), size=min(num_val, len(val_dataset)), replace=False
    )
    val_subset = Subset(val_dataset, val_indices)

    # Training subset (also deterministic)
    train_indices = rng.choice(
        len(train_dataset), size=min(num_train, len(train_dataset)), replace=False
    )
    train_subset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Train the model for one epoch.

    Args:
        model:     A PyTorch nn.Module on the correct device.
        loader:    DataLoader yielding (images, labels) batches.
        optimizer: Optimizer instance.
        criterion: Loss function.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate top-1 accuracy.

    Args:
        model:  A PyTorch nn.Module on the correct device.
        loader: DataLoader yielding (images, labels) batches.

    Returns:
        Accuracy as a float in [0.0, 1.0].
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def train_and_evaluate(
    spec: ArchSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = NAS_EPOCHS,
    lr: float = LEARNING_RATE,
) -> Tuple[float, float]:
    """Build, train, and evaluate a single architecture.

    Args:
        spec:         Architecture specification.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        epochs:       Number of training epochs.
        lr:           Learning rate.

    Returns:
        Tuple of (validation_accuracy, training_wall_time_seconds).
    """
    model = NasCNN(spec, in_channels=3, num_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    t_start = time.time()
    for _epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

    acc = evaluate_accuracy(model, val_loader)
    elapsed = time.time() - t_start

    return acc, elapsed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_accuracy_vs_macs(
    results: List[EvalResult],
    save_path: str = OUTPUT_PLOT,
) -> None:
    """Scatter plot of validation accuracy vs MACs for all sampled architectures.

    Each point is annotated with a compact architecture label showing
    depth, max channels, and min kernel size.

    Args:
        results:   List of EvalResult from the NAS search.
        save_path: File path to save the figure (PNG).
    """
    macs_vals = [r.macs for r in results]
    acc_vals = [r.accuracy * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        macs_vals,
        acc_vals,
        c=acc_vals,
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
    )

    # Annotate each point with a compact label
    for r in results:
        label = (
            f"D{r.arch.depth}_C{max(r.arch.out_channels)}_K{min(r.arch.kernel_sizes)}"
        )
        ax.annotate(
            label,
            (r.macs, r.accuracy * 100),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.7,
        )

    ax.set_xlabel("MACs (Multiply-Accumulate Operations)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title(
        "NAS Random Search: Accuracy vs MACs Trade-off on CIFAR-10",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Accuracy (%)", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nAccuracy vs MACs plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def format_macs(macs: int) -> str:
    """Format a MACs count with human-readable suffix.

    Args:
        macs: Raw MACs integer.

    Returns:
        String like "12.34M".
    """
    if macs >= 1e9:
        return f"{macs / 1e9:.2f}G"
    if macs >= 1e6:
        return f"{macs / 1e6:.2f}M"
    if macs >= 1e3:
        return f"{macs / 1e3:.2f}K"
    return str(macs)


def arch_summary(spec: ArchSpec) -> str:
    """Return a compact one-line string describing the architecture.

    Args:
        spec: Architecture specification.

    Returns:
        String like "D3_C[32,64,128]_K[5,3,7]".
    """
    ch_str = ",".join(str(c) for c in spec.out_channels)
    k_str = ",".join(str(k) for k in spec.kernel_sizes)
    return f"D{spec.depth}_C[{ch_str}]_K[{k_str}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full NAS random search pipeline."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    print("=" * 72)
    print("  LECTURE 07: Simplified NAS Random Search on CIFAR-10")
    print("=" * 72)

    # ---- 1. Print search space ----------------------------------------------
    print(f"\n[1] Search space definition:")
    print(f"  Kernel sizes  : {KERNEL_SIZES}")
    print(f"  Channels      : {CHANNEL_CHOICES}")
    print(f"  Depths        : {DEPTHS}")
    total_configs = sum(
        len(CHANNEL_CHOICES) ** d * len(KERNEL_SIZES) ** d for d in DEPTHS
    )
    print(f"  Total possible architectures: {total_configs}")
    print(f"  Random samples to evaluate : {NUM_SAMPLES}")
    print(f"  Training epochs per arch    : {NAS_EPOCHS}")

    # ---- 2. Load data -------------------------------------------------------
    print(
        f"\n[2] Loading CIFAR-10 (train subset={TRAIN_SUBSET}, val subset={VAL_SUBSET}) ..."
    )
    train_loader, val_loader = get_cifar10_subset()
    print(f"  Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    # ---- 3. Random search ---------------------------------------------------
    print(f"\n[3] Running random search ({NUM_SAMPLES} architectures) ...")
    print(
        f"     {'#':<4} {'Architecture':<35} {'Accuracy':>8} {'MACs':>10} {'Time':>8}"
    )
    print(f"     {'---':<4} {'---':<35} {'---':>8} {'---':>10} {'---':>8}")

    results: List[EvalResult] = []
    total_search_time = 0.0

    for i in range(NUM_SAMPLES):
        spec = random_sample_architecture(rng=rng)
        acc, train_time = train_and_evaluate(
            spec, train_loader, val_loader, epochs=NAS_EPOCHS
        )

        # Build a fresh model for MACs counting (to avoid any side-effects)
        macs_model = NasCNN(spec).to(DEVICE)
        macs = count_macs(macs_model, (3, 32, 32))

        result = EvalResult(arch=spec, accuracy=acc, macs=macs, train_time_s=train_time)
        results.append(result)
        total_search_time += train_time

        print(
            f"     {i + 1:>3d}  {arch_summary(spec):<35} "
            f"{acc * 100:>7.2f}% {format_macs(macs):>9}  {train_time:>6.1f}s"
        )

    print(
        f"  Total search time: {total_search_time:.1f}s ({total_search_time / 60:.1f} min)"
    )

    # ---- 4. Results summary -------------------------------------------------
    print(f"\n[4] Results summary ({len(results)} architectures):")
    accs = [r.accuracy * 100 for r in results]
    macs_list = [r.macs for r in results]
    print(
        f"  Accuracy: min={min(accs):.2f}%,  max={max(accs):.2f}%,  mean={np.mean(accs):.2f}%"
    )
    print(
        f"  MACs:     min={format_macs(min(macs_list))},  "
        f"max={format_macs(max(macs_list))},  "
        f"mean={format_macs(int(np.mean(macs_list)))}"
    )

    # Find best accuracy and efficiency leaders
    best_acc = max(results, key=lambda r: r.accuracy)
    print(
        f"\n  Best accuracy:   {arch_summary(best_acc.arch)} -> {best_acc.accuracy * 100:.2f}%"
    )
    lowest_macs = min(results, key=lambda r: r.macs)
    print(
        f"  Lowest MACs:     {arch_summary(lowest_macs.arch)} -> {format_macs(lowest_macs.macs)}"
    )

    # Simple Pareto-frontier identification (non-dominated architectures)
    pareto: List[EvalResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            # other dominates r if it has both higher accuracy AND lower MACs
            if other.accuracy >= r.accuracy and other.macs <= r.macs:
                if other.accuracy > r.accuracy or other.macs < r.macs:
                    dominated = True
                    break
        if not dominated:
            pareto.append(r)
    print(f"\n  Pareto-frontier architectures: {len(pareto)}")

    # ---- 5. Plot accuracy vs MACs -------------------------------------------
    print(f"\n[5] Plotting accuracy vs MACs trade-off ...")
    plot_accuracy_vs_macs(results, save_path=OUTPUT_PLOT)

    # ---- 6. Done ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(
        f"  Search space:    kernel={KERNEL_SIZES}, ch={CHANNEL_CHOICES}, depth={DEPTHS}"
    )
    print(f"  Total configs:   {total_configs}")
    print(f"  Sampled:         {NUM_SAMPLES}")
    print(
        f"  Training:        {NAS_EPOCHS} epochs CIFAR-10 subset ({TRAIN_SUBSET} samples)"
    )
    print(f"  Best accuracy:   {best_acc.accuracy * 100:.2f}%")
    print(f"  Lowest MACs:     {format_macs(lowest_macs.macs)}")
    print(f"  Pareto frontier: {len(pareto)} architectures")
    print(f"  Plot saved to:   {OUTPUT_PLOT}")
    print("=" * 72)

    print("\nLecture 07 complete.")


if __name__ == "__main__":
    main()
