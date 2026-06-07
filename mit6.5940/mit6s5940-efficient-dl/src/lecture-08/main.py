"""
Evolutionary Search with Latency-Aware NAS on CIFAR-10 (Lecture 08)
====================================================================
Implements an evolutionary Neural Architecture Search (NAS) algorithm
that optimises for both accuracy and inference latency.  The search
space is the same CNN space as Lecture 07.  A **simulated latency
lookup table** provides per-layer latency estimates (in milliseconds)
that mimic real-hardware behaviour where larger kernels, more channels,
and higher resolutions increase cost non-linearly.

The script consists of three stages:

    1. **Random search baseline** -- 20 random architectures, each
       proxy-trained for 3 epochs on CIFAR-10.
    2. **Evolutionary search** -- population of 10 individuals evolved
       over 5 generations using tournament selection, one-point
       crossover, and three mutation operators (kernel, channel, depth).
       Multi-objective fitness uses non-dominated sorting (NSGA-II
       style) so the algorithm naturally explores the Pareto frontier.
    3. **Comparison + visualisation** -- Accuracy-vs-latency scatter
       plot with both random and evolutionary results superimposed,
       plus a summary table comparing the two strategies.

Key concepts:
  - Evolutionary NAS with population / mutation / crossover
  - Latency-aware search via simulated lookup table
  - Pareto-frontier visualisation (accuracy vs latency)
  - Random search vs evolutionary search comparison

All computations run on CPU; no GPU required.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Non-interactive backend so plots can be saved without a display
matplotlib.use("Agg")

# =============================================================================
# Constants
# =============================================================================

# --- Search space ------------------------------------------------------------
KERNEL_SIZES: List[int] = [3, 5, 7]
CHANNEL_CHOICES: List[int] = [16, 32, 64, 128]
DEPTHS: List[int] = [1, 2, 3, 4]

# --- Evolutionary algorithm --------------------------------------------------
POPULATION_SIZE: int = 10  # individuals per generation
NUM_GENERATIONS: int = 5  # evolutionary generations
TOURNAMENT_SIZE: int = 3  # tournament selection size
CROSSOVER_PROB: float = 0.7  # probability of crossover
MUTATION_PROB: float = 0.3  # probability of mutation per individual

# --- NAS experiment ----------------------------------------------------------
NUM_RANDOM_SAMPLES: int = 20  # random search baseline size
NAS_EPOCHS: int = 3  # proxy-training epochs per architecture
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.01

# --- CIFAR-10 data -----------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
TRAIN_SUBSET: int = 5000  # train subset for fast proxy evaluation
VAL_SUBSET: int = 2000  # fixed validation subset

# --- Reproducibility ---------------------------------------------------------
SEED: int = 42

# --- Device & output ---------------------------------------------------------
DEVICE = torch.device("cpu")
OUTPUT_PLOT: str = "nas_accuracy_vs_latency.png"


# =============================================================================
# Data Structures
# =============================================================================


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
        latency_ms: Total estimated inference latency in milliseconds.
        train_time_s: Training wall-clock time in seconds.
        source: "random" or "evolutionary" label for plotting.
    """

    arch: ArchSpec
    accuracy: float
    latency_ms: float
    train_time_s: float
    source: str = "random"


# =============================================================================
# Simulated Latency Lookup Table
# =============================================================================
#
# In real hardware-aware NAS (e.g., ProxylessNAS, MNasNet, FBNet), latency
# is measured on the target device and stored in a lookup table keyed by
# (kernel_size, in_channels, out_channels, height, width).  Here we simulate
# this with a parametric model that captures the key trends:
#
#   latency ~ (kernel_size^2 * in_c * out_c * H_out * W_out) / peak_ops
#
# plus a small penalty for small tensors (launch-overhead effects) and
# a non-linear ceiling for very large layers (memory-bound behaviour).
#
# The table is lazily populated so we only compute entries that are
# actually queried during the search.


class LatencyLookupTable:
    """Simulated latency lookup table for hardware-aware NAS.

    Models per-layer latency (ms) as a function of kernel size, input/output
    channels, and spatial resolution.  Results are cached so repeated queries
    for the same key return instantly.

    Attributes:
        peak_ops_per_ms: Peak throughput in operations per millisecond.
        overhead_ms: Fixed launch overhead per Conv2d layer (ms).
        cache: Internal dict mapping (k, in_c, out_c, h, w) -> latency_ms.
    """

    def __init__(
        self,
        peak_ops_per_ms: float = 1e5,
        overhead_ms: float = 0.02,
    ) -> None:
        self.peak_ops_per_ms = peak_ops_per_ms
        self.overhead_ms = overhead_ms
        self._cache: Dict[Tuple[int, int, int, int, int], float] = {}

    def query(
        self,
        kernel: int,
        in_c: int,
        out_c: int,
        h: int,
        w: int,
        stride: int = 1,
        padding: int = 0,
    ) -> float:
        """Return simulated latency (ms) for one Conv2d layer.

        Args:
            kernel:  Square kernel size.
            in_c:    Input channels.
            out_c:   Output channels.
            h:       Input spatial height.
            w:       Input spatial width.
            stride:  Stride (default 1).
            padding: Padding (default 0).

        Returns:
            Simulated latency in milliseconds.
        """
        key = (kernel, in_c, out_c, h, w)
        if key in self._cache:
            return self._cache[key]

        # Output spatial size
        h_out = (h + 2 * padding - kernel) // stride + 1
        w_out = (w + 2 * padding - kernel) // stride + 1

        # Total MACs for this layer
        macs = out_c * h_out * w_out * in_c * kernel * kernel

        # Base latency: compute-bound portion
        latency_compute = macs / self.peak_ops_per_ms

        # Memory-bound penalty for very large layers
        elements = out_c * h_out * w_out
        if elements > 100_000:
            latency_compute *= 1.3  # +30% penalty due to memory bandwidth

        # Launch overhead + compute
        latency_ms = self.overhead_ms + latency_compute

        # Small non-linear scaling to simulate hardware pipeline effects
        if kernel >= 5:
            latency_ms *= 1.15  # extra cost for larger kernels on real hardware

        self._cache[key] = latency_ms
        return latency_ms

    def estimate_model_latency(
        self,
        spec: ArchSpec,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
    ) -> float:
        """Estimate total inference latency for a full architecture.

        Simulates a forward pass through the VGG-style backbone (Conv2d ->
        MaxPool2d(2) per layer) and sums the per-layer Conv2d latencies.

        Args:
            spec:        Architecture specification.
            input_shape: (C, H, W) of the input image.

        Returns:
            Total simulated latency in milliseconds.
        """
        in_c, h, w = input_shape
        total_ms = 0.0

        for i in range(spec.depth):
            out_c = spec.out_channels[i]
            k = spec.kernel_sizes[i]
            total_ms += self.query(k, in_c, out_c, h, w, stride=1, padding=k // 2)
            # After MaxPool2d(2): spatial halves, channels stay
            h //= 2
            w //= 2
            in_c = out_c

        return total_ms


# =============================================================================
# NAS CNN Builder (shared with Lecture 07)
# =============================================================================


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


# =============================================================================
# CIFAR-10 Data Loading
# =============================================================================


def get_cifar10_subset(
    num_train: int = TRAIN_SUBSET,
    num_val: int = VAL_SUBSET,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 and create fixed training and validation subsets.

    Using smaller subsets keeps NAS search fast on CPU while still providing
    a meaningful accuracy signal for architecture ranking.

    Args:
        num_train: Number of training samples.
        num_val:   Number of validation samples.
        seed:      Random seed for deterministic subset selection.

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
        root="./data", train=True, download=True, transform=transform_train
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )

    rng = np.random.RandomState(seed)
    val_indices = rng.choice(
        len(val_dataset), size=min(num_val, len(val_dataset)), replace=False
    )
    val_subset = Subset(val_dataset, val_indices)

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


# =============================================================================
# Training & Evaluation Utilities
# =============================================================================


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


# =============================================================================
# Search Space: Random Sampler
# =============================================================================


def random_sample_architecture(
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    depth_choices: Sequence[int] = DEPTHS,
    rng: random.Random | None = None,
) -> ArchSpec:
    """Randomly sample an architecture from the search space.

    Args:
        kernel_choices:  Allowed kernel sizes.
        channel_choices: Allowed output channel counts.
        depth_choices:   Allowed network depths.
        rng:             Optional seeded random.Random instance.

    Returns:
        A randomly-sampled ArchSpec.
    """
    if rng is None:
        rng = random.Random()

    depth = rng.choice(list(depth_choices))
    kernel_sizes = [rng.choice(list(kernel_choices)) for _ in range(depth)]
    out_channels = [rng.choice(list(channel_choices)) for _ in range(depth)]

    return ArchSpec(depth=depth, kernel_sizes=kernel_sizes, out_channels=out_channels)


# =============================================================================
# Evolutionary Operators: Mutation
# =============================================================================


def mutate_kernel(
    spec: ArchSpec,
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    rng: random.Random | None = None,
) -> ArchSpec:
    """Mutate the kernel size of one randomly-chosen layer.

    Args:
        spec:           Original architecture.
        kernel_choices: Allowed kernel sizes.
        rng:            Optional seeded Random.

    Returns:
        A new ArchSpec with one kernel mutated.
    """
    if rng is None:
        rng = random.Random()

    new_kernels = list(spec.kernel_sizes)
    idx = rng.randint(0, spec.depth - 1)
    old_k = new_kernels[idx]
    choices = [k for k in kernel_choices if k != old_k]
    if not choices:
        choices = list(kernel_choices)
    new_kernels[idx] = rng.choice(choices)

    return ArchSpec(
        depth=spec.depth,
        kernel_sizes=new_kernels,
        out_channels=list(spec.out_channels),
    )


def mutate_channels(
    spec: ArchSpec,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    rng: random.Random | None = None,
) -> ArchSpec:
    """Mutate the channel count of one randomly-chosen layer.

    Args:
        spec:            Original architecture.
        channel_choices: Allowed output channel counts.
        rng:             Optional seeded Random.

    Returns:
        A new ArchSpec with one channel mutated.
    """
    if rng is None:
        rng = random.Random()

    new_channels = list(spec.out_channels)
    idx = rng.randint(0, spec.depth - 1)
    old_ch = new_channels[idx]
    choices_ch = [c for c in channel_choices if c != old_ch]
    if not choices_ch:
        choices_ch = list(channel_choices)
    new_channels[idx] = rng.choice(choices_ch)

    return ArchSpec(
        depth=spec.depth,
        kernel_sizes=list(spec.kernel_sizes),
        out_channels=new_channels,
    )


def mutate_depth(
    spec: ArchSpec,
    depth_choices: Sequence[int] = DEPTHS,
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    rng: random.Random | None = None,
) -> ArchSpec:
    """Mutate the depth of the architecture (add or remove one layer).

    - If depth is at the minimum, force add.
    - If depth is at the maximum, force remove.
    - Otherwise randomly add or remove.

    When adding, the new layer inherits kernel/channel from a random
    existing layer.  When removing, a random layer is dropped.

    Args:
        spec:            Original architecture.
        depth_choices:   Allowed depths.
        kernel_choices:  Allowed kernel sizes (for new layers).
        channel_choices: Allowed channel counts (for new layers).
        rng:             Optional seeded Random.

    Returns:
        A new ArchSpec with depth changed by +/- 1.
    """
    if rng is None:
        rng = random.Random()

    current_depth = spec.depth
    can_add = current_depth < max(depth_choices)
    can_remove = current_depth > min(depth_choices)

    if can_add and can_remove:
        add_layer = rng.random() < 0.5
    elif can_add:
        add_layer = True
    else:
        add_layer = False  # must remove

    if add_layer:
        # Insert a new layer at a random position
        insert_pos = rng.randint(0, current_depth)
        new_kernel = rng.choice(list(kernel_choices))
        new_channel = rng.choice(list(channel_choices))

        new_kernels = list(spec.kernel_sizes)
        new_channels = list(spec.out_channels)
        new_kernels.insert(insert_pos, new_kernel)
        new_channels.insert(insert_pos, new_channel)

        return ArchSpec(
            depth=current_depth + 1,
            kernel_sizes=new_kernels,
            out_channels=new_channels,
        )
    else:
        # Remove a random layer
        remove_pos = rng.randint(0, current_depth - 1)
        new_kernels = list(spec.kernel_sizes)
        new_channels = list(spec.out_channels)
        new_kernels.pop(remove_pos)
        new_channels.pop(remove_pos)

        return ArchSpec(
            depth=current_depth - 1,
            kernel_sizes=new_kernels,
            out_channels=new_channels,
        )


def mutate(
    spec: ArchSpec,
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    depth_choices: Sequence[int] = DEPTHS,
    rng: random.Random | None = None,
) -> ArchSpec:
    """Apply one random mutation operator to an architecture.

    Picks uniformly among kernel mutation, channel mutation, and depth
    mutation.

    Args:
        spec:            Original architecture.
        kernel_choices:  Allowed kernel sizes.
        channel_choices: Allowed output channel counts.
        depth_choices:   Allowed depths.
        rng:             Optional seeded Random.

    Returns:
        A mutated ArchSpec.
    """
    if rng is None:
        rng = random.Random()

    op = rng.choice(["kernel", "channel", "depth"])
    if op == "kernel":
        return mutate_kernel(spec, kernel_choices, rng)
    elif op == "channel":
        return mutate_channels(spec, channel_choices, rng)
    else:
        return mutate_depth(spec, depth_choices, kernel_choices, channel_choices, rng)


# =============================================================================
# Evolutionary Operators: Crossover
# =============================================================================


def crossover(
    parent1: ArchSpec,
    parent2: ArchSpec,
    rng: random.Random | None = None,
) -> Tuple[ArchSpec, ArchSpec]:
    """One-point crossover on the layer lists of two parent architectures.

    Both parents must have the same depth for crossover to be meaningful.
    If depths differ, the longer parent is truncated to the shorter length
    and a random layer is appended so children have the same depth as the
    longer parent.

    Args:
        parent1: First parent ArchSpec.
        parent2: Second parent ArchSpec.
        rng:     Optional seeded Random.

    Returns:
        Tuple of two child ArchSpecs (child1, child2).
    """
    if rng is None:
        rng = random.Random()

    d1, d2 = parent1.depth, parent2.depth
    min_depth = min(d1, d2)

    if min_depth < 2:
        # Crossover not meaningful for depth 1; return clones
        return (
            ArchSpec(
                depth=d1,
                kernel_sizes=list(parent1.kernel_sizes),
                out_channels=list(parent1.out_channels),
            ),
            ArchSpec(
                depth=d2,
                kernel_sizes=list(parent2.kernel_sizes),
                out_channels=list(parent2.out_channels),
            ),
        )

    # Align to the same depth for crossover
    k1 = list(parent1.kernel_sizes[:min_depth])
    k2 = list(parent2.kernel_sizes[:min_depth])
    ch1 = list(parent1.out_channels[:min_depth])
    ch2 = list(parent2.out_channels[:min_depth])

    # Pick a crossover point (1..min_depth-1)
    point = rng.randint(1, min_depth - 1)

    # Swap tails
    child1_k = k1[:point] + k2[point:]
    child1_ch = ch1[:point] + ch2[point:]
    child2_k = k2[:point] + k1[point:]
    child2_ch = ch2[:point] + ch1[point:]

    # If parents had different depths, preserve the longer one's shape
    # by appending the excess layers from the original parents
    if d1 > min_depth:
        child1_k.extend(parent1.kernel_sizes[min_depth:])
        child2_k.extend(parent1.kernel_sizes[min_depth:])
        child1_ch.extend(parent1.out_channels[min_depth:])
        child2_ch.extend(parent1.out_channels[min_depth:])
    elif d2 > min_depth:
        child1_k.extend(parent2.kernel_sizes[min_depth:])
        child2_k.extend(parent2.kernel_sizes[min_depth:])
        child1_ch.extend(parent2.out_channels[min_depth:])
        child2_ch.extend(parent2.out_channels[min_depth:])

    target_depth = len(child1_k)
    return (
        ArchSpec(depth=target_depth, kernel_sizes=child1_k, out_channels=child1_ch),
        ArchSpec(depth=target_depth, kernel_sizes=child2_k, out_channels=child2_ch),
    )


# =============================================================================
# Evolutionary Operators: Selection
# =============================================================================


def non_dominated_sorting(
    results: List[EvalResult],
) -> List[List[int]]:
    """NSGA-II non-dominated sorting on accuracy and latency.

    Returns a list of fronts, where the first front contains indices of
    all non-dominated individuals, the second front contains indices
    of individuals dominated only by the first front, etc.

    We maximise accuracy and minimise latency.

    Args:
        results: List of EvalResult for the population.

    Returns:
        List of fronts; each front is a list of indices into ``results``.
    """
    n = len(results)
    # domination: S[i] = set of indices that i dominates
    dominates: List[List[int]] = [[] for _ in range(n)]
    # dominated_by_count[i] = how many dominate i
    dominated_by_count: List[int] = [0] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i dominates j if i has >= accuracy AND <= latency
            # with strict inequality on at least one
            better_acc = results[i].accuracy >= results[j].accuracy
            better_lat = results[i].latency_ms <= results[j].latency_ms
            strictly_better = (
                results[i].accuracy > results[j].accuracy
                or results[i].latency_ms < results[j].latency_ms
            )
            if better_acc and better_lat and strictly_better:
                dominates[i].append(j)
                dominated_by_count[j] += 1

    fronts: List[List[int]] = []
    # First front: individuals with dominated_by_count == 0
    current_front = [i for i in range(n) if dominated_by_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front: List[int] = []
        for i in current_front:
            for j in dominates[i]:
                dominated_by_count[j] -= 1
                if dominated_by_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def tournament_select(
    population: List[ArchSpec],
    pop_results: List[EvalResult],
    fronts: List[List[int]],
    tournament_size: int = TOURNAMENT_SIZE,
    rng: random.Random | None = None,
) -> ArchSpec:
    """Tournament selection based on Pareto rank (from non-dominated sorting).

    Randomly selects ``tournament_size`` candidates from the population.
    Returns the one with the best (lowest) Pareto rank, breaking ties
    by preferring higher accuracy.

    Args:
        population:      Current population (list of ArchSpec).
        pop_results:     Corresponding EvalResult for each individual.
        fronts:          Non-dominated sorting fronts (list of index lists).
        tournament_size: Number of individuals in each tournament.
        rng:             Optional seeded Random.

    Returns:
        The selected ArchSpec.
    """
    if rng is None:
        rng = random.Random()

    # Pre-compute rank for each individual
    rank_of: Dict[int, int] = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            rank_of[idx] = rank

    n = len(population)
    # Pick tournament_size random candidates
    candidates = [rng.randint(0, n - 1) for _ in range(tournament_size)]

    best_idx = candidates[0]
    best_rank = rank_of.get(best_idx, 999999)
    best_acc = pop_results[best_idx].accuracy

    for idx in candidates[1:]:
        r = rank_of.get(idx, 999999)
        acc = pop_results[idx].accuracy
        if r < best_rank or (r == best_rank and acc > best_acc):
            best_idx = idx
            best_rank = r
            best_acc = acc

    return copy.deepcopy(population[best_idx])


# =============================================================================
# Evolutionary Algorithm Main Loop
# =============================================================================


def run_evolutionary_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    latency_table: LatencyLookupTable,
    population_size: int = POPULATION_SIZE,
    generations: int = NUM_GENERATIONS,
    rng: random.Random | None = None,
) -> List[EvalResult]:
    """Run the full evolutionary NAS search.

    1. Initialise a random population of ``population_size`` architectures.
    2. Evaluate fitness (accuracy + latency) for each individual.
    3. For each generation:
       a. Perform non-dominated sorting on the current population.
       b. Create offspring via tournament selection, crossover, mutation.
       c. Evaluate offspring.
       d. Combine parents + offspring, select top ``population_size``
          using Pareto rank + accuracy tie-breaker.

    Args:
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        latency_table: Latency lookup table.
        population_size: Size of the population.
        generations:     Number of generations to evolve.
        rng:             Optional seeded Random.

    Returns:
        List of EvalResult for the final population.
    """
    if rng is None:
        rng = random.Random()

    # --- 1. Initialise random population -------------------------------------
    population: List[ArchSpec] = []
    for _ in range(population_size):
        population.append(random_sample_architecture(rng=rng))

    pop_results: List[EvalResult] = []
    print(f"\n  Evaluating initial population ({population_size} architectures) ...")
    for i, spec in enumerate(population):
        acc, train_time = train_and_evaluate(spec, train_loader, val_loader)
        lat = latency_table.estimate_model_latency(spec)
        pop_results.append(
            EvalResult(
                arch=spec,
                accuracy=acc,
                latency_ms=lat,
                train_time_s=train_time,
                source="evolutionary",
            )
        )
        print(
            f"    init [{i + 1:>2d}/{population_size}]  "
            f"{arch_summary(spec):<35}  "
            f"acc={acc * 100:.2f}%  lat={lat:.3f}ms"
        )

    # --- 2. Evolution loop ---------------------------------------------------
    for gen in range(generations):
        print(
            f"\n  --- Generation {gen + 1}/{generations} "
            f"(population {population_size}) ---"
        )

        # Non-dominated sorting of current population
        fronts = non_dominated_sorting(pop_results)
        print(
            f"    Pareto fronts: {len(fronts)}  (front 0: {len(fronts[0])} individuals)"
        )

        # Create offspring
        offspring: List[ArchSpec] = []
        while len(offspring) < population_size:
            # Selection
            p1 = tournament_select(population, pop_results, fronts, rng=rng)
            p2 = tournament_select(population, pop_results, fronts, rng=rng)

            # Crossover
            if rng.random() < CROSSOVER_PROB and p1.depth >= 2 and p2.depth >= 2:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1 = copy.deepcopy(p1)
                c2 = copy.deepcopy(p2)

            # Mutation
            if rng.random() < MUTATION_PROB:
                c1 = mutate(c1, rng=rng)
            if rng.random() < MUTATION_PROB:
                c2 = mutate(c2, rng=rng)

            offspring.append(c1)
            if len(offspring) < population_size:
                offspring.append(c2)

        # Evaluate offspring
        offspring_results: List[EvalResult] = []
        print(f"    Evaluating offspring ({len(offspring)} architectures) ...")
        for i, spec in enumerate(offspring):
            acc, train_time = train_and_evaluate(spec, train_loader, val_loader)
            lat = latency_table.estimate_model_latency(spec)
            offspring_results.append(
                EvalResult(
                    arch=spec,
                    accuracy=acc,
                    latency_ms=lat,
                    train_time_s=train_time,
                    source="evolutionary",
                )
            )

        # --- Environmental selection: keep best population_size individuals ---
        combined_pop = population + offspring
        combined_results = pop_results + offspring_results

        # Rank all combined individuals
        combined_fronts = non_dominated_sorting(combined_results)

        # Select top population_size by Pareto rank (then accuracy tie-break)
        rank_of: Dict[int, int] = {}
        for rank, front in enumerate(combined_fronts):
            for idx in front:
                rank_of[idx] = rank

        # Sort by (rank, -accuracy) -- best first
        ranked_indices = sorted(
            range(len(combined_results)),
            key=lambda i: (rank_of.get(i, 999999), -combined_results[i].accuracy),
        )

        # Keep the best
        kept_indices = ranked_indices[:population_size]
        population = [combined_pop[i] for i in kept_indices]
        pop_results = [combined_results[i] for i in kept_indices]

        # Print generation summary
        gen_accs = [r.accuracy * 100 for r in pop_results]
        gen_lats = [r.latency_ms for r in pop_results]
        print(
            f"    Gen {gen + 1} summary: "
            f"acc mean={np.mean(gen_accs):.2f}%  "
            f"best={np.max(gen_accs):.2f}%  "
            f"lat mean={np.mean(gen_lats):.3f}ms  "
            f"min={np.min(gen_lats):.3f}ms"
        )

    return pop_results


# =============================================================================
# Plotting: Pareto Frontier (Accuracy vs Latency)
# =============================================================================


def plot_pareto_frontier(
    random_results: List[EvalResult],
    evo_results: List[EvalResult],
    save_path: str = OUTPUT_PLOT,
) -> None:
    """Scatter plot of accuracy vs latency with both random and evolutionary results.

    Random search results are shown as blue circles; evolutionary search
    results are shown as red triangles.  The combined Pareto frontier is
    highlighted with a connecting line and filled markers.

    Args:
        random_results: EvalResult list from random search.
        evo_results:    EvalResult list from evolutionary search.
        save_path:      File path for the output PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Random search points ------------------------------------------------
    rand_acc = [r.accuracy * 100 for r in random_results]
    rand_lat = [r.latency_ms for r in random_results]
    ax.scatter(
        rand_lat,
        rand_acc,
        c="steelblue",
        marker="o",
        s=70,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
        label=f"Random Search (n={len(random_results)})",
        zorder=3,
    )

    # --- Evolutionary search points -------------------------------------------
    evo_acc = [r.accuracy * 100 for r in evo_results]
    evo_lat = [r.latency_ms for r in evo_results]
    ax.scatter(
        evo_lat,
        evo_acc,
        c="firebrick",
        marker="^",
        s=90,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        label=f"Evolutionary Search (n={len(evo_results)})",
        zorder=4,
    )

    # --- Pareto frontier (combined) -------------------------------------------
    all_results = random_results + evo_results
    pareto = compute_pareto_frontier(all_results)
    pareto_acc = [r.accuracy * 100 for r in pareto]
    pareto_lat = [r.latency_ms for r in pareto]

    # Sort by latency for connecting line
    pareto_sorted = sorted(pareto, key=lambda r: r.latency_ms)
    sorted_acc = [r.accuracy * 100 for r in pareto_sorted]
    sorted_lat = [r.latency_ms for r in pareto_sorted]

    ax.plot(
        sorted_lat,
        sorted_acc,
        "o-",
        color="darkorange",
        linewidth=2.0,
        markersize=8,
        markerfacecolor="gold",
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=f"Pareto Frontier ({len(pareto)} candidates)",
        zorder=5,
    )

    ax.set_xlabel("Inference Latency (ms)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title(
        "Latency-Aware NAS: Accuracy vs Latency Pareto Frontier",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPareto frontier plot saved to: {save_path}")


def compute_pareto_frontier(results: List[EvalResult]) -> List[EvalResult]:
    """Identify the Pareto frontier (non-dominated set) for accuracy vs latency.

    We maximise accuracy and minimise latency.  An architecture A dominates
    B if A has >= accuracy AND <= latency with at least one strict inequality.

    Args:
        results: List of EvalResult.

    Returns:
        List of non-dominated EvalResult.
    """
    pareto: List[EvalResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            if other.accuracy >= r.accuracy and other.latency_ms <= r.latency_ms:
                if other.accuracy > r.accuracy or other.latency_ms < r.latency_ms:
                    dominated = True
                    break
        if not dominated:
            pareto.append(r)
    return pareto


# =============================================================================
# Utilities
# =============================================================================


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


def format_latency(ms: float) -> str:
    """Format a latency value with appropriate unit.

    Args:
        ms: Latency in milliseconds.

    Returns:
        Formatted string like "2.345ms" or "0.123ms".
    """
    if ms < 0.01:
        return f"{ms * 1000:.2f}us"
    return f"{ms:.3f}ms"


# =============================================================================
# Main Pipeline
# =============================================================================


def main() -> None:
    """Run the full evolutionary NAS pipeline."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    print("=" * 72)
    print("  LECTURE 08: Evolutionary Search with Latency-Aware NAS")
    print("=" * 72)

    # ---- 1. Search space & latency table ------------------------------------
    print(f"\n[1] Search space definition:")
    print(f"  Kernel sizes  : {KERNEL_SIZES}")
    print(f"  Channels      : {CHANNEL_CHOICES}")
    print(f"  Depths        : {DEPTHS}")
    total_configs = sum(
        len(CHANNEL_CHOICES) ** d * len(KERNEL_SIZES) ** d for d in DEPTHS
    )
    print(f"  Total possible architectures: {total_configs}")
    print(f"  Random search samples        : {NUM_RANDOM_SAMPLES}")
    print(f"  Evolution: pop={POPULATION_SIZE}, gens={NUM_GENERATIONS}")
    print(f"  Training epochs per arch     : {NAS_EPOCHS}")

    latency_table = LatencyLookupTable()
    print(
        f"\n  Latency lookup table initialised ({len(latency_table._cache)} entries in cache)"
    )

    # ---- 2. Load CIFAR-10 data ----------------------------------------------
    print(
        f"\n[2] Loading CIFAR-10 (train subset={TRAIN_SUBSET}, val subset={VAL_SUBSET}) ..."
    )
    train_loader, val_loader = get_cifar10_subset()
    print(f"  Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    # ---- 3. Random search baseline ------------------------------------------
    print(
        f"\n[3] Running RANDOM SEARCH baseline ({NUM_RANDOM_SAMPLES} architectures) ..."
    )
    print(
        f"     {'#':<4} {'Architecture':<35} {'Accuracy':>8} {'Latency':>10} {'Time':>8}"
    )
    print(f"     {'---':<4} {'---':<35} {'---':>8} {'---':>10} {'---':>8}")

    random_results: List[EvalResult] = []
    for i in range(NUM_RANDOM_SAMPLES):
        spec = random_sample_architecture(rng=rng)
        acc, train_time = train_and_evaluate(spec, train_loader, val_loader)
        lat = latency_table.estimate_model_latency(spec)
        result = EvalResult(
            arch=spec,
            accuracy=acc,
            latency_ms=lat,
            train_time_s=train_time,
            source="random",
        )
        random_results.append(result)

        print(
            f"     {i + 1:>3d}  {arch_summary(spec):<35} "
            f"{acc * 100:>7.2f}% {format_latency(lat):>9}  {train_time:>6.1f}s"
        )

    # Random search summary
    rand_accs = [r.accuracy * 100 for r in random_results]
    rand_lats = [r.latency_ms for r in random_results]
    rand_pareto = compute_pareto_frontier(random_results)
    print(f"\n  Random search summary:")
    print(
        f"    Accuracy: mean={np.mean(rand_accs):.2f}%, "
        f"min={np.min(rand_accs):.2f}%, max={np.max(rand_accs):.2f}%"
    )
    print(
        f"    Latency:  mean={np.mean(rand_lats):.4f}ms, "
        f"min={np.min(rand_lats):.4f}ms, max={np.max(rand_lats):.4f}ms"
    )
    print(f"    Pareto frontier:  {len(rand_pareto)} candidates")

    # ---- 4. Evolutionary search ---------------------------------------------
    print(f"\n[4] Running EVOLUTIONARY SEARCH ...")
    t_evo_start = time.time()
    evo_results = run_evolutionary_search(
        train_loader,
        val_loader,
        latency_table,
        population_size=POPULATION_SIZE,
        generations=NUM_GENERATIONS,
        rng=rng,
    )
    t_evo_elapsed = time.time() - t_evo_start

    # Evolutionary search summary
    evo_accs = [r.accuracy * 100 for r in evo_results]
    evo_lats = [r.latency_ms for r in evo_results]
    evo_pareto = compute_pareto_frontier(evo_results)
    print(f"\n  Evolutionary search summary:")
    print(f"    Generations: {NUM_GENERATIONS}, population: {POPULATION_SIZE}")
    print(f"    Wall time: {t_evo_elapsed:.1f}s ({t_evo_elapsed / 60:.1f} min)")
    print(
        f"    Accuracy: mean={np.mean(evo_accs):.2f}%, "
        f"min={np.min(evo_accs):.2f}%, max={np.max(evo_accs):.2f}%"
    )
    print(
        f"    Latency:  mean={np.mean(evo_lats):.4f}ms, "
        f"min={np.min(evo_lats):.4f}ms, max={np.max(evo_lats):.4f}ms"
    )
    print(f"    Pareto frontier:  {len(evo_pareto)} candidates")
    print(f"    Latency lookup table: {len(latency_table._cache)} entries cached")

    # ---- 5. Comparison: Random Search vs Evolutionary Search -----------------
    print(f"\n[5] COMPARISON: Random Search vs Evolutionary Search")
    print(f"  {'=' * 60}")
    print(f"  {'Metric':<25} {'Random Search':>16} {'Evolutionary':>16}")
    print(f"  {'-' * 59}")
    print(f"  {'Evaluations':<25} {len(random_results):>16d} {len(evo_results):>16d}")
    print(
        f"  {'Best Accuracy':<25} {np.max(rand_accs):>15.2f}% "
        f"{np.max(evo_accs):>15.2f}%"
    )
    print(
        f"  {'Mean Accuracy':<25} {np.mean(rand_accs):>15.2f}% "
        f"{np.mean(evo_accs):>15.2f}%"
    )
    print(
        f"  {'Min Latency':<25} {format_latency(np.min(rand_lats)):>15}  "
        f"{format_latency(np.min(evo_lats)):>15}"
    )
    print(
        f"  {'Mean Latency':<25} {format_latency(np.mean(rand_lats)):>15}  "
        f"{format_latency(np.mean(evo_lats)):>15}"
    )
    print(
        f"  {'Pareto Frontier Size':<25} {len(rand_pareto):>16d} {len(evo_pareto):>16d}"
    )

    # ---- 6. Plot Pareto frontier --------------------------------------------
    print(f"\n[6] Plotting accuracy vs latency Pareto frontier ...")
    plot_pareto_frontier(random_results, evo_results, save_path=OUTPUT_PLOT)

    # ---- 7. Done ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(
        f"  Search space:    kernel={KERNEL_SIZES}, ch={CHANNEL_CHOICES}, "
        f"depth={DEPTHS} ({total_configs} configs)"
    )
    print(f"  Random search:   {NUM_RANDOM_SAMPLES} samples")
    print(
        f"  Evolution:       pop={POPULATION_SIZE} x gen={NUM_GENERATIONS} "
        f"(tournament={TOURNAMENT_SIZE})"
    )
    print(
        f"  Training:        {NAS_EPOCHS} proxy epochs on {TRAIN_SUBSET} CIFAR-10 samples"
    )
    print(
        f"  Best accuracy:   random={np.max(rand_accs):.2f}%  "
        f"evolutionary={np.max(evo_accs):.2f}%"
    )
    print(
        f"  Best latency:    random={format_latency(np.min(rand_lats))}  "
        f"evolutionary={format_latency(np.min(evo_lats))}"
    )
    print(
        f"  Pareto frontier: random={len(rand_pareto)}  evolutionary={len(evo_pareto)}"
    )
    print(f"  Latency entries: {len(latency_table._cache)} cached")
    print(f"  Plot saved to:   {OUTPUT_PLOT}")
    print("=" * 72)

    print("\nLecture 08 complete.")


if __name__ == "__main__":
    main()
