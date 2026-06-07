"""
Channel Pruning with Frobenius Norm (Lecture 04)
================================================
Implements structured channel pruning for Conv2d layers by ranking output
channels via their Frobenius (L2) norm and removing the least important ones.

Key concepts:
  - frobenius_importance: ranks Conv2d output channels by ||W_i||_F
  - channel_prune: builds a smaller model by keeping only top-k channels
  - fine_tune: retrains the pruned model for 5 epochs to recover accuracy
  - compare_metrics: contrasts accuracy, params, MACs, and latency before vs after

All computations run on CPU; no GPU required.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 10
INPUT_CHANNELS: int = 3
IMAGE_SIZE: int = 32
BATCH_SIZE: int = 64
NUM_TRAIN: int = 2000  # synthetic training samples
NUM_TEST: int = 500  # synthetic test samples
PRUNE_RATIO: float = 0.3  # fraction of channels to prune per layer
FINE_TUNE_EPOCHS: int = 5
INITIAL_EPOCHS: int = 10
LR: float = 0.01
WARMUP_RUNS: int = 10
TIMED_RUNS: int = 100
SEED: int = 42
SAVE_PATH: str = "pruned_model.pth"


# ===========================================================================
# Model Definition
# ===========================================================================


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU block used as a building block.

    Args:
        in_c:  Input channels.
        out_c: Output channels.
        stride: Conv2d stride (default 1).
    """

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DemoCNN(nn.Module):
    """A small CNN with 4 ConvBlock layers for channel pruning experiments.

    Architecture:
        ConvBlock(3,   64,  stride=1)
        ConvBlock(64,  128, stride=2)
        ConvBlock(128, 256, stride=1)
        ConvBlock(256, 256, stride=2)
        AdaptiveAvgPool2d(1) -> Flatten -> Linear(256, 10)

    Args:
        num_classes: Number of output classes (default 10).
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.block1 = ConvBlock(3, 64, stride=1)
        self.block2 = ConvBlock(64, 128, stride=2)
        self.block3 = ConvBlock(128, 256, stride=1)
        self.block4 = ConvBlock(256, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @property
    def blocks(self) -> List[ConvBlock]:
        """Return all ConvBlock modules in forward order."""
        return [self.block1, self.block2, self.block3, self.block4]


class PrunedCNN(nn.Module):
    """A CNN built from a list of channel configurations after pruning.

    This class receives pre-computed channel counts so that pruned models
    can be instantiated with exactly the right architecture.

    Args:
        channels: List of (in_c, out_c, stride) tuples for each ConvBlock.
        num_classes: Number of output classes.
    """

    def __init__(
        self, channels: List[Tuple[int, int, int]], num_classes: int = NUM_CLASSES
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for in_c, out_c, stride in channels:
            self.blocks.append(ConvBlock(in_c, out_c, stride))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        final_out = channels[-1][1] if channels else 0
        self.classifier = nn.Linear(final_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===========================================================================
# Data Utilities
# ===========================================================================


def _create_synthetic_data(
    n: int, c: int, h: int, w: int, num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic images and random labels.

    Args:
        n:           Number of samples.
        c:           Number of channels.
        h, w:        Spatial dimensions.
        num_classes: Number of label classes.

    Returns:
        Tuple of (images, labels).
    """
    images = torch.randn(n, c, h, w)
    labels = torch.randint(0, num_classes, (n,))
    return images, labels


# ===========================================================================
# Training & Evaluation
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> float:
    """Train the model for one epoch.

    Args:
        model:      A PyTorch nn.Module.
        images:     Training images (N, C, H, W).
        labels:     Training labels (N,).
        batch_size: Mini-batch size.
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
    """Evaluate top-1 accuracy.

    Args:
        model:      A PyTorch nn.Module.
        images:     Image tensor (N, C, H, W).
        labels:     Label tensor (N,).
        batch_size: Evaluation batch size.

    Returns:
        Accuracy as a float in [0.0, 1.0].
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
# Frobenius Norm Importance Ranking
# ===========================================================================


def frobenius_importance(weight: torch.Tensor) -> torch.Tensor:
    """Compute per-output-channel importance via Frobenius norm.

    For a Conv2d weight tensor of shape [C_out, C_in, K, K], the Frobenius
    norm of each output channel (filter) is:

        ||W[i, :, :, :]||_F = sqrt( sum( W[i, :, :, :] ** 2 ) )

    Channels with smaller Frobenius norm contribute less to the output
    and are therefore candidates for pruning.

    Args:
        weight: Conv2d weight tensor of shape (C_out, C_in, K, K).

    Returns:
        A 1-D tensor of shape (C_out,) containing the importance score
        for each output channel.
    """
    c_out = weight.size(0)
    # Flatten all but the first dimension, then compute L2 norm
    return weight.view(c_out, -1).norm(p=2, dim=1)


def select_top_channels(importance: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    """Select which output channels to keep based on importance scores.

    Args:
        importance:  Importance scores of shape (C_out,).
        prune_ratio: Fraction of channels to prune (0.0 to 1.0).

    Returns:
        Sorted indices of channels to keep, shape (C_out * (1-prune_ratio),).
    """
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in [0, 1); got {prune_ratio}")

    num_channels = importance.size(0)
    num_keep = max(1, int(num_channels * (1.0 - prune_ratio)))
    _, top_indices = torch.topk(importance, num_keep)
    return torch.sort(top_indices).values


# ===========================================================================
# Channel Pruning
# ===========================================================================


def channel_prune(original: DemoCNN, prune_ratio: float) -> PrunedCNN:
    """Prune channels from every ConvBlock using Frobenius norm ranking.

    The algorithm:
      1. For each ConvBlock, compute Frobenius norm of every output channel.
      2. Sort channels by importance and keep the top (1 - prune_ratio).
      3. Build a new PrunedCNN with reduced channel counts.
      4. Copy the kept weights from the original model into the new one.

    Because pruning output channels of block i reduces the input channels
    of block i+1, the kept-input indices for block i are the kept-output
    indices of block i-1.  The first block always keeps all 3 input channels.

    Args:
        original:    The trained DemoCNN to prune.
        prune_ratio: Fraction of output channels to remove per block.

    Returns:
        A new PrunedCNN instance with reduced channel counts and copied weights.
    """
    original.eval()

    # ---- Step 1: decide which output channels to keep per block ------------
    kept_outputs: List[torch.Tensor] = []
    for block in original.blocks:
        imp = frobenius_importance(block.conv.weight.data)
        kept = select_top_channels(imp, prune_ratio)
        kept_outputs.append(kept)

    # ---- Step 2: build new channel configs ---------------------------------
    new_channels: List[Tuple[int, int, int]] = []
    prev_kept_out = torch.arange(INPUT_CHANNELS)  # all input channels kept for block1

    for i, block in enumerate(original.blocks):
        in_c = prev_kept_out.size(0)
        out_c = kept_outputs[i].size(0)
        stride = block.conv.stride[0]
        new_channels.append((in_c, out_c, stride))
        prev_kept_out = kept_outputs[i]

    # ---- Step 3: instantiate pruned model ----------------------------------
    pruned = PrunedCNN(new_channels, num_classes=original.classifier.out_features)

    # ---- Step 4: copy weights -----------------------------------------------
    prev_kept_out = torch.arange(INPUT_CHANNELS)

    for i, (orig_block, kept_out) in enumerate(zip(original.blocks, kept_outputs)):
        new_block = pruned.blocks[i]

        # Copy Conv2d weight: select kept output and appropriate input channels
        new_block.conv.weight.data.copy_(
            orig_block.conv.weight.data[kept_out][:, prev_kept_out]
        )

        # Copy BatchNorm parameters for the kept output channels
        new_block.bn.weight.data.copy_(orig_block.bn.weight.data[kept_out])
        new_block.bn.bias.data.copy_(orig_block.bn.bias.data[kept_out])
        new_block.bn.running_mean.data.copy_(orig_block.bn.running_mean.data[kept_out])
        new_block.bn.running_var.data.copy_(orig_block.bn.running_var.data[kept_out])

        prev_kept_out = kept_out

    # Copy classifier: input dimension corresponds to last block's kept outputs
    pruned.classifier.weight.data.copy_(
        original.classifier.weight.data[:, prev_kept_out]
    )
    pruned.classifier.bias.data.copy_(original.classifier.bias.data)

    return pruned


# ===========================================================================
# Fine-tuning
# ===========================================================================


def fine_tune(
    model: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int = FINE_TUNE_EPOCHS,
    lr: float = LR,
) -> None:
    """Fine-tune a pruned model to recover accuracy.

    Args:
        model:        Pruned model to fine-tune (modified in-place).
        train_images: Training images.
        train_labels: Training labels.
        test_images:  Test images for tracking accuracy.
        test_labels:  Test labels for tracking accuracy.
        epochs:       Number of fine-tuning epochs.
        lr:           Learning rate.
    """
    print(f"\n  Fine-tuning for {epochs} epochs (lr={lr}) ...")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_images, train_labels, lr=lr)
        acc = evaluate_accuracy(model, test_images, test_labels)
        print(f"    Epoch {epoch:>2d}/{epochs}  loss={loss:.4f}  acc={acc:.4f}")


# ===========================================================================
# Metrics
# ===========================================================================


def count_params(model: nn.Module) -> int:
    """Count total parameters in the model.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        Total number of parameters (trainable + non-trainable).
    """
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, input_shape: Tuple[int, int, int]) -> int:
    """Estimate total Conv2d MACs via a forward hook.

    Only Conv2d layers are counted; BatchNorm, pooling, and FC layers
    contribute negligible compute relative to convolutions.

    Args:
        model:       A PyTorch nn.Module.
        input_shape: (C, H, W) of a single input sample.

    Returns:
        Total estimated MACs across all Conv2d layers.
    """
    model.eval()
    total_macs = 0
    dummy = torch.randn(1, *input_shape)

    def _hook(
        module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor, /
    ) -> None:
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            x = inp[0]  # (1, C_in, H_in, W_in)
            c_in = x.shape[1]
            h_in = x.shape[2]
            w_in = x.shape[3]
            h_out = out.shape[2]
            w_out = out.shape[3]
            # MACs = C_out * H_out * W_out * C_in * K * K
            macs = (
                module.out_channels
                * h_out
                * w_out
                * c_in
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
            total_macs += macs

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(dummy)

    for h in handles:
        h.remove()

    return total_macs


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    warmup: int = WARMUP_RUNS,
    repeats: int = TIMED_RUNS,
) -> float:
    """Measure average forward-pass latency on CPU.

    Args:
        model:       A PyTorch nn.Module.
        input_shape: (C, H, W) of a single input sample.
        warmup:      Number of untimed warmup iterations.
        repeats:     Number of timed iterations.

    Returns:
        Average per-inference latency in milliseconds.
    """
    model.eval()
    dummy = torch.randn(1, *input_shape)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    end = time.perf_counter()

    return (end - start) / repeats * 1000.0  # ms


def print_comparison(
    label: str,
    accuracy: float,
    params: int,
    macs: int,
    latency_ms: float,
) -> None:
    """Print a single row of the comparison table.

    Args:
        label:      Model description (e.g. "Original" or "Pruned + FT").
        accuracy:   Top-1 accuracy.
        params:     Parameter count.
        macs:       Total Conv2d MACs.
        latency_ms: Inference latency in milliseconds.
    """
    print(
        f"  {label:<20}  acc={accuracy:.4f}  "
        f"params={params:>9,}  MACs={macs:>12,}  latency={latency_ms:.3f} ms"
    )


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Run the full channel pruning pipeline."""
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  LECTURE 04: Channel Pruning with Frobenius Norm")
    print("=" * 70)

    # ---- 1. Create synthetic data ------------------------------------------
    print("\n[1] Generating synthetic dataset ...")
    train_images, train_labels = _create_synthetic_data(
        NUM_TRAIN, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES
    )
    test_images, test_labels = _create_synthetic_data(
        NUM_TEST, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES
    )
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")

    # ---- 2. Build and train the original model -----------------------------
    print(f"\n[2] Building DemoCNN and training for {INITIAL_EPOCHS} epochs ...")
    model = DemoCNN(num_classes=NUM_CLASSES)

    for epoch in range(1, INITIAL_EPOCHS + 1):
        loss = train_one_epoch(model, train_images, train_labels)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2d}  loss={loss:.4f}")

    original_acc = evaluate_accuracy(model, test_images, test_labels)
    print(f"  Original accuracy: {original_acc:.4f}")

    # ---- 3. Frobenius norm sanity check ------------------------------------
    print("\n[3] Frobenius norm importance ranking (sanity check) ...")
    sample_conv = model.block1.conv.weight.data  # [64, 3, 3, 3]
    importance = frobenius_importance(sample_conv)
    print(f"  block1.conv weight shape: {tuple(sample_conv.shape)}")
    print(
        f"  Importance scores (first 8 of {importance.size(0)}): "
        f"{importance[:8].tolist()}"
    )
    print(f"  Top-5 channel indices: {torch.topk(importance, 5).indices.tolist()}")

    # ---- 4. Channel pruning ------------------------------------------------
    print(f"\n[4] Pruning {PRUNE_RATIO * 100:.0f}% of channels per layer ...")
    for i, block in enumerate(model.blocks):
        out_c = block.conv.out_channels
        keep = max(1, int(out_c * (1.0 - PRUNE_RATIO)))
        pruned_c = out_c - keep
        print(f"  block{i + 1}: {out_c} -> {keep} output channels ({pruned_c} pruned)")

    pruned_model = channel_prune(model, PRUNE_RATIO)
    pruned_params_before_ft = count_params(pruned_model)
    print(
        f"  Pruned model parameters: {pruned_params_before_ft:,} "
        f"(was {count_params(model):,})"
    )

    # ---- 5. Fine-tune the pruned model -------------------------------------
    print(f"\n[5] Fine-tuning pruned model ({FINE_TUNE_EPOCHS} epochs) ...")
    fine_tune(pruned_model, train_images, train_labels, test_images, test_labels)

    # ---- 6. Compare metrics ------------------------------------------------
    print("\n[6] Comparison: Original vs Pruned (+ fine-tuned)")
    input_shape = (INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    # Original metrics
    orig_params = count_params(model)
    orig_macs = estimate_macs(model, input_shape)
    orig_latency = measure_latency(model, input_shape)
    orig_accuracy = original_acc

    # Pruned metrics
    pruned_accuracy = evaluate_accuracy(pruned_model, test_images, test_labels)
    pruned_params = count_params(pruned_model)
    pruned_macs = estimate_macs(pruned_model, input_shape)
    pruned_latency = measure_latency(pruned_model, input_shape)

    print(
        f"\n  {'':<20}  {'Accuracy':>8}  {'Params':>10}  {'MACs':>13}  {'Latency':>12}"
    )
    print(f"  {'':->20}  {'':->8}  {'':->10}  {'':->13}  {'':->12}")
    print_comparison("Original", orig_accuracy, orig_params, orig_macs, orig_latency)
    print_comparison(
        "Pruned + FT", pruned_accuracy, pruned_params, pruned_macs, pruned_latency
    )

    # ---- 7. Summary statistics ---------------------------------------------
    print(f"\n  Reduction summary (prune_ratio={PRUNE_RATIO}):")
    print(
        f"    Accuracy:  {orig_accuracy:.4f} -> {pruned_accuracy:.4f}  "
        f"({(pruned_accuracy - orig_accuracy) * 100:+.2f}%)"
    )
    print(
        f"    Params:    {orig_params:,} -> {pruned_params:,}  "
        f"({(1 - pruned_params / orig_params) * 100:.1f}% reduction)"
    )
    print(
        f"    MACs:      {orig_macs:,} -> {pruned_macs:,}  "
        f"({(1 - pruned_macs / orig_macs) * 100:.1f}% reduction)"
    )
    print(
        f"    Latency:   {orig_latency:.3f} -> {pruned_latency:.3f} ms  "
        f"({(1 - pruned_latency / orig_latency) * 100:.1f}% reduction)"
    )

    # ---- 8. Save pruned model ----------------------------------------------
    print(f"\n[7] Saving pruned model to '{SAVE_PATH}' ...")
    torch.save(pruned_model.state_dict(), SAVE_PATH)
    import os

    file_size_kb = os.path.getsize(SAVE_PATH) / 1024
    print(f"  Saved: {SAVE_PATH} ({file_size_kb:.1f} KiB)")

    # ---- 9. Done -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model: DemoCNN (4 ConvBlocks)")
    print(f"  Synthetic data: {NUM_TRAIN} train / {NUM_TEST} test")
    print(f"  Prune ratio: {PRUNE_RATIO} (per-layer channel pruning)")
    print(f"  Method: Frobenius norm importance ranking")
    print(f"  Original params: {orig_params:,}  |  Pruned params: {pruned_params:,}")
    print(
        f"  Original accuracy: {orig_accuracy:.4f}  "
        f"|  Pruned accuracy: {pruned_accuracy:.4f}"
    )
    print(f"  Pruned model saved to: {SAVE_PATH}")
    print("=" * 70)

    print("\nLecture 04 complete.")


if __name__ == "__main__":
    main()
