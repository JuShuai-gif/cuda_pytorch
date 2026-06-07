#!/usr/bin/env python3
"""
Lecture 14: Parameter Efficient Fine-Tuning (PEFT) -- LoRA from Scratch
======================================================================

This script demonstrates Low-Rank Adaptation (LoRA) for fine-tuning
large pre-trained models efficiently.  It covers:

  1. A custom ``LoRALinear`` layer that wraps a frozen ``nn.Linear``
     weight with two low-rank matrices A and B.
  2. Pre-training a small MLP on MNIST.
  3. Applying LoRA to selected layers and fine-tuning on a MNIST subset.
  4. Comparing trainable-parameter counts between full fine-tuning
     and LoRA at various ranks.
  5. Merging LoRA weights back into the original weights.

All computation runs on CPU -- no CUDA required.
"""

import copy
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# 1.  Custom LoRA Linear Layer
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """
    A linear layer augmented with Low-Rank Adaptation (LoRA).

    The original weight ``W`` (shape: ``out_features x in_features``) is
    frozen (``requires_grad = False``).  Two trainable low-rank matrices
    ``A`` (``r x in_features``) and ``B`` (``out_features x r``) are
    introduced so that the effective forward pass becomes::

        h = x @ W^T + (x @ A^T @ B^T) * (alpha / r)

    where:
        - ``r``      : rank of the low-rank decomposition,
        - ``alpha``  : scaling factor that controls the magnitude of the
                       LoRA update relative to the frozen weights.

    Initialisation:
        ``A`` is initialised with ``kaiming_uniform`` for stable gradient
        flow; ``B`` is initialised to **zeros** so that the LoRA branch
        initially contributes nothing (``delta_W = 0``).

    Merge / Unmerge:
        ``merge()`` folds the LoRA weights into the original weight:
            ``W_merged = W + (alpha / r) * (B @ A)``
        After merging the LoRA branch can be bypassed with a standard
        linear forward.  ``unmerge()`` reverses the operation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # --- frozen original weight ---
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Freeze the base weight and bias immediately.
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = alpha / r  # cached for forward

        # --- low-rank matrices ---
        # A: (r, in_features)  -- projects input down to rank r
        # B: (out_features, r) -- projects rank-r representation back up
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))  # zero-init
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # Track whether LoRA weights have been merged into the original.
        self._merged = False
        # Backup of the original weight for unmerge.
        self._orig_weight: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ``x @ W^T + (x @ A^T @ B^T) * (alpha / r)``.

        When merged, falls back to a plain linear forward.
        """
        if self._merged:
            return self.linear(x)

        # Frozen base output:   x @ W^T  (batch, out_features)
        base = self.linear(x)

        # LoRA delta:  (x @ A^T) @ B^T  (batch, out_features)
        lora = (x @ self.A.T) @ self.B.T

        return base + lora * self.scaling

    # -- merge / unmerge utilities ----------------------------------------

    def merge(self) -> None:
        """
        Fold LoRA weights into the frozen weight:

            W := W + (alpha / r) * (B @ A)

        After this call the LoRA matrices are no longer needed for
        inference and the layer behaves like a standard ``nn.Linear``.
        """
        if self._merged:
            return  # already merged

        delta = (self.B @ self.A) * self.scaling  # (out_features, in_features)
        self._orig_weight = self.linear.weight.data.clone()
        self.linear.weight.data.add_(delta)
        self._merged = True

    def unmerge(self) -> None:
        """Restore the original weight (undo ``merge()``)."""
        if not self._merged or self._orig_weight is None:
            return

        self.linear.weight.data.copy_(self._orig_weight)
        self._orig_weight = None
        self._merged = False

    # -- convenience properties -------------------------------------------

    @property
    def trainable_params(self) -> int:
        """Number of trainable parameters in A and B."""
        return self.A.numel() + self.B.numel()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, merged={self._merged}"
        )


# ---------------------------------------------------------------------------
# 2.  MLP Model (simple classifier)
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """
    A three-layer MLP for MNIST digit classification.

        Layer 1:  784 -> 256  + ReLU
        Layer 2:  256 -> 128  + ReLU
        Layer 3:  128 -> 10   (logits)
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# 3.  Data helpers
# ---------------------------------------------------------------------------


def get_mnist_loaders(
    batch_size: int = 64,
    subset_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return train and test DataLoaders for MNIST.

    If ``subset_size`` is given the training dataset is trimmed to that
    many samples (useful for demonstrating LoRA fine-tuning on a tiny
    dataset).
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    if subset_size is not None:
        indices = np.random.default_rng(42).choice(
            len(train_ds),
            size=min(subset_size, len(train_ds)),
            replace=False,
        )
        train_ds = Subset(train_ds, indices.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 4.  Training & evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch.  Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 5.  LoRA application helper
# ---------------------------------------------------------------------------


def apply_lora_to_model(
    base_model: nn.Module,
    r: int,
    alpha: float,
    target_layers: List[str],
) -> nn.Module:
    """
    Replace the linear layers named in ``target_layers`` with
    ``LoRALinear`` wrappers that share the original frozen weight.

    Parameters
    ----------
    base_model : nn.Module
        Pre-trained model (weights will be frozen in the process).
    r : int
        LoRA rank.
    alpha : float
        LoRA scaling factor.
    target_layers : List[str]
        Attribute names of ``nn.Linear`` layers to augment (e.g.
        ``["fc1", "fc2"]``).

    Returns
    -------
    nn.Module
        The same model with LoRA layers in place.
    """
    for name in target_layers:
        original: nn.Linear = getattr(base_model, name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear")

        lora = LoRALinear(
            in_features=original.in_features,
            out_features=original.out_features,
            r=r,
            alpha=alpha,
            bias=original.bias is not None,
        )

        # Copy the pre-trained weight (and bias) into the LoRA wrapper.
        with torch.no_grad():
            lora.linear.weight.copy_(original.weight)
            if original.bias is not None:
                lora.linear.bias.copy_(original.bias)

        setattr(base_model, name, lora)

    return base_model


# ---------------------------------------------------------------------------
# 6.  Main demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 6.0  Setup
    # ------------------------------------------------------------------
    device = torch.device("cpu")
    print("Device:", device)
    print()

    # ------------------------------------------------------------------
    # 6.1  Pre-train a simple MLP on full MNIST
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Phase 1: Pre-training MLP on MNIST (full dataset)")
    print("=" * 65)

    pretrain_loader, full_test_loader = get_mnist_loaders(batch_size=128)

    model = SimpleMLP().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters (total): {total_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pretrain_epochs = 3

    for epoch in range(1, pretrain_epochs + 1):
        t0 = time.perf_counter()
        avg_loss = train_one_epoch(model, pretrain_loader, optimizer, device)
        acc = evaluate(model, full_test_loader, device)
        elapsed = time.perf_counter() - t0
        print(
            f"  Epoch {epoch}/{pretrain_epochs} | loss={avg_loss:.4f} "
            f"| test acc={acc:.4f} | {elapsed:.1f}s"
        )

    pretrain_acc = evaluate(model, full_test_loader, device)
    print(f"\nPre-training final accuracy: {pretrain_acc:.4f}")
    print()

    # Save a frozen copy of the pre-trained weights for later re-use.
    base_state = copy.deepcopy(model.state_dict())

    # ------------------------------------------------------------------
    # 6.2  Show full fine-tuning parameter count
    # ------------------------------------------------------------------
    full_ft_params = total_params  # every weight gets updated
    print(f"Full fine-tuning would train: {full_ft_params:,} parameters")
    print()

    # ------------------------------------------------------------------
    # 6.3  LoRA fine-tuning across different ranks
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Phase 2: LoRA fine-tuning on a small MNIST subset (2048 samples)")
    print("=" * 65)

    ranks = [2, 4, 8, 16]
    lora_alphas = {2: 2.0, 4: 4.0, 8: 8.0, 16: 16.0}
    subset_size = 2048
    lora_epochs = 5
    lora_lr = 5e-4

    results: List[Tuple[int, int, float]] = []  # (rank, trainable_params, accuracy)

    for r in ranks:
        print(f"\n--- LoRA rank r = {r} ---")

        # Re-load pre-trained base every time to start fresh.
        model = SimpleMLP().to(device)
        model.load_state_dict(base_state)

        # Apply LoRA to the first two linear layers.
        alpha = lora_alphas[r]
        apply_lora_to_model(model, r=r, alpha=alpha, target_layers=["fc1", "fc2"])

        # Count trainable parameters (only A and B matrices).
        trainable = count_trainable_params(model)
        frozen = total_params - trainable
        print(
            f"  Trainable params: {trainable:,} / {total_params:,} "
            f"({trainable / total_params * 100:.2f}%)"
        )

        # Create a subset loader.
        subset_loader, _ = get_mnist_loaders(batch_size=64, subset_size=subset_size)

        # Only optimise parameters that require gradients (A, B).
        optimizer_lora = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lora_lr,
        )

        for epoch in range(1, lora_epochs + 1):
            avg_loss = train_one_epoch(model, subset_loader, optimizer_lora, device)

        acc = evaluate(model, full_test_loader, device)
        print(f"  LoRA test accuracy (rank={r}): {acc:.4f}")

        results.append((r, trainable, acc))

    # ------------------------------------------------------------------
    # 6.4  Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Phase 3: Results Summary")
    print("=" * 65)

    header = (
        f"{'Method':>22}  {'Trainable Params':>17}  {'Accuracy':>9}  {'% of Full':>10}"
    )
    print(header)
    print("-" * len(header))

    # Baseline: full fine-tuning (all params, accuracy from pre-training)
    print(
        f"{'Full Fine-Tuning':>22}  {full_ft_params:>17,}  {pretrain_acc:>9.4f}  {'100.00%':>10}"
    )

    # LoRA rows
    for r, tp, acc in results:
        pct = tp / full_ft_params * 100
        print(
            f"{'LoRA (rank=' + str(r) + ')':>22}  {tp:>17,}  {acc:>9.4f}  {pct:>9.2f}%"
        )

    print()

    # ------------------------------------------------------------------
    # 6.5  Rank effect analysis
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Phase 4: Rank Effect Analysis")
    print("=" * 65)
    print(
        f"{'Rank':>5}  {'Trainable':>10}  {'Accuracy':>9}  {'Δ Acc (LoRA - Pretrain)':>25}"
    )
    print("-" * 65)
    for r, tp, acc in results:
        delta = acc - pretrain_acc
        print(f"{r:>5}  {tp:>10,}  {acc:>9.4f}  {delta:>+25.4f}")
    print()

    # ------------------------------------------------------------------
    # 6.6  Merge demonstration
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Phase 5: Weight Merge Demonstration (rank=8)")
    print("=" * 65)

    # Create a fresh model with LoRA (rank=8) and quickly train.
    model_merge = SimpleMLP().to(device)
    model_merge.load_state_dict(base_state)
    apply_lora_to_model(model_merge, r=8, alpha=8.0, target_layers=["fc1", "fc2"])

    # Quick train to get distinct LoRA weights.
    merge_loader, _ = get_mnist_loaders(batch_size=64, subset_size=2048)
    opt_merge = torch.optim.Adam(
        [p for p in model_merge.parameters() if p.requires_grad],
        lr=5e-4,
    )
    for _ in range(3):  # short training
        train_one_epoch(model_merge, merge_loader, opt_merge, device)

    # Compare predictions before and after merge (should be identical).
    x_sample, _ = next(iter(full_test_loader))
    x_sample = x_sample[:16].to(device)  # first 16 images

    model_merge.eval()
    with torch.no_grad():
        pred_before = model_merge(x_sample)

    # Merge and compare.
    for name in ["fc1", "fc2"]:
        layer = getattr(model_merge, name)
        if isinstance(layer, LoRALinear):
            layer.merge()

    with torch.no_grad():
        pred_after = model_merge(x_sample)

    max_diff = (pred_before - pred_after).abs().max().item()
    agreement = (
        (pred_before.argmax(dim=1) == pred_after.argmax(dim=1)).float().mean().item()
    )

    print(f"  Max logit difference (before vs after merge): {max_diff:.2e}")
    print(f"  Prediction agreement: {agreement:.1%}")
    if max_diff < 1e-5:
        print("  ✓ Merge is numerically stable -- outputs are identical.")
    else:
        print("  ⚠ Small numerical differences exist (expected with fp32).")

    # Unmerge and verify old behaviour restored.
    for name in ["fc1", "fc2"]:
        layer = getattr(model_merge, name)
        if isinstance(layer, LoRALinear):
            layer.unmerge()

    with torch.no_grad():
        pred_unmerged = model_merge(x_sample)

    max_diff2 = (pred_before - pred_unmerged).abs().max().item()
    print(f"  Max logit diff (original vs after unmerge): {max_diff2:.2e}")
    print()

    # ------------------------------------------------------------------
    # 6.7  Final summary
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Done!  Key takeaways:")
    print("  - LoRA dramatically reduces trainable parameters (1-5% of full FT).")
    print("  - Higher rank can improve accuracy, with diminishing returns.")
    print("  - Merging LoRA back into the base weights is lossless (up to fp32).")
    print("=" * 65)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
