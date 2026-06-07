"""
Vision Transformer Efficiency Analysis (Lecture 16)

Builds a Vision Transformer (ViT) and a comparable ResNet-style CNN from
scratch, then profiles them across different patch/image sizes to
understand the efficiency trade-offs between convolution and self-attention.

Modules implemented:
  - PatchEmbedding: splits an image into non-overlapping patches and
    projects each to a d_model-dimensional vector.
  - TransformerBlock: standard pre-LN block with Multi-Head Self-Attention
    and a two-layer MLP (GELU activation).
  - VisionTransformer: stacks PatchEmbedding, learned positional embeddings,
    N transformer blocks, and a linear classification head.
  - SimpleCNN: a ResNet-style convolutional backbone with three stages
    (each containing a residual block) followed by GAP + FC.

The script also:
  - Counts parameters and estimates FLOPs (MACs) for both models.
  - Compares ViT and CNN across {4, 8, 16} patch sizes and
    {32, 64, 96} image sizes.
  - Extracts and visualises attention maps from the last transformer block.
  - Outputs a structured summary table to stdout.

Dependencies: torch, numpy, matplotlib (all CPU-only; no CUDA required).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Utility: Parameter Counting
# ===========================================================================


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) for *model*.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        A tuple of (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===========================================================================
# Utility: FLOPs (MACs) Counter via Forward Hooks
# ===========================================================================


class _FlopsHook:
    """Accumulates Conv2d and Linear MACs during a single forward pass.

    Conventions (matching common ML literature usage where "FLOPs" ≈ MACs):
        - Conv2d:  out_c * out_h * out_w * (in_c / groups) * k_h * k_w
        - Linear:  in_features * out_features  (per row of the last dim)

    BatchNorm, LayerNorm, activation, pooling, and residual-add operations
    are *not* counted -- they contribute < 1 % of total compute.
    """

    def __init__(self) -> None:
        self.total_macs: int = 0
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- hook callbacks ---------------------------------------------------

    def _conv_hook(
        self,
        module: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        x = inp[0]  # (N, C_in, H_in, W_in)
        in_c = x.shape[1]
        out_c = module.out_channels  # type: ignore[union-attr]
        k_h, k_w = module.kernel_size  # type: ignore[union-attr]
        groups: int = module.groups  # type: ignore[union-attr]
        out_h, out_w = out.shape[2], out.shape[3]
        self.total_macs += out_c * out_h * out_w * (in_c // groups) * k_h * k_w

    def _linear_hook(
        self,
        module: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        x = inp[0]
        # x shape: (*prefix, in_features)
        # Each "row" in the suffix dim does a [in_f, out_f] matmul.
        rows = x.numel() // module.in_features  # type: ignore[union-attr]
        self.total_macs += rows * module.in_features * module.out_features  # type: ignore[union-attr]

    # -- public API -------------------------------------------------------

    def register(self, model: nn.Module) -> None:
        """Attach forward hooks to every Conv2d / Linear inside *model*."""
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self._handles.append(m.register_forward_hook(self._conv_hook))
            elif isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))

    def remove(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


def estimate_macs(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """Run one forward pass through *model* and return total estimated MACs.

    Args:
        model:        PyTorch module (will be set to eval mode).
        input_tensor: A single-sample tensor with batch dim (1, C, H, W).

    Returns:
        Estimated multiply-accumulate count for the forward pass.
    """
    model.eval()
    hook = _FlopsHook()
    hook.register(model)
    with torch.no_grad():
        _ = model(input_tensor)
    hook.remove()
    return hook.total_macs


# ===========================================================================
# ViT Building Blocks
# ===========================================================================


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and project to *d_model*.

    Uses a strided Conv2d as an efficient equivalent of the common
    "unfold + Linear" pattern.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int = 3,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches to embedding vectors.

        Args:
            x: (B, C, H, W) input image.

        Returns:
            Tensor of shape (B, num_patches, d_model).
        """
        x = self.proj(x)  # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: MHA + 2-layer MLP, each with residual.

    Follows the ViT paper (Dosovitskiy et al., 2021) which uses pre-norm
    and GELU activations.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Any:
        """Apply one transformer block.

        Args:
            x:               (B, seq_len, d_model) input.
            return_attention: If True, also return the attention weights from
                              the self-attention layer.

        Returns:
            If return_attention=False: (B, seq_len, d_model).
            If return_attention=True:  ((B, seq_len, d_model), attn_weights).
        """
        # Self-attention sub-block (pre-norm)
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed,
            normed,
            normed,
            need_weights=return_attention,
            average_attn_weights=False,  # per-head weights for visualisation
        )
        x = x + attn_out

        # MLP sub-block (pre-norm)
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn_weights
        return x


class VisionTransformer(nn.Module):
    """A small Vision Transformer (ViT) for tiny image classification.

    Architecture:
        PatchEmbedding -> [CLS] token + pos_embed ->
        N x TransformerBlock -> LayerNorm -> extract [CLS] -> Linear head
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        d_model: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.num_patches = self.patch_embed.num_patches

        # Learned [CLS] token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        # Truncated normal init for pos/CLS embeddings, following the ViT paper
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Linear / Conv layers use default PyTorch init which is fine

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Any:
        """Forward pass through the Vision Transformer.

        Args:
            x:               (B, C, H, W) input image batch.
            return_attention: If True, return attention weights from the
                              *last* transformer block.

        Returns:
            If return_attention=False: (B, num_classes) logits.
            If return_attention=True:
                ((B, num_classes) logits, (B, n_heads, S, S) attn weights).
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d_model)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer blocks
        attn_weights = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                x, attn_weights = block(x, return_attention=True)
            else:
                x = block(x)

        # Final normalisation and classification (use [CLS] token)
        x = self.norm(x)
        logits = self.head(x[:, 0])  # (B, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits

    @property
    def d_model(self) -> int:
        """Convenience accessor for the embedding dimension."""
        return self.patch_embed.proj.out_channels

    @property
    def num_heads(self) -> int:
        """Convenience accessor for the number of attention heads."""
        return self.blocks[0].attn.num_heads

    @property
    def depth(self) -> int:
        """Convenience accessor for the number of transformer blocks."""
        return len(self.blocks)

    def estimate_attention_macs(
        self,
        batch_size: int = 1,
        seq_len: int | None = None,
    ) -> int:
        """Return the MACs contributed by Q@K^T and softmax(QK^T)@V matmuls.

        These operations happen inside nn.MultiheadAttention and are *not*
        captured by linear-layer hooks, so we add them separately.

        Two matmuls per head per block:
            Q @ K^T  :  seq_len * d_head * seq_len   MACs
            attn @ V :  seq_len * seq_len * d_head   MACs

        Summed over heads:  2 * seq_len^2 * d_model  MACs per block.

        Args:
            batch_size: Number of samples in the batch.
            seq_len:    Sequence length (patches + 1 CLS token).  If None,
                        computed from img_size / patch_size.

        Returns:
            Total MACs from attention matmuls across all blocks.
        """
        if seq_len is None:
            seq_len = self.num_patches + 1
        return self.depth * batch_size * 2 * seq_len * seq_len * self.d_model


# ===========================================================================
# ResNet-style CNN for Comparison
# ===========================================================================


class ResidualBlock(nn.Module):
    """A basic residual block with two 3x3 convolutions."""

    expansion: int = 1  # for compatibility with bottleneck variants

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 shortcut when dimensions change
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class SimpleCNN(nn.Module):
    """ResNet-style convolutional backbone for small-image classification.

    Three stages, each doubling the channel count and halving spatial
    resolution (stride-2 in the first residual block of stages 2 and 3).
    Global average pooling reduces to a single feature vector before the
    final fully-connected classification head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 24,
    ) -> None:
        """Initialise SimpleCNN.

        Args:
            in_channels: Number of input image channels (3 for RGB).
            num_classes: Number of output classes.
            base_width:  Width of the first stage; doubles each subsequent
                         stage (base_width -> 2x -> 4x -> 8x).
        """
        super().__init__()
        w = base_width

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualBlock(w, w * 2, stride=2)
        self.layer2 = ResidualBlock(w * 2, w * 4, stride=2)
        self.layer3 = ResidualBlock(w * 4, w * 8, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) input image.

        Returns:
            (B, num_classes) logits.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ===========================================================================
# Visualisation: Attention Map
# ===========================================================================


def visualise_attention(
    model: VisionTransformer,
    input_tensor: torch.Tensor,
    save_path: str = "/tmp/vit_attention_map.png",
) -> None:
    """Extract attention weights from the last block and save a heatmap.

    The figure shows two panels:
      (a) Average attention over all heads (S x S).
      (b) Per-head attention maps in a grid.

    Args:
        model:        A VisionTransformer instance.
        input_tensor: (1, C, H, W) input image tensor.
        save_path:    File-system path for the output PNG.
    """
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(input_tensor, return_attention=True)

    # attn_weights shape: (B, n_heads, seq_len, seq_len)
    if attn_weights is None:
        raise RuntimeError(
            "Attention weights were not returned.  "
            "Ensure return_attention=True was passed to the model."
        )
    attn = attn_weights[0].cpu().numpy()  # (n_heads, S, S)
    n_heads, S, _ = attn.shape
    avg_attn = attn.mean(axis=0)  # (S, S)

    # Build the figure
    cols = min(4, n_heads)
    rows = math.ceil((n_heads + 1) / cols)  # +1 for the average panel
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows),
        squeeze=False,
    )

    # Panel 1/row 1: average over heads
    ax = axes[0, 0]
    im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
    ax.set_title("Average over heads")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused subplots in the first row
    for c in range(1, cols):
        axes[0, c].set_visible(False)

    # Remaining panels: one per head
    for h in range(n_heads):
        r = (h + 1) // cols
        c = (h + 1) % cols
        ax = axes[r, c]
        im = ax.imshow(attn[h], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h + 1}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide any trailing empty subplots
    total_panels = 1 + n_heads
    for idx in range(total_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"ViT Attention Maps  (patch={model.patch_embed.patch_size}, "
        f"img={model.patch_embed.img_size},  "
        f"d={model.d_model},  heads={model.num_heads},  "
        f"depth={model.depth})",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Attention map saved to {save_path}")


# ===========================================================================
# Comparison Table Printer
# ===========================================================================


def print_comparison_table(
    results: List[Dict[str, Any]],
) -> None:
    """Format and print a params / FLOPs summary table.

    Args:
        results: List of dicts, each with keys:
            img_size, patch_size, vit_params, vit_macs,
            cnn_params, cnn_macs.
    """
    header = (
        f"{'Img':>4}  {'Patch':>5}  "
        f"{'ViT Params':>12}  {'ViT MACs':>13}  "
        f"{'CNN Params':>12}  {'CNN MACs':>13}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  MODEL EFFICIENCY COMPARISON: ViT vs CNN")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['img_size']:>4}  {r['patch_size']:>5}  "
            f"{r['vit_params']:>12,}  {r['vit_macs']:>13,}  "
            f"{r['cnn_params']:>12,}  {r['cnn_macs']:>13,}"
        )
    print(sep)
    print()


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    # ---- Config ----------------------------------------------------------
    PATCH_SIZES = [4, 8, 16]
    IMG_SIZES = [32, 64, 96]
    NUM_CLASSES = 10
    IN_CHANNELS = 3

    # ViT hyper-parameters (small model for CPU-friendly profiling)
    VIT_D_MODEL = 128
    VIT_DEPTH = 4
    VIT_N_HEADS = 4
    VIT_MLP_RATIO = 4.0

    # CNN hyper-parameters (chosen to give ~same param count as the ViT)
    CNN_BASE_WIDTH = 24

    # ----------------------------------------------------------------------
    print("=" * 64)
    print("  VISION TRANSFORMER EFFICIENCY -- Lecture 16")
    print("=" * 64)
    print(
        f"  ViT config:  d_model={VIT_D_MODEL}, depth={VIT_DEPTH}, "
        f"heads={VIT_N_HEADS}, mlp_ratio={VIT_MLP_RATIO}"
    )
    print(f"  CNN config:  base_width={CNN_BASE_WIDTH} (ResNet-style, 3 stages)")
    print()

    # ---- 1. Build models and count params once per (img_size) ------------
    results: List[Dict[str, Any]] = []

    # CNN params are independent of patch_size (but vary slightly with
    # image size due to BN running stats which are not counted).
    # We build one CNN per image size for clarity.
    cnn_cache: Dict[int, Tuple[nn.Module, int]] = {}
    for img_size in IMG_SIZES:
        cnn = SimpleCNN(
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
            base_width=CNN_BASE_WIDTH,
        )
        cnn_total, _ = count_parameters(cnn)
        cnn_cache[img_size] = (cnn, cnn_total)

    # ---- 2. Loop over all configurations ---------------------------------
    for img_size in IMG_SIZES:
        for patch_size in PATCH_SIZES:
            # Skip if patch size does not divide image size
            if img_size % patch_size != 0:
                continue

            print(f"  Profiling: img={img_size}, patch={patch_size} ...")

            # ---- Build ViT -----------------------------------------------
            vit = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=IN_CHANNELS,
                num_classes=NUM_CLASSES,
                d_model=VIT_D_MODEL,
                depth=VIT_DEPTH,
                n_heads=VIT_N_HEADS,
                mlp_ratio=VIT_MLP_RATIO,
            )
            vit_total, _ = count_parameters(vit)

            # ---- Count ViT FLOPs -----------------------------------------
            dummy = torch.randn(1, IN_CHANNELS, img_size, img_size)
            vit_macs = estimate_macs(vit, dummy)

            # Add the attention matmul MACs (Q@K^T + attn@V) that are not
            # captured by Linear-hooks.
            seq_len = vit.num_patches + 1  # +1 for [CLS] token
            attn_matmul_macs = vit.estimate_attention_macs(
                batch_size=1,
                seq_len=seq_len,
            )
            vit_macs += attn_matmul_macs

            # ---- Count CNN FLOPs -----------------------------------------
            cnn, cnn_total = cnn_cache[img_size]
            cnn_macs = estimate_macs(cnn, dummy)

            results.append(
                {
                    "img_size": img_size,
                    "patch_size": patch_size,
                    "vit_params": vit_total,
                    "vit_macs": vit_macs,
                    "cnn_params": cnn_total,
                    "cnn_macs": cnn_macs,
                }
            )

    # ---- 3. Print comparison table ---------------------------------------
    print_comparison_table(results)

    # ---- 4. Attention map visualisation ----------------------------------
    print("  Generating attention map visualisation ...")
    demo_img_size = 64
    demo_patch_size = 8
    vit_demo = VisionTransformer(
        img_size=demo_img_size,
        patch_size=demo_patch_size,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        d_model=VIT_D_MODEL,
        depth=VIT_DEPTH,
        n_heads=VIT_N_HEADS,
        mlp_ratio=VIT_MLP_RATIO,
    )
    demo_input = torch.randn(1, IN_CHANNELS, demo_img_size, demo_img_size)
    visualise_attention(vit_demo, demo_input)

    # ---- 5. Summary comparison -------------------------------------------
    # Pick one representative config for the summary
    rep = results[0]  # first entry (e.g., img=32, patch=4)
    print()
    print("=" * 64)
    print("  SUMMARY: ViT vs CNN")
    print("=" * 64)
    print(f"  Image size:          {rep['img_size']}x{rep['img_size']}")
    print(f"  Patch size (ViT):    {rep['patch_size']}")
    print(f"  ViT parameters:      {rep['vit_params']:>12,}")
    print(f"  CNN parameters:      {rep['cnn_params']:>12,}")
    print(f"  ViT MACs:            {rep['vit_macs']:>12,}")
    print(f"  CNN MACs:            {rep['cnn_macs']:>12,}")
    print()
    print("  Key characteristics:")
    print("    - ViT FLOPs scale *quadratically* with sequence length")
    print("      (i.e., (img_size / patch_size)^2).")
    print("    - CNN FLOPs scale *linearly* with spatial resolution")
    print("      (convolution is a local, weight-sharing operation).")
    print("    - ViT has *global* receptive field from layer 1 (MHA),")
    print("      whereas CNN builds it hierarchically.")
    print("    - ViT uses learnable positional embeddings; CNN is")
    print("      translation-equivariant by design (inductive bias).")
    print("    - At small image sizes, ViT and CNN are comparable;")
    print("      at large image sizes, the quadratic attention cost")
    print("      makes vanilla ViT much more expensive than CNN.")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
