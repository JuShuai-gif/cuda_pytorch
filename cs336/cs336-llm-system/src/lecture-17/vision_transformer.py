"""
Vision Transformer (ViT) from scratch.

Implements the full ViT pipeline:
  - Patch embedding: split image → linear projection
  - Learned 1D positional encoding + CLS token
  - Multi-head self-attention (from scratch)
  - MLP with GELU activation
  - Stacked transformer encoder blocks (N=4)
  - Classification head on CLS token output
  - Training on synthetic image data with accuracy tracking
  - Attention map visualization saved as PNG
"""

from __future__ import annotations

from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (built from scratch)
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    No reliance on nn.MultiheadAttention — purely built from linear layers,
    reshape, and matmul operations.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Self-attention forward pass.

        Args:
            x: (B, N, embed_dim) input sequence.
            return_attention: if True, also return attention weights.

        Returns:
            Output tensor (B, N, embed_dim), and optionally attention (B, H, N, N).
        """
        B, N, D = x.shape

        # Linear projection to Q, K, V and split across heads
        qkv = self.qkv(x)  # (B, N, 3 * D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_output = attn_weights @ v  # (B, H, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(attn_output)

        if return_attention:
            return out, attn_weights
        return out


# ---------------------------------------------------------------------------
# MLP Block with GELU
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """Two-layer MLP with GELU activation and dropout."""

    def __init__(
        self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block: MHA → add&norm → MLP → add&norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_ratio, dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and project to embeddings."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            "image_size must be divisible by patch_size"
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Use a Conv2d as patch projection (equivalent to linear per-patch)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, num_patches, embed_dim)."""
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Full Vision Transformer
# ---------------------------------------------------------------------------


class VisionTransformer(nn.Module):
    """Vision Transformer from scratch (Dosovitskiy et al., 2021)."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Learned 1D positional encoding (one per patch + CLS)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        self.dropout = nn.Dropout(dropout)

        # Stack of transformer encoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: CLS token → logits
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_attentions: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass: (B, C, H, W) → (B, num_classes)."""
        B = x.size(0)

        # Patch embedding + CLS token + positional encoding
        x = self.patch_embed(x)  # (B, N_patches, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N_patches+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer blocks
        attentions: list[torch.Tensor] = []
        for blk in self.blocks:
            if return_attentions:
                x, attn = blk(x, return_attention=True)
                attentions.append(attn)
            else:
                x = blk(x)

        x = self.norm(x)

        # CLS token → classification
        x = self.head(x[:, 0])  # (B, num_classes)

        if return_attentions:
            return x, attentions
        return x


# ---------------------------------------------------------------------------
# Synthetic image dataset
# ---------------------------------------------------------------------------


class SyntheticImageDataset(torch.utils.data.Dataset):
    """Random images with class-specific patterns for classification."""

    def __init__(
        self,
        num_samples: int = 600,
        num_classes: int = 10,
        image_size: int = 32,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cls = idx % self.num_classes
        # Deterministic pseudo-random per sample, with class-specific bias
        torch.manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size)
        image = image + (cls / self.num_classes) * 1.0
        return image, cls


# ---------------------------------------------------------------------------
# Attention map visualization
# ---------------------------------------------------------------------------


def visualize_attention(
    model: VisionTransformer,
    image: torch.Tensor,
    save_path: str = "vit_attention_maps.png",
) -> None:
    """Visualize attention maps from all heads in the last transformer block.

    Saves a grid of heatmaps to a PNG file.
    """
    model.eval()
    with torch.no_grad():
        _, attentions = model(image.unsqueeze(0), return_attentions=True)

    # Take attention from the last block
    attn_last = attentions[-1]  # (1, num_heads, N+1, N+1)
    num_heads = attn_last.size(1)

    # Use CLS token's attention to patches (exclude CLS self-attention)
    cls_attn = attn_last[0, :, 0, 1:]  # (num_heads, N_patches)

    num_patches = cls_attn.size(1)
    patch_grid = int(num_patches**0.5)

    fig, axes = plt.subplots(1, num_heads, figsize=(3 * num_heads, 3), squeeze=False)
    for h in range(num_heads):
        attn_map = cls_attn[h].reshape(patch_grid, patch_grid).cpu().numpy()
        im = axes[0, h].imshow(attn_map, cmap="viridis", aspect="equal")
        axes[0, h].set_title(f"Head {h + 1}")
        axes[0, h].axis("off")
        plt.colorbar(im, ax=axes[0, h], fraction=0.046)

    fig.suptitle("ViT CLS Token Attention to Patches (Last Block)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention maps saved to {save_path}")


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("Vision Transformer (ViT) from Scratch")
    print("=" * 60)

    # Hyperparameters
    image_size = 32
    patch_size = 4
    embed_dim = 128
    depth = 4
    num_heads = 4
    num_classes = 10
    batch_size = 64
    num_epochs = 20
    lr = 1e-3

    # Build model
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        dropout=0.1,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input: {image_size}x{image_size} RGB images")
    print(
        f"Patches: {patch_size}x{patch_size} → {model.patch_embed.num_patches} patches"
    )
    print(f"Embed dim: {embed_dim}, Depth: {depth}, Heads: {num_heads}")

    # Synthetic dataset
    train_dataset = SyntheticImageDataset(
        num_samples=500, num_classes=num_classes, image_size=image_size
    )
    test_dataset = SyntheticImageDataset(
        num_samples=200, num_classes=num_classes, image_size=image_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining: {num_epochs} epochs, batch_size={batch_size}, lr={lr}\n")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += images.size(0)
        test_acc = test_correct / test_total

        if epoch % 4 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Loss: {total_loss / total:.4f} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Test Acc: {test_acc:.2%}"
            )

    print(f"\nFinal test accuracy: {test_acc:.2%}")

    # Visualize attention maps
    sample_img, _label = test_dataset[0]
    visualize_attention(model, sample_img, save_path="vit_attention_maps.png")

    print("Done!")


if __name__ == "__main__":
    main()
