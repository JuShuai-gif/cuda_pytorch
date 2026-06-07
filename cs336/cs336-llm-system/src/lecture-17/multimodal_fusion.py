"""
Multimodal Fusion Strategies Comparison.

Implements and compares three fusion strategies:
  a) Early Fusion: concatenate raw inputs → shared transformer
  b) Late Fusion: separate encoders → concatenated embeddings → classifier
  c) Cross-Attention Fusion: self-attention + cross-attention between modalities

Uses synthetic data (random images + random token sequences) to evaluate
each strategy on a binary classification task (matching vs non-matching pair).
Compares parameter counts, FLOPs, and accuracy.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: one sequence attends to another.

    Query comes from x, key/value come from y (context).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attention: x attends to context.

        Args:
            x: (B, Nx, D) query sequence.
            context: (B, Ny, D) key/value sequence.

        Returns:
            (B, Nx, D) output.
        """
        B, Nx, D = x.shape
        Ny = context.size(1)

        q = self.q_proj(x).reshape(B, Nx, self.num_heads, self.head_dim)
        kv = self.kv_proj(context).reshape(B, Ny, 2, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # (B, H, Nx, head_dim)
        k = kv[:, :, 0].permute(0, 2, 1, 3)  # (B, H, Ny, head_dim)
        v = kv[:, :, 1].permute(0, 2, 1, 3)  # (B, H, Ny, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nx, Ny)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, Nx, head_dim)
        out = out.transpose(1, 2).reshape(B, Nx, D)
        return self.out_proj(out)


class PositionalEncoding(nn.Module):
    """Learned 1D positional encoding."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Shared Encoders (used by Late Fusion and Cross-Attention Fusion)
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """Lightweight CNN → flat embedding."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32→16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16→8
            nn.Conv2d(64, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.view(x.size(0), -1)


class TextEncoder(nn.Module):
    """Small transformer encoder for text tokens."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x) * math.sqrt(self.token_emb.embedding_dim)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=None, is_causal=False)
        x = x.mean(dim=1)  # (B, d_model)
        return self.proj(x)


# ===================================================================
# Strategy A: Early Fusion
# ===================================================================


class EarlyFusion(nn.Module):
    """Concatenate raw pixels + token embeddings → shared transformer → classifier."""

    def __init__(
        self,
        image_size: int = 32,
        seq_len: int = 16,
        vocab_size: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.seq_len = seq_len

        # Flatten image to seq of patches (4x4 patches, 16 patches)
        self.patch_proj = nn.Conv2d(
            3, d_model, kernel_size=8, stride=8
        )  # 32→4x4 patches
        self.num_img_patches = (image_size // 8) ** 2  # 16

        # Text embedding to d_model
        self.text_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding for the combined sequence
        total_len = self.num_img_patches + seq_len
        self.pos_enc = PositionalEncoding(d_model, max_len=total_len)

        # Shared transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image: (B, 3, 32, 32)
            text: (B, seq_len) token indices

        Returns:
            (B, num_classes) logits
        """
        B = image.size(0)

        # Patchify image
        img_tokens = self.patch_proj(image)  # (B, D, 4, 4)
        img_tokens = img_tokens.flatten(2).transpose(1, 2)  # (B, 16, D)

        # Embed text tokens
        txt_tokens = self.text_emb(text)  # (B, seq_len, D)

        # Concatenate
        combined = torch.cat([img_tokens, txt_tokens], dim=1)  # (B, 16+seq_len, D)
        combined = self.pos_enc(combined)

        # Shared transformer
        out = self.transformer(combined, mask=None, is_causal=False)

        # Pool (mean over all tokens) and classify
        pooled = out.mean(dim=1)  # (B, D)
        return self.classifier(pooled)


# ===================================================================
# Strategy B: Late Fusion
# ===================================================================


class LateFusion(nn.Module):
    """Independent image/text encoders → concatenate embeddings → classifier."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        self.txt_encoder = TextEncoder(embed_dim=embed_dim)

        # Fusion + classification
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        img_emb = self.img_encoder(image)  # (B, embed_dim)
        txt_emb = self.txt_encoder(text)  # (B, embed_dim)
        fused = torch.cat([img_emb, txt_emb], dim=1)  # (B, 2*embed_dim)
        return self.classifier(fused)


# ===================================================================
# Strategy C: Cross-Attention Fusion
# ===================================================================


class CrossAttentionFusionBlock(nn.Module):
    """One fusion block: text self-attn → text→image cross-attn → MLP."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, text_seq: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Text undergoes self-attn + cross-attn to image features."""
        # Self-attention on text
        x = text_seq
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # Cross-attention: text attends to image
        x = x + self.cross_attn(self.norm2(x), image_features)

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class CrossAttentionFusion(nn.Module):
    """Fusion via cross-attention: text transformer with cross-attention to image.

    Image is encoded into a sequence of feature vectors (e.g., patch features).
    Text is processed through transformer layers that include cross-attention
    to the image feature sequence.
    """

    def __init__(
        self,
        image_size: int = 32,
        seq_len: int = 16,
        vocab_size: int = 256,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Image encoder to produce a sequence of features
        self.img_proj = nn.Conv2d(
            3, embed_dim, kernel_size=8, stride=8
        )  # 32→4x4 patches
        self.num_img_features = (image_size // 8) ** 2  # 16
        self.img_pos_enc = PositionalEncoding(embed_dim, max_len=self.num_img_features)

        # Text embedding
        self.text_emb = nn.Embedding(vocab_size, embed_dim)
        self.text_pos_enc = PositionalEncoding(embed_dim, max_len=seq_len)

        # Cross-attention fusion blocks
        self.fusion_blocks = nn.ModuleList(
            [CrossAttentionFusionBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        B = image.size(0)

        # Encode image as sequence
        img_feats = self.img_proj(image)  # (B, D, 4, 4)
        img_feats = img_feats.flatten(2).transpose(1, 2)  # (B, 16, D)
        img_feats = self.img_pos_enc(img_feats)

        # Embed text
        text_seq = self.text_emb(text) * math.sqrt(self.embed_dim)
        text_seq = self.text_pos_enc(text_seq)

        # Cross-attention fusion
        for block in self.fusion_blocks:
            text_seq = block(text_seq, img_feats)

        # Pool text sequence and classify
        pooled = text_seq.mean(dim=1)  # (B, D)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Synthetic multimodal dataset
# ---------------------------------------------------------------------------


class SyntheticMultimodalDataset(torch.utils.data.Dataset):
    """Pairs of (image, text, label) where label=1 if matching, 0 otherwise."""

    def __init__(
        self,
        num_samples: int = 400,
        num_classes: int = 2,  # binary: matching or not
        image_size: int = 32,
        seq_len: int = 16,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Half are matching pairs, half are not
        label = idx % self.num_classes  # 0 or 1
        torch.manual_seed(idx)

        # Image with class-specific bias
        image = torch.randn(3, self.image_size, self.image_size)
        if label == 1:
            # Matching: both modalities share the same class signal
            image = image + 0.5
            text = torch.randint(128, 256, (self.seq_len,))
        else:
            # Non-matching: different signal
            image = image - 0.5
            text = torch.randint(0, 128, (self.seq_len,))

        return image, text.long(), label


# ---------------------------------------------------------------------------
# Training & evaluation utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
) -> int:
    """Estimate FLOPs via PyTorch profiler (if available) or a simple heuristic."""
    try:
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.no_grad():
                model(image, text)
        total_flops = sum(
            event.flops for event in prof.key_averages() if event.flops is not None
        )
        return total_flops
    except (ImportError, TypeError, AttributeError):
        # Fallback: rough estimate based on parameter count
        return count_parameters(model) * 100  # heuristic


def train_one_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    name: str = "Model",
) -> float:
    """Train a fusion model and return test accuracy."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, texts, labels in train_loader:
            images, texts, labels = (
                images.to(device),
                texts.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            logits = model(images, texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, texts, labels in test_loader:
                images, texts, labels = (
                    images.to(device),
                    texts.to(device),
                    labels.to(device),
                )
                logits = model(images, texts)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc

    print(f"  {name}: best test accuracy = {best_acc:.2%}")
    return best_acc


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 66)
    print("Multimodal Fusion Strategies Comparison")
    print("=" * 66)

    # Synthetic dataset
    seq_len = 16
    image_size = 32
    train_dataset = SyntheticMultimodalDataset(
        num_samples=400,
        num_classes=2,
        image_size=image_size,
        seq_len=seq_len,
    )
    test_dataset = SyntheticMultimodalDataset(
        num_samples=100,
        num_classes=2,
        image_size=image_size,
        seq_len=seq_len,
    )
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Sample inputs for shape / FLOPs demo
    sample_img = torch.randn(1, 3, image_size, image_size)
    sample_txt = torch.randint(0, 256, (1, seq_len))

    # Build models
    models: list[Tuple[str, nn.Module]] = [
        (
            "Early Fusion",
            EarlyFusion(
                image_size=image_size,
                seq_len=seq_len,
                d_model=128,
                nhead=4,
                num_layers=2,
                num_classes=2,
            ),
        ),
        ("Late Fusion", LateFusion(embed_dim=128, num_classes=2)),
        (
            "Cross-Attn Fusion",
            CrossAttentionFusion(
                image_size=image_size,
                seq_len=seq_len,
                embed_dim=128,
                num_heads=4,
                num_layers=2,
                num_classes=2,
            ),
        ),
    ]

    # --- Part 1: Forward pass shapes ---
    print("\n--- Forward pass output shapes ---")
    for name, model in models:
        model.eval()
        with torch.no_grad():
            out = model(sample_img, sample_txt)
        print(
            f"  {name}: input ({sample_img.shape}, {sample_txt.shape}) → output {out.shape}"
        )

    # --- Part 2: Parameter counts and complexity ---
    print("\n--- Model complexity comparison ---")
    header = f"{'Strategy':<20} {'Params':>10} {'FLOPs (est)':>14}"
    print(header)
    print("-" * len(header))
    results: list[Tuple[str, int, int, float]] = []
    for name, model in models:
        params = count_parameters(model)
        model_copy = (
            type(model)(
                **{
                    k: v
                    for k, v in model.__dict__.items()
                    if not k.startswith("_") and k not in ("training",)
                }
            )
            if hasattr(model, "__init__")
            else model
        )
        flops = estimate_flops(model, sample_img, sample_txt)
        print(f"  {name:<20} {params:>10,} {flops:>14,}")

        # --- Part 3: Train and compare accuracy ---
        print(f"\n  Training {name}...")
        acc = train_one_model(
            model,
            train_loader,
            test_loader,
            num_epochs=15,
            lr=2e-3,
            device=device,
            name=name,
        )
        results.append((name, params, flops, acc))

    # --- Final comparison table ---
    print("\n" + "=" * 72)
    print("Final Comparison Table")
    print("=" * 72)
    print(f"{'Strategy':<20} {'Params':>10} {'FLOPs':>10} {'Accuracy':>10}")
    print("-" * 56)
    for name, params, flops, acc in results:
        print(f"{name:<20} {params:>10,} {flops:>10,} {acc:>9.2%}")

    print("\nSummary:")
    print("  - Early Fusion: processes concatenated raw inputs together;")
    print("    learns cross-modal interactions from the start.")
    print("  - Late Fusion: processes modalities independently then merges;")
    print("    modular and easy to extend with new modalities.")
    print("  - Cross-Attention Fusion: lets one modality selectively attend")
    print("    to the other; allows fine-grained inter-modal alignment.")
    print("\nDone!")


if __name__ == "__main__":
    main()
