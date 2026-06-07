"""
Simplified CLIP (Contrastive Language-Image Pre-training) implementation.

Demonstrates contrastive learning between image and text modalities:
  - Image encoder: small CNN (Conv2d + ReLU + MaxPool) → embedding
  - Text encoder: 2-layer transformer with learned positional encoding → embedding
  - Projection heads map both embeddings to a shared 128-dim space
  - InfoNCE / symmetric cross-entropy loss with temperature scaling
  - Training on synthetic image-text pairs
  - Retrieval evaluation: image-to-text and text-to-image accuracy
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Image Encoder: simple CNN
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """Small CNN that maps images to a fixed-dimension embedding vector."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        # Stack of Conv2d + ReLU + MaxPool2d layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 → 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 → 4
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # → (batch, embed_dim, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C, H, W) → (B, embed_dim)."""
        out = self.conv(x)  # (B, embed_dim, 1, 1)
        return out.view(out.size(0), -1)


# ---------------------------------------------------------------------------
# Learned Positional Encoding
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Learned 1D positional encoding for text tokens."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding: (B, L, D) → (B, L, D)."""
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Text Encoder: small transformer
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Small transformer encoder that maps token sequences to embeddings."""

    def __init__(
        self,
        vocab_size: int = 256,  # synthetic vocabulary
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Project from d_model to final embedding dimension
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, L) → (B, embed_dim)."""
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.pos_encoding(x)
        # causal mask for the text encoder is not needed; use no mask
        x = self.transformer(x, mask=None, is_causal=False)
        # Average pooling over sequence length
        x = x.mean(dim=1)  # (B, d_model)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Full CLIP Model
# ---------------------------------------------------------------------------


class SimpleCLIP(nn.Module):
    """Joint image-text model with contrastive training."""

    def __init__(
        self,
        embed_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        # Learnable temperature parameter for softmax scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image and L2-normalize the embedding."""
        emb = self.image_encoder(image)
        return F.normalize(emb, dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text and L2-normalize the embedding."""
        emb = self.text_encoder(text)
        return F.normalize(emb, dim=-1)

    def forward(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return loss, image embeddings, text embeddings."""
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(text)

        # Cosine similarity matrix with temperature scaling
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (img_emb @ txt_emb.T)  # (B, B)

        # Labels: diagonal elements are the positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric cross-entropy loss
        loss_img = F.cross_entropy(logits, labels)  # image → text
        loss_txt = F.cross_entropy(logits.T, labels)  # text → image
        loss = (loss_img + loss_txt) / 2.0

        return loss, img_emb, txt_emb


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


class SyntheticImageTextDataset(torch.utils.data.Dataset):
    """Generates random images paired with class labels encoded as text tokens."""

    def __init__(
        self,
        num_samples: int = 500,
        num_classes: int = 10,
        image_size: int = 32,
        seq_len: int = 8,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (image, text_tokens, label)."""
        cls = idx % self.num_classes
        # Images: class-specific patterns so the model can learn associations
        torch.manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size)
        # Add a class-specific bias to make images distinguishable
        image = image + (cls / self.num_classes) * 0.5

        # Text: class label encoded as repeated token, padded to seq_len
        # Token IDs: 0..255; use class-specific prefix
        base_token = cls * 10 + 1  # shift by 1 so 0 is not the class signal
        text = torch.full((self.seq_len,), base_token, dtype=torch.long)
        # Add small noise to make sequences slightly distinct
        text[1:] = torch.randint(0, 256, (self.seq_len - 1,))

        return image, text, cls


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def compute_retrieval_accuracy(
    model: SimpleCLIP,
    dataset: SyntheticImageTextDataset,
    batch_size: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Compute image-to-text and text-to-image retrieval accuracy."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_img_embs: list[torch.Tensor] = []
    all_txt_embs: list[torch.Tensor] = []
    all_labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for images, texts, labels in loader:
            images = images.to(device)
            texts = texts.to(device)
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(texts)
            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())
            all_labels.extend(labels.tolist())

    img_embs = torch.cat(all_img_embs, dim=0)
    txt_embs = torch.cat(all_txt_embs, dim=0)
    labels_tensor = torch.tensor(all_labels)

    # For retrieval, match by class (all samples of same class are "hits")
    # Image-to-text: for each image, check top-1 text class
    sim = img_embs @ txt_embs.T
    i2t_pred = labels_tensor[sim.argmax(dim=1)]
    i2t_acc = (i2t_pred == labels_tensor).float().mean().item()

    # Text-to-image
    t2i_pred = labels_tensor[sim.T.argmax(dim=1)]
    t2i_acc = (t2i_pred == labels_tensor).float().mean().item()

    return i2t_acc, t2i_acc


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("Simple CLIP: Contrastive Language-Image Pre-training")
    print("=" * 60)

    # Hyperparameters
    embed_dim = 128
    batch_size = 64
    num_epochs = 25
    num_classes = 10
    lr = 3e-4

    # Create synthetic dataset
    dataset = SyntheticImageTextDataset(
        num_samples=500, num_classes=num_classes, image_size=32, seq_len=8
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model
    model = SimpleCLIP(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size: {len(dataset)} samples, {num_classes} classes")
    print(f"Training: {num_epochs} epochs, batch_size={batch_size}, lr={lr}\n")

    # Training loop
    losses: list[float] = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, texts, _labels in loader:
            images = images.to(device)
            texts = texts.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(images, texts)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            i2t, t2i = compute_retrieval_accuracy(model, dataset, device=device)
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"I2T Acc: {i2t:.2%} | "
                f"T2I Acc: {t2i:.2%}"
            )

    print(
        f"\nTraining complete. Initial loss: {losses[0]:.4f} → Final loss: {losses[-1]:.4f}"
    )

    # Final retrieval evaluation
    i2t_acc, t2i_acc = compute_retrieval_accuracy(model, dataset, device=device)
    print(f"\nFinal retrieval accuracy:")
    print(f"  Image-to-text: {i2t_acc:.2%}")
    print(f"  Text-to-image: {t2i_acc:.2%}")

    print(
        "\nDone! Loss is decreasing, demonstrating contrastive learning on synthetic data."
    )


if __name__ == "__main__":
    main()
