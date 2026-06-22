"""
从零实现的 Vision Transformer (ViT)。

实现完整的 ViT 流水线：
  - Patch embedding：切分图像 → 线性投影
  - 可学习 1D 位置编码 + CLS token
  - 多头自注意力（从零实现）
  - 带 GELU 激活的 MLP
  - 堆叠的 transformer 编码器块（N=4）
  - 基于 CLS token 输出的分类头
  - 在合成图像数据上训练并跟踪准确率
  - 注意力图可视化并保存为 PNG
"""

from __future__ import annotations

from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # 无头环境下的非交互式后端
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 多头自注意力（从零实现）
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """多头缩放点积自注意力。

    不依赖 nn.MultiheadAttention——完全由线性层、reshape 和 matmul 操作构建。
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
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
        """自注意力前向传播。

        Args:
            x: (B, N, embed_dim) 输入序列。
            return_attention: 若为 True，同时返回注意力权重。

        Returns:
            输出张量 (B, N, embed_dim)，以及可选的注意力权重 (B, H, N, N)。
        """
        B, N, D = x.shape

        # 线性投影到 Q、K、V 并拆分到多个头
        qkv = self.qkv(x)  # (B, N, 3 * D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和 values
        attn_output = attn_weights @ v  # (B, H, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(attn_output)

        if return_attention:
            return out, attn_weights
        return out


# ---------------------------------------------------------------------------
# 带 GELU 的 MLP 块
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """两层 MLP，使用 GELU 激活和 dropout。"""

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
# Transformer 编码器块
# ---------------------------------------------------------------------------


class TransformerEncoderBlock(nn.Module):
    """单个 transformer 编码器块：MHA → add&norm → MLP → add&norm。"""

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
    """将图像切分为不重叠的 patch 并投影为 embedding。"""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "image_size 必须能被 patch_size 整除"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 使用 Conv2d 作为 patch 投影（等价于每个 patch 做线性投影）
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, num_patches, embed_dim)。"""
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ---------------------------------------------------------------------------
# 完整 Vision Transformer
# ---------------------------------------------------------------------------


class VisionTransformer(nn.Module):
    """从零实现的 Vision Transformer（Dosovitskiy et al., 2021）。"""

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

        # CLS token（可学习）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # 可学习 1D 位置编码（每个 patch + CLS 各一个）
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        self.dropout = nn.Dropout(dropout)

        # transformer 编码器块的堆叠
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头：CLS token → logits
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
        """前向传播：(B, C, H, W) → (B, num_classes)。"""
        B = x.size(0)

        # Patch embedding + CLS token + 位置编码
        x = self.patch_embed(x)  # (B, N_patches, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N_patches+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer 块
        attentions: list[torch.Tensor] = []
        for blk in self.blocks:
            if return_attentions:
                x, attn = blk(x, return_attention=True)
                attentions.append(attn)
            else:
                x = blk(x)

        x = self.norm(x)

        # CLS token → 分类
        x = self.head(x[:, 0])  # (B, num_classes)

        if return_attentions:
            return x, attentions
        return x


# ---------------------------------------------------------------------------
# 合成图像数据集
# ---------------------------------------------------------------------------


class SyntheticImageDataset(torch.utils.data.Dataset):
    """带有类别特定模式的随机图像，用于分类任务。"""

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
        # 每个样本的确定性伪随机数，带类别特定偏置
        torch.manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size)
        image = image + (cls / self.num_classes) * 1.0
        return image, cls


# ---------------------------------------------------------------------------
# 注意力图可视化
# ---------------------------------------------------------------------------


def visualize_attention(
    model: VisionTransformer,
    image: torch.Tensor,
    save_path: str = "vit_attention_maps.png",
) -> None:
    """可视化最后一个 transformer 块中所有头的注意力图。

    将热力图网格保存为 PNG 文件。
    """
    model.eval()
    with torch.no_grad():
        _, attentions = model(image.unsqueeze(0), return_attentions=True)

    # 取最后一个块的注意力
    attn_last = attentions[-1]  # (1, num_heads, N+1, N+1)
    num_heads = attn_last.size(1)

    # 使用 CLS token 对 patch 的注意力（排除 CLS 自注意力）
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
# 演示
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("Vision Transformer (ViT) from Scratch")
    print("=" * 60)

    # 超参数
    image_size = 32
    patch_size = 4
    embed_dim = 128
    depth = 4
    num_heads = 4
    num_classes = 10
    batch_size = 64
    num_epochs = 20
    lr = 1e-3

    # 构建模型
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

    # 合成数据集
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

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining: {num_epochs} epochs, batch_size={batch_size}, lr={lr}\n")

    # 训练循环
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

        # 在测试集上评估
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

    # 可视化注意力图
    sample_img, _label = test_dataset[0]
    visualize_attention(model, sample_img, save_path="vit_attention_maps.png")

    print("Done!")


if __name__ == "__main__":
    main()
