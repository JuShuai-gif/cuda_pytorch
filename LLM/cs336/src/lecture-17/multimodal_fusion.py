"""
多模态融合策略对比。

实现并比较三种融合策略：
  a) Early Fusion（早期融合）：拼接原始输入 → 共享 transformer
  b) Late Fusion（晚期融合）：独立编码器 → 拼接 embedding → 分类器
  c) Cross-Attention Fusion（交叉注意力融合）：自注意力 + 模态间交叉注意力

使用合成数据（随机图像 + 随机 token 序列）评估每种策略
在二分类任务（匹配 vs 不匹配对）上的表现。
比较参数数量、FLOPs 和准确率。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 通用构建块
# ---------------------------------------------------------------------------


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力：一个序列关注另一个序列。

    Query 来自 x，Key/Value 来自 y（上下文）。
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
        """交叉注意力：x 关注 context。

        Args:
            x: (B, Nx, D) query 序列。
            context: (B, Ny, D) key/value 序列。

        Returns:
            (B, Nx, D) 输出。
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
    """可学习 1D 位置编码。"""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# 共享编码器（Late Fusion 和 Cross-Attention Fusion 使用）
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """轻量 CNN → 扁平化 embedding。"""

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
    """文本 token 的小型 transformer 编码器。"""

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
        x = x.mean(dim=1)  # (B, d_model) 在序列长度上做平均池化
        return self.proj(x)


# ===================================================================
# 策略 A：Early Fusion（早期融合）
# ===================================================================


class EarlyFusion(nn.Module):
    """拼接原始像素 + token embedding → 共享 transformer → 分类器。"""

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

        # 将图像展平为 patch 序列（4x4 patches，共 16 个 patch）
        self.patch_proj = nn.Conv2d(
            3, d_model, kernel_size=8, stride=8
        )  # 32→4x4 patches
        self.num_img_patches = (image_size // 8) ** 2  # 16

        # Text embedding 投影到 d_model
        self.text_emb = nn.Embedding(vocab_size, d_model)

        # 组合序列的位置编码
        total_len = self.num_img_patches + seq_len
        self.pos_enc = PositionalEncoding(d_model, max_len=total_len)

        # 共享 transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            image: (B, 3, 32, 32)
            text: (B, seq_len) token 索引

        Returns:
            (B, num_classes) logits
        """
        B = image.size(0)

        # 将图像切分为 patch
        img_tokens = self.patch_proj(image)  # (B, D, 4, 4)
        img_tokens = img_tokens.flatten(2).transpose(1, 2)  # (B, 16, D)

        # 嵌入文本 token
        txt_tokens = self.text_emb(text)  # (B, seq_len, D)

        # 拼接
        combined = torch.cat([img_tokens, txt_tokens], dim=1)  # (B, 16+seq_len, D)
        combined = self.pos_enc(combined)

        # 共享 transformer
        out = self.transformer(combined, mask=None, is_causal=False)

        # 池化（对所有 token 求均值）并分类
        pooled = out.mean(dim=1)  # (B, D)
        return self.classifier(pooled)


# ===================================================================
# 策略 B：Late Fusion（晚期融合）
# ===================================================================


class LateFusion(nn.Module):
    """独立的图像/文本编码器 → 拼接 embedding → 分类器。"""

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        self.txt_encoder = TextEncoder(embed_dim=embed_dim)

        # 融合 + 分类
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
        fused = torch.cat([img_emb, txt_emb], dim=1)  # (B, 2*embed_dim) 拼接
        return self.classifier(fused)


# ===================================================================
# 策略 C：Cross-Attention Fusion（交叉注意力融合）
# ===================================================================


class CrossAttentionFusionBlock(nn.Module):
    """一个融合块：文本自注意力 → 文本→图像交叉注意力 → MLP。"""

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
        """文本先做自注意力，然后对图像特征做交叉注意力。"""
        # 文本自注意力
        x = text_seq
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # 交叉注意力：文本关注图像
        x = x + self.cross_attn(self.norm2(x), image_features)

        # MLP 前馈网络
        x = x + self.mlp(self.norm3(x))
        return x


class CrossAttentionFusion(nn.Module):
    """通过交叉注意力进行融合：文本 transformer 对图像做交叉注意力。

    图像被编码为特征向量序列（例如 patch 特征）。
    文本通过包含对图像特征序列做交叉注意力的 transformer 层进行处理。
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

        # 图像编码器，生成特征序列
        self.img_proj = nn.Conv2d(
            3, embed_dim, kernel_size=8, stride=8
        )  # 32→4x4 patches
        self.num_img_features = (image_size // 8) ** 2  # 16
        self.img_pos_enc = PositionalEncoding(embed_dim, max_len=self.num_img_features)

        # 文本 embedding
        self.text_emb = nn.Embedding(vocab_size, embed_dim)
        self.text_pos_enc = PositionalEncoding(embed_dim, max_len=seq_len)

        # 交叉注意力融合块
        self.fusion_blocks = nn.ModuleList(
            [CrossAttentionFusionBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        B = image.size(0)

        # 将图像编码为序列
        img_feats = self.img_proj(image)  # (B, D, 4, 4)
        img_feats = img_feats.flatten(2).transpose(1, 2)  # (B, 16, D)
        img_feats = self.img_pos_enc(img_feats)

        # 嵌入文本
        text_seq = self.text_emb(text) * math.sqrt(self.embed_dim)
        text_seq = self.text_pos_enc(text_seq)

        # 交叉注意力融合
        for block in self.fusion_blocks:
            text_seq = block(text_seq, img_feats)

        # 池化文本序列并分类
        pooled = text_seq.mean(dim=1)  # (B, D)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# 合成多模态数据集
# ---------------------------------------------------------------------------


class SyntheticMultimodalDataset(torch.utils.data.Dataset):
    """（图像, 文本, 标签）三元组，label=1 表示匹配，0 表示不匹配。"""

    def __init__(
        self,
        num_samples: int = 400,
        num_classes: int = 2,  # 二分类：匹配或不匹配
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
        # 一半是匹配对，一半不是
        label = idx % self.num_classes  # 0 或 1
        torch.manual_seed(idx)

        # 带类别特定偏置的图像
        image = torch.randn(3, self.image_size, self.image_size)
        if label == 1:
            # 匹配：两个模态共享相同的类别信号
            image = image + 0.5
            text = torch.randint(128, 256, (self.seq_len,))
        else:
            # 不匹配：使用不同的信号
            image = image - 0.5
            text = torch.randint(0, 128, (self.seq_len,))

        return image, text.long(), label


# ---------------------------------------------------------------------------
# 训练与评估工具
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """返回可训练参数的总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
) -> int:
    """通过 PyTorch profiler（如果可用）或简单启发式方法估算 FLOPs。"""
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
        # 回退方案：基于参数数量的粗略估算
        return count_parameters(model) * 100  # 启发式估算


def train_one_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    name: str = "Model",
) -> float:
    """训练一个融合模型并返回测试准确率。"""
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

        # 评估
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
# 演示
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 66)
    print("Multimodal Fusion Strategies Comparison")
    print("=" * 66)

    # 合成数据集
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

    # 用于形状/FLOPs 演示的示例输入
    sample_img = torch.randn(1, 3, image_size, image_size)
    sample_txt = torch.randint(0, 256, (1, seq_len))

    # 构建模型
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

    # --- 第一部分：前向传播输出形状 ---
    print("\n--- Forward pass output shapes ---")
    for name, model in models:
        model.eval()
        with torch.no_grad():
            out = model(sample_img, sample_txt)
        print(
            f"  {name}: input ({sample_img.shape}, {sample_txt.shape}) → output {out.shape}"
        )

    # --- 第二部分：参数数量和复杂度 ---
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

        # --- 第三部分：训练并比较准确率 ---
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

    # --- 最终对比表格 ---
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
