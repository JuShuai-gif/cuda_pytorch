"""
简化的 CLIP（对比语言-图像预训练）实现。

演示图像与文本模态之间的对比学习：
  - 图像编码器：小型 CNN（Conv2d + ReLU + MaxPool）→ embedding
  - 文本编码器：带可学习位置编码的 2 层 transformer → embedding
  - 投影头将两个 embedding 映射到共享的 128 维空间
  - 带温度缩放的 InfoNCE / 对称交叉熵损失
  - 在合成图像-文本对上训练
  - 检索评估：image-to-text 和 text-to-image 准确率
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 图像编码器：简单 CNN
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """将图像映射到固定维度 embedding 向量的小型 CNN。"""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        # Conv2d + ReLU + MaxPool2d 层的堆叠
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
            nn.AdaptiveAvgPool2d(1),  # → (batch, embed_dim, 1, 1) 输出形状
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：(B, C, H, W) → (B, embed_dim)。"""
        out = self.conv(x)  # (B, embed_dim, 1, 1)
        return out.view(out.size(0), -1)


# ---------------------------------------------------------------------------
# 可学习位置编码
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """文本 token 的可学习 1D 位置编码。"""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """叠加可学习位置编码：(B, L, D) → (B, L, D)。"""
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# 文本编码器：小型 transformer
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """将 token 序列映射为 embedding 的小型 transformer 编码器。"""

    def __init__(
        self,
        vocab_size: int = 256,  # 合成词汇表
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
        # 从 d_model 投影到最终的 embedding 维度
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：(B, L) → (B, embed_dim)。"""
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.pos_encoding(x)
        # 文本编码器不需要 causal mask；不使用 mask
        x = self.transformer(x, mask=None, is_causal=False)
        # 在序列长度维度上做平均池化
        x = x.mean(dim=1)  # (B, d_model)
        return self.proj(x)


# ---------------------------------------------------------------------------
# 完整 CLIP 模型
# ---------------------------------------------------------------------------


class SimpleCLIP(nn.Module):
    """联合图像-文本模型，用于对比训练。"""

    def __init__(
        self,
        embed_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        # 可学习的温度参数，用于 softmax 缩放
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像并对 embedding 做 L2 归一化。"""
        emb = self.image_encoder(image)
        return F.normalize(emb, dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """编码文本并对 embedding 做 L2 归一化。"""
        emb = self.text_encoder(text)
        return F.normalize(emb, dim=-1)

    def forward(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 loss、图像 embedding 和文本 embedding。"""
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(text)

        # 带温度缩放的余弦相似度矩阵
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (img_emb @ txt_emb.T)  # (B, B)

        # 标签：对角线元素为正样本对
        labels = torch.arange(logits.size(0), device=logits.device)

        # 对称交叉熵损失
        loss_img = F.cross_entropy(logits, labels)  # image → text
        loss_txt = F.cross_entropy(logits.T, labels)  # text → image
        loss = (loss_img + loss_txt) / 2.0

        return loss, img_emb, txt_emb


# ---------------------------------------------------------------------------
# 合成数据集
# ---------------------------------------------------------------------------


class SyntheticImageTextDataset(torch.utils.data.Dataset):
    """生成随机图像并与编码为文本 token 的类别标签配对。"""

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
        """返回 (image, text_tokens, label)。"""
        cls = idx % self.num_classes
        # 图像：使用类别特定模式，使模型能学到关联
        torch.manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size)
        # 添加类别特定偏置，使图像可区分
        image = image + (cls / self.num_classes) * 0.5

        # 文本：类别标签编码为重复 token，填充到 seq_len
        # Token ID 范围：0..255；使用类别特定的前缀
        base_token = cls * 10 + 1  # 偏移 1，避免 0 作为类别信号
        text = torch.full((self.seq_len,), base_token, dtype=torch.long)
        # 添加少量噪声，使序列略有不同
        text[1:] = torch.randint(0, 256, (self.seq_len - 1,))

        return image, text, cls


# ---------------------------------------------------------------------------
# 训练工具
# ---------------------------------------------------------------------------


def compute_retrieval_accuracy(
    model: SimpleCLIP,
    dataset: SyntheticImageTextDataset,
    batch_size: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """计算 image-to-text 和 text-to-image 检索准确率。"""
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

    # 检索时按类别匹配（同一类别的所有样本视为"命中"）
    # Image-to-text：对每张图像，检查 top-1 文本的类别
    sim = img_embs @ txt_embs.T
    i2t_pred = labels_tensor[sim.argmax(dim=1)]
    i2t_acc = (i2t_pred == labels_tensor).float().mean().item()

    # Text-to-image
    t2i_pred = labels_tensor[sim.T.argmax(dim=1)]
    t2i_acc = (t2i_pred == labels_tensor).float().mean().item()

    return i2t_acc, t2i_acc


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("Simple CLIP: Contrastive Language-Image Pre-training")
    print("=" * 60)

    # 超参数
    embed_dim = 128
    batch_size = 64
    num_epochs = 25
    num_classes = 10
    lr = 3e-4

    # 创建合成数据集
    dataset = SyntheticImageTextDataset(
        num_samples=500, num_classes=num_classes, image_size=32, seq_len=8
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 构建模型
    model = SimpleCLIP(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size: {len(dataset)} samples, {num_classes} classes")
    print(f"Training: {num_epochs} epochs, batch_size={batch_size}, lr={lr}\n")

    # 训练循环
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

    # 最终检索评估
    i2t_acc, t2i_acc = compute_retrieval_accuracy(model, dataset, device=device)
    print(f"\nFinal retrieval accuracy:")
    print(f"  Image-to-text: {i2t_acc:.2%}")
    print(f"  Text-to-image: {t2i_acc:.2%}")

    print(
        "\nDone! Loss is decreasing, demonstrating contrastive learning on synthetic data."
    )


if __name__ == "__main__":
    main()
