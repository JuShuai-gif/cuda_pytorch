"""
第 16 讲：Vision Transformer 效率分析

从零构建 Vision Transformer (ViT) 和可比较的 ResNet 风格 CNN，
并在不同 patch/图像尺寸下分析它们的计算效率，从而理解
卷积与自注意力之间的效率权衡。

实现的模块：
  - PatchEmbedding：将图像分割成不重叠的 patch，
    并将每个 patch 投影到 d_model 维向量。
  - TransformerBlock：标准 pre-LN 块，包含多头自注意力 (MHA)
    和两层 MLP（GELU 激活）。
  - VisionTransformer：堆叠 PatchEmbedding、可学习的位置嵌入、
    N 个 Transformer 块以及线性分类头。
  - SimpleCNN：ResNet 风格的卷积骨干网络，包含三个阶段
    （每个阶段含残差块），后接全局平均池化 + 全连接层。

本脚本还会：
  - 统计两个模型的参数量并估算 FLOPs (MACs)。
  - 在 {4, 8, 16} 的 patch 尺寸和 {32, 64, 96} 的图像尺寸下比较 ViT 与 CNN。
  - 提取并可视化最后一个 Transformer 块的注意力图。
  - 向 stdout 输出结构化的汇总表格。

依赖：torch、numpy、matplotlib（均为 CPU 版本；不需要 CUDA）。
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
# 工具函数：参数统计
# ===========================================================================


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """返回 *model* 的 (总参数量, 可训练参数量)。

    Args:
        model: PyTorch nn.Module 实例。

    Returns:
        元组 (total_parameters, trainable_parameters)。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===========================================================================
# 工具函数：通过前向钩子统计 FLOPs (MACs)
# ===========================================================================


class _FlopsHook:
    """在单次前向传播中累加 Conv2d 和 Linear 的 MACs。

    约定（与常见 ML 文献中 "FLOPs" ≈ MACs 的用法一致）：
        - Conv2d:  out_c * out_h * out_w * (in_c / groups) * k_h * k_w
        - Linear:  in_features * out_features  （最后一维的每一行）

    BatchNorm、LayerNorm、激活、池化和残差相加操作
    不计入 —— 它们在总计算量中占比 < 1 %。
    """

    def __init__(self) -> None:
        self.total_macs: int = 0
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- 钩子回调函数 -------------------------------------------------------

    def _conv_hook(
        self,
        module: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        """Conv2d 前向钩子：统计卷积 MACs。"""
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
        """Linear 前向钩子：统计全连接 MACs。

        x 形状为 (*prefix, in_features)，
        末尾维度的每一"行"执行一次 [in_f, out_f] 矩阵乘。
        """
        x = inp[0]
        rows = x.numel() // module.in_features  # type: ignore[union-attr]
        self.total_macs += rows * module.in_features * module.out_features  # type: ignore[union-attr]

    # -- 公开 API -----------------------------------------------------------

    def register(self, model: nn.Module) -> None:
        """将前向钩子绑定到 *model* 中的每个 Conv2d / Linear 层。"""
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self._handles.append(m.register_forward_hook(self._conv_hook))
            elif isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))

    def remove(self) -> None:
        """移除所有已注册的钩子。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()


def estimate_macs(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """在 *model* 上运行一次前向传播，返回估算的总 MACs。

    Args:
        model:        PyTorch 模块（将被设为 eval 模式）。
        input_tensor: 单样本张量，带 batch 维度 (1, C, H, W)。

    Returns:
        前向传播的估算乘加操作次数。
    """
    model.eval()
    hook = _FlopsHook()
    hook.register(model)
    with torch.no_grad():
        _ = model(input_tensor)
    hook.remove()
    return hook.total_macs


# ===========================================================================
# ViT 构建模块
# ===========================================================================


class PatchEmbedding(nn.Module):
    """将图像分割成不重叠的 patch 并投影到 *d_model* 维。

    使用步长等于 patch 尺寸的 Conv2d 来实现，
    等价于常见的 "unfold + Linear" 模式，但更高效。
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

        # 使用 strided Conv2d 实现 patch 分割与投影
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将图像 patch 投影为嵌入向量。

        Args:
            x: (B, C, H, W) 输入图像。

        Returns:
            形状为 (B, num_patches, d_model) 的张量。
        """
        x = self.proj(x)  # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer 块：MHA + 两层 MLP，各自带残差连接。

    遵循 ViT 论文（Dosovitskiy 等, 2021）的做法，使用 pre-norm
    和 GELU 激活函数。
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

        # MLP 隐藏层维度 = d_model * mlp_ratio
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
        """执行一个 Transformer 块。

        Args:
            x:               (B, seq_len, d_model) 输入。
            return_attention: 如果为 True，还返回自注意力层的注意力权重。

        Returns:
            如果 return_attention=False： (B, seq_len, d_model)。
            如果 return_attention=True：  ((B, seq_len, d_model), attn_weights)。
        """
        # 自注意力子块（pre-norm）
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed,
            normed,
            normed,
            need_weights=return_attention,
            average_attn_weights=False,  # 保留每个头的权重以便可视化
        )
        x = x + attn_out

        # MLP 子块（pre-norm）
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn_weights
        return x


class VisionTransformer(nn.Module):
    """用于小尺寸图像分类的简易 Vision Transformer (ViT)。

    架构：
        PatchEmbedding -> [CLS] token + pos_embed ->
        N x TransformerBlock -> LayerNorm -> 提取 [CLS] -> 线性分类头
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

        # 可学习的 [CLS] token 和位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model),
        )

        # 堆叠多个 Transformer 块
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
        """对位置嵌入和 CLS token 使用截断正态初始化，遵循 ViT 论文。"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Linear / Conv 层使用 PyTorch 默认初始化即可

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Any:
        """Vision Transformer 前向传播。

        Args:
            x:               (B, C, H, W) 输入图像批次。
            return_attention: 如果为 True，返回 *最后一个* Transformer 块的注意力权重。

        Returns:
            如果 return_attention=False： (B, num_classes) logits。
            如果 return_attention=True：
                ((B, num_classes) logits, (B, n_heads, S, S) 注意力权重)。
        """
        B = x.shape[0]

        # Patch 嵌入
        x = self.patch_embed(x)  # (B, N, d_model)

        # 在序列最前面添加 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d_model)

        # 加上位置编码
        x = x + self.pos_embed

        # 依次经过每个 Transformer 块
        attn_weights = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                # 仅在最后一个块返回注意力权重
                x, attn_weights = block(x, return_attention=True)
            else:
                x = block(x)

        # 最终归一化并使用 [CLS] token 进行分类
        x = self.norm(x)
        logits = self.head(x[:, 0])  # (B, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits

    @property
    def d_model(self) -> int:
        """便捷属性：获取嵌入维度。"""
        return self.patch_embed.proj.out_channels

    @property
    def num_heads(self) -> int:
        """便捷属性：获取注意力头数。"""
        return self.blocks[0].attn.num_heads

    @property
    def depth(self) -> int:
        """便捷属性：获取 Transformer 块的数量。"""
        return len(self.blocks)

    def estimate_attention_macs(
        self,
        batch_size: int = 1,
        seq_len: int | None = None,
    ) -> int:
        """返回 Q@K^T 和 softmax(QK^T)@V 矩阵乘法的 MACs。

        这些操作发生在 nn.MultiheadAttention 内部，
        线性层钩子无法捕获，因此我们单独计算。

        每个块每个头有两个矩阵乘：
            Q @ K^T  :  seq_len * d_head * seq_len   MACs
            attn @ V :  seq_len * seq_len * d_head   MACs

        对所有头求和：每个块 2 * seq_len^2 * d_model  MACs。

        Args:
            batch_size: 批次中的样本数。
            seq_len:    序列长度（patch 数 + 1 个 CLS token）。
                        如果为 None，则根据 img_size / patch_size 计算。

        Returns:
            所有块中注意力矩阵乘的总 MACs。
        """
        if seq_len is None:
            seq_len = self.num_patches + 1
        return self.depth * batch_size * 2 * seq_len * seq_len * self.d_model


# ===========================================================================
# 用于对比的 ResNet 风格 CNN
# ===========================================================================


class ResidualBlock(nn.Module):
    """基础残差块，包含两个 3x3 卷积。"""

    expansion: int = 1  # 与 bottleneck 变体保持兼容

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

        # 当维度发生变化时使用 1x1 快捷连接
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差块前向传播。

        Args:
            x: (B, C, H, W) 输入张量。

        Returns:
            形状相同的输出张量。
        """
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
    """用于小图像分类的 ResNet 风格卷积骨干网络。

    三个阶段，每个阶段都使通道数翻倍并使空间分辨率
    减半（阶段 2 和 3 的第一个残差块使用 stride=2）。
    全局平均池化将特征图压缩为单个特征向量，
    然后输入最终的全连接分类头。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 24,
    ) -> None:
        """初始化 SimpleCNN。

        Args:
            in_channels: 输入图像通道数（RGB 为 3）。
            num_classes: 输出类别数。
            base_width:  第一阶段宽度；后续阶段依次翻倍
                         (base_width -> 2x -> 4x -> 8x)。
        """
        super().__init__()
        w = base_width

        # 输入 stem 层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )

        # 三个残差阶段，空间分辨率逐阶段减半
        self.layer1 = ResidualBlock(w, w * 2, stride=2)
        self.layer2 = ResidualBlock(w * 2, w * 4, stride=2)
        self.layer3 = ResidualBlock(w * 4, w * 8, stride=2)

        # 全局平均池化 + 分类头
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, C, H, W) 输入图像。

        Returns:
            (B, num_classes) logits。
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
# 可视化：注意力图
# ===========================================================================


def visualise_attention(
    model: VisionTransformer,
    input_tensor: torch.Tensor,
    save_path: str = "/tmp/vit_attention_map.png",
) -> None:
    """从最后一个块提取注意力权重并保存为热力图。

    图中包含两个面板：
      (a) 所有头的平均注意力 (S x S)。
      (b) 每个头的注意力图，以网格排列。

    Args:
        model:        VisionTransformer 实例。
        input_tensor: (1, C, H, W) 输入图像张量。
        save_path:    输出 PNG 文件的文件系统路径。
    """
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(input_tensor, return_attention=True)

    # attn_weights 形状: (B, n_heads, seq_len, seq_len)
    if attn_weights is None:
        raise RuntimeError(
            "Attention weights were not returned.  "
            "Ensure return_attention=True was passed to the model."
        )
    attn = attn_weights[0].cpu().numpy()  # (n_heads, S, S)
    n_heads, S, _ = attn.shape
    avg_attn = attn.mean(axis=0)  # (S, S)

    # 构建子图网格
    cols = min(4, n_heads)
    rows = math.ceil((n_heads + 1) / cols)  # +1 给平均面板留位置
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows),
        squeeze=False,
    )

    # 第 1 行第 1 列面板：所有头的平均注意力
    ax = axes[0, 0]
    im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
    ax.set_title("Average over heads")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 隐藏第一行的其余子图
    for c in range(1, cols):
        axes[0, c].set_visible(False)

    # 其余面板：每个头一张
    for h in range(n_heads):
        r = (h + 1) // cols
        c = (h + 1) % cols
        ax = axes[r, c]
        im = ax.imshow(attn[h], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h + 1}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 隐藏多余的空白子图
    total_panels = 1 + n_heads
    for idx in range(total_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    # 设置总标题
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
# 比较表格打印
# ===========================================================================


def print_comparison_table(
    results: List[Dict[str, Any]],
) -> None:
    """格式化并打印参数/FLOPs 汇总表格。

    Args:
        results: 字典列表，每个字典包含以下键：
            img_size, patch_size, vit_params, vit_macs,
            cnn_params, cnn_macs。
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
# 主函数
# ===========================================================================


def main() -> None:
    # ---- 配置参数 ----------------------------------------------------------
    PATCH_SIZES = [4, 8, 16]
    IMG_SIZES = [32, 64, 96]
    NUM_CLASSES = 10
    IN_CHANNELS = 3

    # ViT 超参数（小模型，适合 CPU 分析）
    VIT_D_MODEL = 128
    VIT_DEPTH = 4
    VIT_N_HEADS = 4
    VIT_MLP_RATIO = 4.0

    # CNN 超参数（选择使得参数量与 ViT 大致相同）
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

    # ---- 1. 构建模型并统计参数量 --------------------------------------------
    results: List[Dict[str, Any]] = []

    # CNN 的参数量与 patch_size 无关（但因 BN 运行统计量会随图像尺寸略有变化，
    # 此处不统计这些统计量）。
    # 为清晰起见，每个图像尺寸构建一个 CNN。
    cnn_cache: Dict[int, Tuple[nn.Module, int]] = {}
    for img_size in IMG_SIZES:
        cnn = SimpleCNN(
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
            base_width=CNN_BASE_WIDTH,
        )
        cnn_total, _ = count_parameters(cnn)
        cnn_cache[img_size] = (cnn, cnn_total)

    # ---- 2. 遍历所有配置 ---------------------------------------------------
    for img_size in IMG_SIZES:
        for patch_size in PATCH_SIZES:
            # 如果 patch 尺寸不能整除图像尺寸则跳过
            if img_size % patch_size != 0:
                continue

            print(f"  Profiling: img={img_size}, patch={patch_size} ...")

            # ---- 构建 ViT -------------------------------------------------
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

            # ---- 统计 ViT FLOPs -------------------------------------------
            dummy = torch.randn(1, IN_CHANNELS, img_size, img_size)
            vit_macs = estimate_macs(vit, dummy)

            # 加上线性钩子无法捕获的注意力矩阵乘 MACs（Q@K^T + attn@V）
            seq_len = vit.num_patches + 1  # +1 是 [CLS] token
            attn_matmul_macs = vit.estimate_attention_macs(
                batch_size=1,
                seq_len=seq_len,
            )
            vit_macs += attn_matmul_macs

            # ---- 统计 CNN FLOPs -------------------------------------------
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

    # ---- 3. 打印比较表格 ---------------------------------------------------
    print_comparison_table(results)

    # ---- 4. 注意力图可视化 -------------------------------------------------
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

    # ---- 5. 汇总比较 -------------------------------------------------------
    # 选一个代表性配置做汇总展示
    rep = results[0]  # 第一项（例如 img=32, patch=4）
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
