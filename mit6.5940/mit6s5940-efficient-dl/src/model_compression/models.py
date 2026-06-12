"""
Model definitions for compression benchmarks.

Provides:
- SmallCNN: a compact CNN for CIFAR-10-style tasks
- TransformerAttentionBlock: a single Transformer attention/FFN block
- VLAActionHead: MLP action head for VLA/robot action chunk prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """A compact CNN suitable for CIFAR-10 (3x32x32 -> 10 classes).

    Architecture: Conv -> Conv -> Conv -> FC -> FC
    Designed to be small enough for fast CPU benchmarks while still
    exhibiting meaningful compression characteristics.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 32,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.conv2 = nn.Conv2d(base_width, base_width * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_width * 2)
        self.conv3 = nn.Conv2d(base_width * 2, base_width * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_width * 4)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(base_width * 4 * 4 * 4, base_width * 8)
        self.fc2 = nn.Linear(base_width * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerAttentionBlock(nn.Module):
    """A single Transformer attention block with pre-LayerNorm.

    Consists of multi-head self-attention followed by a 2-layer FFN.
    This is representative of the layers found in BERT/ViT/LLM decoders.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        ffn_size: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.ln1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn1 = nn.Linear(hidden_size, ffn_size)
        self.ffn2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        # Self-attention
        residual = x
        x_norm = self.ln1(x)
        q = self.q_proj(x_norm).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        x = residual + attn_out

        # FFN
        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn2(F.gelu(self.ffn1(x_norm)))
        ffn_out = self.dropout(ffn_out)
        x = residual + ffn_out

        return x


class VLAActionHead(nn.Module):
    """VLA-style action head for robot action chunk prediction.

    Takes visual features + robot state, outputs action chunks.
    Typical in ACT (Action Chunking Transformer) and similar VLA architectures.
    The action head is typically an MLP that maps combined features
    (vision embedding + proprioceptive state) to action chunks.
    """

    def __init__(
        self,
        vision_feature_dim: int = 256,
        state_dim: int = 7,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_action_chunks: int = 100,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        input_dim = vision_feature_dim + state_dim
        self.num_action_chunks = num_action_chunks
        self.action_dim = action_dim
        output_dim = num_action_chunks * action_dim

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([vision_features, robot_state], dim=-1)
        out = self.net(x)
        return out.view(-1, self.num_action_chunks, self.action_dim)


class SimpleMLP(nn.Module):
    """A simple multi-layer perceptron for baseline comparisons.

    Architecture: input -> hidden layers -> output
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | None = None,
        output_dim: int = 10,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Return total number of all parameters (including non-trainable)."""
    return sum(p.numel() for p in model.parameters())
