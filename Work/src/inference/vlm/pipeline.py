"""VLM inference pipeline stages.

Decomposes a vision-language model into the stages a serving system actually
executes, so each can be measured independently:

    Image bytes -> decode -> preprocess (resize/normalize) -> H2D
                -> vision encoder -> connector -> language model -> output

A real VLM (LLaVA/Qwen-VL) uses a ViT vision encoder, a projection/connector
that maps vision tokens into the LLM embedding space, and a transformer LLM.
Here we use a small from-scratch ViT + a linear connector + a small transformer
decoder, which is enough to expose the latency *structure* (which stage
dominates, and where the CPU/GPU boundary sits).
"""

from __future__ import annotations

import io
import math

import torch
from torch import nn
from torchvision import transforms


class SimpleViT(nn.Module):
    """Patch-embed + N transformer-encoder layers, no pretraining."""

    def __init__(self, image_size=224, patch_size=16, hidden=384, layers=6, heads=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_patches, hidden))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.hidden = hidden
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, P, hidden)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos
        return self.encoder(tokens)  # (B, 1+P, hidden)


class VLM(nn.Module):
    def __init__(self, llm_hidden=512, llm_layers=4):
        super().__init__()
        self.vision = SimpleViT()
        self.connector = nn.Linear(self.vision.hidden, llm_hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=llm_hidden, nhead=8, dim_feedforward=llm_hidden * 4,
            batch_first=True, activation="gelu")
        self.llm = nn.TransformerEncoder(enc_layer, num_layers=llm_layers)

    def vision_encode(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            v = self.vision(img)
            v = self.connector(v)
        return v

    def llm_forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.llm(vision_tokens)


def make_image_bytes(size: int = 224, seed: int = 0) -> bytes:
    """Return a synthetic JPEG image as bytes (the real VLM input path)."""
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def decode_image(data: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess(img) -> torch.Tensor:
    return _preprocess(img)  # (3, 224, 224) float32 on CPU
