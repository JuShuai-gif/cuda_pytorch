#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic SmolVLA / pi0-style Vision-Language-Action policy (~705M params).

Built so the bundled profiler runs end-to-end without a real checkpoint:

  - explicit attention via torch.matmul (so fvcore counts attention MACs),
  - top-level submodules named vision_encoder / language_encoder /
    fusion_transformer / action_head so module_splitter classifies them,
  - the action head emits a full action chunk, so its cost scales with the
    chunk horizon (used by the chunk-rollout cost model).

The exact parameter count is not meant to match any real model bit-for-bit;
it is sized to land near 700M for a realistic profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VLAConfig:
    # vision
    img_size: int = 224
    patch: int = 16
    vis_dim: int = 768
    vis_depth: int = 32
    # language
    vocab: int = 8000
    txt_len: int = 32
    txt_dim: int = 768
    txt_depth: int = 16
    # fusion
    fus_dim: int = 1024
    fus_depth: int = 22
    # action
    act_depth: int = 5
    chunk_steps: int = 50
    action_dim: int = 7
    heads: int = 8


class MHA(nn.Module):
    """Explicit multi-head attention (matmul based, fvcore-countable)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.scale = self.dh**-0.5

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        b, n, _ = x.shape
        m = ctx.shape[1]
        q = self.q(x).view(b, n, self.h, self.dh).transpose(1, 2)
        k = self.k(ctx).view(b, m, self.h, self.dh).transpose(1, 2)
        v = self.v(ctx).view(b, m, self.h, self.dh).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, cross: bool = False):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = MHA(dim, heads)
        self.cross = cross
        if cross:
            self.nc = nn.LayerNorm(dim)
            self.cattn = MHA(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x, ctx=None):
        x = x + self.attn(self.n1(x))
        if self.cross and ctx is not None:
            x = x + self.cattn(self.nc(x), ctx)
        x = x + self.mlp(self.n2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, c: VLAConfig):
        super().__init__()
        n = (c.img_size // c.patch) ** 2
        self.patch_embed = nn.Conv2d(3, c.vis_dim, c.patch, c.patch)
        self.pos = nn.Parameter(torch.zeros(1, n, c.vis_dim))
        self.blocks = nn.ModuleList(
            [Block(c.vis_dim, c.heads) for _ in range(c.vis_depth)]
        )
        self.norm = nn.LayerNorm(c.vis_dim)

    def forward(self, img):
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        x = x + self.pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class LanguageEncoder(nn.Module):
    def __init__(self, c: VLAConfig):
        super().__init__()
        self.embed = nn.Embedding(c.vocab, c.txt_dim)
        self.pos = nn.Parameter(torch.zeros(1, c.txt_len, c.txt_dim))
        self.blocks = nn.ModuleList(
            [Block(c.txt_dim, c.heads) for _ in range(c.txt_depth)]
        )
        self.norm = nn.LayerNorm(c.txt_dim)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class FusionTransformer(nn.Module):
    def __init__(self, c: VLAConfig):
        super().__init__()
        self.vis_proj = nn.Linear(c.vis_dim, c.fus_dim)
        self.txt_proj = nn.Linear(c.txt_dim, c.fus_dim)
        self.blocks = nn.ModuleList(
            [Block(c.fus_dim, c.heads) for _ in range(c.fus_depth)]
        )
        self.norm = nn.LayerNorm(c.fus_dim)

    def forward(self, vis, txt):
        x = torch.cat([self.vis_proj(vis), self.txt_proj(txt)], dim=1)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class ActionHead(nn.Module):
    def __init__(self, c: VLAConfig):
        super().__init__()
        self.chunk_steps = c.chunk_steps
        self.queries = nn.Parameter(torch.zeros(1, c.chunk_steps, c.fus_dim))
        self.blocks = nn.ModuleList(
            [Block(c.fus_dim, c.heads, cross=True) for _ in range(c.act_depth)]
        )
        self.head = nn.Sequential(
            nn.Linear(c.fus_dim, 2 * c.fus_dim),
            nn.GELU(),
            nn.Linear(2 * c.fus_dim, c.action_dim),
        )

    def forward(self, memory):
        b = memory.shape[0]
        x = self.queries.expand(b, -1, -1)
        for blk in self.blocks:
            x = blk(x, memory)
        return self.head(x)  # (B, chunk_steps, action_dim)


class SyntheticVLA(nn.Module):
    def __init__(self, c: VLAConfig | None = None):
        super().__init__()
        self.c = c or VLAConfig()
        self.vision_encoder = VisionEncoder(self.c)
        self.language_encoder = LanguageEncoder(self.c)
        self.fusion_transformer = FusionTransformer(self.c)
        self.action_head = ActionHead(self.c)

    def forward(self, image, tokens):
        vis = self.vision_encoder(image)
        txt = self.language_encoder(tokens)
        mem = self.fusion_transformer(vis, txt)
        return self.action_head(mem)

    def dummy_inputs(self, batch: int = 1, device: str = "cpu"):
        img = torch.randn(batch, 3, self.c.img_size, self.c.img_size, device=device)
        tok = torch.randint(0, self.c.vocab, (batch, self.c.txt_len), device=device)
        return (img, tok)


def build_synthetic_vla(preset: str = "705M") -> SyntheticVLA:
    """Factory. Currently a single ~700M preset; extend with more sizes."""
    if preset in ("705M", "700M", "default"):
        return SyntheticVLA(VLAConfig())
    raise ValueError(f"Unknown preset: {preset}")
