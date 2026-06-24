#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLA module splitter.

Classifies every parameter / sub-module of a Vision-Language-Action policy
into one of four canonical functional groups:

    vision    - image backbone / patch embedding / visual encoder
    language  - text / instruction encoder, tokenizer-side projections
    fusion    - cross-modal transformer that mixes vision + language tokens
    action    - action expert / action head producing the action chunk

The classification is keyword-based on the *fully qualified* parameter name
(``model.named_parameters()``). Keyword tables are configurable so the same
tool works on SmolVLA, pi0.5, OpenVLA, RT-2 style checkpoints whose internal
naming differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VISION_KEYS = (
    "vision",
    "image",
    "img",
    "visual",
    "patch_embed",
    "backbone",
    "vit",
    "siglip",
    "dino",
    "resnet",
    "conv_stem",
    "pixel",
)
LANGUAGE_KEYS = (
    "language",
    "text",
    "lang",
    "token_embed",
    "word_embed",
    "instruction",
    "prompt",
    "llm",
    "gemma",
    "qwen",
    "t5",
)
ACTION_KEYS = (
    "action",
    "act_head",
    "policy_head",
    "expert",
    "flow",
    "diffusion_head",
    "motor",
    "joint",
    "ee_",
    "gripper",
)
FUSION_KEYS = (
    "fusion",
    "cross",
    "mixer",
    "connector",
    "projector",
    "adapter",
    "multimodal",
    "mm_",
    "vlm",
    "transformer",
)

CATEGORIES = ("vision", "language", "fusion", "action")


@dataclass
class SplitConfig:
    """Keyword tables used to classify modules.

    Order of precedence is: action -> vision -> language -> fusion.
    Action is checked first because action heads are frequently nested inside
    a fusion transformer namespace and would otherwise be misattributed.
    """

    vision_keys: tuple[str, ...] = VISION_KEYS
    language_keys: tuple[str, ...] = LANGUAGE_KEYS
    action_keys: tuple[str, ...] = ACTION_KEYS
    fusion_keys: tuple[str, ...] = FUSION_KEYS
    default: str = "fusion"
    overrides: dict[str, str] = field(default_factory=dict)


def classify(name: str, config: SplitConfig | None = None) -> str:
    """Map a parameter / module name to one of CATEGORIES."""
    config = config or SplitConfig()
    low = name.lower()

    for prefix, category in config.overrides.items():
        if low.startswith(prefix.lower()):
            return category

    if any(k in low for k in config.action_keys):
        return "action"
    if any(k in low for k in config.vision_keys):
        return "vision"
    if any(k in low for k in config.language_keys):
        return "language"
    if any(k in low for k in config.fusion_keys):
        return "fusion"
    return config.default


def split_by_category(
    per_name: dict[str, float],
    config: SplitConfig | None = None,
) -> dict[str, float]:
    """Aggregate a {qualified_name: value} mapping into the 4 VLA groups."""
    out = {c: 0.0 for c in CATEGORIES}
    for name, value in per_name.items():
        out[classify(name, config)] += value
    return out


def as_fractions(grouped: dict[str, float]) -> dict[str, float]:
    """Convert absolute per-group values to fractions of the total."""
    total = sum(grouped.values()) or 1.0
    return {c: grouped.get(c, 0.0) / total for c in CATEGORIES}
