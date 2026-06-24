#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-backend MACs analysis for VLA policies.

Backend precedence (``backend="auto"``):

    1. fvcore       - trace based, gives total MACs AND a per-module breakdown,
                      good coverage for transformer matmul / attention.
    2. torchprofile - TorchScript trace, best op coverage / accuracy for
                      transformers; total MACs only (per-module filled by hook).
    3. ptflops      - hook based with aten-level support; counts attention
                      matmul (more accurate than thop); total MACs only.
    4. thop         - hook based, light, total MACs only; attention undercounted.
    5. hook         - built-in fallback estimator for Linear / Conv, always
                      available, also fills the per-module breakdown when the
                      chosen total backend cannot provide one.

Unit convention: every backend here returns **MACs** (1 multiply-add).
fvcore internally counts 1 MAC == 1 flop, so ``FlopCountAnalysis.total()`` is
already a MAC count and needs no division by two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from . import module_splitter as ms

logger = logging.getLogger(__name__)


@dataclass
class MacsStats:
    total_macs: float
    per_module: dict[str, float]  # top-level submodule -> MACs
    per_category: dict[str, float]  # vision/language/fusion/action
    category_fraction: dict[str, float]
    backend: str
    unsupported_ops: dict[str, int] = field(default_factory=dict)


def _as_inputs(dummy_input: Any) -> tuple:
    if isinstance(dummy_input, (tuple, list)):
        return tuple(dummy_input)
    return (dummy_input,)


# --------------------------------------------------------------------------- #
# fvcore backend
# --------------------------------------------------------------------------- #
def _fvcore(model: nn.Module, inputs: tuple):
    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model, inputs)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    total = float(flops.total())  # already MACs

    per_module: dict[str, float] = {}
    by_module = flops.by_module()
    for path, val in by_module.items():
        if not path:  # root aggregate
            continue
        if "." in path:  # keep only depth-1, mutually exclusive submodules
            continue
        per_module[path] = float(val)

    unsupported = {str(k): int(v) for k, v in flops.unsupported_ops().items()}
    return total, per_module, unsupported


# --------------------------------------------------------------------------- #
# thop backend
# --------------------------------------------------------------------------- #
def _thop(model: nn.Module, inputs: tuple) -> float:
    from thop import profile

    macs, _ = profile(model, inputs=inputs, verbose=False)
    return float(macs)


# --------------------------------------------------------------------------- #
# torchprofile backend (TorchScript trace, best op coverage for transformers)
# --------------------------------------------------------------------------- #
def _torchprofile(model: nn.Module, inputs: tuple) -> float:
    from torchprofile import profile_macs

    was_training = model.training
    model.eval()
    with torch.no_grad():
        macs = profile_macs(model, inputs)
    if was_training:
        model.train()
    return float(macs)


# --------------------------------------------------------------------------- #
# ptflops backend (hook based; aten-level support counts attention matmul)
# --------------------------------------------------------------------------- #
def _ptflops(model: nn.Module, inputs: tuple) -> float:
    import inspect

    from ptflops import get_model_complexity_info

    # ptflops builds the batch itself; for multi-input models we map our
    # prebuilt tensors onto the forward()'s positional parameter names.
    names = [p for p in inspect.signature(model.forward).parameters if p != "self"]
    batch = dict(zip(names, inputs))
    if not batch:
        raise RuntimeError("ptflops: could not bind inputs to forward() args")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            model,
            (1,),
            input_constructor=lambda _res: batch,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
    if was_training:
        model.train()
    return float(macs)


# --------------------------------------------------------------------------- #
# built-in hook estimator (always available)
# --------------------------------------------------------------------------- #
def _hook_estimate(model: nn.Module, inputs: tuple) -> dict[str, float]:
    """Per-leaf MACs for Linear / Conv via forward hooks. Returns {name: macs}."""
    counts: dict[str, float] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            macs = 0.0
            if isinstance(module, nn.Linear):
                out_elems = out.numel() / out.shape[-1] if out.ndim else 1
                macs = out_elems * module.in_features * module.out_features
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                out_spatial = out.numel() / (out.shape[0] * out.shape[1])
                k = 1
                for ks in module.kernel_size:
                    k *= ks
                macs = (
                    out.shape[0]
                    * out.shape[1]
                    * out_spatial
                    * (module.in_channels // module.groups)
                    * k
                )
            if macs:
                counts[name] = counts.get(name, 0.0) + macs

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(*inputs)
    if was_training:
        model.train()
    for h in handles:
        h.remove()
    return counts


def _rollup_to_top(per_leaf: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, val in per_leaf.items():
        top = name.split(".")[0]
        out[top] = out.get(top, 0.0) + val
    return out


# --------------------------------------------------------------------------- #
# public entry
# --------------------------------------------------------------------------- #
def compute_macs(
    model: nn.Module,
    dummy_input: Any,
    backend: str = "auto",
    config: ms.SplitConfig | None = None,
) -> MacsStats:
    inputs = _as_inputs(dummy_input)
    total: float | None = None
    per_module: dict[str, float] = {}
    unsupported: dict[str, int] = {}
    used = "none"

    order = (
        ["fvcore", "torchprofile", "ptflops", "thop", "hook"]
        if backend == "auto"
        else [backend]
    )

    for be in order:
        try:
            if be == "fvcore":
                total, per_module, unsupported = _fvcore(model, inputs)
                used = "fvcore"
                break
            if be == "torchprofile":
                total = _torchprofile(model, inputs)
                used = "torchprofile"
                break
            if be == "ptflops":
                total = _ptflops(model, inputs)
                used = "ptflops"
                break
            if be == "thop":
                total = _thop(model, inputs)
                used = "thop"
                break
            if be == "torchprofile":
                total = _torchprofile(model, inputs)
                used = "torchprofile"
                break
            if be == "thop":
                total = _thop(model, inputs)
                used = "thop"
                break
            if be == "hook":
                per_module = _rollup_to_top(_hook_estimate(model, inputs))
                total = sum(per_module.values())
                used = "hook"
                break
        except Exception as exc:  # noqa: BLE001 - resilience across backends
            logger.warning("MACs backend '%s' failed: %s", be, exc)
            continue

    if total is None:
        raise RuntimeError("All MACs backends failed; check the dummy input.")

    # If the chosen total backend gave no per-module breakdown, fill it via hooks.
    if not per_module:
        try:
            per_module = _rollup_to_top(_hook_estimate(model, inputs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Per-module hook estimate failed: %s", exc)

    per_category = {c: 0.0 for c in ms.CATEGORIES}
    for c, v in ms.split_by_category(per_module, config).items():
        per_category[c] = v

    return MacsStats(
        total_macs=total,
        per_module=per_module,
        per_category=per_category,
        category_fraction=ms.as_fractions(per_category),
        backend=used,
        unsupported_ops=unsupported,
    )
