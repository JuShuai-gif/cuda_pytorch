"""
Model loading utilities for converting between configurations and checkpoints.

Provides:
- Load HuggingFace checkpoint weights
- Convert between model configurations
- Weight mapping between architectures
- Safe state dict loading with strict key matching

Supports loading from:
- HuggingFace format (safetensors, pytorch .bin files)
- Custom checkpoints
"""

from __future__ import annotations

from typing import Any, Optional
import fnmatch
import json
import os

import torch
import torch.nn as nn


class WeightMapper:
    """Map weights between different model architectures.

    Handles the conversion of state dict keys from one naming convention
    to another, enabling weight transfer between architectures.
    """

    # Standard HuggingFace Llama key patterns to our implementation
    HF_LLAMA_TO_OURS: dict[str, str] = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "model.layers.{i}.input_layernorm.weight": "layers.{i}.input_norm.weight",
        "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attn_norm.weight",
        "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
        "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate_proj.weight",
        "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.mlp.up_proj.weight",
        "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.mlp.down_proj.weight",
    }

    @staticmethod
    def map_hf_to_ours(
        hf_state_dict: dict[str, torch.Tensor],
        num_layers: int,
    ) -> dict[str, torch.Tensor]:
        """Convert HuggingFace Llama state dict to our naming convention.

        Args:
            hf_state_dict: State dict with HuggingFace key names.
            num_layers: Number of transformer layers in the model.

        Returns:
            State dict with our implementation key names.
        """
        mapped: dict[str, torch.Tensor] = {}

        for hf_key, tensor in hf_state_dict.items():
            # Skip non-weight keys
            if not isinstance(tensor, torch.Tensor):
                continue

            our_key: Optional[str] = None

            # Try layer-specific patterns
            for i in range(num_layers):
                pattern = hf_key.replace(f".{i}.", ".{i}.")
                for hf_pat, our_pat in WeightMapper.HF_LLAMA_TO_OURS.items():
                    formatted_hf = hf_pat.format(i=i)
                    formatted_our = our_pat.format(i=i)
                    if fnmatch.fnmatch(hf_key, formatted_hf):
                        our_key = formatted_our
                        break
                if our_key is not None:
                    break

            # Try non-layer patterns
            if our_key is None:
                for hf_pat, our_pat in WeightMapper.HF_LLAMA_TO_OURS.items():
                    if fnmatch.fnmatch(hf_key, hf_pat):
                        our_key = our_pat
                        break

            if our_key is not None:
                mapped[our_key] = tensor

        return mapped

    @staticmethod
    def merge_kv_heads(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Merge separate key/value weight projections if needed.

        Some implementations use separate W_k and W_v weight matrices
        that need to be concatenated for our projection layers.
        """
        # Our implementation uses combined projections, so no merging needed
        # This is a placeholder for architectures that differ
        return state_dict


def load_hf_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    strict: bool = False,
    map_location: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[str]:
    """Load weights from a HuggingFace format checkpoint.

    Supports both safetensors and PyTorch .bin files.

    Args:
        model: The target model to load weights into.
        checkpoint_dir: Directory containing model files.
        strict: If True, raises error on missing/unexpected keys.
        map_location: Device to map tensors to ("cpu", "cuda", etc.).
        dtype: Optional dtype to cast weights to.
    Returns:
        List of loaded weight file paths.
    """
    import glob

    loaded_files: list[str] = []
    full_state_dict: dict[str, torch.Tensor] = {}

    # Try to load safetensors first, fall back to pytorch .bin
    safetensor_pattern = os.path.join(checkpoint_dir, "model*.safetensors")
    bin_pattern = os.path.join(checkpoint_dir, "pytorch_model*.bin")

    weight_files: list[str] = sorted(glob.glob(safetensor_pattern)) or sorted(
        glob.glob(bin_pattern)
    )

    for weight_file in weight_files:
        ext = os.path.splitext(weight_file)[1]
        if ext == ".safetensors":
            try:
                from safetensors.torch import load_file

                shard = load_file(weight_file, device=map_location or "cpu")
            except ImportError:
                raise ImportError(
                    "safetensors is required to load .safetensors files. "
                    "Install with: pip install safetensors"
                )
        else:
            shard = torch.load(
                weight_file,
                map_location=map_location or "cpu",
                weights_only=True,
            )

        full_state_dict.update(shard)
        loaded_files.append(weight_file)

    if not full_state_dict:
        raise FileNotFoundError(f"No weight files found in {checkpoint_dir}")

    # Optionally cast to target dtype
    if dtype is not None:
        full_state_dict = {k: v.to(dtype=dtype) for k, v in full_state_dict.items()}

    # Map HuggingFace keys to our naming convention
    from .configs import Llama3Config

    if hasattr(model, "num_layers"):
        num_layers = model.num_layers
    else:
        num_layers = 32

    mapped_state_dict = WeightMapper.map_hf_to_ours(full_state_dict, num_layers)

    # Load into model
    if mapped_state_dict:
        missing, unexpected = model.load_state_dict(mapped_state_dict, strict=strict)
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    return loaded_files


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    map_location: Optional[str] = None,
) -> None:
    """Load a standard PyTorch checkpoint (.pt/.pth file).

    Args:
        model: Target model.
        checkpoint_path: Path to the checkpoint file.
        strict: If True, raises on missing/unexpected keys.
        map_location: Device mapping.
    """
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=map_location or "cpu",
        weights_only=True,
    )

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)


def convert_config_to_llama3(
    config: Any,
) -> dict[str, Any]:
    """Convert a generic model config to Llama3Model constructor args.

    Args:
        config: Source configuration object (HF config or dict).

    Returns:
        Dictionary of keyword arguments for LlamaModel constructor.
    """
    if isinstance(config, dict):
        cfg = config
    else:
        cfg = vars(config) if not isinstance(config, dict) else config

    result: dict[str, Any] = {}

    # Map common config keys
    key_mapping: dict[str, str] = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_kv_heads",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_seq_len",
        "rope_theta": "rope_theta",
        "rms_norm_eps": "norm_eps",
        "attention_dropout": "attn_dropout",
        "hidden_dropout": "resid_dropout",
        "tie_word_embeddings": "tie_word_embeddings",
    }

    for src_key, dst_key in key_mapping.items():
        if src_key in cfg:
            result[dst_key] = cfg[src_key]

    return result


def save_checkpoint(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    **extra: Any,
) -> None:
    """Save a model checkpoint with optional training state.

    Args:
        model: The model to save.
        save_path: Output file path.
        optimizer: Optional optimizer state to save.
        epoch: Current epoch number.
        global_step: Current global step.
        **extra: Additional items to save in the checkpoint.
    """
    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if global_step is not None:
        checkpoint["global_step"] = global_step
    checkpoint.update(extra)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(checkpoint, save_path)


# Quick test
if __name__ == "__main__":
    import tempfile

    # Test with our own model (save and load roundtrip)
    from .configs import llama3_8b_config

    # Small test model
    config = {
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_layers": 2,
        "num_heads": 8,
        "num_kv_heads": 2,
        "intermediate_size": 512,
        "max_seq_len": 128,
    }

    from .llama import LlamaModel

    model = LlamaModel(**config)  # type: ignore[arg-type]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        ckpt_path = os.path.join(tmpdir, "model.pt")
        save_checkpoint(model, ckpt_path, epoch=0, global_step=0)

        # Load into new model
        model2 = LlamaModel(**config)  # type: ignore[arg-type]
        load_checkpoint(model2, ckpt_path)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch: {n1}"
        print(f"Save/load roundtrip: OK")

    # Test weight mapper
    hf_state = {
        "model.embed_tokens.weight": torch.randn(1000, 256),
        "model.layers.0.input_layernorm.weight": torch.randn(256),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
    }
    mapped = WeightMapper.map_hf_to_ours(hf_state, num_layers=2)
    assert len(mapped) == 3, f"Expected 3 keys, got {len(mapped)}"
    print(f"WeightMapper: OK, mapped {len(mapped)} keys")

    # Test config conversion
    hf_config = {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
    }
    converted = convert_config_to_llama3(hf_config)
    assert converted["hidden_size"] == 4096
    assert converted["num_layers"] == 32
    assert converted["num_kv_heads"] == 8
    print(f"Config conversion: OK")

    print("\nAll model_loader tests passed!")
