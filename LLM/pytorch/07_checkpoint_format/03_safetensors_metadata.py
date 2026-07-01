"""Checkpoint Format case study 3: SafeTensors + JSON metadata pipeline.

Companion script for checkpoint_format/checkpoint_format.md. Covers:
  1. Save model + optimizer with SafeTensors + JSON
  2. Load with version compatibility check
  3. Migration from torch.save to safetensors

Run:
    python 03_safetensors_metadata.py
"""

import sys
import json
import os

import torch

SAVE_DIR = "/tmp/safetensors_demo"


def exp_save_pipeline():
    print("=" * 60)
    print("1. SafeTensors + JSON: dual-file save")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    has_safetensors = False
    try:
        from safetensors.torch import save_file
        has_safetensors = True
    except ImportError:
        print("  safetensors not installed (pip install safetensors)")

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 10),
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run one step
    x = torch.randn(8, 128)
    loss = model(x).sum()
    loss.backward()
    optim.step()

    # Step 1: Model weights -> SafeTensors
    if has_safetensors:
        from safetensors.torch import save_file as st_save
        st_save(model.state_dict(), f"{SAVE_DIR}/model.safetensors")
        print(f"  Saved: model.safetensors ({os.path.getsize(f'{SAVE_DIR}/model.safetensors'):,} bytes)")
    else:
        torch.save(model.state_dict(), f"{SAVE_DIR}/model.pt")
        print(f"  Saved: model.pt (fallback, no safetensors)")

    # Step 2: Non-tensor metadata -> JSON
    metadata = {
        "model_config": {
            "type": "Sequential",
            "layers": [
                {"type": "Linear", "in_features": 128, "out_features": 256},
                {"type": "GELU"},
                {"type": "Linear", "in_features": 256, "out_features": 10},
            ],
        },
        "training": {
            "epoch": 1,
            "step": 100,
            "optimizer": "Adam",
            "lr": 0.001,
            "loss": loss.item(),
        },
    }
    with open(f"{SAVE_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: metadata.json")

    # Step 3: Optimizer tensor state -> SafeTensors (separate file)
    if has_safetensors:
        optim_sd = optim.state_dict()
        optim_tensors = {}
        for param_id, state in optim_sd["state"].items():
            for key, val in state.items():
                if torch.is_tensor(val):
                    optim_tensors[f"param_{param_id}_{key}"] = val
        if optim_tensors:
            from safetensors.torch import save_file as st_save
            st_save(optim_tensors, f"{SAVE_DIR}/optim.safetensors")
            print(f"  Saved: optim.safetensors ({os.path.getsize(f'{SAVE_DIR}/optim.safetensors'):,} bytes)")
    print()


def exp_load_pipeline():
    print("=" * 60)
    print("2. SafeTensors + JSON: dual-file load with validation")
    print("=" * 60)

    has_safetensors = True
    try:
        from safetensors.torch import load_file
    except ImportError:
        has_safetensors = False

    # Load metadata first
    try:
        with open(f"{SAVE_DIR}/metadata.json", "r") as f:
            metadata = json.load(f)
        print(f"  Loaded metadata:")
        print(f"    Model type: {metadata['model_config']['type']}")
        print(f"    Training step: {metadata['training']['step']}")
        print(f"    Loss: {metadata['training']['loss']:.4f}")

        # Validate version compatibility
        arch = metadata["model_config"]["layers"]
        expected_total = sum(layer["in_features"] * layer["out_features"]
                           for layer in arch
                           if layer["type"] == "Linear")
        print(f"    Expected weight count: ~{expected_total}")
    except FileNotFoundError:
        print(f"  metadata.json not found (run part 1 first)")

    # Load model weights
    if has_safetensors and os.path.exists(f"{SAVE_DIR}/model.safetensors"):
        from safetensors.torch import load_file
        sd = load_file(f"{SAVE_DIR}/model.safetensors")
        print(f"\n  Loaded model.safetensors:")
        for k, v in sd.items():
            print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}")

        # Reconstruct model and load
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 10),
        )
        model.load_state_dict(sd)
        print(f"\n  Model reconstructed successfully")
    else:
        print(f"\n  model.safetensors not found or safetensors not installed")
    print()


def exp_migration():
    print("=" * 60)
    print("3. Migration: torch.save -> SafeTensors")
    print("=" * 60)

    print(f"  Migration checklist:")
    print(f"    1. Start with existing torch.save files:")
    print(f"       ckpt = torch.load('old.pt', weights_only=True)")
    print(f"")
    print(f"    2. Extract tensors only:")
    print(f"       tensors = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}")
    print(f"") if has_safetensors else None
    print(f"    3. Save tensors via safetensors:")
    print(f"       from safetensors.torch import save_file")
    print(f"       save_file(tensors, 'new.safetensors')")
    print(f"")
    print(f"    4. Save non-tensor metadata separately:")
    print(f"       metadata = {k: v for k, v in ckpt.items() if not isinstance(v, torch.Tensor)}")
    print(f"       json.dump(metadata, open('metadata.json', 'w'))")
    print(f"")
    print(f"    5. Verify roundtrip:")
    print(f"       sd = load_file('new.safetensors')")
    print(f"       model.load_state_dict(sd)")
    print()

    # Cleanup
    for f in os.listdir(SAVE_DIR):
        os.remove(os.path.join(SAVE_DIR, f))
    os.rmdir(SAVE_DIR)


EXPERIMENTS = {
    "save": exp_save_pipeline,
    "load": exp_load_pipeline,
    "migrate": exp_migration,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint_format case 3] DONE")


if __name__ == "__main__":
    main()
