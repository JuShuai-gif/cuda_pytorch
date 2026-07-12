"""Checkpoint Format case study 4: checkpoint versioning and compatibility.

Companion script for checkpoint_format/checkpoint_format.md. Covers:
  1. Checkpoint versioning strategy
  2. Load partial / upgrade checkpoint
  3. Module version migration

Run:
    python 04_version_migration.py
"""

import sys

import torch


def exp_version_tagging():
    print("=" * 60)
    print("1. Embed version info in checkpoint")
    print("=" * 60)

    model = torch.nn.Linear(16, 8)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "version": "2.1.0",
        "model_class": "Linear",
        "config": {"in_features": 16, "out_features": 8},
        "torch_version": torch.__version__,
        "created_at": "2024-01-01",
    }

    torch.save(checkpoint, "/tmp/versioned_ckpt.pt")
    print(f"  Checkpoint saved with version metadata")

    # Load and check version
    ckpt = torch.load("/tmp/versioned_ckpt.pt", weights_only=True)
    print(f"  Version: {ckpt.get('version', 'unknown')}")
    print(f"  Config:  {ckpt.get('config', {})}")

    # Version check: refuse to load if too old
    required_version = "2.0.0"
    ckpt_version = ckpt.get("version", "0.0.0")
    if ckpt_version >= required_version:
        print(f"  Version compatible: {ckpt_version} >= {required_version}")
    else:
        print(f"  Version incompatible: need migration")

    import os; os.remove("/tmp/versioned_ckpt.pt")
    print()


def exp_partial_load():
    print("=" * 60)
    print("2. Partial checkpoint loading (incremental)")
    print("=" * 60)

    class FullModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(32, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
            )
            self.decoder = torch.nn.Linear(32, 16)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = FullModel()
    sd = model.state_dict()

    # Save only encoder
    encoder_sd = {k: v for k, v in sd.items() if "encoder" in k}
    torch.save(encoder_sd, "/tmp/encoder.pt")

    # Load encoder into new model with different decoder
    new_model = FullModel()
    new_model.load_state_dict(encoder_sd, strict=False)
    print(f"  Loaded encoder params, decoder = random init")
    print(f"  -> strict=False allows partial loading")

    import os; os.remove("/tmp/encoder.pt")
    print()


def exp_upgrade_path():
    print("=" * 60)
    print("3. Upgrade checkpoint: add/remove layers")
    print("=" * 60)

    # Old model: one Linear
    old_sd = {"weight": torch.randn(8, 4), "bias": torch.randn(8)}

    # New model: two Linears (linear0 + linear1)
    new_model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Linear(8, 8),
    )

    # Manual rename to fit new model
    upgraded_sd = {
        "0.weight": old_sd["weight"],
        "0.bias": old_sd["bias"],
    }
    # linear1 uses default random init (not loaded)
    missing, unexpected = new_model.load_state_dict(upgraded_sd, strict=False)
    print(f"  Missing keys: {missing}")
    print(f"  Unexpected keys: {unexpected}")
    print(f"  -> linear0 loaded from old ckpt, linear1 random init")
    print()


EXPERIMENTS = {
    "version": exp_version_tagging,
    "partial": exp_partial_load,
    "upgrade": exp_upgrade_path,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint_format case 4] DONE")


if __name__ == "__main__":
    main()
