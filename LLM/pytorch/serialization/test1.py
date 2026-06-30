"""Serialization demo: save/load, weights_only, mmap, cross-device.

Companion script for serialization/serialization.md.
  1. basics:                torch.save → torch.load
  2. weights_only:          security vs full pickle
  3. mmap:                  zero-copy loading
  4. cross_device:          map_location to different GPU
  5. state_dict hooks:      custom save/load behavior

Run:
    python test1.py                 # full demo
    python test1.py basic           # basic save/load
    python test1.py weights_only    # weights_only security
    python test1.py mmap            # mmap zero-copy loading
    python test1.py cross_device    # cross-device loading
"""

import sys
import os
import tempfile
import torch
import torch.nn as nn


# ============ 1. Basic save/load ============
def exp_basic():
    print("=" * 60)
    print("1. Basic: torch.save → torch.load")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    model.eval()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(model.state_dict(), tmp.name)
    fsize = os.path.getsize(tmp.name) / 1024
    print(f"  Saved: {tmp.name} ({fsize:.1f} KB)")

    # Load
    sd = torch.load(tmp.name, weights_only=True)
    print(f"  Loaded keys: {list(sd.keys())}")
    print(f"  Types: weight={sd['0.weight'].dtype}, bias={sd['0.bias'].dtype}")

    # Restore to new model
    model2 = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    model2.load_state_dict(sd)
    print(f"  Restored match: {torch.allclose(model[0].weight, model2[0].weight)}")

    os.unlink(tmp.name)
    print()


# ============ 2. weights_only security ============
def exp_weights_only():
    print("=" * 60)
    print("2. weights_only: secure loading")
    print("=" * 60)

    # Save a complex object (dict with custom class)
    class MyConfig:
        def __init__(self):
            self.lr = 0.001
            self.name = "test"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save({"model": torch.randn(3), "config": MyConfig()}, tmp.name)

    # weights_only=True: only tensors & basic types
    try:
        data_safe = torch.load(tmp.name, weights_only=True)
        print(f"  weights_only=True:  model={data_safe['model'].shape}, config=...")
        if "config" in data_safe:
            print(f"    config loaded as: {type(data_safe['config']).__name__}")
    except Exception as e:
        print(f"  weights_only=True:  FAILED — {type(e).__name__}")

    # weights_only=False: full pickle (dangerous for untrusted files)
    try:
        data_full = torch.load(tmp.name, weights_only=False)
        print(f"  weights_only=False: config type={type(data_full['config']).__name__}")
        print(f"    config.lr={data_full['config'].lr}")
    except Exception as e:
        print(f"  weights_only=False: FAILED — {type(e).__name__}")

    print(f"  → weights_only=True prevents code execution from untrusted checkpoints")
    print(f"  → use it UNLESS you need to restore custom non-tensor objects")
    os.unlink(tmp.name)
    print()


# ============ 3. mmap zero-copy loading ============
def exp_mmap():
    print("=" * 60)
    print("3. mmap: zero-copy tensor loading")
    print("=" * 60)

    # Save a large tensor
    big_tensor = torch.randn(4096, 4096)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save({"weight": big_tensor}, tmp.name)
    fsize = os.path.getsize(tmp.name) / 1e6
    print(f"  Saved: {fsize:.1f} MB")

    # Normal load: copies data to new storage
    data_normal = torch.load(tmp.name, weights_only=True)
    print(f"  Normal load: tensor.sum()={data_normal['weight'].sum().item():.2f}")

    # mmap load: shares storage with file
    data_mmap = torch.load(tmp.name, weights_only=True, mmap=True)
    print(f"  mmap load:    tensor.sum()={data_mmap['weight'].sum().item():.2f}")
    print(f"  → mmap avoids the full copy, tensor directly reads from disk mapping")

    os.unlink(tmp.name)
    del data_normal, data_mmap
    print()


# ============ 4. Cross-device loading ============
def exp_cross_device():
    print("=" * 60)
    print("4. Cross-device: map_location")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    # Save a model on CUDA
    model = nn.Linear(4, 3).cuda()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(model.state_dict(), tmp.name)

    # Load: force to CPU
    sd_cpu = torch.load(tmp.name, map_location="cpu", weights_only=True)
    print(f"  map_location='cpu':  weight device={sd_cpu['weight'].device}")

    # Load: to specific GPU
    sd_gpu = torch.load(tmp.name, map_location="cuda:0", weights_only=True)
    print(f"  map_location='cuda:0': weight device={sd_gpu['weight'].device}")

    # Load: stay on original device
    sd_keep = torch.load(tmp.name, weights_only=True)
    print(f"  no map_location:      weight device={sd_keep['weight'].device}")

    os.unlink(tmp.name)
    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "weights_only": exp_weights_only,
    "mmap": exp_mmap,
    "cross_device": exp_cross_device,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[serialization demo] DONE")


if __name__ == "__main__":
    main()
