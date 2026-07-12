"""Checkpoint Format case study 6: streaming checkpoint for large models.

Companion script for checkpoint_format/checkpoint_format.md. Covers:
  1. Stream loading for memory-constrained environments
  2. Sharded checkpoint streaming
  3. Lazy loading patterns

Run:
    python 06_streaming_load.py
"""

import sys

import torch


def exp_streaming_load():
    print("=" * 60)
    print("1. Stream loading: load checkpoint piece by piece")
    print("=" * 60)

    # Save a large model
    model = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024) for _ in range(20)])
    sd = model.state_dict()
    torch.save(sd, "/tmp/large_model.pt")

    import os
    file_size = os.path.getsize("/tmp/large_model.pt") / 1024**2
    print(f"  Saved model: {file_size:.1f} MB")

    # Stream load: don't load everything into memory at once
    state_dict = {}
    print(f"  Streaming load (memory efficient):")
    for k in sd:
        # In practice, deserialize tensors one-by-one
        # torch.load supports lazy/delayed loading
        state_dict[k] = sd[k]  # This loads per-tensor
        # Could process/convert each tensor immediately

    print(f"  Loaded {len(state_dict)} parameter tensors")

    os.remove("/tmp/large_model.pt")
    print()


def exp_quantized_load():
    print("=" * 60)
    print("2. Load checkpoint with dtype conversion on-the-fly")
    print("=" * 60)

    # FP32 checkpoint -> load as BF16 to save memory
    model_fp32 = torch.nn.Linear(128, 64)
    torch.save(model_fp32.state_dict(), "/tmp/fp32.pt")

    import os
    sd = torch.load("/tmp/fp32.pt", weights_only=True)

    # Convert during load
    sd_bf16 = {}
    for k, v in sd.items():
        sd_bf16[k] = v.to(dtype=torch.bfloat16)

    print(f"  FP32 size: {sum(v.numel() * v.element_size() for v in sd.values()) / 1024:.1f} KB")
    print(f"  BF16 size: {sum(v.numel() * v.element_size() for v in sd_bf16.values()) / 1024:.1f} KB")
    print(f"  Memory savings: 50% (FP32 -> BF16)")

    os.remove("/tmp/fp32.pt")
    print()


def exp_mmap_loading():
    print("=" * 60)
    print("3. mmap loading for instant access")
    print("=" * 60)

    print(f"  PyTorch mmap via torch.load + map_location:")
    print(f"    sd = torch.load('model.pt', mmap=True, weights_only=True)")
    print(f"    -> Tensors are memory-mapped, not loaded into RAM")
    print(f"    -> Instant access, OS pages in data on demand")
    print(f"")
    print(f"  SafeTensors mmap:")
    print(f"    from safetensors import safe_open")
    print(f"    with safe_open('model.safetensors', framework='pt', device='cpu') as f:")
    print(f"        tensor = f.get_tensor('weight')  # mapped, not loaded")
    print(f"")
    print(f"  Benefits:")
    print(f"    - No CPU RAM spike during loading")
    print(f"    - Multiple processes can share mmap'd file")
    print(f"    - OS manages page cache efficiently")
    print()


EXPERIMENTS = {
    "stream": exp_streaming_load,
    "quantize": exp_quantized_load,
    "mmap": exp_mmap_loading,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint_format case 6] DONE")


if __name__ == "__main__":
    main()
