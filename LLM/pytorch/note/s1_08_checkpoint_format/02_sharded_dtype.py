"""Checkpoint Format case study 2: FSDP sharding, map_location, and dtype.

Companion script for checkpoint_format/checkpoint_format.md. Covers:
  1. map_location for device conversion
  2. dtype conversion during load
  3. FSDP checkpoint planning

Run:
    python 02_sharded_dtype.py
"""

import sys

import torch


def exp_map_location():
    print("=" * 60)
    print("1. map_location: device conversion during load")
    print("=" * 60)

    model = torch.nn.Linear(16, 8)
    sd = model.state_dict()

    torch.save(sd, "/tmp/map_test.pt")

    # Load to different device
    if torch.cuda.is_available():
        sd_cuda = torch.load("/tmp/map_test.pt", map_location="cuda:0", weights_only=True)
        for k, v in sd_cuda.items():
            print(f"  {k}: device={v.device}, shape={list(v.shape)}")
    else:
        sd_cpu = torch.load("/tmp/map_test.pt", map_location="cpu", weights_only=True)

    # Load to specific device function
    def to_device(tensor, device):
        return tensor.to(device)

    if torch.cuda.is_available():
        sd_func = torch.load("/tmp/map_test.pt", map_location=lambda storage, loc: storage.cuda(0), weights_only=True)
        print(f"\n  With lambda map_location: all on CUDA")

    import os
    os.remove("/tmp/map_test.pt")
    print()


def exp_dtype_conversion():
    print("=" * 60)
    print("2. Dtype conversion during/after load")
    print("=" * 60)

    # Save in float32
    model = torch.nn.Linear(64, 32)
    sd_fp32 = model.state_dict()
    torch.save(sd_fp32, "/tmp/fp32_model.pt")

    # Load and convert to bfloat16
    sd_loaded = torch.load("/tmp/fp32_model.pt", weights_only=True)

    if torch.cuda.is_available():
        # Convert to bf16
        for k in sd_loaded:
            sd_loaded[k] = sd_loaded[k].to(dtype=torch.bfloat16, device="cuda")
        print(f"  Original dtype: {sd_fp32['weight'].dtype}")
        print(f"  Converted dtype: {sd_loaded['weight'].dtype}")
        print(f"  Converted device: {sd_loaded['weight'].device}")
        print(f"  Shape preserved: {list(sd_loaded['weight'].shape)}")

    # If loaded data is fp32 torch.save but model uses fp16
    model_fp16 = torch.nn.Linear(64, 32, dtype=torch.float16)
    # The load_state_dict raises no error because PyTorch auto-casts
    # (unless `strict` dtype checking is enabled)
    print(f"\n  Auto dtype conversion: load_state_dict auto-handles dtype mismatch")

    import os
    os.remove("/tmp/fp32_model.pt")
    print()


def exp_sharded_checkpoint():
    print("=" * 60)
    print("3. FSDP/Sharded checkpoint planning")
    print("=" * 60)

    print(f"  Distributed checkpoint (FSDP):")
    print(f"    Each rank saves its own shard:")
    print(f"      rank_0/__0_0.distcp   (shard metadata)")
    print(f"      rank_0/__1_0.distcp   (tensor data)")
    print(f"      rank_1/__0_0.distcp")
    print(f"      ...")
    print(f"")

    print(f"  Loading a sharded checkpoint:")
    print(f"    from torch.distributed.checkpoint import FileSystemReader")
    print(f"    reader = FileSystemReader('/path/to/checkpoint/')")
    print(f"    state_dict = load(reader, metadata)")
    print(f"")

    print(f"  Resharding (world_size changed):")
    print(f"    1. Save with world_size=N -> N shard files")
    print(f"    2. Load with world_size=M:")
    print(f"       - Need to consolidate shards first (or use planner)")
    print(f"       - Each rank reads partial data from all saved ranks")
    print(f"       - torch.distributed.checkpoint.planner handles this")
    print(f"")

    print(f"  Best practices for large model checkpoints:")
    print(f"    1. Use SafeTensors for final model distribution")
    print(f"    2. Use torch.distributed.checkpoint for training resumption")
    print(f"    3. Use DeepSpeed/FSDP checkpointing for sharded saves")
    print()


EXPERIMENTS = {
    "map": exp_map_location,
    "dtype": exp_dtype_conversion,
    "sharded": exp_sharded_checkpoint,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint_format case 2] DONE")


if __name__ == "__main__":
    main()
