"""DDP Reducer case study 4: model parallel + data parallel interaction.

Companion script for distributed_techniques/ddp_reducer/ddp_reducer.md. Covers:
  1. DDP auto-bucket behavior with model parallelism
  2. Param grouping across devices
  3. Bucket allocation for irregular models

Run:
    python 04_model_parallel_ddp.py
"""

import sys

import torch


def exp_irregular_models():
    print("=" * 60)
    print("1. Bucket allocation for irregular model structure")
    print("=" * 60)

    # Irregular: huge and tiny params mixed
    class IrregularModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(50000, 4096)  # large
            self.head = torch.nn.Linear(4096, 50000)       # large
            self.layers = torch.nn.Sequential(*[
                torch.nn.Linear(4096, 4096) for _ in range(24)
            ])

        def forward(self, x):
            x = self.embed(x)
            x = self.layers(x)
            return self.head(x)

    model = IrregularModel()
    param_sizes = []

    for name, param in model.named_parameters():
        size_mb = param.numel() * 4 / (1024**2)
        param_sizes.append((name[-50:], size_mb))

    param_sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"  Param count: {len(param_sizes)}")
    print(f"  Top 5 largest:")
    for name, size in param_sizes[:5]:
        print(f"    {name}: {size:.1f} MB")

    print(f"\n  DDP bucket allocation:")
    print(f"    Huge params (embed/head): get own buckets, no overlap")
    print(f"    Layer params: grouped together if < bucket_cap")
    print(f"    -> Embed backward finishes last -> head bucket allreduce at the end")
    print()


def exp_ddp_ordering():
    print("=" * 60)
    print("2. Parameter registration order effect on bucket")
    print("=" * 60)

    class Ordered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Register child modules in order
            self.a = torch.nn.Linear(256, 512)
            self.b = torch.nn.Linear(512, 256)
            self.norm = torch.nn.LayerNorm(256)

        def forward(self, x):
            return self.norm(self.b(torch.relu(self.a(x))))

    model = Ordered()

    print(f"  Parameter registration order:")
    for name, param in model.named_parameters():
        print(f"    {name:30s}: {param.numel() * 4 / 1024:.1f} KB")

    print(f"\n  Backward order: norm -> b -> a")
    print(f"  DDP constructs buckets in registration order")
    print(f"  -> If backward order != registration order,")
    print(f"     early buckets may wait for later params to finish backward")
    print()


def exp_broadcast_buffers():
    print("=" * 60)
    print("3. broadcast_buffers and BatchNorm interaction")
    print("=" * 60)

    class BNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(64)
            self.linear = torch.nn.Linear(64, 10)

        def forward(self, x):
            return self.linear(self.bn(x))

    model = BNModel()
    print(f"  BN buffers:")
    for name, buf in model.named_buffers():
        print(f"    {name}: shape={list(buf.shape)}, persistent={buf not in model._non_persistent_buffers_set}")

    print(f"\n  DDP broadcast_buffers=True (default):")
    print(f"    Syncs BN running_mean/var from rank 0 to all ranks")
    print(f"    Ensures eval mode consistency across workers")
    print()


EXPERIMENTS = {
    "irregular": exp_irregular_models,
    "order": exp_ddp_ordering,
    "buffers": exp_broadcast_buffers,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ddp_reducer case 4] DONE")


if __name__ == "__main__":
    main()
