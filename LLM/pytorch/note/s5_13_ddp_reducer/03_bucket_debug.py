"""DDP Reducer case study 3: debug bucket distribution and performance.

Companion script for distributed_techniques/ddp_reducer/ddp_reducer.md. Covers:
  1. Model parameter size distribution analysis
  2. Simulate bucket grouping
  3. Performance metrics

Run:
    python 03_bucket_debug.py
"""

import sys

import torch


def exp_param_analysis():
    print("=" * 60)
    print("1. Model parameter size distribution")
    print("=" * 60)

    # Build a transformer-like model
    model = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
        num_layers=6,
    )

    total_params = 0
    total_grad_mb = 0
    param_sizes = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            size_mb = num * 4 / (1024**2)
            total_params += num
            total_grad_mb += size_mb
            param_sizes.append((name[-60:], size_mb))

    param_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"  Total params: {total_params:,}")
    print(f"  Total grad size: {total_grad_mb:.1f} MB")
    print(f"  Top 5 largest parameters:")
    for name, size in param_sizes[:5]:
        print(f"    {name:50s} {size:.2f} MB")

    default_bucket = 25  # MB
    est_buckets = max(1, int(total_grad_mb / default_bucket))
    print(f"\n  With bucket_cap_mb=25: ~{est_buckets} buckets")
    print(f"  With bucket_cap_mb=10: ~{max(1, int(total_grad_mb / 10))} buckets")
    print()


def exp_bucket_sim():
    print("=" * 60)
    print("2. Simulate DDP bucket grouping")
    print("=" * 60)

    # Simplified bucket simulation
    bucket_cap_mb = 25

    class SimBucket:
        def __init__(self, idx):
            self.idx = idx
            self.params = []
            self.total_mb = 0

        def can_add(self, size_mb):
            return self.total_mb + size_mb <= bucket_cap_mb

        def add(self, name, size_mb):
            self.params.append(name)
            self.total_mb += size_mb

    # Simulated parameter list (names and sizes in MB)
    param_list = [
        ("embedding.weight", 200),
        ("layers.0.attn.w_qkv", 50),
        ("layers.0.mlp.w1", 40),
        ("layers.0.mlp.w2", 40),
        ("layers.1.attn.w_qkv", 50),
        ("layers.1.mlp.w1", 40),
        ("layers.2.attn.w_qkv", 50),
        ("layers.2.mlp.w2", 40),
        ("norm.weight", 0.001),
        ("head.weight", 80),
    ]

    buckets = []
    current_bucket = SimBucket(0)

    for name, size in param_list:
        if size > bucket_cap_mb:
            # Oversized param gets its own bucket
            b = SimBucket(len(buckets))
            b.add(name, size)
            buckets.append(b)
        else:
            if not current_bucket.can_add(size):
                buckets.append(current_bucket)
                current_bucket = SimBucket(len(buckets))
            current_bucket.add(name, size)

    if current_bucket.params:
        buckets.append(current_bucket)

    print(f"  Bucket size threshold: {bucket_cap_mb} MB")
    print(f"  Number of buckets: {len(buckets)}")
    for b in buckets:
        params = ", ".join(n[-30:] for n in b.params)
        print(f"    Bucket {b.idx}: {b.total_mb:.1f} MB ({params})")

    print(f"\n  Observations:")
    print(f"    1. Large params (>bucket_cap) get their own bucket -> no overlap")
    print(f"    2. Small params are grouped -> better overlap")
    print(f"    3. Parameter registration order determines grouping")
    print()


def exp_performance_tips():
    print("=" * 60)
    print("3. Performance optimization tips")
    print("=" * 60)

    tips = [
        ("Register order", "Register params in backward-order for better overlap"),
        ("Bucket cap", "Start with 25MB, tune up for large models, down for slow networks"),
        ("Grad accumulation", "Use no_sync() to batch gradients before allreduce"),
        ("Mixed precision", "FP16 gradients halve communication, train with AMP"),
        ("NCCL env", "Set NCCL_IB_DISABLE=0 for InfiniBand, NCCL_SOCKET_IFNAME for RoCE"),
        ("Profile", "Use nsys/torch.profiler to verify overlap"),
    ]

    for title, desc in tips:
        print(f"  {title:20s}: {desc}")


EXPERIMENTS = {
    "params": exp_param_analysis,
    "bucket": exp_bucket_sim,
    "tips": exp_performance_tips,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ddp_reducer case 3] DONE")


if __name__ == "__main__":
    main()
