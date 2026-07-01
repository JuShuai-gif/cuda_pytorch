"""DDP Reducer case study 2: gradient accumulation and communication hook.

Companion script for distributed_techniques/ddp_reducer/ddp_reducer.md. Covers:
  1. Gradient accumulation with DDP (no_sync)
  2. Custom communication hook for gradient compression
  3. Timing analysis

Run:
    python 02_grad_accum_comm_hook.py
"""

import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _maybe_init():
    try:
        if not dist.is_initialized():
            dist.init_process_group("gloo")
        return True
    except Exception:
        return False


def exp_no_sync_grad_accum():
    print("=" * 60)
    print("1. Gradient accumulation with DDP no_sync")
    print("=" * 60)

    # Simulated gradient accumulation
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x)).sum()

    model = SimpleModel()

    accumulation_steps = 4
    print(f"  Gradient accumulation: {accumulation_steps} steps")

    # Without DDP (simulated single-GPU workflow)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for step in range(accumulation_steps):
        x = torch.randn(8, 32)
        loss = model(x)
        (loss / accumulation_steps).backward()
        print(f"    Step {step+1}: grad norm = {model.linear.weight.grad.norm().item():.4f}")

    optimizer.step()
    optimizer.zero_grad()

    print(f"\n  In real DDP training:")
    print(f"    Steps 1..N-1: use model.no_sync() context")
    print(f"    Step N:        default sync (allreduce triggers)")
    print(f"")
    print(f"    # Pseudo code:")
    print(f"    for step in range(acc_steps):")
    print(f"        ctx = model.no_sync() if step < acc_steps - 1 else nullcontext()")
    print(f"        with ctx:")
    print(f"            loss = model(x)")
    print(f"            (loss / acc_steps).backward()")
    print(f"    optimizer.step()")
    print()


def exp_comm_hook_patterns():
    print("=" * 60)
    print("2. Communication hook patterns")
    print("=" * 60)

    print(f"  Hook 1: PowerSGD (low-rank compression)")
    print(f"    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook")
    print(f"    ddp_model.register_comm_hook(state, powerSGD_hook)")
    print(f"")

    print(f"  Hook 2: FP16 gradient compression")
    print(f"    # Custom hook: cast to fp16 before allreduce, cast back")
    print(f"    def fp16_compress_hook(state, bucket):")
    print(f"        tensor = bucket.buffer().half()")
    print(f"        fut = state.process_group.allreduce(tensor)")
    print(f"        def decompress(fut):")
    print(f"            return fut.value().float()")
    print(f"        return fut.then(decompress)")
    print(f"    ddp_model.register_comm_hook(None, fp16_compress_hook)")
    print(f"")

    print(f"  Hook 3: Gradient sparsification")
    print(f"    # Only send top-k gradient elements")
    print(f"    # Use default_hooks with custom state buffer")
    print(f"    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks")
    print(f"")

    print(f"  Benefits: reduce communication bandwidth")
    print(f"  Cost:     precision loss on gradients")
    print()


def exp_network_env_analysis():
    print("=" * 60)
    print("3. Network environment analysis for DDP tuning")
    print("=" * 60)

    print(f"  Check network latency (intra-node vs inter-node):")
    print(f"    # Intra-node (NVLink): ~300 GB/s")
    print(f"    # Inter-node (IB/RoCE): ~25-100 GB/s")
    print(f"")

    print(f"  Network bandwidth rules of thumb for bucket sizing:")
    print(f"    NVLink (intra):  bucket_cap_mb=25 (default OK)")
    print(f"    IB 100Gb/s:      bucket_cap_mb=10 (smaller = more overlap)")
    print(f"    IB 25Gb/s:       bucket_cap_mb=5  (even smaller)")
    print(f"    Ethernet 10Gb/s: bucket_cap_mb=2  (smallest)")
    print(f"")

    print(f"  Gradient size estimate for common models:")
    print(f"    ResNet-50:      ~100MB total grad")
    print(f"    GPT-2 Small:    ~500MB total grad")
    print(f"    LLaMA-7B:       ~14GB total grad (need FSDP!)")

    # DDP becomes inefficient when single param > bucket_cap_mb
    large_tensor = torch.randn(100 * 1024 * 1024 // 4)  # 100MB float32
    print(f"\n  When a single tensor ({large_tensor.numel() * 4 / 1024 / 1024:.0f}MB)")
    print(f"  exceeds bucket_cap_mb:")
    print(f"    -> It gets its own bucket")
    print(f"    -> No overlap benefit for that parameter")
    print(f"    -> Consider FSDP or ZeRO for models with large params")


EXPERIMENTS = {
    "accum": exp_no_sync_grad_accum,
    "hook": exp_comm_hook_patterns,
    "network": exp_network_env_analysis,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ddp_reducer case 2] DONE")


if __name__ == "__main__":
    main()
