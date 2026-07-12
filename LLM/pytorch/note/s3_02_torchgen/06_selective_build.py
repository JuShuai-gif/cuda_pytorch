"""torchgen case study 6: selective build for mobile/edge deployment.

Companion script for torchgen/torchgen.md. Covers:
  1. Selective build concepts
  2. Operator allowlist generation
  3. Reducing binary size for mobile

Run:
    python 06_selective_build.py
"""

import sys

import torch


def exp_selective_build():
    print("=" * 60)
    print("1. Selective build: reduce binary size")
    print("=" * 60)

    print(f"  torchgen supports selective build for mobile:")
    print(f"    - Only compile ops actually used by the model")
    print(f"    - Reduces binary from ~200MB to ~5-10MB")
    print(f"")
    print(f"  Workflow:")
    print(f"    1. Trace model to list all used ops:")
    print(f"       from torchgen.selective_build.selector import SelectiveBuilder")
    print(f"       builder = SelectiveBuilder.from_yaml('model_ops.yaml')")
    print(f"")
    print(f"    2. Generate operator allowlist:")
    print(f"       python torchgen/gen.py --op_registration_allowlist=allowlist.yaml")
    print(f"")

    # Show which ops a simple model uses
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 8),
    )

    try:
        from torch.fx import symbolic_trace
        gm = symbolic_trace(model)
        ops_used = set()
        for node in gm.graph.nodes:
            if node.op in ("call_module", "call_function", "call_method"):
                ops_used.add(str(node.target).split("(")[0])

        print(f"  Simple Sequential model uses {len(ops_used)} op types:")
        for op in sorted(ops_used):
            print(f"    {op}")
    except Exception:
        # Fallback: manually list ops
        print(f"  Typical ops for Sequential(Linear+ReLU+Linear):")
        print(f"    aten::linear, aten::addmm, aten::relu")

    print(f"\n  Selective build can exclude:")
    print(f"    - Conv/Pool ops not used by the model")
    print(f"    - Quantization ops")
    print(f"    - Sparse ops")
    print()


def exp_backend_static_registration():
    print("=" * 60)
    print("2. Static vs dynamic op registration")
    print("=" * 60)

    print(f"  Dynamic registration (default):")
    print(f"    All ops registered via TORCH_LIBRARY at startup")
    print(f"    + Full feature set")
    print(f"    - Slower startup, larger memory footprint")
    print(f"")

    print(f"  Static registration (SELECTIVE_BUILD):")
    print(f"    Only whitelisted ops are registered")
    print(f"    + Fast startup, small binary")
    print(f"    - Must pre-declare all needed ops")
    print(f"    - torchgen/gen.py --op_registration_allowlist=ops.yaml")
    print(f"")

    # Check which dispatch keys an op uses
    ops_of_interest = ["add", "matmul", "relu", "conv2d"]
    for op in ops_of_interest:
        try:
            has_cpu = torch._C._dispatch_has_kernel_for_dispatch_key(op, "CPU")
            has_cuda = torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA")
            has_meta = torch._C._dispatch_has_kernel_for_dispatch_key(op, "Meta")
            k = sum([has_cpu, has_cuda, has_meta])
            print(f"  {op:10s}: {k} backend kernels registered")
        except Exception:
            print(f"  {op:10s}: (check requires PyTorch >= 2.x)")
    print()


EXPERIMENTS = {
    "selective": exp_selective_build,
    "static": exp_backend_static_registration,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[torchgen case 6] DONE")


if __name__ == "__main__":
    main()
