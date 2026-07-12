"""torchgen case study 7: add custom backend dispatch key via torchgen.

Companion script for torchgen/torchgen.md. Covers:
  1. Custom backend registration pattern
  2. Dispatch key propagation
  3. Backend fallback chains

Run:
    python 07_custom_backend_key.py
"""

import sys

import torch


def exp_backend_override_pattern():
    print("=" * 60)
    print("1. Custom dispatch key registration pattern")
    print("=" * 60)

    # Register an op with a fake custom backend
    lib = torch.library.Library("backend_demo", "DEF")
    lib.define("backend_add(Tensor a, Tensor b) -> Tensor")

    @torch.library.impl("backend_demo::backend_add", "CompositeImplicitAutograd")
    def default_impl(a, b):
        return a + b

    # To add a custom backend (e.g., "XLA"), you'd:
    # 1. Register a new DispatchKey in c10/core/DispatchKey.h
    # 2. torchgen generates RegisterMyBackend.cpp from native_functions.yaml
    # 3. Implement kernels for each op

    print(f"  Custom backend checklist:")
    print(f"    1. Add DispatchKey to c10/core/DispatchKey.h")
    print(f"    2. Add dispatch entries to native_functions.yaml:")
    print(f"       dispatch:")
    print(f"         MyBackend: my_backend_kernel_name")
    print(f"    3. Run torchgen to generate RegisterMyBackend.cpp")
    print(f"    4. Implement kernels")
    print(f"    5. Build PyTorch with new backend enabled")
    print()


def exp_backend_fallback():
    print("=" * 60)
    print("2. Backend fallback chain demo")
    print("=" * 60)

    lib = torch.library.Library("fallback_demo", "DEF")
    lib.define("custom_op(Tensor x) -> Tensor")

    # Layer 1: no custom backend -> CompositeImplicitAutograd fallback
    @torch.library.impl("fallback_demo::custom_op", "CompositeImplicitAutograd")
    def composite_fallback(x):
        return x * 5

    # Layer 2: Register CPU override -> CPU uses this instead
    @torch.library.impl("fallback_demo::custom_op", "CPU")
    def cpu_override(x):
        return x * 50

    x = torch.randn(3)
    y_cpu = torch.ops.fallback_demo.custom_op(x)
    print(f"  CPU: {y_cpu.tolist()} (expect x*50, CPU overrides CompositeImplicitAutograd)")

    x_meta = torch.empty(3, device="meta")
    y_meta = torch.ops.fallback_demo.custom_op(x_meta)
    print(f"  Meta: shape={list(y_meta.shape)} (uses CompositeImplicitAutograd)")

    print(f"\n  Fallback chain: CPU > CompositeImplicitAutograd > error")
    print()


def exp_aten_override():
    print("=" * 60)
    print("3. Override ATen op at a specific dispatch key")
    print("=" * 60)

    # Show that ATen ops can also have custom kernels at specific keys
    print(f"  To override an ATen op for a custom backend:")
    print(f"")
    print(f"  C++:")
    print(f"    TORCH_LIBRARY_IMPL(aten, MyBackend, m) {{")
    print(f"        m.impl('add.Tensor', my_add_kernel);")
    print(f"        m.impl('mul.Tensor', my_mul_kernel);")
    print(f"    }}")
    print(f"")
    print(f"  Python:")
    print(f"    @torch.library.impl('aten::add.Tensor', 'MyBackend')")
    print(f"    def my_add(a, b, alpha=1):")
    print(f"        return my_custom_add(a, b, alpha)")
    print(f"")
    print(f"  This is how torch_xla, torch_npu etc. work")
    print()


EXPERIMENTS = {
    "pattern": exp_backend_override_pattern,
    "fallback": exp_backend_fallback,
    "aten": exp_aten_override,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[torchgen case 7] DONE")


if __name__ == "__main__":
    main()
