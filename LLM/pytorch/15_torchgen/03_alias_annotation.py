"""torchgen case study 3: out= variant, alias annotation, and YAML inspection.

Companion script for torchgen/torchgen.md. Covers:
  1. out= variant behavior and dispatch
  2. alias annotation (a!) impact on autograd
  3. manual YAML schema inspection tips

Run:
    python 03_alias_annotation.py
"""

import sys

import torch


def exp_out_variant():
    print("=" * 60)
    print("1. out= variant and functional/out dispatch comparison")
    print("=" * 60)

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    # Functional: returns new tensor
    z_func = torch.add(x, y)
    print(f"  Functional add: shape={list(z_func.shape)}")

    # out variant: writes to pre-allocated tensor
    out = torch.empty(3, 4)
    z_out = torch.add(x, y, out=out)
    print(f"  Out variant: z_out is out = {z_out is out}")
    print(f"  same data_ptr: {z_out.data_ptr() == out.data_ptr()}")

    # Check dispatch table for both
    print(f"\n  Dispatch table for 'aten::add.out':")
    table = torch._C._dispatch_dump_table("aten::add.out")
    for line in table.strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")

    print(f"\n  Key: out= variant has ((a!)) annotation in YAML")
    print(f"  (a!) = input is mutated (write operation)")
    print(f"  This affects: Functionalization, Autograd in-place detection")
    print()


def exp_alias_impact():
    print("=" * 60)
    print("2. Alias annotation impact on autograd detection")
    print("=" * 60)

    # (a!) annotation means the op is in-place
    # Autograd uses this to do version counter checks

    x = torch.randn(3, requires_grad=True)
    print(f"  Before: x version = {x._version}")

    # add_ is marked Tensor(a!) in YAML -> in-place detected
    x.add_(1.0)
    print(f"  After add_(1.0): x version = {x._version} (incremented)")

    # Normal add is NOT in-place -> no version bump
    y = x + 1.0
    print(f"  After x + 1.0: x version = {x._version} (unchanged, x not modified)")

    # Version check prevents backward through modified tensors
    x2 = torch.randn(3, requires_grad=True)
    y2 = x2 * 2
    loss = y2.sum()
    x2.add_(0.5)  # in-place mutation AFTER the computation graph was built
    try:
        loss.backward()
        print(f"  Backward succeeded (unexpected)")
    except RuntimeError as e:
        print(f"\n  backward() ERROR: {str(e)[:100]}")
        print(f"  -> in-place mutation on x2 invalidated the saved forward version")

    print(f"\n  YAML annotation Tensor(a!) -> autograd version counter check")
    print(f"  Without (a!), version would NOT be bumped -> silent bugs")
    print()


def exp_structured_kernel_check():
    print("=" * 60)
    print("3. Structured kernel vs unstructured registration")
    print("=" * 60)

    # Structured kernels: torchgen generates meta+impl stubs
    # Unstructured: single monolithic registration

    structured_ops = ["conv2d", "batch_norm", "max_pool2d"]
    unstructured_ops = ["add", "mul", "relu"]

    print(f"  Structured kernels (torchgen generates set_output / set_meta):")
    for op in structured_ops:
        has_cpu = torch._C._dispatch_has_kernel_for_dispatch_key(op, "CPU")
        has_meta = torch._C._dispatch_has_kernel_for_dispatch_key(op, "Meta")
        print(f"    {op:20s}: CPU={has_cpu}, Meta={has_meta}")

    print(f"\n  Unstructured / Composite ops:")
    for op in unstructured_ops:
        has_comp = torch._C._dispatch_has_kernel_for_dispatch_key(
            op, "CompositeImplicitAutograd"
        )
        print(f"    {op:20s}: CompositeImplicitAutograd={has_comp}")

    print(f"\n  To find structured/unstructured in YAML:")
    print(f"    rg 'structured: true' aten/src/ATen/native/native_functions.yaml")
    print()


def exp_yaml_search_guide():
    print("=" * 60)
    print("4. Guide: search native_functions.yaml from Python")
    print("=" * 60)

    guide = """
  Locate torchgen source (in PyTorch repo):
    git clone https://github.com/pytorch/pytorch

  Common searches in native_functions.yaml:
    # Find an op schema
    rg -n "func: silu" aten/src/ATen/native/native_functions.yaml

    # Find all out= variants
    rg -n "func:.*\\.out" aten/src/ATen/native/native_functions.yaml | head -10

    # Find alias annotations
    rg -n "Tensor\\(a!\\)" aten/src/ATen/native/native_functions.yaml | head -10

    # Find CompositeImplicitAutograd entries
    rg -n "CompositeImplicitAutograd:" aten/src/ATen/native/native_functions.yaml | head -10

    # Find structured kernels
    rg -n "structured: true" aten/src/ATen/native/native_functions.yaml

  Examine generated code:
    rg -n "aten::silu" build/aten/src/ATen/RegisterCPU.cpp
    rg -n "torch.*silu" build/aten/src/ATen/Functions.cpp
"""
    print(guide)


EXPERIMENTS = {
    "out": exp_out_variant,
    "alias": exp_alias_impact,
    "structured": exp_structured_kernel_check,
    "yaml": exp_yaml_search_guide,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[torchgen case 3] DONE")


if __name__ == "__main__":
    main()
