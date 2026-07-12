"""torchgen case study 1: trace op dispatch table and generated code.

Companion script for torchgen/torchgen.md. Covers:
  1. dump dispatch table for any ATen op
  2. verify operator handle and schema
  3. search generated Register code signatures

Run:
    python 01_trace_dispatch.py              # full demo
    python 01_trace_dispatch.py add          # trace aten::add
    python 01_trace_dispatch.py matmul       # trace aten::matmul
    python 01_trace_dispatch.py silu         # trace aten::silu
"""

import sys

import torch


def dump_op(table_text, op_name):
    """Parse dispatch dump output and print structured info."""
    print(f"\n  Dispatch table for '{op_name}':")
    print(f"  {'-' * 50}")
    for line in table_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            kernel = parts[1] if len(parts) > 1 else "?"
            print(f"    {key:40s} -> {kernel}")


def exp_dispatch_dump(op_name="aten::add.Tensor"):
    print("=" * 60)
    print(f"1. Dispatch table dump: {op_name}")
    print("=" * 60)

    handle = getattr(torch.ops.aten, op_name.split("::")[-1], None)
    if handle is None:
        print(f"  op '{op_name}' not found in torch.ops.aten")
        return

    print(f"  Operator handle: {handle}")
    print(f"  Default overload: {handle.default}")

    table = torch._C._dispatch_dump_table(op_name)
    dump_op(table, op_name)
    print()


def exp_schema_inspect():
    print("=" * 60)
    print("2. Inspect operator schema")
    print("=" * 60)

    ops_to_inspect = [
        "aten::add.Tensor",
        "aten::add.out",
        "aten::matmul",
    ]

    for op_name in ops_to_inspect:
        print(f"\n  Schema for '{op_name}':")
        try:
            overload_name = op_name.split("::")[-1]
            op = getattr(torch.ops.aten, overload_name)
            print(f"    overloaded: {op.overloads()}")
            for ov in op.overloads():
                print(f"    {ov}: {getattr(op, ov)._schema}")
        except Exception as e:
            print(f"    ERROR: {e}")
    print()


def exp_check_generated():
    print("=" * 60)
    print("3. Verify kernel presence via dispatch key")
    print("=" * 60)

    ops_to_check = ["add", "mul", "matmul", "relu"]

    for op_name in ops_to_check:
        print(f"\n  Checking '{op_name}':")
        for key in ["CPU", "CUDA", "Meta", "CompositeImplicitAutograd"]:
            try:
                has = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, key)
                status = "YES" if has else "no "
                print(f"    {key:30s}: {status}")
            except Exception:
                print(f"    {key:30s}: API not available (PyTorch >= 2.x needed)")

    # Tip: how to search in source
    print(f"\n  To inspect generated code, search in PyTorch build dir:")
    print(f"    rg -n 'aten::add.Tensor' build/aten/src/ATen/RegisterCPU.cpp")
    print(f"    rg -n 'func: add\\\\.Tensor' aten/src/ATen/native/native_functions.yaml")
    print()


def exp_works_with_compile():
    print("=" * 60)
    print("4. Check if op works with torch.compile")
    print("=" * 60)

    @torch.compile
    def f(x, y):
        return torch.add(x, y)

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    result = f(x, y)
    print(f"  torch.add via compile: OK, shape={list(result.shape)}")

    @torch.compile
    def g(x):
        return torch.nn.functional.silu(x)

    x = torch.randn(3, 4)
    result = g(x)
    print(f"  F.silu via compile:   OK, shape={list(result.shape)}")

    print(f"\n  torch.compile works because:")
    print(f"    1. These ops have Meta kernel (torchgen generates meta stubs)")
    print(f"    2. Dynamo can fallback to CompositeImplicitAutograd if needed")
    print()


EXPERIMENTS = {
    "add": lambda: exp_dispatch_dump("aten::add.Tensor"),
    "matmul": lambda: exp_dispatch_dump("aten::matmul"),
    "schema": exp_schema_inspect,
    "check": exp_check_generated,
    "compile": exp_works_with_compile,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else ["add", "schema", "compile"]
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[torchgen case 1] DONE")


if __name__ == "__main__":
    main()
