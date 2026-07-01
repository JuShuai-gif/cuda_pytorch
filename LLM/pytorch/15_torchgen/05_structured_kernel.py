"""torchgen case study 5: structured kernel and codegen output.

Companion script for torchgen/torchgen.md. Covers:
  1. Structured kernel TORCH_META_FUNC / TORCH_IMPL_FUNC
  2. Generated code inspection from Python
  3. Adding a new op to native_functions.yaml

Run:
    python 05_structured_kernel.py
"""

import sys

import torch


def exp_structured_pattern():
    print("=" * 60)
    print("1. Structured kernel pattern: TORCH_META_FUNC + TORCH_IMPL_FUNC")
    print("=" * 60)

    # Structured kernel example: avg_pool2d
    # In C++ source (aten/src/ATen/native/Normalization.cpp):
    # TORCH_META_FUNC(avg_pool2d)(const Tensor& self, IntArrayRef kernel_size, ...)
    # {
    #     // set inferred output shape and dtype
    #     set_output(sizes, options);
    # }
    # TORCH_IMPL_FUNC(avg_pool2d_out_cpu)(const Tensor& self, IntArrayRef kernel_size, ...)
    # {
    #     // actual CPU computation using inferred shapes
    # }

    print(f"  Structured kernel splits op into two parts:")
    print(f"    TORCH_META_FUNC:  shape/dtype inference (used by Meta/FakeTensor)")
    print(f"    TORCH_IMPL_FUNC:  actual computation for each backend")

    # torchgen generates stubs connecting meta and impl:
    print(f"\n  torchgen generates:")
    print(f"    1. RegisterCPU.cpp stubs: meta() -> set_output, then impl()")
    print(f"    2. The meta func is reused by Meta/Fake dispatch keys")
    print(f"    3. Reduces code duplication between backends")

    # Demonstrate with a real structured op: avg_pool2d
    x = torch.randn(1, 3, 32, 32)

    # Eager
    y = torch.nn.functional.avg_pool2d(x, 3)
    print(f"\n  avg_pool2d: {list(x.shape)} -> {list(y.shape)}")
    print(f"  Shape inference doesn't require running kernel")
    print()


def exp_compile_generated_pattern():
    print("=" * 60)
    print("2. torch.compile traces through structured kernel")
    print("=" * 60)

    # Structured ops work well with compile because meta is built-in
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),   # structured
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),  # structured
    )

    compiled = torch.compile(model)

    x = torch.randn(2, 3, 32, 32)

    # Forward trace
    with torch.no_grad():
        y = compiled(x)
        print(f"  Input:    {list(x.shape)}")
        print(f"  Output:   {list(y.shape)}")
        expected = list(model(x).shape)
        print(f"  Expected: {expected}")
        print(f"  Match:    {list(y.shape) == expected}")

    print(f"\n  Structured ops have meta kernel by construction")
    print(f"  -> torch.compile traces shapes without running actual kernels")
    print()


def exp_generated_code_structure():
    print("=" * 60)
    print("3. What torchgen generates for each op")
    print("=" * 60)

    print(f"  For op aten::add.Tensor, torchgen generates:")
    print(f"")
    print(f"  build/aten/src/ATen/Functions.cpp:")
    print(f"    Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {{")
    print(f"        return op_registry.call('aten::add', self, other, alpha);")
    print(f"    }}")
    print(f"")
    print(f"  build/aten/src/ATen/RegisterCPU.cpp:")
    print(f"    TORCH_LIBRARY_IMPL(aten, CPU, m) {{")
    print(f"        m.impl('add.Tensor', TORCH_FN(aten::native::add_kernel));")
    print(f"    }}")
    print(f"")
    print(f"  build/aten/src/ATen/Operators.cpp:")
    print(f"    OperatorHandle op_add = c10::Dispatcher::singleton()")
    print(f"        .registerSchema(FunctionSchema::parse('add.Tensor(...)'));")
    print(f"")
    print(f"  torch/csrc/autograd/generated/python_variable_methods.cpp:")
    print(f"    {'add', (PyCFunction)THPVariable_add, ...}")
    print()

    print(f"  To see actual generated code (in PyTorch repo):")
    print(f"    rg -n 'aten::add.Tensor' build/aten/src/ATen/RegisterCPU.cpp | head -5")
    print(f"    rg -n 'void add\(' build/aten/src/ATen/Functions.h | head -5")
    print()


EXPERIMENTS = {
    "structured": exp_structured_pattern,
    "compile": exp_compile_generated_pattern,
    "generated": exp_generated_code_structure,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[torchgen case 5] DONE")


if __name__ == "__main__":
    main()
