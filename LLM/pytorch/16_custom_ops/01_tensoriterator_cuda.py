"""Custom Ops case study 1: TensorIterator + CUDA elementwise kernel.

Companion script for custom_ops/custom_ops.md. Covers:
  1. TensorIterator for elementwise ops
  2. CPU and CUDA kernel registration
  3. Broadcasting and type promotion

Run:
    python 01_tensoriterator_cuda.py
"""

import sys

import torch


def exp_tensoriterator_basics():
    print("=" * 60)
    print("1. TensorIterator usage pattern")
    print("=" * 60)

    # In PyTorch C++ code, TensorIterator is used like:
    print(f"  C++ pattern:")
    print(f"    auto iter = TensorIteratorConfig()")
    print(f"        .add_output(result)")
    print(f"        .add_input(a)")
    print(f"        .add_input(b)")
    print(f"        .build();")
    print(f"    ")
    print(f"    // CPU:")
    print(f"    cpu_kernel(iter, [](float a, float b) -> float {{ return a * b + a; }});")
    print(f"    ")
    print(f"    // CUDA:")
    print(f"    gpu_kernel(iter, []GPU_LAMBDA(float a, float b) -> float {{")
    print(f"        return a * b + a;")
    print(f"    }});")

    # In Python, torch.ops already uses TensorIterator internally
    # We can simulate similar functionality
    x = torch.randn(4, 8)
    y = torch.randn(1, 8)  # broadcastable

    # Elementwise ops handle broadcasting via TensorIterator
    z = x * y + x  # same as lambda: a * b + a
    print(f"\n  Python elementwise (uses TensorIterator internally):")
    print(f"    x shape: {list(x.shape)}")
    print(f"    y shape: {list(y.shape)} (broadcast)")
    print(f"    z shape: {list(z.shape)}")
    print(f"    Broadcasting handled automatically")
    print()


def exp_custom_op_demo():
    print("=" * 60)
    print("2. Register a custom elementwise op in Python")
    print("=" * 60)

    # Define namespace and op
    lib = torch.library.Library("tidi", "DEF")
    lib.define("scaled_add(Tensor a, Tensor b, float scale) -> Tensor")

    @torch.library.impl("tidi::scaled_add", "CPU")
    def scaled_add_cpu(a, b, scale):
        # Uses native ops which internally use TensorIterator
        return a + b * scale

    if torch.cuda.is_available():
        @torch.library.impl("tidi::scaled_add", "CUDA")
        def scaled_add_cuda(a, b, scale):
            return a + b * scale

    # Test
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    result = torch.ops.tidi.scaled_add(a, b, 2.0)
    print(f"  scaled_add(a, b, 2.0): shape={list(result.shape)}")
    print(f"  Correct: {torch.allclose(result, a + b * 2.0)}")

    # Works with broadcasting
    a = torch.randn(3, 4)
    b = torch.randn(4)  # broadcast
    result = torch.ops.tidi.scaled_add(a, b, 3.0)
    print(f"  Broadcasting test: OK, shape={list(result.shape)}")

    # Works on CUDA
    if torch.cuda.is_available():
        a_cuda = torch.randn(3, 4, device="cuda")
        b_cuda = torch.randn(3, 4, device="cuda")
        result_cuda = torch.ops.tidi.scaled_add(a_cuda, b_cuda, 1.5)
        print(f"  CUDA test: OK, device={result_cuda.device}")
    print()


def exp_cpp_extension_demo():
    print("=" * 60)
    print("3. cpp_extension: build custom CUDA kernel from Python")
    print("=" * 60)

    print(f"  Example setup.py for a custom CUDA extension:")
    print(f"  ```python")
    print(f"  from torch.utils.cpp_extension import CppExtension, BuildExtension")
    print(f"  from setuptools import setup")
    print(f"  ")
    print(f"  setup(")
    print(f"      name='my_cuda_ext',")
    print(f"      ext_modules=[")
    print(f"          CppExtension(")
    print(f"              'my_cuda_ext',")
    print(f"              ['my_kernel.cpp', 'my_kernel.cu'],")
    print(f"          ),")
    print(f"      ],")
    print(f"      cmdclass={'build_ext': BuildExtension},")
    print(f"  )")
    print(f"  ```")

    try:
        from torch.utils.cpp_extension import check_inline
        print(f"\n  Check C++ compiler availability:")
        result = check_inline(verbose=False)
        print(f"    Compiler available: {result}")
    except ImportError:
        print(f"\n  cpp_extension check: not available")

    print(f"\n  Inline compilation (for quick testing):")
    print(f"  ```python")
    print(f"  from torch.utils.cpp_extension import load_inline")
    print(f"  source = '''")
    print(f"  #include <torch/extension.h>")
    print(f"  torch::Tensor my_func(torch::Tensor x, float alpha) {")
    print(f"      return x * alpha;")
    print(f"  }")
    print(f"  '''")
    print(f"  module = load_inline(name='inline_test', cpp_sources=source)")
    print(f"  ```")
    print()


EXPERIMENTS = {
    "iter": exp_tensoriterator_basics,
    "custom": exp_custom_op_demo,
    "extension": exp_cpp_extension_demo,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[custom_ops case 1] DONE")


if __name__ == "__main__":
    main()
