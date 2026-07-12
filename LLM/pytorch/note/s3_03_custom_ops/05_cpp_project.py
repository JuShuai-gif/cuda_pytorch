"""Custom Ops case study 5: C++ extension project structure.

Companion script for custom_ops/custom_ops.md. Covers:
  1. Full C++/CUDA extension project layout
  2. setup.py + LoadExtension
  3. JIT compile vs pre-built

Run:
    python 05_cpp_project.py
"""

import sys

import torch


def exp_project_structure():
    print("=" * 60)
    print("1. Complete C++/CUDA extension project structure")
    print("=" * 60)

    project_tree = """
  my_extension/
    setup.py                 # Build script
    my_extension.cpp         # TORCH_LIBRARY registration
    my_extension_kernel.cu   # CUDA kernel implementation
    my_extension.h           # Header (optional)
  """

    print(project_tree)

    print(f"  setup.py:")
    print(f"  ```python")
    print(f"  from torch.utils.cpp_extension import CUDAExtension, BuildExtension")
    print(f"  from setuptools import setup")
    print(f"  ")
    print(f"  setup(")
    print(f"      name='my_extension',")
    print(f"      ext_modules=[")
    print(f"          CUDAExtension(")
    print(f"              name='my_extension._C',")
    print(f"              sources=['my_extension.cpp', 'my_extension_kernel.cu'],")
    print(f"          ),")
    print(f"      ],")
    print(f"      cmdclass={'build_ext': BuildExtension}")
    print(f"  )")
    print(f"  ```")
    print()


def exp_jit_compile():
    print("=" * 60)
    print("2. JIT compile vs pre-built extension")
    print("=" * 60)

    # JIT compile example (inline)
    cpp_source = """
    #include <torch/extension.h>
    torch::Tensor fast_scale(torch::Tensor x, float alpha) {
        return x * alpha;
    }

    TORCH_LIBRARY(jit_demo, m) {
        m.def("fast_scale(Tensor x, float alpha) -> Tensor");
        m.impl("fast_scale", torch::kCPU, &fast_scale);
    }
    """

    try:
        from torch.utils.cpp_extension import load_inline
        module = load_inline(
            name="jit_demo_ext",
            cpp_sources=cpp_source,
            functions=["fast_scale"],
            verbose=False,
        )

        x = torch.randn(3)
        y = torch.ops.jit_demo.fast_scale(x, 3.0)
        print(f"  JIT compile OK: {y.tolist()}")
        print(f"  -> load_inline compiles from Python at import time")
    except Exception as e:
        print(f"  JIT compile: {str(e)[:100]}")

    print(f"\n  JIT compile pros: quick prototyping, no setup.py")
    print(f"  JIT compile cons: slow first import, no caching")
    print(f"  Pre-built: fast import, reproducible, CICD friendly")
    print()


def exp_abi_compatibility():
    print("=" * 60)
    print("3. ABI compatibility and torch version")
    print("=" * 60)

    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA version:    {torch.version.cuda}")
    print(f"  C++ ABI flag:    {torch._C._GLIBCXX_USE_CXX11_ABI}")

    print(f"\n  ABI compatibility checklist:")
    print(f"    1. Match torch's C++ ABI (_GLIBCXX_USE_CXX11_ABI)")
    print(f"    2. Match CUDA toolkit version")
    print(f"    3. cpp_extension handles these via _get_build_environment()")
    print(f"    4. Pre-built extensions must match torch version and CUDA")
    print()


EXPERIMENTS = {
    "structure": exp_project_structure,
    "jit": exp_jit_compile,
    "abi": exp_abi_compatibility,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[custom_ops case 5] DONE")


if __name__ == "__main__":
    main()
