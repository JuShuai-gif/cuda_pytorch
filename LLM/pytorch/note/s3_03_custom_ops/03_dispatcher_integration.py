"""Custom Ops case study 3: dispatcher integration and TORCH_LIBRARY pattern.

Companion script for custom_ops/custom_ops.md. Covers:
  1. Dispatcher integration: verify custom op in global dispatch table
  2. TORCH_LIBRARY vs torch.library equivalences
  3. Performance benchmarking of custom op

Run:
    python 03_dispatcher_integration.py
"""

import sys
import time

import torch


def exp_dispatcher_verify():
    print("=" * 60)
    print("1. Verify custom op in global Dispatcher")
    print("=" * 60)

    lib = torch.library.Library("dispdemo", "DEF")
    lib.define("custom_elu(Tensor x, float alpha=1.0) -> Tensor")

    @torch.library.impl("dispdemo::custom_elu", "CPU")
    def custom_elu_cpu(x, alpha):
        return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

    if torch.cuda.is_available():
        @torch.library.impl("dispdemo::custom_elu", "CUDA")
        def custom_elu_cuda(x, alpha):
            return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

    # 1. Check op exists in torch.ops
    print(f"  Op handle: {torch.ops.dispdemo.custom_elu}")
    print(f"  Default overload: {torch.ops.dispdemo.custom_elu.default}")

    # 2. Dump dispatch table
    table = torch._C._dispatch_dump_table("dispdemo::custom_elu")
    print(f"\n  Dispatch table:")
    for line in table.strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")

    # 3. Check which dispatch keys have kernels
    has_cpu = torch._C._dispatch_has_kernel_for_dispatch_key("dispdemo::custom_elu", "CPU")
    has_cuda = torch._C._dispatch_has_kernel_for_dispatch_key("dispdemo::custom_elu", "CUDA")
    will_prefix = "  Has kernel:"
    print(f"\n{will_prefix} CPU={has_cpu}, CUDA={has_cuda}")

    # 4. Verify calling through dispatcher
    x = torch.randn(4, 8)
    y = torch.ops.dispdemo.custom_elu(x, 1.0)
    expected = torch.where(x > 0, x, torch.exp(x) - 1)
    print(f"  Functional correctness: {torch.allclose(y, expected)}")
    print()


def exp_torch_library_equivalences():
    print("=" * 60)
    print("2. Python torch.library vs C++ TORCH_LIBRARY")
    print("=" * 60)

    equivalences = [
        ("Define schema", "lib.define(schema)", 'm.def("name", schema)'),
        ("Register CPU", 'lib.impl(name, "CPU", fn)', "m.impl(name, kCPU, fn)"),
        ("Register CUDA", 'lib.impl(name, "CUDA", fn)', "m.impl(name, kCUDA, fn)"),
        ("Register Meta", 'lib.impl(name, "Meta", fn)', "m.impl(name, kMeta, fn)"),
        ("Register Autograd", 'lib.impl(name, "Autograd", fn)', "m.impl(name, kAutograd, fn)"),
        ("Call op", "torch.ops.myops.fn(args)", "at::fn(args) or torch::fn(args)"),
    ]

    for purpose, python_api, cpp_api in equivalences:
        print(f"  {purpose:20s}:")
        print(f"    Python: {python_api}")
        print(f"    C++:    {cpp_api}")

    print(f"\n  Key: Both APIs call Dispatcher::registerOp/registerKernel under the hood")
    print()


def exp_performance_benchmark():
    print("=" * 60)
    print("3. Custom op vs native op performance comparison")
    print("=" * 60)

    # The custom ELU vs torch.nn.functional.elu
    x = torch.randn(4096, 4096)

    n_warmup = 5
    n_iter = 50

    # Warmup
    for _ in range(n_warmup):
        torch.ops.dispdemo.custom_elu(x, 1.0)
        torch.nn.functional.elu(x, alpha=1.0)

    # Custom op
    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch.ops.dispdemo.custom_elu(x, 1.0)
    t_custom = (time.perf_counter() - t0) / n_iter

    # Native op
    t1 = time.perf_counter()
    for _ in range(n_iter):
        torch.nn.functional.elu(x, alpha=1.0)
    t_native = (time.perf_counter() - t1) / n_iter

    print(f"  Tensor size: {list(x.shape)} (float32)")
    print(f"  Custom ELU:  {t_custom*1000:.3f} ms")
    print(f"  Native ELU:  {t_native*1000:.3f} ms")
    if t_native > 0:
        print(f"  Ratio:       {t_custom / t_native:.2f}x")
        print(f"  -> Custom has ~{(t_custom / t_native - 1) * 100:.0f}% overhead from Python dispatch")

    if torch.cuda.is_available():
        x_cuda = torch.randn(4096, 4096, device="cuda")
        for _ in range(n_warmup):
            torch.ops.dispdemo.custom_elu(x_cuda, 1.0)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(n_iter):
            torch.ops.dispdemo.custom_elu(x_cuda, 1.0)
        torch.cuda.synchronize()
        t_custom_cuda = (time.perf_counter() - t2) / n_iter

        print(f"\n  CUDA custom ELU: {t_custom_cuda*1000:.3f} ms")
        print(f"  -> CUDA kernel is same speed as composing torch.where + exp internally")
    print()


EXPERIMENTS = {
    "dispatcher": exp_dispatcher_verify,
    "equivalence": exp_torch_library_equivalences,
    "benchmark": exp_performance_benchmark,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[custom_ops case 3] DONE")


if __name__ == "__main__":
    main()
