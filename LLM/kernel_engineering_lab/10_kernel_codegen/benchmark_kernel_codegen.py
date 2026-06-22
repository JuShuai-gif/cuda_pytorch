#!/usr/bin/env python3
"""
Benchmark: Codegen-generated kernels vs PyTorch eager vs torch.compile.

Measures:
  - Generated fused kernel throughput vs PyTorch sequential ops
  - Generated kernels vs torch.compile (inductor)
  - Fused kernel vs separate kernel dispatch
  - Codegen overhead (time spent generating source + compiling)
  - Various tensor sizes

Run: python 10_kernel_codegen/benchmark_kernel_codegen.py
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
import triton

from ir import Graph, OpType
from triton_codegen import TritonCodeGenerator

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def _cuda_available() -> bool:
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.")
        return False
    return True


def _format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    else:
        return f"{seconds:.3f} s"


def benchmark_kernel(
    fn: Any,
    *args: Any,
    warmup: int = 10,
    repeat: int = 100,
    **kwargs: Any,
) -> tuple[float, float]:
    """Benchmark a function, returning (mean_time_sec, std_time_sec)."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mean = elapsed / repeat
    return mean, 0.0


def bench_elementwise_fusion() -> None:
    """Benchmark fused codegen kernels vs eager vs torch.compile."""
    if not _cuda_available():
        return

    print("=" * 80)
    print("  CODGEN: Elementwise Fusion Performance")
    print("=" * 80)

    sizes = [
        (1024,),
        (4096,),
        (16384,),
        (65536,),
        (262144,),
        (1048576,),
        (16, 1024),
        (64, 1024),
        (256, 1024),
        (128, 4096),
    ]

    results: list[list[str]] = []
    header = ["Shape", "PyTorch Eager", "torch.compile", "Codegen Fused", "Speedup vs Eager"]

    for shape in sizes:
        N = int(torch.tensor(shape).prod().item())

        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        y = torch.randn(shape, device="cuda", dtype=torch.float32)

        # --- PyTorch eager: add + relu ---
        def eager_fn():
            return F.relu(x + y)

        eager_t, _ = benchmark_kernel(eager_fn, warmup=5, repeat=50)

        # --- torch.compile ---
        compiled_fn = torch.compile(eager_fn, backend="inductor")
        compiled_fn()  # warmup compile
        compiled_t, _ = benchmark_kernel(compiled_fn, warmup=5, repeat=50)

        # --- Codegen fused ---
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        relu = g.add_node(OpType.RELU, [add], name="relu")
        g.outputs.append(relu)

        # Time codegen (source generation + compilation)
        t0 = time.perf_counter()
        cg = TritonCodeGenerator(block_size=1024)
        src = cg.generate_elementwise_fusion([add, relu], g)
        gen_out = cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape})
        torch.cuda.synchronize()
        codegen_compile_t = time.perf_counter() - t0

        def codegen_fn():
            out = cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape})
            torch.cuda.synchronize()
            return out

        # Re-measure without compile overhead
        codegen_t, _ = benchmark_kernel(
            lambda: cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape}),
            warmup=3,
            repeat=30,
        )

        speedup = eager_t / codegen_t if codegen_t > 0 else 0
        results.append(
            [
                str(shape),
                _format_time(eager_t),
                _format_time(compiled_t),
                _format_time(codegen_t),
                f"{speedup:.2f}x" if speedup >= 1 else f"{(1 / speedup):.2f}x slower",
            ]
        )

    if tabulate:
        print(tabulate(results, headers=header, tablefmt="grid", stralign="right"))
    else:
        print(f"{'Shape':<20} {'Eager':>12} {'Compile':>12} {'Codegen':>12} {'Speedup':>15}")
        for row in results:
            print(f"{row[0]:<20} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>15}")


def bench_reduction_kernels() -> None:
    """Benchmark generated reduction kernels vs PyTorch."""
    if not _cuda_available():
        return

    print("\n" + "=" * 80)
    print("  CODGEN: Reduction Kernel Performance")
    print("=" * 80)

    configs = [
        ("Softmax", OpType.SOFTMAX),
        ("LayerNorm", OpType.LAYERNORM),
        ("RMSNorm", OpType.RMSNORM),
    ]

    shapes = [(4, 256), (16, 512), (64, 1024), (128, 4096)]

    for op_name, op_type in configs:
        print(f"\n--- {op_name} ---")

        results: list[list[str]] = []
        for shape in shapes:
            x = torch.randn(shape, device="cuda", dtype=torch.float32)

            g = Graph()
            inp = g.add_node(OpType.CONSTANT, [], name="x")
            g.inputs.append(inp)
            node = g.add_node(op_type, [inp], name=op_type.value)
            g.outputs.append(node)

            cg = TritonCodeGenerator()
            src = cg.generate_reduction(node, g)

            def pytorch_fn():
                if op_type == OpType.SOFTMAX:
                    return F.softmax(x, dim=-1)
                elif op_type == OpType.LAYERNORM:
                    return F.layer_norm(x, [shape[-1]], weight=None, bias=None, eps=1e-5)
                else:
                    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
                    return x * rms

            py_t, _ = benchmark_kernel(pytorch_fn, warmup=5, repeat=30)

            def codegen_fn():
                return cg.compile_and_run(src, {"x": x}, {"result": shape})

            cg_t, _ = benchmark_kernel(codegen_fn, warmup=5, repeat=30)

            speedup = py_t / cg_t if cg_t > 0 else 0
            results.append(
                [
                    str(shape),
                    _format_time(py_t),
                    _format_time(cg_t),
                    f"{speedup:.2f}x" if speedup >= 1 else f"{(1 / speedup):.2f}x slower",
                ]
            )

        if tabulate:
            print(
                tabulate(
                    results,
                    headers=["Shape", "PyTorch", "Codegen", "Speedup"],
                    tablefmt="simple",
                    stralign="right",
                )
            )
        else:
            for row in results:
                print(f"  {row[0]:<16} Py:{row[1]:>10}  CG:{row[2]:>10}  {row[3]}")


def bench_fused_vs_separate() -> None:
    """Compare fused codegen kernel vs separate kernel dispatches."""
    if not _cuda_available():
        return

    print("\n" + "=" * 80)
    print("  CODGEN: Fused vs Separate Kernel Dispatch")
    print("=" * 80)

    shape = (262144,)
    x = torch.randn(shape, device="cuda")
    y = torch.randn(shape, device="cuda")

    # --- Separate kernels: 3 dispatches (add, mul, relu) ---
    g_sep = Graph()
    a = g_sep.add_node(OpType.CONSTANT, [], name="a")
    b = g_sep.add_node(OpType.CONSTANT, [], name="b")
    g_sep.inputs.extend([a, b])
    add_n = g_sep.add_node(OpType.ADD, [a, b], name="add_sep")
    g_sep.outputs.append(add_n)

    cg1 = TritonCodeGenerator()
    src1 = cg1.generate_elementwise_fusion([add_n], g_sep)

    g_sep2 = Graph()
    x2 = g_sep2.add_node(OpType.CONSTANT, [], name="tmp")
    y2 = g_sep2.add_node(OpType.CONSTANT, [], name="mul_in")
    g_sep2.inputs.extend([x2, y2])
    mul_n = g_sep2.add_node(OpType.MUL, [x2, y2], name="mul_sep")
    g_sep2.outputs.append(mul_n)

    cg2 = TritonCodeGenerator()
    src2 = cg2.generate_elementwise_fusion([mul_n], g_sep2)

    g_sep3 = Graph()
    z = g_sep3.add_node(OpType.CONSTANT, [], name="tmp2")
    g_sep3.inputs.append(z)
    relu_n = g_sep3.add_node(OpType.RELU, [z], name="relu_sep")
    g_sep3.outputs.append(relu_n)

    cg3 = TritonCodeGenerator()
    src3 = cg3.generate_elementwise_fusion([relu_n], g_sep3)

    def separate_dispatch():
        t1 = cg1.compile_and_run(src1, {"a": x, "b": y}, {"r1": shape})["r1"]
        t2 = cg2.compile_and_run(src2, {"tmp": t1, "mul_in": y}, {"r2": shape})["r2"]
        t3 = cg3.compile_and_run(src3, {"tmp2": t2}, {"r3": shape})["r3"]
        return t3

    sep_t, _ = benchmark_kernel(separate_dispatch, warmup=3, repeat=30)

    # --- Fused: single kernel ---
    g_fused = Graph()
    a_f = g_fused.add_node(OpType.CONSTANT, [], name="a")
    b_f = g_fused.add_node(OpType.CONSTANT, [], name="b")
    g_fused.inputs.extend([a_f, b_f])
    add_f = g_fused.add_node(OpType.ADD, [a_f, b_f], name="add")
    mul_f = g_fused.add_node(OpType.MUL, [add_f, b_f], name="mul")
    relu_f = g_fused.add_node(OpType.RELU, [mul_f], name="relu")
    g_fused.outputs.append(relu_f)

    cg_f = TritonCodeGenerator()
    src_f = cg_f.generate_elementwise_fusion([add_f, mul_f, relu_f], g_fused)

    def fused_dispatch():
        return cg_f.compile_and_run(src_f, {"a": x, "b": y}, {"result": shape})["result"]

    fused_t, _ = benchmark_kernel(fused_dispatch, warmup=3, repeat=30)

    ref = F.relu((x + y) * y)
    fused_out = fused_dispatch()
    assert torch.allclose(fused_out, ref, atol=1e-3)

    speedup = sep_t / fused_t if fused_t > 0 else 0
    print(f"  Shape:               {shape}")
    print(f"  Separate (3 kernels): {_format_time(sep_t)}")
    print(f"  Fused   (1 kernel):   {_format_time(fused_t)}")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  Global memory savings: 4 fewer global reads/writes")


def bench_codegen_overhead() -> None:
    """Measure time spent in code generation and compilation."""
    if not _cuda_available():
        return

    print("\n" + "=" * 80)
    print("  CODGEN: Code Generation Overhead")
    print("=" * 80)

    shape = (4096,)
    x = torch.randn(shape, device="cuda")
    y = torch.randn(shape, device="cuda")

    # Build graph once
    g = Graph()
    a = g.add_node(OpType.CONSTANT, [], name="a")
    b = g.add_node(OpType.CONSTANT, [], name="b")
    g.inputs.extend([a, b])
    add = g.add_node(OpType.ADD, [a, b], name="add")
    relu = g.add_node(OpType.RELU, [add], name="relu")
    g.outputs.append(relu)

    # Time source generation only
    t0 = time.perf_counter()
    cg = TritonCodeGenerator()
    src = cg.generate_elementwise_fusion([add, relu], g)
    gen_t = time.perf_counter() - t0
    print(f"  Source generation:   {_format_time(gen_t)}")

    # Time compilation (first run includes JIT)
    t0 = time.perf_counter()
    _ = cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape})
    torch.cuda.synchronize()
    compile_t = time.perf_counter() - t0
    print(f"  First call (JIT):    {_format_time(compile_t)}")

    # Time subsequent calls (cached)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape})
    torch.cuda.synchronize()
    cached_t = (time.perf_counter() - t0) / 100
    print(f"  Cached call (avg):   {_format_time(cached_t)}")

    print(f"\n  Source lines:        {len(src.splitlines())}")
    print(f"  Source chars:        {len(src)}")

    # Print the generated source for inspection
    print(f"\n  --- Generated kernel source ({cg._meta.kernel_name}) ---")
    print(src)


if __name__ == "__main__":
    bench_elementwise_fusion()
    bench_reduction_kernels()
    bench_fused_vs_separate()
    bench_codegen_overhead()
