#!/usr/bin/env python3
"""
Benchmark 自定义 CUDA kernel 与 PyTorch 内置算子的性能对比。

比较项:
  1. FlashAttention vs torch.nn.functional.scaled_dot_product_attention
  2. PagedAttention vs 手动参考实现
  3. RMSNorm CUDA vs PyTorch 手动 vs torch.compile
  4. LayerNorm CUDA vs torch.nn.LayerNorm vs torch.compile
  5. Fused residual+norm vs 顺序 add+norm
  6. SiLU/GELU CUDA vs torch.nn.functional.*
  7. SwiGLU CUDA vs 手动 gate * silu(up)
  8. Fused bias+activation vs 顺序 bias add + activation
  9. Online softmax vs torch.softmax
   10. Warp reduce vs naive reduce vs torch.sum
   11. Tiled Matmul vs torch.matmul vs torch.compile

运行: python 01_cuda_basics/benchmark_cuda_basics.py
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import torch

# 确保可以导入 benchmarks 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    benchmark_kernel,
    benchmark_torch,
    compare_kernels,
)

# ---------------------------------------------------------------------------
# 确保 CUDA 扩展已编译
# ---------------------------------------------------------------------------


def _ensure_kernels_built() -> bool:
    """构建 CUDA 扩展（如果尚未编译）。"""
    try:
        import cuda_kernels  # noqa: F401

        return True
    except ImportError:
        setup_py = Path(__file__).parent / "setup.py"
        result = subprocess.run(
            [sys.executable, str(setup_py), "build_ext", "--inplace"],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"扩展构建失败:\n{result.stderr}")
            return False
        return True


# ---------------------------------------------------------------------------
# 基准测试配置
# ---------------------------------------------------------------------------
CONFIG = BenchmarkConfig(warmup_steps=10, measure_steps=50, repeat=3)

# ---------------------------------------------------------------------------
# 常用 LLM 配置
# ---------------------------------------------------------------------------
ATTENTION_CONFIGS = [
    # (batch, n_heads, seq_len, head_dim)
    (2, 4, 64, 64),
    (1, 8, 128, 64),
    (2, 4, 128, 128),
    (1, 8, 256, 64),
    (1, 8, 512, 64),
]

NORM_CONFIGS = [
    # (rows, hidden_dim)
    (16, 768),
    (128, 768),
    (16, 4096),
    (128, 4096),
    (256, 4096),
]

ACTIVATION_SIZES = [
    1024,
    2**16,  # 64K
    2**20,  # 1M
    8 * 2**20,  # 8M
]

REDUCTION_SIZES = [
    2**10,  # 1K
    2**15,  # 32K
    2**20,  # 1M
    2**24,  # 16M
]


# ============================================================================
# FlashAttention Benchmark
# ============================================================================
def bench_flash_attention() -> None:
    """对比 FlashAttention 与 PyTorch scaled_dot_product_attention。"""
    import cuda_kernels

    for batch, n_heads, seq_len, head_dim in ATTENTION_CONFIGS:
        scale = 1.0 / math.sqrt(head_dim)
        shape = (batch, n_heads, seq_len, head_dim)

        Q = torch.randn(shape, device="cuda", dtype=torch.float16)
        K = torch.randn(shape, device="cuda", dtype=torch.float16)
        V = torch.randn(shape, device="cuda", dtype=torch.float16)
        O = torch.empty(shape, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  FlashAttention - batch={batch} heads={n_heads} seq={seq_len} d_head={head_dim}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda q, k, v, o, s: cuda_kernels.flash_attention_fwd(q, k, v, o, s, False),
                args=(Q, K, V, O, scale),
                name="flash_attention_fwd",
                config=CONFIG,
            )
        )

        results.append(
            benchmark_torch(
                torch.nn.functional.scaled_dot_product_attention,
                Q,
                K,
                V,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                name="torch.sdpa",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# PagedAttention Benchmark
# ============================================================================
def bench_paged_attention() -> None:
    """对比 PagedAttention 与手动 K/V 收集后做 attention。"""
    import cuda_kernels

    configs = [
        (4, 64, 16, 32),  # num_heads, head_dim, block_size, context_len
        (8, 64, 16, 128),
        (4, 128, 16, 128),
        (8, 64, 16, 256),
    ]

    for num_heads, head_dim, block_size, context_len in configs:
        scale = 1.0 / math.sqrt(head_dim)
        num_blocks = (context_len + block_size - 1) // block_size * 2

        K_cache, V_cache = cuda_kernels.allocate_kv_cache(
            num_blocks, block_size, num_heads, head_dim
        )
        K_cache.normal_()
        V_cache.normal_()

        # 使用连续的 block
        num_needed = (context_len + block_size - 1) // block_size
        block_table = list(range(num_needed))
        while len(block_table) < 4:
            block_table.append(-1)

        context_lens = torch.tensor([context_len], device="cuda", dtype=torch.int32)
        block_tables = torch.tensor([block_table], device="cuda", dtype=torch.int32)

        Q = torch.randn(num_heads, head_dim, device="cuda", dtype=torch.float16)
        O = torch.empty(num_heads, head_dim, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  PagedAttention - {num_heads}h {head_dim}d ctx_len={context_len}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda q, kc, vc, bt, cl, o, s: cuda_kernels.paged_attention(
                    q, kc, vc, bt, cl, o, s
                ),
                args=(Q, K_cache, V_cache, block_tables, context_lens, O, scale),
                name="paged_attention",
                config=CONFIG,
            )
        )

        # 手动参考：收集 K/V 后做 attention
        # 收集实际用到的 K/V
        K_ref_list = []
        V_ref_list = []
        collected = 0
        for bi in block_table:
            if bi < 0:
                break
            take = min(block_size, context_len - collected)
            K_ref_list.append(K_cache[bi, :take].reshape(-1, num_heads, head_dim))
            V_ref_list.append(V_cache[bi, :take].reshape(-1, num_heads, head_dim))
            collected += take

        K_ref = torch.cat(K_ref_list, dim=0)
        V_ref = torch.cat(V_ref_list, dim=0)

        def reference_paged_attn(q, k, v, scale):
            # q: [nh, hd], k: [ctx, nh, hd], v: [ctx, nh, hd]
            q_f = q.float().unsqueeze(0)  # [1, nh, hd]
            k_f = k.float().permute(1, 2, 0)  # [nh, hd, ctx]
            v_f = v.float().permute(1, 0, 2)  # [nh, ctx, hd]
            scores = torch.bmm(q_f, k_f) * scale  # [1, nh, ctx]
            attn_w = torch.softmax(scores, dim=-1)
            out_ref = torch.bmm(attn_w, v_f).squeeze(0)  # [nh, hd]
            return out_ref.to(torch.float16)

        results.append(
            benchmark_kernel(
                fn=lambda q, k, v, s: reference_paged_attn(q, k, v, s),
                args=(Q, K_ref, V_ref, scale),
                name="manual_reference",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# RMSNorm Benchmark
# ============================================================================
def bench_rmsnorm() -> None:
    """对比 RMSNorm CUDA vs PyTorch 手动实现 vs torch.compile。"""
    import cuda_kernels

    for rows, hidden_dim in NORM_CONFIGS:
        eps = 1e-5
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  RMSNorm - rows={rows} hidden_dim={hidden_dim}")
        print(f"{'=' * 60}")

        results = []

        # CUDA kernel
        results.append(
            benchmark_kernel(
                fn=lambda x_, w_, o_: cuda_kernels.rmsnorm_fwd(x_, w_, o_, eps),
                args=(x, weight, out),
                name="cuda_rmsnorm",
                config=CONFIG,
            )
        )

        # PyTorch 手动参考
        def manual_rmsnorm(x_, w_, eps_):
            xf = x_.float()
            wf = w_.float()
            rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps_)
            return (xf / rms * wf).to(x_.dtype)

        results.append(
            benchmark_kernel(
                fn=lambda x_, w_: manual_rmsnorm(x_, w_, eps),
                args=(x, weight),
                name="torch_manual_rmsnorm",
                config=CONFIG,
            )
        )

        # torch.compile 版本
        compiled_rmsnorm = torch.compile(manual_rmsnorm, dynamic=False)

        # warmup torch.compile
        for _ in range(5):
            compiled_rmsnorm(x, weight, eps)
        torch.cuda.synchronize()

        results.append(
            benchmark_kernel(
                fn=lambda x_, w_: compiled_rmsnorm(x_, w_, eps),
                args=(x, weight),
                name="torch_compile_rmsnorm",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# LayerNorm Benchmark
# ============================================================================
def bench_layernorm() -> None:
    """对比 LayerNorm CUDA vs torch.nn.LayerNorm vs torch.compile。"""
    import cuda_kernels

    for rows, hidden_dim in NORM_CONFIGS:
        eps = 1e-5
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  LayerNorm - rows={rows} hidden_dim={hidden_dim}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, w_, b_, o_: cuda_kernels.layernorm_fwd(x_, w_, b_, o_, eps),
                args=(x, weight, bias, out),
                name="cuda_layernorm",
                config=CONFIG,
            )
        )

        # torch.nn.LayerNorm
        ln = torch.nn.LayerNorm(hidden_dim, eps=eps, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            ln.weight.copy_(weight.float().to(torch.float16))
            ln.bias.copy_(bias.float().to(torch.float16))

        results.append(
            benchmark_torch(
                ln,
                x,
                name="torch.nn.LayerNorm",
                config=CONFIG,
            )
        )

        # torch.compile LayerNorm
        ln_compiled = torch.compile(ln, dynamic=False)
        for _ in range(5):
            ln_compiled(x)
        torch.cuda.synchronize()

        results.append(
            benchmark_torch(
                ln_compiled,
                x,
                name="torch_compile_LayerNorm",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# Fused Residual + Norm Benchmark
# ============================================================================
def bench_fused_residual_norm() -> None:
    """对比融合残差+LayerNorm vs 顺序 add+LayerNorm。"""
    import cuda_kernels

    for rows, hidden_dim in [
        (128, 768),
        (256, 768),
        (128, 4096),
        (256, 4096),
    ]:
        eps = 1e-5
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        residual = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  Fused Residual + LayerNorm - rows={rows} hidden_dim={hidden_dim}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, r_, w_, b_, o_: cuda_kernels.fused_residual_layernorm(
                    x_, r_, w_, b_, o_, eps
                ),
                args=(x, residual, weight, bias, out),
                name="fused_residual_layernorm",
                config=CONFIG,
            )
        )

        ln = torch.nn.LayerNorm(hidden_dim, eps=eps, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            ln.weight.copy_(weight.float().to(torch.float16))
            ln.bias.copy_(bias.float().to(torch.float16))

        def sequential_add_ln(x_, r_):
            y = x_ + r_
            return ln(y)

        results.append(
            benchmark_kernel(
                fn=lambda x_, r_: sequential_add_ln(x_, r_),
                args=(x, residual),
                name="sequential_add+LayerNorm",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# Activation Benchmark
# ============================================================================
def bench_activations() -> None:
    """对比 SiLU/GELU/SwiGLU CUDA vs PyTorch 参考实现。"""
    import cuda_kernels

    for n in ACTIVATION_SIZES:
        # SiLU
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.empty(n, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  SiLU - n={n:,}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, o_: cuda_kernels.silu_fwd(x_, o_),
                args=(x, out),
                name="cuda_silu",
                config=CONFIG,
            )
        )

        results.append(
            benchmark_torch(
                torch.nn.functional.silu,
                x.float(),
                name="torch.silu",
                config=CONFIG,
            )
        )

        compare_kernels(results)

    # GELU
    for n in ACTIVATION_SIZES:
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.empty(n, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  GELU - n={n:,}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, o_: cuda_kernels.gelu_fwd(x_, o_),
                args=(x, out),
                name="cuda_gelu",
                config=CONFIG,
            )
        )

        results.append(
            benchmark_torch(
                torch.nn.functional.gelu,
                x.float(),
                approximate="tanh",
                name="torch.gelu",
                config=CONFIG,
            )
        )

        compare_kernels(results)

    # SwiGLU
    for n in ACTIVATION_SIZES:
        gate = torch.randn(n, device="cuda", dtype=torch.float16)
        up = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.empty(n, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  SwiGLU - n={n:,}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda g_, u_, o_: cuda_kernels.swiglu_fwd(g_, u_, o_),
                args=(gate, up, out),
                name="cuda_swiglu",
                config=CONFIG,
            )
        )

        def manual_swiglu(g_, u_):
            return (g_.float() * torch.nn.functional.silu(u_.float())).to(torch.float16)

        results.append(
            benchmark_kernel(
                fn=lambda g_, u_: manual_swiglu(g_, u_),
                args=(gate, up),
                name="torch_swiglu",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# Fused Bias + Activation Benchmark
# ============================================================================
def bench_fused_bias_activation() -> None:
    """对比融合 bias+activation vs 顺序计算。"""
    import cuda_kernels

    configs = [
        (128, 768),
        (128, 4096),
        (512, 768),
        (512, 4096),
    ]

    for rows, hidden_dim in configs:
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  Fused Bias+ReLU - rows={rows} hidden_dim={hidden_dim}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, b_, o_: cuda_kernels.fused_bias_relu(x_, b_, o_),
                args=(x, bias, out),
                name="fused_bias_relu",
                config=CONFIG,
            )
        )

        def sequential_bias_relu(x_, b_):
            return torch.nn.functional.relu(x_.float() + b_.float()).to(torch.float16)

        results.append(
            benchmark_kernel(
                fn=lambda x_, b_: sequential_bias_relu(x_, b_),
                args=(x, bias),
                name="sequential_bias+relu",
                config=CONFIG,
            )
        )

        compare_kernels(results)

    # Fused bias + GELU
    for rows, hidden_dim in configs:
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  Fused Bias+GELU - rows={rows} hidden_dim={hidden_dim}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, b_, o_: cuda_kernels.fused_bias_gelu(x_, b_, o_),
                args=(x, bias, out),
                name="fused_bias_gelu",
                config=CONFIG,
            )
        )

        def sequential_bias_gelu(x_, b_):
            return torch.nn.functional.gelu(x_.float() + b_.float(), approximate="tanh").to(
                torch.float16
            )

        results.append(
            benchmark_kernel(
                fn=lambda x_, b_: sequential_bias_gelu(x_, b_),
                args=(x, bias),
                name="sequential_bias+gelu",
                config=CONFIG,
            )
        )

        compare_kernels(results)

    # Fused bias + SiLU
    for rows, hidden_dim in configs:
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  Fused Bias+SiLU - rows={rows} hidden_dim={hidden_dim}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, b_, o_: cuda_kernels.fused_bias_silu(x_, b_, o_),
                args=(x, bias, out),
                name="fused_bias_silu",
                config=CONFIG,
            )
        )

        def sequential_bias_silu(x_, b_):
            return torch.nn.functional.silu(x_.float() + b_.float()).to(torch.float16)

        results.append(
            benchmark_kernel(
                fn=lambda x_, b_: sequential_bias_silu(x_, b_),
                args=(x, bias),
                name="sequential_bias+silu",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# Online Softmax Benchmark
# ============================================================================
def bench_softmax() -> None:
    """对比 online softmax vs torch.softmax。"""
    import cuda_kernels

    configs = [
        (16, 128),
        (128, 128),
        (16, 512),
        (128, 512),
        (16, 4096),
    ]

    for rows, cols in configs:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16) * 0.5
        out = torch.empty(rows, cols, device="cuda", dtype=torch.float16)

        print(f"\n{'=' * 60}")
        print(f"  Online Softmax - rows={rows} cols={cols}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, o_: cuda_kernels.online_softmax(x_, o_),
                args=(x, out),
                name="online_softmax",
                config=CONFIG,
            )
        )

        results.append(
            benchmark_torch(
                torch.softmax,
                x.float(),
                dim=-1,
                name="torch.softmax",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# Reduction Benchmark
# ============================================================================
def bench_reduction() -> None:
    """对比 warp reduce / naive reduce / torch.sum。"""
    import cuda_kernels

    for n in REDUCTION_SIZES:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        print(f"\n{'=' * 60}")
        print(f"  Reduction - n={n:,}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_: cuda_kernels.warp_reduce_sum(x_),
                args=(x,),
                name="warp_reduce_sum",
                config=CONFIG,
            )
        )

        results.append(
            benchmark_kernel(
                fn=lambda x_: cuda_kernels.naive_reduce_sum(x_),
                args=(x,),
                name="naive_reduce_sum",
                config=CONFIG,
            )
        )

        results.append(
            benchmark_kernel(
                fn=lambda x_: cuda_kernels.full_warp_reduction(x_),
                args=(x,),
                name="full_warp_reduction",
                config=CONFIG,
            )
        )

        results.append(
            benchmark_torch(
                torch.sum,
                x,
                name="torch.sum",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# Vector Add Benchmark
# ============================================================================
def bench_vector_add() -> None:
    """对比 CUDA vector_add vs torch.add。"""
    import cuda_kernels

    sizes = [2**10, 2**15, 2**20, 2**24]

    for n in sizes:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        print(f"\n{'=' * 60}")
        print(f"  Vector Add - n={n:,}")
        print(f"{'=' * 60}")

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x_, y_: cuda_kernels.vector_add(x_, y_),
                args=(a, b),
                name="cuda_vector_add",
                config=CONFIG,
            )
        )

        results.append(
            benchmark_torch(
                torch.add,
                a,
                b,
                name="torch.add",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# Matmul Benchmark — CUDA tiled vs torch.matmul vs torch.compile
# ============================================================================
# LLM 推理中矩阵乘法的典型场景：
#   - 单 token 解码（M=1）：batch=1, seq_len=1 的自回归投影
#   - Prefill（M=64..2048）：批量预填充的投影和 FFN
#   - FFN 上投影（N = hidden_dim * 8/3 * 2）：Gate + Up 的扩展维度
#   - FFN 下投影（K = 大，N = hidden_dim）：降回 hidden_dim
MATMUL_CONFIGS = [
    # (M, N, K, description)
    (1, 4096, 4096, "decode projection: Q/K/V/O"),
    (64, 4096, 4096, "small prefill projection"),
    (128, 8192, 4096, "prefill FFN gate"),
    (128, 4096, 14336, "LLaMA-3 FFN up"),
    (128, 14336, 4096, "LLaMA-3 FFN down"),
    (1024, 4096, 4096, "medium prefill QKV"),
    (1, 4096, 8192, "decode projection (large head)"),
    (2048, 4096, 4096, "large prefill attention out"),
]


def bench_matmul() -> None:
    """对比 CUDA tiled matmul vs torch.matmul vs torch.compile。"""
    import cuda_kernels

    for M, N, K, desc in MATMUL_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"  Tiled Matmul - M={M} N={N} K={K}  ({desc})")
        print(f"{'=' * 60}")

        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C_cuda = torch.empty(M, N, device="cuda", dtype=torch.float16)

        results = []

        # CUDA tiled matmul
        results.append(
            benchmark_kernel(
                fn=lambda a_, b_, c_: cuda_kernels.tiled_matmul(a_, b_, c_),
                args=(A, B, C_cuda),
                name="cuda_tiled_matmul",
                config=CONFIG,
            )
        )

        # torch.matmul（cuBLAS 后端）
        results.append(
            benchmark_kernel(
                fn=lambda a_, b_: torch.matmul(a_.float(), b_.float()).to(torch.float16),
                args=(A, B),
                name="torch.matmul",
                config=CONFIG,
            )
        )

        # torch.compile matmul —— 使用 torch.compile 加速
        def compiled_matmul_fn(a_, b_):
            return torch.matmul(a_.float(), b_.float()).to(torch.float16)

        compiled_fn = torch.compile(compiled_matmul_fn, dynamic=False)
        for _ in range(5):
            compiled_fn(A, B)
        torch.cuda.synchronize()

        results.append(
            benchmark_kernel(
                fn=lambda a_, b_: compiled_fn(a_, b_),
                args=(A, B),
                name="torch.compile_matmul",
                config=CONFIG,
            )
        )

        compare_kernels(results)


# ============================================================================
# 主函数
# ============================================================================
def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA 不可用，退出。")
        sys.exit(1)

    if not _ensure_kernels_built():
        print("CUDA 扩展构建失败，退出。")
        sys.exit(1)

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print()

    print("=" * 60)
    print("  1. FlashAttention Benchmark")
    print("=" * 60)
    bench_flash_attention()

    print("\n" + "=" * 60)
    print("  2. PagedAttention Benchmark")
    print("=" * 60)
    bench_paged_attention()

    print("\n" + "=" * 60)
    print("  3. RMSNorm Benchmark")
    print("=" * 60)
    bench_rmsnorm()

    print("\n" + "=" * 60)
    print("  4. LayerNorm Benchmark")
    print("=" * 60)
    bench_layernorm()

    print("\n" + "=" * 60)
    print("  5. Fused Residual + Norm Benchmark")
    print("=" * 60)
    bench_fused_residual_norm()

    print("\n" + "=" * 60)
    print("  6. Activation Benchmark (SiLU / GELU / SwiGLU)")
    print("=" * 60)
    bench_activations()

    print("\n" + "=" * 60)
    print("  7. Fused Bias + Activation Benchmark")
    print("=" * 60)
    bench_fused_bias_activation()

    print("\n" + "=" * 60)
    print("  8. Online Softmax Benchmark")
    print("=" * 60)
    bench_softmax()

    print("\n" + "=" * 60)
    print("  9. Reduction Benchmark")
    print("=" * 60)
    bench_reduction()

    print("\n" + "=" * 60)
    print("  10. Vector Add Benchmark")
    print("=" * 60)
    bench_vector_add()

    print("\n" + "=" * 60)
    print("  11. Tiled Matmul Benchmark")
    print("=" * 60)
    bench_matmul()

    print("\n" + "=" * 60)
    print("  All benchmarks completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
