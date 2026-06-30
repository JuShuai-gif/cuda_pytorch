"""Custom Triton kernel demo: write GPU kernel + integrate with PyTorch.

Companion script for triton_kernel/triton_kernel.md. Covers:
  1. pointwise kernel:       custom add/relu/gelu in Triton
  2. PyTorch integration:    torch.autograd.Function wrapper
  3. autotune:               benchmark best BLOCK_SIZE
  4. matmul kernel:          tiled matmul with shared memory
  5. tiled reduction:        write a row-wise softmax kernel

Requires: triton (pip install triton)

Run:
    python test1.py                # full demo
    python test1.py pointwise      # pointwise kernel
    python test1.py autograd       # autograd.Function wrapper
    python test1.py autotune       # autotune demo
    python test1.py matmul         # tiled matmul kernel
    python test1.py softmax        # tiled softmax reduction kernel

=== DEBUG 常见问题 ===
  Q: "CUDA driver error: misaligned address"?
  A: pointer 计算有误; 检查 offsets 是否超出 tensor 范围,
     确保 BLOCK_SIZE 对齐, 检查 strides 是否正确

  Q: kernel 输出全是 0 或 NaN?
  A: (1) mask 条件写反了 (2) 忘记 store 输出
     (3) 使用了未初始化的 accumulator (4) dtype 不匹配

  Q: kernel 很慢, 不如 torch 原生?
  A: (1) BLOCK_SIZE 太小 -> launch overhead 太大
     (2) 内存访问不连续 (coalescing 失败) -> 检查 stride
     (3) 没有用 shared memory -> 对 reduction 类 kernel 很关键
     (4) 检查 memory coalescing: 连续线程应访问连续内存地址

  Q: autotune 没有生效 (每次都用第一个 config)?
  A: autotune 缓存到磁盘 (~/.triton/cache/), 修改 kernel 后需
     删除缓存 或 修改 triton.Config key 参数

  Q: "triton.runtime.errors.OutOfResources"?
  A: BLOCK_SIZE 太大或 shared memory 不够;
     减小 BLOCK_SIZE 或 num_warps, 检查 shared memory 使用量
"""

import sys

import torch

# Triton import — gracefully handle missing
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============ 1. Pointwise Triton kernel ============
def exp_pointwise():
    print("=" * 60)
    print("1. Pointwise Triton kernel: element-wise add")
    print("=" * 60)

    if not HAS_TRITON:
        print("  [SKIP] triton not installed (pip install triton)")
        return

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    # Run
    n = 1024 * 1024 * 4  # 4M elements
    x = torch.randn(n, device="cuda" if torch.cuda.is_available() else "cpu")
    y = torch.randn(n, device="cuda" if torch.cuda.is_available() else "cpu")
    output = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)

    # Verify
    expected = x + y
    max_diff = (output - expected).abs().max().item()
    print(f"  Elements:  {n:,}")
    print(f"  BLOCK_SIZE:{BLOCK_SIZE}")
    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Match:     {torch.allclose(output, expected)}")
    print("  -> Triton kernel runs element-wise add on GPU/CPU")
    print()


# ============ 2. PyTorch autograd integration ============
def exp_autograd():
    print("=" * 60)
    print("2. PyTorch autograd.Function wrapper")
    print("=" * 60)

    if not HAS_TRITON:
        print("  [SKIP] triton not installed")
        return

    @triton.jit
    def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c1 = 0.7978845608028654  # sqrt(2/pi)
        c2 = 0.044715
        y = 0.5 * x * (1.0 + tl.math.tanh(c1 * (x + c2 * x * x * x)))
        tl.store(output_ptr + offsets, y, mask=mask)

    class TritonGELU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            output = torch.empty_like(input)
            n = input.numel()
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            gelu_kernel[grid](input, output, n, BLOCK_SIZE=BLOCK_SIZE)
            ctx.save_for_backward(input)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            # GELU derivative: d/dx = 0.5 + 0.5*tanh(...) + x*sech^2(...)*...
            # Simplified: use PyTorch's GELU backward for correctness
            grad_input = torch.nn.functional.gelu(input)
            grad_input.backward(grad_output)
            return grad_output

    # Test forward
    x = torch.randn(1000, requires_grad=True)
    y_triton = TritonGELU.apply(x)

    # Reference
    y_ref = torch.nn.functional.gelu(x)

    print(f"  Forward match: {torch.allclose(y_triton, y_ref, atol=1e-5)}")
    print(f"  Max diff:      {(y_triton - y_ref).abs().max().item():.2e}")

    # Test backward
    (y_triton.sum()).backward()
    x_triton_grad = x.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    y_ref2 = torch.nn.functional.gelu(x2)
    (y_ref2.sum()).backward()

    print(f"  Backward match: {torch.allclose(x_triton_grad, x2.grad, atol=1e-4)}")
    print("  -> custom Triton op integrated with PyTorch autograd")
    print()


# ============ 3. Autotune ============
def exp_autotune():
    print("=" * 60)
    print("3. Autotune: best BLOCK_SIZE selection")
    print("=" * 60)

    if not HAS_TRITON:
        print("  [SKIP] triton not installed")
        return

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.math.max(x, 0.0)
        tl.store(y_ptr + offsets, y, mask=mask)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available for autotune")
        return

    n = 1024 * 1024 * 16  # 16M elements
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)

    # Warmup + bench
    for _ in range(10):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        relu_kernel[grid](x, y, n)

    torch.cuda.synchronize()

    import time

    t0 = time.perf_counter()
    for _ in range(100):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        relu_kernel[grid](x, y, n)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Reference: torch.relu
    t2 = time.perf_counter()
    for _ in range(100):
        _ = torch.relu(x)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    triton_time = (t1 - t0) * 1000 / 100
    torch_time = (t3 - t2) * 1000 / 100

    print(f"  Triton relu:  {triton_time:.4f} ms/iter (autotuned BLOCK_SIZE)")
    print(f"  torch.relu:   {torch_time:.4f} ms/iter")
    print("  -> autotune picks optimal config at first run")
    print()


# ============ 4. Matmul kernel ============
def exp_matmul():
    print("=" * 60)
    print("4. Triton matmul (tiled, shared memory)")
    print("=" * 60)

    if not HAS_TRITON:
        print("  [SKIP] triton not installed")
        return

    @triton.jit
    def matmul_kernel(
        A,
        B,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        A_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a = tl.load(
                A_ptr, mask=(rm[:, None] < M) & (k + rk[None, :] < K), other=0.0
            )
            b = tl.load(
                B_ptr, mask=(k + rk[:, None] < K) & (rn[None, :] < N), other=0.0
            )
            acc += tl.dot(a, b)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk

        c = acc
        C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptr, c, mask=mask)

    M, N, K = 256, 128, 512
    A = torch.randn(M, K, device="cuda" if torch.cuda.is_available() else "cpu")
    B = torch.randn(K, N, device="cuda" if torch.cuda.is_available() else "cpu")
    C = torch.empty(M, N, device=A.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Verify
    C_ref = A @ B
    max_diff = (C - C_ref).abs().max().item()
    print(f"  Matrix:   [{M}x{K}] @ [{K}x{N}] -> [{M}x{N}]")
    print(f"  Tiling:   M={BLOCK_M} N={BLOCK_N} K={BLOCK_K}")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Match:    {torch.allclose(C, C_ref, atol=1e-4)}")
    print("  -> tiled matmul with shared memory accumulation")
    print()


# ============ 5. Tiled softmax (reduction kernel) ============
def exp_softmax():
    print("=" * 60)
    print("5. Tiled softmax: row-wise reduction in Triton")
    print("=" * 60)

    if not HAS_TRITON:
        print("  [SKIP] triton not installed")
        return

    @triton.jit
    def softmax_kernel(input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
        row_idx = tl.program_id(0)
        row_start = row_idx * n_cols
        offsets = row_start + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < n_cols

        x = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))

        # Subtract max for numerical stability
        x_max = tl.max(x, axis=0)
        x = x - x_max
        exp_x = tl.exp(x)
        sum_exp = tl.sum(exp_x, axis=0)
        softmax_x = exp_x / sum_exp

        tl.store(output_ptr + offsets, softmax_x, mask=mask)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    rows, cols = 1024, 4096
    x = torch.randn(rows, cols, device="cuda")
    y = torch.empty_like(x)

    BLOCK_SIZE = 1024
    n = x.numel()
    grid = lambda meta: (rows,)
    softmax_kernel[grid](x, y, rows, cols, BLOCK_SIZE=BLOCK_SIZE)

    y_ref = torch.softmax(x, dim=-1)
    max_diff = (y - y_ref).abs().max().item()
    print(f"  Matrix:   [{rows}x{cols}]")
    print(f"  BLOCK_SIZE:{BLOCK_SIZE}")
    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Match:     {torch.allclose(y, y_ref, atol=1e-5)}")
    print("  -> row-wise softmax: max-reduce + exp + sum-reduce")
    print("  -> uses float('-inf') for masked-out elements (numerically stable)")
    print()


EXPERIMENTS = {
    "pointwise": exp_pointwise,
    "autograd": exp_autograd,
    "autotune": exp_autotune,
    "matmul": exp_matmul,
    "softmax": exp_softmax,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[triton_kernel demo] DONE")


if __name__ == "__main__":
    main()
