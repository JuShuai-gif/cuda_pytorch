"""Debugging experiment: out-of-bounds access error handling.

Prereq: tilelang installed (see 12_编译与安装指南.md).
"""

import torch

import tilelang
import tilelang.language as T


@tilelang.jit
def bad(A, B):
    n = T.const("n")
    A: T.Tensor((n,), T.float32)
    B: T.Tensor((n,), T.float32)
    C = T.empty((n,), T.float32)
    with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
        for i in T.Parallel(256):
            C[bx * 256 + i + 1000000] = A[bx * 256 + i] + B[bx * 256 + i]
    return C


def main():
    k = bad.compile(n=1024)
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    try:
        out = k(a, b)
        torch.cuda.synchronize()
        print("no error? out[:3] =", out[:3])
    except Exception as e:
        print("ERROR:", type(e).__name__, e)
        print("(safe memory access may have inserted guards; see 16/23 docs)")


if __name__ == "__main__":
    main()
