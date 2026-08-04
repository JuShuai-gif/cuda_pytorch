"""JIT cache experiment: verify cache key, kernel source, rebuild.

Prereq: tilelang installed (see 12_编译与安装指南.md).
"""

import os

import torch

import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B):
    n = T.const("n")
    A: T.Tensor((n,), T.float32)
    B: T.Tensor((n,), T.float32)
    C = T.empty((n,), T.float32)
    with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
        for i in T.Parallel(256):
            C[bx * 256 + i] = A[bx * 256 + i] + B[bx * 256 + i]
    return C


def main():
    k = add.compile(n=1024)
    print("kernel type:", k.get_kernel_type())
    src = k.get_kernel_source()
    print("kernel source head:\n", src[:400])

    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    out = k(a, b)
    torch.cuda.synchronize()
    print("result ok:", torch.allclose(out, a + b))

    cache_root = os.path.expanduser("~/.tilelang/cache")
    print("\ncache dir:", cache_root, "exists:", os.path.isdir(cache_root))
    if os.path.isdir(cache_root):
        for root, dirs, _files in os.walk(cache_root):
            for d in dirs:
                print("  ", os.path.join(root, d))

    # 强制重编译
    k2 = add.compile(n=1024, rebuild=True)
    print("\nrebuild done, same type:", k2.get_kernel_type())


if __name__ == "__main__":
    main()
