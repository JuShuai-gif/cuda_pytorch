"""01_vector_add: entry-level vector add kernel.

Prereq: tilelang installed (see 12_编译与安装指南.md).
"""

import torch

import tilelang
import tilelang.language as T


@tilelang.jit
def vector_add(A, B):
    n = T.const("n")
    A: T.Tensor((n,), T.float32)
    B: T.Tensor((n,), T.float32)
    C = T.empty((n,), T.float32)
    with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
        for i in T.Parallel(256):
            C[bx * 256 + i] = A[bx * 256 + i] + B[bx * 256 + i]
    return C


def main():
    n = 1 << 20
    k = vector_add.compile(n=n)
    print("kernel type:", k.get_kernel_type())
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = k(a, b)
    torch.cuda.synchronize()
    print("result ok:", torch.allclose(c, a + b))

    # timing vs torch
    from tilelang.profiler import do_bench

    lat_tl = do_bench(lambda: k(a, b), warmup=25, rep=100)
    lat_torch = do_bench(lambda: a + b, warmup=25, rep=100)
    gbytes = 3 * n * 4 / 1e9
    print(f"tilelang: {lat_tl * 1e6:.1f} us, {gbytes / (lat_tl * 1e-3):.1f} GB/s")
    print(f"torch:    {lat_torch * 1e6:.1f} us, {gbytes / (lat_torch * 1e-3):.1f} GB/s")


if __name__ == "__main__":
    main()
