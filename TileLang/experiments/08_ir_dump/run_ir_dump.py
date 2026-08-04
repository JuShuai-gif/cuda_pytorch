"""Dump IR between passes for a simple vector-add kernel.

Prereq: tilelang installed (see 12_编译与安装指南.md).
"""

import os

import torch

import tilelang
import tilelang.language as T
from tilelang.transform import PassConfigKey

DUMP_DIR = "/tmp/dump_ir"


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
    k = add.compile(
        n=1024,
        pass_configs={
            PassConfigKey.TL_ENABLE_DUMP_IR: True,
            PassConfigKey.TL_DUMP_IR_DIR: DUMP_DIR,
        },
    )
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    out = k(a, b)
    torch.cuda.synchronize()
    print("result ok:", torch.allclose(out, a + b))
    files = sorted(os.listdir(DUMP_DIR))
    print(f"\n{len(files)} IR files in {DUMP_DIR}:")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
