"""JIT cache experiment for Triton.

Prereq: triton installed (see 12_编译与安装指南.md).
"""

import os

import torch

import triton
import triton.language as tl


@triton.jit
def add(x, y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offs, tl.load(x + offs) + 1, mask=offs < n)


def main():
    x = torch.randn(1024, device="cuda")
    y = torch.empty_like(x)
    add[(1,)](x, y, 1024, BLOCK=1024)
    torch.cuda.synchronize()
    print("result ok:", torch.allclose(y, x + 1))

    cache = os.path.expanduser("~/.triton/cache")
    print("cache dir:", cache, "exists:", os.path.isdir(cache))
    if os.path.isdir(cache):
        for root, _dirs, files in os.walk(cache):
            for f in files:
                print("  ", os.path.join(root, f))

    # TRITON_ALWAYS_COMPILE forces recompilation
    print("\nset TRITON_ALWAYS_COMPILE=1 to force rebuild")


if __name__ == "__main__":
    main()
