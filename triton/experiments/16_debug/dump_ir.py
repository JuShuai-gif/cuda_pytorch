"""Dump IR stages for a simple vector-add kernel.

Prereq: triton installed (see 12_编译与安装指南.md).
"""

import os

import torch

import triton
import triton.language as tl

DUMP_DIR = "/tmp/triton_dump"


@triton.jit
def add(x, y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offs, tl.load(x + offs) + 1, mask=offs < n)


def main():
    os.environ.setdefault("TRITON_DUMP_IR", "1")
    os.environ.setdefault("TRITON_DUMP_DIR", DUMP_DIR)
    x = torch.randn(1024, device="cuda")
    y = torch.empty_like(x)
    add[(1,)](x, y, 1024, BLOCK=1024)
    torch.cuda.synchronize()
    print("result ok:", torch.allclose(y, x + 1))
    if os.path.isdir(DUMP_DIR):
        print(f"\nfiles in {DUMP_DIR}:")
        for f in sorted(os.listdir(DUMP_DIR)):
            print("  ", f)


if __name__ == "__main__":
    main()
