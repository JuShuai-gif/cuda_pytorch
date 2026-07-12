"""
C++ / CUDA level debugging example.

Goal: attach gdb to this running Python process and break inside
PyTorch's C++/CUDA kernels while `torch.add` executes.

Why the input() pause: gdb must attach BEFORE the op runs, so the
script prints its PID and waits. See README.md for the exact steps.

Reliable breakpoints to set in gdb (all hit by torch.add):
  b at::TensorIteratorBase::build          # every elementwise op builds an iterator
  b at::native::structured_ufunc_add_CPU_out::impl   # CPU add (anon ns, use tab-complete)
  # scalar kernel lives at aten/src/ATen/native/ufunc/add.h:16  -> self + alpha * other
"""

import os

import torch


def main() -> None:
    print(f"PID = {os.getpid()}")
    print("Attach gdb now:  (gdb) Attach to Python  -> pick this PID")
    print("Set breakpoint:  b at::TensorIteratorBase::build   then  continue")
    input("Press ENTER here once gdb is attached and continued... ")

    a = torch.randn(1000)
    b = torch.randn(1000)

    # CPU add: breakpoint at::TensorIteratorBase::build fires here.
    c = a + b
    print("cpu add done, sample:", c[0].item())

    if torch.cuda.is_available():
        ag = a.cuda()
        bg = b.cuda()
        # CUDA add: dispatch routes to the CUDA add kernel.
        cg = ag + bg
        torch.cuda.synchronize()
        print("cuda add done, sample:", cg[0].item())


if __name__ == "__main__":
    main()
