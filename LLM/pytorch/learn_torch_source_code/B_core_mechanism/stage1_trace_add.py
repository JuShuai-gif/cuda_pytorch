"""
Stage 1 hands-on: trace torch.add from Python down to the C++ kernel.

Run under gdb to see the real call stack:
    conda activate torch_env
    gdb -q -batch -x trace_add.gdb --args python trace_add.py

The breakpoint at::TensorIteratorBase::build is hit 3 times here:
  hit 1, 2 -> torch.ones(4) for a and b (creation uses fill_ -> TensorIterator)
  hit 3    -> torch.add (this is the one we care about; trace_add.gdb skips
              the first two hits with `continue`)
"""

import torch

a = torch.ones(4)
b = torch.ones(4)
print("about to add")
c = torch.add(a, b)  # <-- the call we trace
print("result:", c.tolist())
