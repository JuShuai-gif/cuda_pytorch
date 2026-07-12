"""
Trace t.view(): a view shares the SAME Storage, only builds a new TensorImpl
with different sizes/strides/storage_offset.

Run (plain, shows shared storage):
    python trace_view.py
Run (with gdb, shows the C++ call chain):
    gdb -q -batch -x trace_view.gdb --args python trace_view.py

See tensor.md section "十三、view 视图链路" for the annotated stacks.
"""

import torch

base = torch.arange(12.0)  # 1D, contiguous
print("=== about to call base.view(3, 4) ===")
v = base.view(3, 4)  # zero-copy view

# Evidence that the view shares the base's Storage (no data copy):
print("base.data_ptr :", base.data_ptr())
print("v.data_ptr    :", v.data_ptr(), "(== base: same buffer)")
print(
    "same storage  :",
    base.untyped_storage().data_ptr() == v.untyped_storage().data_ptr(),
)
print("base.shape/stride:", tuple(base.shape), base.stride())
print("v.shape/stride   :", tuple(v.shape), v.stride())

# Mutating the view mutates the base (shared memory):
v[0, 0] = 999.0
print("after v[0,0]=999 -> base[0] =", base[0].item())
