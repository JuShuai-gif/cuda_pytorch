"""
Trace torch.empty(2,3) from Python down to C++ and back to a Python object.

Run:
    conda activate torch_env
    cd /home/ghr/code/cuda_pytorch/LLM/pytorch/note/s1_01_tensor
    gdb -q -batch -x trace_empty.gdb --args python trace_empty.py

See tensor.md section "十二、Python <-> C++ 调用链路" for the annotated stacks.
"""

import torch
import os; print(f"PID: {os.getpid()}")
print("=== about to call torch.empty(2, 3) ===")
t = torch.empty(2, 3)
print("result:", t.shape, t.dtype, t.data_ptr())
