# gdb script for Stage 1: trace torch.add
# Usage: gdb -q -batch -x stage1_trace_add.gdb --args python stage1_trace_add.py
set pagination off
set breakpoint pending on

# TensorIteratorBase::build is hit by every elementwise op.
break at::TensorIteratorBase::build
run

# Skip the two hits from a = ones(4) and b = ones(4).
continue
continue

echo \n===== torch.add call stack (Python -> C++ kernel) =====\n
# Show the meaningful frames. Skip the WrapFunctionIntoFunctor boilerplate
# frames mentally; the "real" frames are:
#   VariableType::add_Tensor   (autograd layer, records backward)
#   wrapper_CPU_add_Tensor     (CPU backend kernel entry)
#   structured_add_Tensor::meta (shape inference + build iterator)
#   TensorIteratorBase::build  (broadcast, dtype promotion, output alloc)
bt 18

echo \n===== inspect: which dispatch key are we on? =====\n
# frame 8 is the redispatch from Autograd down to the backend
frame 8
echo \n(continue to let the program finish)\n
continue
