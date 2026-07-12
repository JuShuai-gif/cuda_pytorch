import torch
from torch.utils.cpp_extension import load_inline

# C++ source: define a kernel and register it into the dispatcher via TORCH_LIBRARY.
cpp = r'''
#include <torch/torch.h>

// The actual CPU kernel: y = x^2 + bias
at::Tensor square_plus_cpu(const at::Tensor& x, double bias) {
  return x * x + bias;
}

// 1) declare the operator schema in our own namespace "myops"
TORCH_LIBRARY(myops, m) {
  m.def("square_plus(Tensor x, float bias) -> Tensor");
}

// 2) register the CPU implementation for that schema
TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("square_plus", &square_plus_cpu);
}
'''

mod = load_inline(
    name="myops_ext",
    cpp_sources=cpp,
    functions=[],            # we expose via the dispatcher, not pybind
    verbose=False,
)

x = torch.tensor([1.0, 2.0, 3.0])
# Call through the dispatcher, exactly like a built-in aten op:
y = torch.ops.myops.square_plus(x, 10.0)
print("result:", y.tolist())         # [11, 14, 19]

# Inspect its dispatch table -> shows our CPU registration
print("=== myops::square_plus dispatch table ===")
print(torch._C._dispatch_dump_table("myops::square_plus"))
