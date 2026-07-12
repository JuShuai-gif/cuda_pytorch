import torch

def f(x, y):
    a = x @ y
    b = torch.relu(a)
    return b + 1.0

# 1) Dynamo capture: show the FX graph via a custom backend
def show_backend(gm, example_inputs):
    print("=== FX graph captured by Dynamo (frontend) ===")
    print(gm.graph)
    return gm.forward

torch._dynamo.reset()
g = torch.compile(f, backend=show_backend)
g(torch.randn(4, 4), torch.randn(4, 4))

# 2) Inductor backend: generate real Triton/C++ kernels, dump the code path
torch._dynamo.reset()
h = torch.compile(f, backend="inductor")
out = h(torch.randn(64, 64), torch.randn(64, 64))
print("=== inductor result shape:", tuple(out.shape))
