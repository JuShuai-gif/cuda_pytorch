import torch
x = torch.ones(4, requires_grad=True)
y = (x * x).sum()          # forward: builds graph. grad_fn: MulBackward0 -> SumBackward0
print("about to backward")
y.backward()               # <-- trace this
print("x.grad:", x.grad.tolist())   # expected 2*x = [2,2,2,2]
