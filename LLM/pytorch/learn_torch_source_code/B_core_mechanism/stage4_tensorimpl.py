import torch
t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print("about to inspect", t.shape, t.stride())
print(t.sum().item())   # a place to break
