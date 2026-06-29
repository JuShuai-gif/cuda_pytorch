import torch

x = torch.tensor(2.0).requires_grad_()
y = torch.tensor(3.0).requires_grad_()

z = x * x * y

# grad_x = torch.autograd.grad(outputs=z, inputs=x) # 这表示 dz/dx
# grad_y = torch.autograd.grad(outputs=z, inputs=y)
# print(grad_x[0], grad_y[0])


# 问题在于第一次前向传播后，计算图被释放了，所以需要显式保留图。
# 对 backward 来说也是一样的。


grad_x = torch.autograd.grad(outputs=z, inputs=x, retain_graph=True)  # 这表示 dz/dx
grad_y = torch.autograd.grad(outputs=z, inputs=y)
print(grad_x[0], grad_y[0])
