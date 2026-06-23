import torch


# 直通估计器(STE, Straight-Through Estimator)实现的“伪量化(fake quant)”
# 量化中的 round 是阶梯函数，导数几乎处处为 0，直接反向传播会让梯度消失、
# 权重无法更新。STE 的核心技巧：前向照常量化，反向时假装量化是恒等函数，
# 把上游梯度原样传回，从而让量化网络可以正常训练(QAT)。
class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        # 前向：量化到整数(round + clamp)，再乘回 scale 得到反量化结果
        q = torch.round(x / scale).clamp(qmin, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：STE —— 把量化视作恒等函数，梯度原样传回
        # 其余三个输入(scale/qmin/qmax)不需要梯度，返回 None
        return grad_output, None, None, None


def fake_quant(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    # scale 用 detach 计算，把它当作常量，不参与反向传播
    scale = x.detach().abs().max().clamp(min=1e-8) / qmax
    return FakeQuantSTE.apply(x, scale, qmin, qmax)


# 对照组：不使用 STE，直接用 torch.round 做伪量化
# torch.round 的梯度处处为 0，因此经过它的链路梯度会全部变成 0(梯度消失)。
def fake_quant_no_ste(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    scale = x.detach().abs().max().clamp(min=1e-8) / qmax
    q = torch.round(x / scale).clamp(qmin, qmax)
    return q * scale


# ---------------- 实验 1：STE vs 无 STE 的梯度对比 ----------------
# 同样的前向计算，唯一区别是反向是否用 STE。
# 有 STE：梯度正常回传，权重可训练；无 STE：梯度被 round 清零，权重学不动。
torch.manual_seed(0)
w = torch.randn(16, requires_grad=True)
y = (fake_quant(w, bits=8) ** 2).mean()
y.backward()
print("[有 STE] w.grad[:5] =", w.grad[:5])
print("[有 STE] 梯度范数    =", float(w.grad.norm()))

w2 = w.detach().clone().requires_grad_(True)
y2 = (fake_quant_no_ste(w2, bits=8) ** 2).mean()
y2.backward()
print("\n[无 STE] w.grad[:5] =", w2.grad[:5])
print("[无 STE] 梯度范数    =", float(w2.grad.norm()), " <- round 不可导，梯度消失")


# ---------------- 实验 2：不同 bit 宽度的量化质量(SQNR) ----------------
# bit 越多 → 量化步长(scale)越小 → 量化误差越小 → SQNR 越高。
def sqnr_db(x, x_hat):
    s = torch.mean(x.float() ** 2)
    n = torch.mean((x.float() - x_hat.float()) ** 2).clamp(min=1e-12)
    return 10 * torch.log10(s / n)


print("\n[不同 bit 宽度的量化质量]")
x = torch.randn(4096)
for bits in (2, 4, 8):
    x_hat = fake_quant_no_ste(x, bits=bits)
    print(f"  {bits}-bit: SQNR = {float(sqnr_db(x, x_hat)):6.2f} dB")
print("结论：低比特量化误差更大，但配合 STE 仍能参与 QAT 训练，从而把精度损失学回来。")
