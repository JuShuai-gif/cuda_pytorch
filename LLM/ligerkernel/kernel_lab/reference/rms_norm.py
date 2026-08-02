"""
数学公式：

y_i = x_i / sqrt(mean_j(x_j^2) + eps) * w_i


- rms_norm_reference：纯 PyTorch 实现的参考版本（学习指南第 2 步）。
                    因为所有操作都是可微的，完全支持 autograd 反向传播，所以也可以作为正确性测试的"标准答案"（ground truth）。

- LlamaRMSNorm：用 torch.nn.Module 包装的版本，带有可学习的 weight 参数（nn.Parameter，
                    默认 requires_grad=True），因此可以像其他模块一样参与训练，权重会在反向传播中更新。

两者核心区别：数学计算完全一样（都是 output = x / sqrt(mean(x^2) + eps) * w），
区别只在"形态"：

- rms_norm_reference 是普通函数，weight 由调用方传入，函数本身不持有任何状态。
  何时用：作为 CUDA 内核正确性测试的 ground truth——给定相同的 x/weight，
  只要你的内核输出和它一致，就算算对了。它不参与训练，只做"验算"。

- LlamaRMSNorm 是 nn.Module，weight 是模块内部的 nn.Parameter（requires_grad=True）。
  何时用：作为模型中的归一化层参与训练，训练时优化器会更新它的 weight，
  state_dict 也能自动保存/加载。它等价于 rms_norm_reference 的"可训练封装版"。

一句话：验证算得对不对用 reference，训练模型用 LlamaRMSNorm。
"""

import torch
import torch.nn as nn


def rms_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Reference RMSNorm.

    Args:
        x: 输入
        weight: 权重
        eps: 微小变量

    Returns:
        归一化之后的张量
    """
    # 转成 float32 类型再返回一个新的张量
    x_float = x.float()

    # x_float.pow(2)：对每个元素平方 → x_j²
    # .mean(dim=-1, keepdim=True)：沿最后一维（dim=-1）求平均，keepdim=True 保留维度
    # keepdim = True很关键：结果保持 (batch, 1) 形状，这样后面第 31 行才能和 x_float（形状 (batch, hidden)）直接做逐元素除法/广播。
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)

    # variance + eps：加上一个很小的数（如 1e-5），防止分母为 0
    # torch.rsqrt(...)：计算平方根的倒数 1/√x，所以 rsqrt(variance + eps) = 1 / sqrt(mean(x²) + eps)
    # x_float * ...：输入逐元素乘上这个缩放系数，得到归一化结果
    """
    为什么要用 rsqrt 而不是 x / torch.sqrt(...)？
    因为 rsqrt 是 GPU 上的单个指令，比"先 sqrt 再 div"更快也更精确（误差更小），
    这正是 CUDA 内核里想复现的等价实现
    """
    normalized = x_float * torch.rsqrt(variance + eps)

    return (normalized * weight.float()).to(x.dtype)


# 为什么 class LlamaRMSNorm(nn.Module) 要继承 nn.Module？
# 继承后 PyTorch 免费提供以下标准能力：
#   - 自动管理参数：nn.Parameter 赋给 self.weight 后，module.parameters()/named_parameters()
#     会自动收集到它，requires_grad 属性也由此生效
#   - 可训练：能被优化器（optimizer.step()）更新权重，符合"像其他模块一样参与训练"
#   - 前向调用：module(x) 会触发 forward(x)，并自动处理 __call__ 里的钩子机制（hooks）
#   - 状态管理：state_dict()/load_state_dict() 自动保存/加载权重，model.to(device) 自动迁移参数
#   - 可组合：能嵌进更大的 nn.Module（如 nn.Sequential、Transformer 块）层层嵌套
#   - 切换训练/推理：.train()/.eval()、torch.no_grad() 等自动生效
# 不继承的话，参数收集、子模块递归、状态保存这些都要自己手写，工作量大且容易出错。
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # nn.Parameter defaults to requires_grad=True.
        # 初始化 hidden_size 个全 1 的可学习权重，形状为 (hidden_size,)，
        # 对应每个维度一个缩放系数；requires_grad=True 表示训练时会被优化器更新。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接复用纯函数的 reference 实现：
        # 把模块持有的 self.weight 和 self.eps 传进去完成归一化。
        # 这样"数学计算"只有一份，两个版本不会算出不同结果。
        return rms_norm_reference(x, self.weight, self.eps)


if __name__ == "__main__":
    # 固定随机种子，保证每次运行结果可复现
    torch.manual_seed(0)
    x = torch.randn(3, 5)  # 输入：batch=3, hidden=5
    w = torch.randn(5, requires_grad=True)  # 权重：5 个维度各一个缩放系数，开启求导
    eps = 1e-6

    # 检查 1：前向是否正确。
    # 用"链式写法"手算一遍同样的公式作为 expected，和函数输出逐元素比对。
    # expected = x / sqrt(mean(x²) + eps) * w（等价写法，只换了下运算顺序）
    y = rms_norm_reference(x, w, eps)
    expected = x * x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt() * w
    assert torch.allclose(y, expected, atol=1e-6), "forward mismatch"

    # 检查 2：LlamaRMSNorm 是否能作为可训练模块使用。
    module = LlamaRMSNorm(5, eps=eps)
    assert module.weight.requires_grad is True, "weight must require grad"
    # 把模块里全 1 的初始权重，覆盖成和前面测试一致的 w（.detach() 断开梯度，只拷贝数值）
    module.weight.data.copy_(w.detach())
    # 前向 + 标量求和 + 反向，验证 autograd 整条链路能跑通
    module(x).sum().backward()
    # 梯度必须存在且数值有限（没有 NaN/Inf），说明反传正确
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()

    # 检查 3：手算 weight 的梯度做交叉验证。
    # 因为 loss = sum(norm * w) 对 w 求导就是 norm 沿 batch 维求和，
    # 所以 grad_w_manual = sum_batch(norm)，其中 norm = x / sqrt(mean(x²) + eps)
    norm = x * x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    grad_w_manual = norm.sum(dim=0)
    # 把自动微分得到的 grad 和手算结果比对，一致说明反向传播算对了
    assert torch.allclose(module.weight.grad, grad_w_manual, atol=1e-6), (
        "weight grad mismatch"
    )

    print("forward OK, weight.requires_grad OK, weight.grad OK")
