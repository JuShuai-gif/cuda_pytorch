import copy
import torch
import torch.nn as nn
from ofa.utils import Hswish, Hsigmoid, MyConv2d

from ofa.utils.layers import ResidualBlock
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.mobilenet import InvertedResidual

__all__ = ["count_model_size", "count_activation_size", "profile_memory_cost"]


def count_model_size(net, trainable_param_bits=32, frozen_param_bits=8, print_log=True):
    """统计模型的参数内存占用，区分可训练参数和冻结参数。

    TinyTL 的核心分析工具：冻结参数可以用更低位数存储（如 8bit），
    可训练参数需要保持训练精度（通常 32bit）。两者的存储成本差异很大。

    Args:
            trainable_param_bits: 可训练参数的存储位数（默认 32bit=FP32）
            frozen_param_bits: 冻结参数的存储位数（默认 8bit=INT8），为 None 时使用 32bit
            print_log: 是否打印内存统计日志
    Returns:
            model_size: 模型总参数字节数
    """
    frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits

    trainable_param_size = 0
    frozen_param_size = 0
    for p in net.parameters():
        if p.requires_grad:
            trainable_param_size += trainable_param_bits / 8 * p.numel()
        else:
            frozen_param_size += frozen_param_bits / 8 * p.numel()
    model_size = trainable_param_size + frozen_param_size
    if print_log:
        print(
            "Total: %d" % model_size,
            "\tTrainable: %d (data bits %d)"
            % (trainable_param_size, trainable_param_bits),
            "\tFrozen: %d (data bits %d)" % (frozen_param_size, frozen_param_bits),
        )
    # Byte
    return model_size


def count_activation_size(
    net, input_size=(1, 3, 224, 224), require_backward=True, activation_bits=32
):
    """统计训练过程中的峰值激活内存。

    通过注册 forward hook 追踪每一层的输入/输出大小，再替换每层的 forward 方法
    模拟训练时的激活内存累积和释放过程，计算反向传播所需的梯度激活和临时激活。

    Args:
            net: 待分析的网络
            input_size: 输入张量的 shape (batch, C, H, W)
            require_backward: 是否需要反向传播（仅推理时设为 False）
            activation_bits: 激活值的存储位数（默认 32bit）

    Returns:
            (peak_activation_size, grad_activation_size): 峰值激活内存和各层梯度激活内存
    """
    act_byte = activation_bits / 8
    model = copy.deepcopy(net)

    # ---- 定义各层类型的激活钩子函数 ----

    # noinspection PyArgumentList
    def count_convNd(m, x, y):
        """卷积层的内存统计：输入激活（反向需要）+ 输入+部分输出的临时内存"""
        # 反向传播需要的梯度激活：保存前向输入以计算梯度
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # 推理阶段的临时内存：输入 + 输出 / groups（考虑深度可分离卷积）
        m.tmp_activations = torch.Tensor(
            [x[0].numel() * act_byte + y.numel() * act_byte // m.groups]
        )  # bytes

    # noinspection PyArgumentList
    def count_linear(m, x, y):
        """全连接层的内存统计"""
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        m.tmp_activations = torch.Tensor(
            [x[0].numel() * act_byte + y.numel() * act_byte]
        )  # bytes

    # noinspection PyArgumentList
    def count_bn(m, x, _):
        """BatchNorm/GroupNorm 层的内存统计"""
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    # noinspection PyArgumentList
    def count_relu(m, x, _):
        """ReLU 类激活函数的内存统计：反向传播仅需 1 bit 掩码"""
        if require_backward:
            # ReLU 反向只需要 1-bit 掩码来标记哪些输入 > 0
            m.grad_activations = torch.Tensor([x[0].numel() / 8])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    # noinspection PyArgumentList
    def count_smooth_act(m, x, _):
        """平滑激活函数（Sigmoid/Tanh/Hswish 等）的内存统计：
        反向传播需要完整保存输入激活以计算梯度"""
        if require_backward:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    # ---- 注册钩子函数到网络的所有叶子节点 ----
    def add_hooks(m_):
        """递归为网络的叶子模块注册 forward hook 和内存统计缓冲区"""
        if len(list(m_.children())) > 0:
            return

        # 为每个叶子模块注册两个缓冲区，存储该层的激活内存统计
        m_.register_buffer("grad_activations", torch.zeros(1))
        m_.register_buffer("tmp_activations", torch.zeros(1))

        # 根据模块类型分配对应的 hook 函数
        if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d, MyConv2d]:
            fn = count_convNd
        elif type(m_) in [nn.Linear]:
            fn = count_linear
        elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]:
            fn = count_bn
        elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
            fn = count_relu
        elif type(m_) in [nn.Sigmoid, nn.Tanh, Hswish, Hsigmoid]:
            fn = count_smooth_act
        else:
            fn = None

        if fn is not None:
            _handler = m_.register_forward_hook(fn)

    # 第一阶段：用 hook 记录每层的激活大小
    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(model.parameters().__next__().device)
    with torch.no_grad():
        model(x)

    # 第二阶段：替换 forward，模拟训练时的内存分配和释放，追踪峰值
    memory_info_dict = {
        "peak_activation_size": torch.zeros(1),  # 整个前向过程中的峰值激活内存
        "grad_activation_size": torch.zeros(1),  # 所有需要反向传播的层的梯度激活累积
        "residual_size": torch.zeros(1),  # 当前残差连接的输入内存（暂存等待相加）
    }

    for m in model.modules():
        if len(list(m.children())) == 0:
            # 为每个叶子模块包装 forward，在调用前更新峰值内存统计
            def new_forward(_module):
                def lambda_forward(_x):
                    # 当前激活内存 = 临时激活 + 已累积的梯度激活 + 残差暂存
                    current_act_size = (
                        _module.tmp_activations
                        + memory_info_dict["grad_activation_size"]
                        + memory_info_dict["residual_size"]
                    )
                    # 更新峰值
                    memory_info_dict["peak_activation_size"] = max(
                        current_act_size, memory_info_dict["peak_activation_size"]
                    )
                    # 累加本层的梯度激活（反向时需要保存）
                    memory_info_dict["grad_activation_size"] += _module.grad_activations
                    return _module.old_forward(_x)

                return lambda_forward

            m.old_forward = m.forward
            m.forward = new_forward(m)

        # 检测残差连接块：残差输入在 shortcut 计算期间需要暂存在内存中
        if (
            (isinstance(m, ResidualBlock) and m.shortcut is not None)
            or (isinstance(m, InvertedResidual) and m.use_res_connect)
            or type(m) in [BasicBlock, Bottleneck]
        ):

            def new_forward(_module):
                def lambda_forward(_x):
                    # 记录残差输入的大小，在主路径计算完成前需保留在内存中
                    memory_info_dict["residual_size"] = _x.numel() * act_byte
                    result = _module.old_forward(_x)
                    # 残差相加后释放输入
                    memory_info_dict["residual_size"] = 0
                    return result

                return lambda_forward

            m.old_forward = m.forward
            m.forward = new_forward(m)

    # 用替换后的 forward 再跑一次，获取真实的内存使用峰值和梯度激活量
    with torch.no_grad():
        model(x)

    return memory_info_dict["peak_activation_size"].item(), memory_info_dict[
        "grad_activation_size"
    ].item()


def profile_memory_cost(
    net,
    input_size=(1, 3, 224, 224),
    require_backward=True,
    activation_bits=32,
    trainable_param_bits=32,
    frozen_param_bits=8,
    batch_size=8,
):
    """综合分析训练内存成本：参数量 + 激活量 × batch_size。

    TinyTL 场景下的核心分析入口：
    - 冻结主干权重（8bit 存储）+ 冻结卷积梯度为 0（不需要存梯度激活）
    - 仅 bias/BN/lite_residual 可训练 → 显著降低参数内存和激活内存

    Returns:
            memory_cost: 总内存成本（字节）
            memory_cost_dict: 分项统计 {'param_size': ..., 'act_size': ...}
    """
    # 统计参数内存（区分可训练和冻结）
    param_size = count_model_size(
        net, trainable_param_bits, frozen_param_bits, print_log=True
    )

    # 统计激活内存（考虑反向传播需求）
    activation_size, _ = count_activation_size(
        net, input_size, require_backward, activation_bits
    )

    # 总内存 = 参数 + 激活 × batch_size（激活随 batch 线性增长）
    memory_cost = activation_size * batch_size + param_size
    return memory_cost, {"param_size": param_size, "act_size": activation_size}
