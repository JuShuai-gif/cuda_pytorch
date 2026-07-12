# ============================================================================
# pytorch_utils.py —— PyTorch 模型分析与性能统计工具
#
# 本文件提供一系列用于分析和统计 PyTorch 模型性能的工具函数，包括：
#   1. rm_bn_from_net / rm_bn    —— 从网络中移除 BN 层（用于 FLOPs / 参数统计）
#   2. get_net_device            —— 获取模型所在的设备
#   3. count_parameters          —— 统计模型的可训练参数量
#   4. count_net_flops           —— 统计模型的理论 FLOPs（浮点运算次数）
#   5. count_peak_activation_size—— 统计模型推理时的峰值激活内存占用
#
# 为什么需要这些工具？
#   在 MCUNet 部署到微控制器的场景中，模型必须满足严格的硬件资源限制：
#   - Flash 空间有限（通常 < 256KB）：对应模型参数存储
#   - RAM 空间有限（通常 < 256KB）：对应特征图（激活值）的内存占用
#   - 计算能力有限（通常 < 100M MACs）：对应模型 FLOPs
#   这些工具帮助开发者在训练/搜索阶段精确评估候选架构的资源需求，确保
#   搜索出的网络在目标 MCU 上能实际运行。
# ============================================================================

import copy
import torch
import torch.nn as nn

# __all__ 控制 from pytorch_utils import * 时暴露的公共接口
__all__ = [
    "rm_bn_from_net",
    "get_net_device",
    "count_parameters",
    "count_net_flops",
    "count_peak_activation_size",
]

""" 网络性能分析工具 """


# ============================================================================
# rm_bn_from_net
# ============================================================================
# 功能：从网络的计算图和参数统计中移除 BatchNorm 层的影响。
#
# 设计意图：
#   在推理阶段，BN 层通常会被融合到前一个卷积层中（通过合并卷积权重和
#   BN 参数），因此 BN 本身不会引入额外的计算量或参数量。在进行 FLOPs
#   和参数量统计时，我们应该将 BN 层排除，以避免重复计数。
#
# 实现方式：
#   1. 将 BN 层的 forward 替换为恒等映射（lambda x: x），这样在统计
#      FLOPs 时 BN 层被视为无运算。
#   2. 删除 BN 层的 weight/bias/running_mean/running_var，这样在统计
#      参数量时 BN 的参数不被计入。
#
# 注意：
#   这个函数直接修改传入的模型，不会创建副本。调用者应该先深拷贝模型
#   再调用此函数。
# ============================================================================
def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # 将 forward 替换为恒等映射：输入是什么就输出什么
            # 这样在 FLOPs 统计工具看来，BN 层不产生任何计算量
            m.forward = lambda x: x
            # 删除 BN 的参数，使其不计入总参数量
            del m.weight
            del m.bias
            del m.running_mean
            del m.running_var


# ============================================================================
# rm_bn
# ============================================================================
# 功能：递归地从模块中删除所有 BN 层，替换为 nn.Identity。
#
# 与 rm_bn_from_net 的区别：
#   rm_bn_from_net 只是将 BN 的 forward 改为恒等映射并删除参数，但 BN
#   层本身仍然存在于网络结构中。而 rm_bn 将整个 BN 模块替换为 Identity，
#   网络结构中不再存在 BN 层。
#
# 参数：
#   module —— 输入模块（会递归处理子模块）
#
# 返回值：
#   删除了 BN 层的新模块
# ============================================================================
def rm_bn(module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        # 将 BN 层替换为 Identity
        module_output = nn.Identity()

    # 递归处理子模块
    for name, child in module.named_children():
        module_output.add_module(name, rm_bn(child))
    del module
    return module_output


# ============================================================================
# get_net_device
# ============================================================================
# 功能：获取模型所在的设备（CPU 或 CUDA 设备索引）。
#
# 实现：
#   通过获取模型的第一个参数所在的设备来推断。PyTorch 的 nn.Parameter
#   有一个 .device 属性，可以返回参数所在的设备。
#
# 使用场景：
#   在数据加载或创建随机输入时，需要将张量放在与模型相同的设备上。
# ============================================================================
def get_net_device(net):
    return net.parameters().__next__().device


# ============================================================================
# count_parameters
# ============================================================================
# 功能：统计模型的可训练参数量（仅统计 requires_grad=True 的参数）。
#
# 返回值：
#   参数量（整数），以"个"为单位，不是以 M 为单位。
#
# 使用场景：
#   用于评估模型的大小是否适合目标硬件的 Flash/RAM 限制。
#   以 MCUNet 为例，通常目标参数量在 1M 以下。
# ============================================================================
def count_parameters(net):
    # p.numel() 返回张量的元素总数
    # 只统计 requires_grad=True 的参数（排除冻结的参数）
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


# ============================================================================
# count_net_flops
# ============================================================================
# 功能：统计模型的理论 FLOPs（浮点运算次数），通常以 MACs（乘加运算）表示。
#
# 实现方式：
#   使用 torchprofile 库的 profile_macs 函数，它通过一次虚拟前向传播，
#   使用 PyTorch 的 forward hook 机制自动统计每层的 MACs。
#
# 为什么需要事先移除 BN？
#   如 rm_bn_from_net 所述，推理时 BN 会融合到卷积中，所以单独统计 BN
#   的 FLOPs 会导致重复计算。移除 BN 后统计结果更准确。
#
# 参数：
#   model     —— 目标模型
#   data_shape—— 输入数据的形状，例如 (1, 3, 224, 224)
#
# 返回值：
#   FLOPs/MACs 数量（整数）
# ============================================================================
def count_net_flops(model, data_shape):
    from torchprofile import profile_macs

    # 深拷贝模型（避免修改原始模型）
    model = copy.deepcopy(model)
    # 移除 BN 层（BN 会融合到 Conv 中，不单独产生计算量）
    rm_bn_from_net(model)
    # profile_macs 通过注册 forward hook 来统计每层的 MACs
    total_macs = profile_macs(model, torch.randn(*data_shape).to(get_net_device(model)))
    del model
    return total_macs


# ============================================================================
# count_peak_activation_size
# ============================================================================
# 功能：计算模型推理时的峰值激活内存占用。
#
# 为什么需要这个指标？
#   对于部署到 MCU 的模型，最关键的限制因素之一是 RAM 大小。模型参数可以
#   存储在 Flash 中（只读），但特征图（激活值）必须在 RAM 中读写。峰值激活
#   内存占用决定了网络在推理过程中最多需要多少 RAM。
#
#   注意：这里统计的是"激活值"占用的内存，不包括模型参数。
#
# 实现方法：
#   1. 为所有 Conv2d、Linear 和 MobileInvertedResidualBlock 注册 forward hook
#   2. 运行一次前向传播，通过 hook 记录每层输入/输出的元素数
#   3. 对于普通卷积层：内存占用 = 输入大小 + 输出大小（假设输入和输出同时存在）
#   4. 对于倒残差块（有 shortcut 的）：需要考虑残差路径需要额外保留一份输入
#   5. 取所有层中的最大值作为峰值激活内存
#
# 参数：
#   net        —— ProxylessNASNets 模型
#   data_shape —— 输入数据形状，默认 (1, 3, 224, 224)
#
# 返回值：
#   峰值激活内存占用（以元素数为单位，乘以 dtype 的字节数得到实际字节数）
# ============================================================================
def count_peak_activation_size(net, data_shape=(1, 3, 224, 224)):
    from ..tinynas.nn.networks import MobileInvertedResidualBlock

    # ====================================================================
    # record_in_out_size（forward hook 函数）
    # 功能：记录层的输入和输出张量的元素总数。
    # x.numel() = batch * C * H * W
    # y.numel() = batch * C' * H' * W'
    # ====================================================================
    def record_in_out_size(m, x, y):
        x = x[0]
        m.input_size = torch.Tensor([x.numel()])
        m.output_size = torch.Tensor([y.numel()])

    # ====================================================================
    # add_io_hooks
    # 功能：为 Conv2d、Linear 和 MobileInvertedResidualBlock 注册
    #       record_in_out_size hook，并创建对应的 buffer。
    # ====================================================================
    def add_io_hooks(m_):
        m_type = type(m_)
        if m_type in [nn.Conv2d, nn.Linear, MobileInvertedResidualBlock]:
            # 注册 buffer 存储输入/输出元素数
            m_.register_buffer("input_size", torch.zeros(1))
            m_.register_buffer("output_size", torch.zeros(1))
            # 注册 forward hook
            m_.register_forward_hook(record_in_out_size)

    # ====================================================================
    # count_conv_mem
    # 功能：计算单个卷积层或全连接层的内存占用。
    #
    # 内存计算假设：
    #   在推理时，输入张量和输出张量都需要在内存中同时存在。
    #   所以该层占用的内存 = input_size + output_size（元素数）。
    #
    # 注意：
    #   m 可以是包装类型（如 ConvBNReLU），其 .conv 属性才是真正的 Conv2d。
    # ====================================================================
    def count_conv_mem(m):
        if m is None:
            return 0
        # 解包装：找到真正的 Conv2d 或 Linear 模块
        if hasattr(m, "conv"):
            m = m.conv
        elif hasattr(m, "linear"):
            m = m.linear
        assert isinstance(m, (nn.Conv2d, nn.Linear))
        return m.input_size.item() + m.output_size.item()

    # ====================================================================
    # count_block
    # 功能：计算倒残差块（MobileInvertedResidualBlock）的峰值内存占用。
    #
    # 倒残差块包含三个子层（pointwise1 → depthwise → pointwise2），
    # 计算过程是串行的，所以内存峰值取决于这三步中的最大值。
    #
    # 分两种情况：
    #   1. 有残差连接（shortcut 存在且有效）：
    #      在执行 depthwise 时，残差分支需要保留 pointwise1 的输入特征图
    #      供后续相加使用。所以 depthwise 阶段的内存占用需要加上
    #      residual_size（即 pointwise1 的输入大小）。
    #   2. 无残差连接：直接取三个子层的最大值。
    # ====================================================================
    def count_block(m):
        from ..tinynas.nn.modules import ZeroLayer

        assert isinstance(m, MobileInvertedResidualBlock)

        # 如果主路被禁用（None 或 ZeroLayer），不占用内存
        if m.mobile_inverted_conv is None or isinstance(
            m.mobile_inverted_conv, ZeroLayer
        ):
            return 0
        elif m.shortcut is None or isinstance(m.shortcut, ZeroLayer):
            # 无残差连接：只需考虑三个子层的峰值
            return max(
                [
                    count_conv_mem(m.mobile_inverted_conv.inverted_bottleneck),
                    count_conv_mem(m.mobile_inverted_conv.depth_conv),
                    count_conv_mem(m.mobile_inverted_conv.point_linear),
                ]
            )
        else:
            # 有残差连接：depthwise 阶段需要额外保留残差输入
            # residual_size = pointwise1 的输入大小
            residual_size = (
                m.mobile_inverted_conv.inverted_bottleneck.conv.input_size.item()
            )
            return max(
                [
                    count_conv_mem(m.mobile_inverted_conv.inverted_bottleneck),
                    # depthwise 阶段多占一份残差输入
                    count_conv_mem(m.mobile_inverted_conv.depth_conv) + residual_size,
                    count_conv_mem(m.mobile_inverted_conv.point_linear),
                ]
            )

    # ====================================================================
    # 主逻辑
    # ====================================================================

    # 如果模型被 DataParallel 包裹，先取实际的 module
    if isinstance(net, nn.DataParallel):
        net = net.module
    # 深拷贝，避免修改原始模型
    net = copy.deepcopy(net)

    from ..tinynas.nn.networks import ProxylessNASNets

    # 验证模型类型
    assert isinstance(net, ProxylessNASNets)

    # 注册 all hooks
    net.apply(add_io_hooks)

    # 执行虚拟前向传播，触发 hooks 记录输入/输出大小
    with torch.no_grad():
        _ = net(torch.randn(*data_shape).to(net.parameters().__next__().device))

    # 收集所有层的峰值内存占用
    mem_list = [
        count_conv_mem(net.first_conv),  # 首层卷积
        count_conv_mem(net.feature_mix_layer),  # 特征混合层（可能为 None）
        count_conv_mem(net.classifier),  # 分类器
    ] + [count_block(blk) for blk in net.blocks]  # 所有残差块

    del net
    # 峰值 = 所有层中最大的内存占用
    return max(mem_list)
