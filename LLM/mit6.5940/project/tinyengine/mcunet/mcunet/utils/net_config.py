# ============================================================================
# net_config.py —— 网络配置提取工具
#
# 功能：
#   本文件导出一个核心函数 get_network_config_with_activation_shape，它的作用
#   是运行一次模型前向传播，通过 PyTorch 的 forward hook 机制自动记录网络
#   中每一层的输入和输出张量形状，然后生成一个详细的配置字典。
#
# 为什么需要这个功能？
#   在 MCUNet 部署到微控制器（MCU）时，需要精确知道每层的：
#     - 输入/输出通道数（用于分配缓冲区）
#     - 特征图的空间分辨率（用于计算内存占用）
#     - 卷积核大小、步长、分组数（用于生成目标平台的推理代码）
#   手动计算这些信息容易出错，本工具通过一次自动前向传播自动收集所有信息。
#
# 适用模型：
#   本工具目前专为 ProxylessNASNets 架构设计（MCUNet 的基础架构），
#   对其他架构可能需要适配。
# ============================================================================

import torch
import torch.nn as nn

# __all__ 控制 from net_config import * 时暴露的公共接口
__all__ = ["get_network_config_with_activation_shape"]


# ============================================================================
# record_in_out_shape（forward hook 函数）
# ============================================================================
# 功能：PyTorch forward hook，自动记录卷积层的输入和输出形状。
#
# PyTorch 的 register_forward_hook 会在模块的 forward 执行后被调用，
# 参数如下：
#   m —— 模块本身（即被 hook 的层）
#   x —— 输入张量元组（通常 x[0] 是真正的输入）
#   y —— 输出张量
#
# 此 hook 将输入/输出形状记录为模块的 buffer（持久化张量），
# 方便后续的配置提取函数读取。
# ============================================================================
def record_in_out_shape(m, x, y):
    # x 是 tuple，取第一个元素（真正的输入张量）
    x = x[0]
    # 将形状信息注册为 buffer，这样即使模型进入 eval 模式也能访问
    # 使用 Tensor 而不是元组，是因为 PyTorch buffer 要求是 Tensor 类型
    m.input_shape = torch.Tensor(list(x.shape))
    m.output_shape = torch.Tensor(list(y.shape))


# ============================================================================
# record_residual_shape（forward hook 函数）
# ============================================================================
# 功能：记录倒残差块（MobileInvertedResidualBlock）的输出形状，
#       并判断是否存在实际的残差连接。
#
# 设计背景：
#   在 MCUNet 的 ProxylessNASNets 架构中，每个 residual block 可能包含
#   一个 shortcut（残差连接）。但在搜索过程中，某些配置可能使 shortcut
#   被禁用（设置为 None）或被替换为 ZeroLayer（表示没有连接）。
#   本 hook 只在 shortcut 实际存在且有非零输出时才记录形状。
# ============================================================================
def record_residual_shape(m, x, y):
    from ..tinynas.nn.modules import ZeroLayer

    # 判断残差路径是否有效：
    # 1. mobile_inverted_conv 为 None → 主路为空，不记录
    # 2. mobile_inverted_conv 为 ZeroLayer → 主路无实际计算，不记录
    # 3. shortcut 为 None → 无残差连接，不记录
    # 4. shortcut 为 ZeroLayer → 残差被掩码掉，不记录
    # 只在 shortcut 实际存在且生效时（有特征图输出）才记录 output_shape
    if m.mobile_inverted_conv is None or isinstance(m.mobile_inverted_conv, ZeroLayer):
        pass
    elif m.shortcut is None or isinstance(m.shortcut, ZeroLayer):
        pass
    else:
        # 残差连接存在且有效，记录输出形状（用于后续计算残差分支占用的内存）
        m.output_shape = torch.Tensor(list(y.shape))


# ============================================================================
# add_activation_shape_hook
# ============================================================================
# 功能：为 Conv2d 和 MobileInvertedResidualBlock 注册 forward hook。
#
# 这个函数被 model.apply() 调用，遍历模型的所有子模块，对指定类型的
# 模块注册相应的 hook：
#   - nn.Conv2d: 注册 record_in_out_shape hook
#   - MobileInvertedResidualBlock: 注册 record_residual_shape hook
#
# 同时为这些模块创建 input_shape / output_shape buffer，确保 hook
# 执行时有地方存放数据。
# ============================================================================
def add_activation_shape_hook(m_):
    from ..tinynas.nn.networks import MobileInvertedResidualBlock

    m_type = type(m_)
    if m_type == nn.Conv2d:
        # 为 Conv2d 层注册输入/输出形状 buffer
        # 初始为全零张量形状 (4,)，后续被 hook 覆盖
        m_.register_buffer("input_shape", torch.zeros(4))
        m_.register_buffer("output_shape", torch.zeros(4))
        m_.register_forward_hook(record_in_out_shape)
    elif m_type == MobileInvertedResidualBlock:
        # 为倒残差块注册输出形状 buffer（只需要输出形状以判断残差是否存在）
        m_.register_buffer("output_shape", torch.zeros(4))
        m_.register_forward_hook(record_residual_shape)


# ============================================================================
# get_network_config_with_activation_shape
# ============================================================================
# 功能：执行一次模型前向传播，自动收集每层的形状信息，生成详细配置。
#
# 工作流程：
#   1. 深拷贝模型（避免 hook 修改原始模型）
#   2. 将所有层设为 eval 模式（关闭 dropout 等）
#   3. 用 model.apply() 为所有 Conv2d 和 ResidualBlock 注册 hook
#   4. 用随机数据跑一次前向传播，触发所有 hook
#   5. 从 hook 记录的 buffer 中提取每层的配置信息
#   6. 组织为字典格式返回
#
# 参数：
#   model     —— ProxylessNASNets 模型实例
#   device    —— 运行设备的字符串，如 "cpu" 或 "cuda:0"（默认 "cpu"）
#   data_shape—— 输入数据的形状，默认为 (1, 3, 224, 224)
#                 对应 batch_size=1, RGB三通道, 224x224 分辨率
#
# 返回值：
#   嵌套字典，包含以下字段：
#   {
#     "first_conv":   第一层卷积的配置
#     "classifier":   分类头（全连接层）的配置
#     "feature_mix":  特征混合层的配置（可能为 None）
#     "blocks":       [残差块列表]，每块包含 pointwise1/depthwise/pointwise2/sub_residual
#   }
#   每个卷积配置包含：in_channel, in_shape, out_channel, out_shape,
#                    kernel_size, stride, groups, depthwise
# ============================================================================
def get_network_config_with_activation_shape(
    model, device="cpu", data_shape=(1, 3, 224, 224)
):
    from ..tinynas.nn.networks import ProxylessNASNets
    from ..tinynas.nn.modules import ZeroLayer

    # 验证模型类型
    assert isinstance(model, ProxylessNASNets)
    import copy

    # 第一步：深拷贝并设置 eval 模式
    model = copy.deepcopy(model).to(device)
    model.eval()

    # 第二步：为所有目标层注册 forward hook
    model.apply(add_activation_shape_hook)

    # 第三步：执行一次前向传播（随机数据），触发 hook 记录形状
    with torch.no_grad():
        _ = model(torch.randn(*data_shape).to(device))

    # ====================================================================
    # 内部函数：get_conv_cfg
    # ====================================================================
    # 从 conv 模块中提取配置。这里 conv 可能是一个包装类，其 .conv 属性
    # 才是真正的 nn.Conv2d。通过 hook 记录的 input_shape / output_shape
    # 获取特征图的通道数和分辨率。
    # ====================================================================
    def get_conv_cfg(conv):
        conv = conv.conv  # 解包获取真正的 Conv2d 模块
        return {
            "in_channel": int(conv.input_shape[1]),  # 输入通道数
            "in_shape": int(conv.input_shape[2]),  # 输入空间尺寸（H=W）
            "out_channel": int(conv.output_shape[1]),  # 输出通道数
            "out_shape": int(conv.output_shape[2]),  # 输出空间尺寸（H=W）
            "kernel_size": conv.kernel_size[0],  # 卷积核大小
            "stride": conv.stride[0],  # 步长
            "groups": conv.groups,  # 分组数
            "depthwise": conv.groups
            == int(conv.input_shape[1]),  # 是否为 depthwise 卷积
        }

    # ====================================================================
    # 内部函数：get_linear_cfg
    # ====================================================================
    # 从全连接层中提取输入/输出维度。
    # ====================================================================
    def get_linear_cfg(op):
        return {
            "in_channel": op.in_features,
            "out_channel": op.out_features,
        }

    # ====================================================================
    # 内部函数：get_block_cfg
    # ====================================================================
    # 从倒残差块（MobileInvertedResidualBlock）中提取详细配置。
    #
    # 每个倒残差块包含三个子卷积：
    #   1. pointwise1 (inverted_bottleneck): 1x1 升维卷积
    #   2. depthwise: 3x3 depthwise 卷积（空间特征提取）
    #   3. pointwise2 (point_linear): 1x1 降维卷积
    #
    # 此外还有可选的残差连接（shortcut），其配置由 hook 记录的
    # output_shape 决定：如果 output_shape[0] == 0，表示无残差连接。
    # ====================================================================
    def get_block_cfg(block):
        pdp = block.mobile_inverted_conv  # Pointwise-Depthwise-Pointwise 结构

        # 判断是否有残差连接
        # block.output_shape 在 record_residual_shape hook 中设置
        # 如果为 0，说明 hook 没有记录（残差不存在或被掩码）
        if int(block.output_shape[0]) == 0:
            residual = None
        else:
            # 残差连接的配置：输入通道数和空间尺寸
            assert block.output_shape[2] == block.output_shape[3]
            residual = {
                "in_channel": int(block.output_shape[1]),
                "in_shape": int(block.output_shape[2]),
            }

        return {
            "pointwise1": get_conv_cfg(pdp.inverted_bottleneck)
            if pdp.inverted_bottleneck is not None
            else None,
            "depthwise": get_conv_cfg(pdp.depth_conv),
            "pointwise2": get_conv_cfg(pdp.point_linear),
            "residual": residual,
        }

    # ====================================================================
    # 主逻辑：逐层提取配置
    # ====================================================================
    cfg = {}

    # 第一层卷积
    cfg["first_conv"] = get_conv_cfg(model.first_conv)

    # 分类器（全连接层）
    cfg["classifier"] = get_linear_cfg(model.classifier)

    # 特征混合层（可能存在也可能不存在，由网络结构决定）
    if model.feature_mix_layer is not None:
        cfg["feature_mix"] = get_conv_cfg(model.feature_mix_layer)
    else:
        cfg["feature_mix"] = None

    # 遍历所有残差块，跳过被禁用的块
    block_cfg = []
    for block in model.blocks:
        if block.mobile_inverted_conv is None or isinstance(
            block.mobile_inverted_conv, ZeroLayer
        ):
            continue
        block_cfg.append(get_block_cfg(block))
    cfg["blocks"] = block_cfg

    # 释放深拷贝的模型
    del model
    return cfg
