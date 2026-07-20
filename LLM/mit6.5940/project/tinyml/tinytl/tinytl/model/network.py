from ofa.utils.layers import ResidualBlock
from ofa.imagenet_classification.networks import ProxylessNASNets
from .modules import my_set_layer_from_config

__all__ = ["build_residual_block_from_config", "build_network_from_config"]


def build_residual_block_from_config(config):
    """从配置字典构建一个残差块（ResidualBlock）。

    残差块 = 主路径（conv/mobile_inverted_conv）+ shortcut 连接。
    TinyTL 插入 LiteResidualModule 后，config['conv'] 可能就是 LiteResidualModule 的配置，
    经过 my_set_layer_from_config 会正确构建出带轻量旁路的包裹层。
    """
    # 兼容两种配置键名：'conv'（TinyTL 风格）或 'mobile_inverted_conv'（OFA 原生风格）
    conv_config = config["conv"] if "conv" in config else config["mobile_inverted_conv"]
    conv = my_set_layer_from_config(conv_config)

    # shortcut 连接：可能是 Identity 或 1×1 卷积（通道数/分辨率不匹配时）
    shortcut = my_set_layer_from_config(config["shortcut"])

    return ResidualBlock(conv, shortcut)


def build_network_from_config(config):
    """从配置字典构建完整的 TinyTL 网络。

    网络结构：first_conv → N×blocks(ResidualBlock) → feature_mix_layer → classifier
    使用 OFA 的 ProxylessNASNets 作为骨干容器，但其中的层可能已被 LiteResidualModule 包裹。
    """
    # 构建网络的四个主要部分
    first_conv = my_set_layer_from_config(config["first_conv"])  # 第一层卷积（stem）
    feature_mix_layer = my_set_layer_from_config(
        config["feature_mix_layer"]
    )  # 特征混合层（通常为 1×1 Conv）
    classifier = my_set_layer_from_config(config["classifier"])  # 分类头（Linear 层）

    # 构建中间的所有残差块
    blocks = []
    for block_config in config["blocks"]:
        blocks.append(build_residual_block_from_config(block_config))

    # 用 ProxylessNASNets 容器组装完整网络
    net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)

    # 设置 BatchNorm 参数（momentum 和 eps）
    # TinyTL 中 BN 使用较小的 momentum (0.1)，因为训练数据量少、需要更快适应新分布
    if "bn" in config:
        net.set_bn_param(**config["bn"])
    else:
        net.set_bn_param(momentum=0.1, eps=1e-3)

    return net
