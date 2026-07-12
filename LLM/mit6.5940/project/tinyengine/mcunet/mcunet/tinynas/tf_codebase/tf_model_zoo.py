# =============================================================================
# tf_model_zoo.py — TF 模型加载工具
#
# 本文件提供从文件加载预训练 TF 模型的便捷接口。
# 它是 PyTorch → TFLite 转换管线的入口点之一。
#
# 主要功能：
#   1. 从 JSON 配置文件加载网络架构
#   2. 从 pickle 文件加载预训练权重（TF 格式）
#   3. 构建 ProxylessNASNets 的 TF 实例
#
# 使用场景：
#   - 直接加载预训练 TF 模型进行推理
#   - 作为 generate_tflite.py 的基础构件
# =============================================================================

import json
import pickle

from .tf_modules import ProxylessNASNets


def proxyless_base(
    pretrained=True,
    net_config=None,
    net_weight=None,
    graph=None,
    sess=None,
    is_training=True,
    images=None,
    img_size=None,
    only_train=True,
    latency=-1,
):
    """从配置文件加载 ProxylessNAS / MCUNet 的 TF 模型

    这是一个高级便捷函数，用于：
        1. 从 JSON 文件读取网络架构配置
        2. 从 pickle 文件加载预训练 TF 权重
        3. 创建并初始化 ProxylessNASNets 实例

    参数说明:
        pretrained:    是否加载预训练权重。为 True 时需提供 net_weight 路径
        net_config:    网络配置文件路径（JSON 格式）
        net_weight:    预训练权重文件路径（pickle 格式，.tfinit 后缀）
        graph:         外部 TF Graph（可空）
        sess:          外部 TF Session（可空）
        is_training:   训练模式标志
        images:        外部输入图像张量（可空）
        img_size:      输入图像尺寸
        only_train:    是否为仅训练模式（仅影响打印信息）
        latency:       目标延迟（毫秒），用于选择不同延迟档位的模型

    文件路径约定:
        - 配置文件: finetune/{train|train+val}/{latency}ms/net.config
        - 权重文件: finetune/{train|train+val}/{latency}ms/1001.tfinit

    设计说明:
        - 权重文件的扩展名为 .tfinit（TensorFlow Initializer 的缩写）
        - 权重以 pickle 序列化的字典形式存储
        - 类别数为 1001（ImageNet 1000 类 + 1 个背景类）
        - 路径中的 'train' 或 'train+val' 表示使用的数据集划分

    返回值:
        ProxylessNASNets 实例（已初始化，可直接用于推理）
    """
    # ---- 打印延迟信息 ----
    # 根据 only_train 标志区分显示训练集模型和全数据集模型
    if only_train:
        print("#" * 50, "TRAIN {}ms".format(latency), "#" * 50)
    else:
        print("#" * 50, "ALL {}ms".format(latency), "#" * 50)

    # 验证网络配置是否存在
    assert net_config is not None, "Please input a network config"

    # 确定配置前缀：'train' 或 'train+val'
    prefix = "train" if only_train else "train+val"

    # 读取网络架构的 JSON 配置文件
    # 文件路径示例: finetune/train/100ms/net.config
    net_config_json = json.load(
        open("finetune/{}/{}ms/net.config".format(prefix, latency), "r")
    )

    # ---- 加载预训练权重 ----
    if pretrained:
        # 验证权重文件路径
        assert net_weight is not None, "Please specify network weights"
        # 从 pickle 文件加载 TF 权重字典
        # 权重字典的 key 是 TF variable_scope 路径
        # value 是对应的 NumPy 数组
        init = pickle.load(
            open("finetune/{}/{}ms/1001.tfinit".format(prefix, latency), "rb")
        )
    else:
        # 不使用预训练权重，随机初始化
        init = None

    # ---- 创建并初始化网络 ----
    net = ProxylessNASNets(
        net_config_json, init, graph, sess, is_training, images, img_size
    )

    return net
