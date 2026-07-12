import json
import torch

from .tinynas.nn.networks import ProxylessNASNets
from .utils import download_url

__all__ = ["net_id_list", "build_model", "download_tflite"]

# 注意：以下所有内存和延迟数据均基于 TinyEngine 实测
# NET_INFO 是模型库注册表：net_id -> {下载用的文件名, 描述}
NET_INFO = {
    ##### ImageNet 分类模型 ######
    # MCUNet 系列模型
    "mcunet-in0": {
        "net_name": "mcunet-10fps_imagenet",
        "description": "MCUNet 模型，在 STM32F746 上运行 10fps（ImageNet）",
    },
    "mcunet-in1": {
        "net_name": "mcunet-5fps_imagenet",
        "description": "MCUNet 模型，在 STM32F746 上运行 5fps（ImageNet）",
    },
    "mcunet-in2": {
        "net_name": "mcunet-256kb-1mb_imagenet",
        "description": "MCUNet 模型，适配 256KB SRAM 与 1MB Flash（ImageNet）",
    },
    "mcunet-in3": {
        "net_name": "mcunet-320kb-1mb_imagenet",
        "description": "MCUNet 模型，适配 320KB SRAM 与 1MB Flash（ImageNet）",
    },
    "mcunet-in4": {
        "net_name": "mcunet-512kb-2mb_imagenet",
        "description": "MCUNet 模型，适配 512KB SRAM 与 2MB Flash（ImageNet）",
    },
    # 基线对比模型（手工缩放的 MobileNetV2 / ProxylessNet）
    "mbv2-w0.35": {
        "net_name": "mbv2-w0.35-r144_imagenet",
        "description": "缩放版 MobileNetV2，适配 320KB SRAM 与 1MB Flash（ImageNet）",
    },
    "proxyless-w0.3": {
        "net_name": "proxyless-w0.3-r176_imagenet",
        "description": "缩放版 ProxylessNet，适配 320KB SRAM 与 1MB Flash（ImageNet）",
    },
    ##### VWW（Visual Wake Words）模型 ######
    "mcunet-vww0": {
        "net_name": "mcunet-10fps_vww",
        "description": "MCUNet 模型，在 STM32F746 上运行 10fps（VWW）",
    },
    "mcunet-vww1": {
        "net_name": "mcunet-5fps_vww",
        "description": "MCUNet 模型，在 STM32F746 上运行 5fps（VWW）",
    },
    "mcunet-vww2": {
        "net_name": "mcunet-320kb-1mb_vww",
        "description": "MCUNet 模型，适配 320KB SRAM 与 1MB Flash（VWW）",
    },
    ##### 目标检测演示模型 ######
    # 注意：该模型仅提供 tflite 格式
    "person-det": {
        "net_name": "person-det",
        "description": "演示用的行人检测模型",
    },
}

net_id_list = list(NET_INFO.keys())

url_base = "https://hanlab18.mit.edu/projects/tinyml/mcunet/release/"

# 从远程下载模型配置与权重，构建 PyTorch 模型并加载预训练参数


def build_model(net_id, pretrained=True):
    # 检查 net_id 是否合法（必须在 NET_INFO 注册表中）
    assert net_id in NET_INFO, "Invalid net_id! Select one from {})".format(
        list(NET_INFO.keys())
    )
    net_info = NET_INFO[net_id]

    # 构造远程 URL：下载网络结构配置 JSON 和预训练权重 .pth
    net_config_url = url_base + net_info["net_name"] + ".json"
    sd_url = url_base + net_info["net_name"] + ".pth"

    # 下载 JSON 配置文件，获取网络结构参数和输入分辨率
    net_config = json.load(open(download_url(net_config_url)))
    resolution = net_config["resolution"]
    # 根据 JSON 配置构建 ProxylessNAS 模型结构（此时权重随机）
    model = ProxylessNASNets.build_from_config(net_config)

    if pretrained:
        # 下载预训练权重文件，加载到 CPU
        sd = torch.load(download_url(sd_url), map_location="cpu")
        model.load_state_dict(sd["state_dict"])
    # 返回模型、输入分辨率、模型描述
    return model, resolution, net_info["description"]


# ============================================================================
# download_tflite —— 从 MIT HAN Lab 模型库下载指定 net_id 对应的 .tflite 文件
#
# TFLite (TensorFlow Lite) 是经过量化的模型格式（通常为 INT8 量化），
# 可以直接部署到 MCU 上通过 TinyEngine 的代码生成器转换为 C 代码。
#
# 参数:
#   net_id (str)  : 模型标识符，必须存在于 NET_INFO 注册表中，
#                   例如 "mcunet-vww1"、"mcunet-in0" 等
#
# 返回:
#   str : 下载到本地的 .tflite 模型文件的绝对路径。
#         该路径通常为 ~/.torch/mcunet/<net_name>.tflite
#         如果下载失败，返回 None（由 download_url 内部处理）
#
# 使用示例:
#   tflite_path = download_tflite("mcunet-vww1")
#   print(tflite_path)  # 输出: /home/user/.torch/mcunet/mcunet-5fps_vww.tflite
#
# 说明:
#   - 文件会缓存到本地，重复调用不会重复下载
#   - 模型存储在 hanlab18.mit.edu 的公开服务器上
#   - 下载的 .tflite 文件同时包含了权重参数和量化参数（scale/zero_point），
#     可以直接传给 TfliteConvertor 解析并生成 MCU 部署代码
# ============================================================================
def download_tflite(net_id):
    # 第一步：校验 net_id 是否合法
    # NET_INFO 字典中存储了所有支持的模型信息，如果传入不在注册表中的 ID，
    # 直接报错并列出所有可用的选择，避免用户下载不存在的模型
    assert net_id in NET_INFO, "Invalid net_id! Select one from {})".format(
        list(NET_INFO.keys())
    )
    # 第二步：从注册表中查找到该模型的元信息（net_name、description 等）
    net_info = NET_INFO[net_id]

    # 第三步：构造 TFLite 文件的远程下载 URL
    # 文件命名规则为 <net_name>.tflite，存储在远程服务器的根目录下
    # 例如：mcunet-5fps_vww → https://hanlab18.mit.edu/.../mcunet-5fps_vww.tflite
    tflite_url = url_base + net_info["net_name"] + ".tflite"

    # 第四步：调用 download_url 执行实际下载
    # download_url 负责：
    #   1. 创建本地缓存目录 ~/.torch/mcunet/
    #   2. 检查是否已经下载过（避免重复下载）
    #   3. 如果未下载则从远程拉取文件
    #   4. 返回本地缓存文件的绝对路径
    return download_url(tflite_url)  # 返回下载的 tflite 模型文件路径
