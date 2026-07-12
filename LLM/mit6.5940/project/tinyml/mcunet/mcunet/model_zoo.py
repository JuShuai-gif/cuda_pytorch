import json
import torch

from .tinynas.nn.networks import ProxylessNASNets
from .utils import download_url

__all__ = ["net_id_list", "build_model", "download_tflite"]

""" Note: all the memory and latency profiling is done with TinyEngine """
NET_INFO = {
    ##### imagenet models ######
    # mcunet models
    "mcunet-10fps": {
        "net_name": "mcunet-10fps_imagenet",
        "description": "MCUNet model that runs 10fps on STM32F746 (ImageNet)",
    },
    "mcunet-5fps": {
        "net_name": "mcunet-5fps_imagenet",
        "description": "MCUNet model that runs 5fps on STM32F746 (ImageNet)",
    },
    "mcunet-256kB": {
        "net_name": "mcunet-256kb-1mb_imagenet",
        "description": "MCUNet model that fits 256KB SRAM and 1MB Flash (ImageNet)",
    },
    "mcunet-320kB": {
        "net_name": "mcunet-320kb-1mb_imagenet",
        "description": "MCUNet model that fits 320KB SRAM and 1MB Flash (ImageNet)",
    },
    "mcunet-512kB": {
        "net_name": "mcunet-512kb-2mb_imagenet",
        "description": "MCUNet model that fits 512KB SRAM and 2MB Flash (ImageNet)",
    },
    # baseline models
    "mbv2-320kB": {
        "net_name": "mbv2-w0.35-r144_imagenet",
        "description": "scaled MobileNetV2 that fits 320KB SRAM and 1MB Flash (ImageNet)",
    },
    "proxyless-320kB": {
        "net_name": "proxyless-w0.3-r176_imagenet",
        "description": "scaled ProxylessNet that fits 320KB SRAM and 1MB Flash (ImageNet)",
    },
    ##### vww models (to be updated) ######
    "mcunet-10fps-vww": {
        "net_name": "mcunet-10fps_vww",
        "description": "MCUNet model that runs 10fps on STM32F746 (VWW)",
    },
    "mcunet-5fps-vww": {
        "net_name": "mcunet-5fps_vww",
        "description": "MCUNet model that runs 5fps on STM32F746 (VWW)",
    },
    "mcunet-320kB-vww": {
        "net_name": "mcunet-320kb-1mb_vww",
        "description": "MCUNet model that fits 320KB SRAM and 1MB Flash (VWW)",
    },
}

net_id_list = list(NET_INFO.keys())

url_base = "https://hanlab.mit.edu/projects/tinyml/mcunet/release/"


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


def download_tflite(net_id):
    assert net_id in NET_INFO, "Invalid net_id! Select one from {})".format(
        list(NET_INFO.keys())
    )
    net_info = NET_INFO[net_id]
    tflite_url = url_base + net_info["net_name"] + ".tflite"
    return download_url(tflite_url)  # the file path of the downloaded tflite model
