# =============================================================================
# elastic_nn.networks 包初始化文件
#
# 本包提供基于弹性（动态）模块构建的完整超网络架构。
# 这些网络使用 DynamicMBConvLayer / DynamicConvLayer / DynamicLinearLayer
# 等动态层构建，支持在推理时灵活选择子网络结构。
#
# 当前支持的架构：
#   - OFAProxylessNASNets: ProxylessNAS 架构的 OFA 超网络版本
#     （在 MCUNet 中作为搜索空间的基础架构）
# =============================================================================

from .ofa_proxyless import *
