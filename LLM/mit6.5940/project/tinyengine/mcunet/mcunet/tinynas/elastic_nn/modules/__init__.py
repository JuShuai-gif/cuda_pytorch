# =============================================================================
# elastic_nn.modules 包初始化文件
#
# 本包提供构建弹性（动态）神经网络所需的"动态层"和"动态算子"。
# 这些动态模块的特点是：它们的计算图在推理时可以根据需要调整形状，
# 例如卷积核大小、输出通道数、扩展比等。
#
# 导出内容：
#   - DynamicMBConvLayer  : 动态的 Mobile Inverted Bottleneck 卷积层
#   - DynamicConvLayer     : 动态的普通卷积层
#   - DynamicLinearLayer   : 动态的全连接层
#   - DynamicSeparableConv2d : 动态深度可分离卷积
#   - DynamicPointConv2d     : 动态逐点卷积（1x1卷积）
#   - DynamicLinear          : 动态全连接
#   - DynamicBatchNorm2d     : 动态批归一化
#   - DynamicSE              : 动态 Squeeze-and-Excitation 模块
# =============================================================================

from .dynamic_layers import *
from .dynamic_op import *
