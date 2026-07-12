# Code adapted from Once for All: Train One Network and Specialize it for Efficient Deployment
#
# =============================================================================
# dynamic_layers.py — 动态层（Dynamic Layers）
#
# 本文件在 dynamic_op.py 提供的"动态算子"之上，
# 构建了更高层次的"动态层"模块。
# 这些动态层直接对应于实际网络中的构建块（building blocks）。
#
# 核心层类型：
#   - DynamicMBConvLayer  : MobileNetV2/V3 风格的 Inverted Bottleneck 层
#                           （expand → depthwise conv → pointwise conv）
#   - DynamicConvLayer     : 普通卷积层（conv → bn → act）
#   - DynamicLinearLayer   : 全连接层（dropout → linear）
#
# 每个动态层支持在推理时弹性调整其内部结构参数：
#   - 输出通道数（out_channels）
#   - 卷积核大小（kernel_size）
#   - 扩展比（expand_ratio）
#   - 深度（depth，通过上层控制）
# =============================================================================

from collections import OrderedDict
import copy
import torch
import torch.nn as nn

from ...nn.modules import MBInvertedConvLayer, ConvLayer, LinearLayer
from .dynamic_op import *
from ....utils import (
    adjust_bn_according_to_idx,
    copy_bn,
    make_divisible,
    SEModule,
    MyModule,
    val2list,
    get_net_device,
    build_activation,
)

__all__ = ["DynamicMBConvLayer", "DynamicConvLayer", "DynamicLinearLayer"]


class DynamicMBConvLayer(MyModule):
    """
    ============================================================================
    动态 Mobile Inverted Bottleneck 卷积层。

    这是 MobileNetV2 提出的核心构建块，也是 MCUNet/OFA 中最重要的动态层。
    结构（当 expand_ratio > 1 时）：
      输入 → [1x1 conv (expand)] → BN → Act → [depthwise conv] → BN → Act
            → [1x1 conv (project)] → BN → 输出

    当 expand_ratio = 1 时，省略 expansion 阶段（inverted_bottleneck = None）。

    动态维度：
      - in_channel:     输入通道可选值列表
      - out_channel:    输出通道可选值列表
      - kernel_size:    卷积核大小可选值列表（[3, 5, 7] 等）
      - expand_ratio:   扩展比可选值列表（[3, 4, 6] 等）

    权重共享机制：
      三个子模块（inverted_bottleneck, depth_conv, point_linear）各自
      包含一个"动态算子"，它们以最大值初始化，在 forward 时通过切片
      选择当前子网络的配置。所有子网络共享同一组权重参数。
    ============================================================================
    """

    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        expand_ratio_list=6,
        stride=1,
        act_func="relu6",
        use_se=False,
    ):
        """
        初始化动态 MBConv 层。

        参数说明:
          in_channel_list (list[int]): 可能的输入通道数列表（弹性宽度）
          out_channel_list (list[int]): 可能的输出通道数列表
          kernel_size_list (int 或 list[int]): 支持的卷积核大小
          expand_ratio_list (int 或 list[int]): 支持的扩展比
          stride (int): 卷积步长
          act_func (str): 激活函数类型，默认为 'relu6'
          use_se (bool): 是否在 depthwise conv 后添加 SE 模块

        设计决策:
          - 所有子模块都以"最大可能配置"初始化，确保支持所有子网络
          - 最大中间通道 = max(in_channel) * max(expand_ratio)
          - 如果所有扩展比都为 1，则省略 inverted_bottleneck 以节省计算
        """
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = val2list(kernel_size_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se

        # === 构建子模块（全部以最大配置初始化） ===

        # 最大中间通道数 = 最大输入通道 × 最大扩展比
        max_middle_channel = round(
            max(self.in_channel_list) * max(self.expand_ratio_list)
        )

        # 1) Inverted Bottleneck（扩展层）：1x1 卷积，将通道数从 in 扩展到 middle
        #    当 expand_ratio = 1 时不需要扩展，设为 None
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            DynamicPointConv2d(
                                max(self.in_channel_list), max_middle_channel
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        # 2) Depthwise Convolution（深度卷积）：每个通道独立卷
        self.depth_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicSeparableConv2d(
                            max_middle_channel, self.kernel_size_list, self.stride
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )
        if self.use_se:
            # 可选：在 depthwise conv 后添加 SE 层，增强通道注意力
            self.depth_conv.add_module("se", DynamicSE(max_middle_channel))

        # 3) Pointwise Linear（投影层）：1x1 卷积，将通道数从 middle 投影到 out
        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicPointConv2d(
                            max_middle_channel, max(self.out_channel_list)
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        # 当前激活的子网络配置（默认均为最大值，即选择最大子网络）
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        """
        前向传播：根据当前激活的动态配置执行 MBConv。

        流程:
          1. 设置各子模块的激活参数（基于当前选择的子网络配置）
          2. 依次执行：inverted_bottleneck → depth_conv → point_linear

        动态切换机制:
          - 在每次 forward 之前，外部代码（如 set_active_subnet）会修改
            self.active_* 属性，从而控制本次前向使用的子网络结构。
          - 子模块（如 inverted_bottleneck.conv）的 active_out_channel
            被设置为当前配置对应的值，从而使内部的动态算子使用切片后的权重。
        """
        in_channel = x.size(1)

        # 根据当前扩展比动态计算中间通道数，并通知 inverted_bottleneck
        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = make_divisible(
                round(in_channel * self.active_expand_ratio), 8
            )

        # 通知 depth_conv 和 point_linear 当前的激活参数
        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        # 前向传播
        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        """
        返回当前激活子网络的字符串表示，用于调试和日志。
        格式: 'SE(O128, E6.0, K5)' 表示输出128通道，扩展比6，核大小5
        """
        if self.use_se:
            return "SE(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )
        else:
            return "(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )

    @property
    def config(self):
        """返回该层的配置字典，可用于后续重建。"""
        return {
            "name": DynamicMBConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "expand_ratio_list": self.expand_ratio_list,
            "stride": self.stride,
            "act_func": self.act_func,
            "use_se": self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        """从配置字典重建一个动态 MBConv 层。"""
        return DynamicMBConvLayer(**config)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        """
        从动态超网络中提取一个"静态"子网络。

        这是 OFA 框架的关键方法之一：
          训练时使用动态超网络（所有子网络共享权重）；
          部署时调用此方法，将当前激活的子网络提取为一个独立的静态网络，
          这样可以去除动态切片开销，获得更高效的推理。

        参数:
          in_channel (int): 输入通道数
          preserve_weight (bool): 是否从超网络复制训练好的权重

        返回:
          MBInvertedConvLayer: 一个静态的 MBConv 层（非动态），可直接用于推理
        """
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)

        # 创建对应的静态层（MBInvertedConvLayer 是普通静态层）
        sub_layer = MBInvertedConvLayer(
            in_channel,
            self.active_out_channel,
            self.active_kernel_size,
            self.stride,
            self.active_expand_ratio,
            act_func=self.act_func,
            mid_channels=middle_channel,
            use_se=self.use_se,
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            # 如果不保留权重，只返回空壳，通常用于测试架构兼容性
            return sub_layer

        # === 从动态超网络复制权重到静态子网络 ===
        # 复制 inverted_bottleneck（1x1 扩展卷积 + BN）
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[
                    :middle_channel, :in_channel, :, :
                ]
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        # 复制 depthwise conv（深度可分离卷积 + BN + SE）
        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(
                middle_channel, self.active_kernel_size
            ).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            # 复制 SE 模块的权重
            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid, :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.fc.reduce.bias.data[:se_mid]
            )

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel, :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.fc.expand.bias.data[:middle_channel]
            )

        # 复制 point_linear（投影 1x1 卷积 + BN）
        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[
                : self.active_out_channel, :middle_channel, :, :
            ]
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        """
        重新组织中间通道的权重顺序。

        在 OFA 渐进式收缩训练（progressive shrinking）中，
        当从较大的扩展比切换到较小的扩展比时，哪些中间通道被保留会影响性能。
        此方法根据 point_linear 权重的重要性对所有中间通道进行排序，
        重要性高的通道排在前面，这样在小扩展比模式下，
        被切掉的（后面的）通道是最不重要的。

        参数:
          expand_ratio_stage (int): 当前渐进式收缩的阶段索引，
                                    用于确定目标宽度（通道数）
        """
        # 计算 point_linear 权重在输入维度上的 L1 范数作为重要性指标
        # 对每个输入通道，求其对应所有权重参数的绝对值之和
        importance = torch.sum(
            torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3)
        )  # over input ch

        if expand_ratio_stage > 0:
            # 如果处于渐进式收缩阶段，对超出目标宽度的通道赋予负重要性
            # 这样排序时它们会被排到最后，确保被裁剪掉
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width = sorted_expand_list[expand_ratio_stage]
            target_width = round(max(self.in_channel_list) * target_width)
            # 从 target_width 开始，重要性递减（负值），确保排在末尾
            importance[target_width:] = torch.arange(
                0, target_width - importance.size(0), -1
            )

        # 按重要性降序排序，得到通道索引的排序
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

        # 按照排序后的索引重新排列 point_linear 的输入通道维度
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        # 同时也需要调整 depth_conv 的 BN 层和 depthwise conv 的权重顺序
        # 因为 depth_conv 的输出通道对应 point_linear 的输入通道
        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            # 如果使用了 SE 模块，也需要调整 SE 的权重顺序以保持一致
            # se_expand: 输出维度对应中间通道，按排序重新排列
            se_expand = self.depth_conv.se.fc.expand
            se_expand.weight.data = torch.index_select(
                se_expand.weight.data, 0, sorted_idx
            )
            se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
            # se_reduce: 输入维度对应中间通道，按排序重新排列
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(
                se_reduce.weight.data, 1, sorted_idx
            )
            # 对 SE 内部也做一次重要性排序
            se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)

        # TODO: 如果 inverted_bottleneck 不存在（expand_ratio=1），
        #       需要调整前一层的输出通道顺序以匹配
        if self.inverted_bottleneck is not None:
            # 调整 inverted_bottleneck 的 BN 和权重顺序
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            # 返回排序索引，供前一层使用
            return sorted_idx


class DynamicConvLayer(MyModule):
    """
    ============================================================================
    动态普通卷积层。

    对于步长为 2 或非 MBConv 结构的场景（如网络的第一个卷积层），
    使用普通的 Conv → BN → Act 结构。

    相比于 DynamicMBConvLayer，该层更简单：
      - 没有 expansion 和 projection 阶段
      - 只有一个 DynamicPointConv2d 用于通道变换
      - 支持可选的批归一化和激活函数

    注：这里的命名虽然是 PointConv2d，但可以通过 kernel_size 参数
        实现任意大小的卷积（如 3x3、5x5 等），不仅仅是 1x1。
    ============================================================================
    """

    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bn=True,
        act_func="relu6",
    ):
        """
        参数:
          in_channel_list (list[int]):  可能的输入通道数列表
          out_channel_list (list[int]): 可能的输出通道数列表
          kernel_size (int):             卷积核大小
          stride (int):                  步长
          dilation (int):                膨胀率
          use_bn (bool):                 是否使用批归一化
          act_func (str):                激活函数类型
        """
        super(DynamicConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        # 以最大配置初始化动态逐点卷积
        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func, inplace=True)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        """
        前向传播：Conv → BN（可选）→ Act。
        根据 self.active_out_channel 动态选择输出通道数。
        """
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self):
        return "DyConv(O%d, K%d, S%d)" % (
            self.active_out_channel,
            self.kernel_size,
            self.stride,
        )

    @property
    def config(self):
        return {
            "name": DynamicConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        """
        从动态超网络中提取静态子网络（普通卷积层版本）。
        与 DynamicMBConvLayer.get_active_subnet 原理相同。
        """
        sub_layer = ConvLayer(
            in_channel,
            self.active_out_channel,
            self.kernel_size,
            self.stride,
            self.dilation,
            use_bn=self.use_bn,
            act_func=self.act_func,
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(
            self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :]
        )
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer


class DynamicLinearLayer(MyModule):
    """
    ============================================================================
    动态全连接层。

    用于网络的分类器头部（classifier）。
    支持：
      - 输入特征数的弹性选择（对应前一层的输出通道数变化）
      - 可选的 Dropout 正则化
      - 偏置项

    内部使用 DynamicLinear 作为动态算子。
    ============================================================================
    """

    def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0):
        """
        参数:
          in_features_list (list[int]): 可能的输入特征数列表
          out_features (int):           输出特征数（通常是类别数）
          bias (bool):                  是否包含偏置
          dropout_rate (float):         Dropout 比率，0 表示不使用 Dropout
        """
        super(DynamicLinearLayer, self).__init__()

        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list),
            max_out_features=self.out_features,
            bias=self.bias,
        )

    def forward(self, x):
        """前向传播：Dropout（可选）→ Linear。"""
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return "DyLinear(%d)" % self.out_features

    @property
    def config(self):
        return {
            "name": DynamicLinear.__name__,
            "in_features_list": self.in_features_list,
            "out_features": self.out_features,
            "bias": self.bias,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        """
        从动态全连接层中提取静态子网络。
        """
        sub_layer = LinearLayer(
            in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate
        )
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_(
            self.linear.linear.weight.data[: self.out_features, :in_features]
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(
                self.linear.linear.bias.data[: self.out_features]
            )
        return sub_layer
