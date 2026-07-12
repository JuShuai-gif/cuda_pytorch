# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#
# =============================================================================
# ofa_proxyless.py — OFA-ProxylessNAS 超网络架构
#
# 本文件定义了 OFAProxylessNASNets 类，它是 ProxylessNAS 网络架构的
# OFA（Once for All）超网络版本，也是 MCUNet 中弹性搜索空间的核心。
#
# 核心概念：
#   1. "超网络"（Super-Network）：包含所有可能的子网络配置
#   2. 弹性维度：
#      - width_mult（宽度乘子）：控制每层通道数
#      - kernel_size（卷积核大小）：控制 depthwise conv 的核大小 [3,5,7] 等
#      - expand_ratio（扩展比）：控制 MBConv 的扩展比 [3,4,6] 等
#      - depth（深度）：控制每个 stage 的 block 数量
#   3. 训练策略：渐进式收缩（Progressive Shrinking）
#      先训练大子网络，逐渐收缩到小子网络
#
# 与 ProxylessNASNets 的关系：
#   继承自 ProxylessNASNets（定义在 tinynas/nn/networks/ 中）。
#   在 ProxylessNAS 的静态架构基础上，将各层替换为动态版本，
#   从而支持弹性的子网络采样。
# =============================================================================

import copy
import random

from ..modules import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
from ...nn.modules import ConvLayer, IdentityLayer, LinearLayer, MBInvertedConvLayer
from ...nn.networks import ProxylessNASNets, MobileInvertedResidualBlock
from ....utils import make_divisible, val2list

__all__ = ["OFAProxylessNASNets"]


class OFAProxylessNASNets(ProxylessNASNets):
    """
    ============================================================================
    OFA-ProxylessNAS 超网络。

    该类将 ProxylessNAS 的静态网络结构中的每一层替换为动态版本：
      - 普通 ConvLayer → DynamicConvLayer
      - MBInvertedConvLayer → DynamicMBConvLayer
      - LinearLayer → DynamicLinearLayer

    这样替换后，训练好的超网络可以抽取出任意配置的子网络进行部署。

    弹性维度：
      1. width_mult (宽度乘子):
         - 网络每层通道数 = base_width × width_mult
         - width_mult_list 例如 [0.5, 0.625, 0.75, 1.0]
         - 通过 DynamicPointConv2d 的切片机制实现
      2. kernel_size (卷积核大小):
         - 每个 MBConv 的 depthwise conv 可以选不同大小
         - 通过 DynamicSeparableConv2d 实现
      3. expand_ratio (扩展比):
         - MBConv 中扩展层的通道放大倍数
         - 通过 DynamicPointConv2d 的 active_out_channel 实现
      4. depth (深度):
         - 每个 stage 中可以保留不同数量的 block
         - 通过 runtime_depth 动态控制 forward 路径

    关键方法：
      - set_active_subnet():   手动设置当前使用的子网络配置
      - sample_active_subnet(): 随机采样一个子网络配置
      - get_active_subnet():    提取当前配置为静态子网络
      - set_constraint():       设置约束列表（渐进式收缩用）
      - clear_constraint():     清除所有约束
    ============================================================================
    """

    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.1,
        base_stage_width=None,
        width_mult_list=1.0,
        ks_list=3,
        expand_ratio_list=6,
        depth_list=4,
        no_mix_layer=False,
    ):
        """
        初始化 OFA-ProxylessNAS 超网络。

        参数说明:
          n_classes (int): 分类任务的类别数（默认 ImageNet 的 1000 类）
          bn_param (tuple): BN 层的 momentum 和 eps 参数
          dropout_rate (float): 分类器 Dropout 比率
          base_stage_width (str 或 list): 各 stage 的基础宽度
            - 'google': 使用 MobileNetV2 论文的宽度配置
            - None: 使用 ProxylessNAS 的宽度配置
          width_mult_list (float 或 list[float]): 宽度乘子列表 [0.5, 0.75, 1.0]
          ks_list (int 或 list[int]): 卷积核大小列表 [3, 5, 7]
          expand_ratio_list (int 或 list[int]): 扩展比列表 [3, 4, 6]
          depth_list (int 或 list[int]): 每 stage 的深度列表 [2, 3, 4]
          no_mix_layer (bool): 是否移除 classifier 前的 1x1 混合层
                               （为 True 可减少模型大小，适合 MCU 部署）
        """
        # =====================================================================
        # 步骤1：解析弹性维度参数，全部转为列表格式
        # =====================================================================
        self.width_mult_list = val2list(width_mult_list, 1)
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.base_stage_width = base_stage_width

        # 统一排序，保证配置的可重复性
        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        # =====================================================================
        # 步骤2：确定基础宽度配置
        # =====================================================================
        if base_stage_width == "google":
            # MobileNetV2 的宽度配置
            # 各 stage 的基础通道数（不包含第一层和最后一层）
            base_stage_width = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        else:
            # ProxylessNAS 的宽度配置（在 MCUNet 中被广泛使用）
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 1280]

        # =====================================================================
        # 步骤3：计算各 width_mult 下的实际通道数
        # =====================================================================
        # 输入层：3 通道 RGB → input_channel（用 8 的倍数对齐，利于硬件加速）
        input_channel = [
            make_divisible(base_stage_width[0] * width_mult, 8)
            for width_mult in self.width_mult_list
        ]
        # 第一个 block 的输出通道
        first_block_width = [
            make_divisible(base_stage_width[1] * width_mult, 8)
            for width_mult in self.width_mult_list
        ]
        # 最后一层（feature_mix_layer 后的通道数）
        # 注意：当 width_mult > 1.0 时对齐到 8 的倍数，否则保留原值
        last_channel = [
            make_divisible(base_stage_width[-1] * width_mult, 8)
            if width_mult > 1.0
            else base_stage_width[-1]
            for width_mult in self.width_mult_list
        ]

        # =====================================================================
        # 步骤4：构建超网络的各个组件
        # =====================================================================

        # --- 4a. 第一个卷积层（通常 stride=2，做快速下采样）---
        # 根据 width_mult_list 的长度决定是动态还是静态：
        #   - 如果只有一种宽度，用静态 ConvLayer（更高效）
        #   - 如果多种宽度，用动态 DynamicConvLayer（弹性）
        if len(input_channel) == 1:
            first_conv = ConvLayer(
                3,
                max(input_channel),
                kernel_size=3,
                stride=2,
                use_bn=True,
                act_func="relu6",
                ops_order="weight_bn_act",
            )
        else:
            first_conv = DynamicConvLayer(
                in_channel_list=val2list(3, len(input_channel)),
                out_channel_list=input_channel,
                kernel_size=3,
                stride=2,
                act_func="relu6",
            )

        # --- 4b. 第一个 MBConv block（expand_ratio 固定为 1，即无扩展）---
        if len(first_block_width) == 1:
            first_block_conv = MBInvertedConvLayer(
                in_channels=max(input_channel),
                out_channels=max(first_block_width),
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                act_func="relu6",
            )
        else:
            first_block_conv = DynamicMBConvLayer(
                in_channel_list=input_channel,
                out_channel_list=first_block_width,
                kernel_size_list=3,
                expand_ratio_list=1,
                stride=1,
                act_func="relu6",
            )
        first_block = MobileInvertedResidualBlock(first_block_conv, None)

        input_channel = first_block_width  # 更新当前输入通道数为第一个 block 的输出

        # --- 4c. 主网络：多个 stage，每个 stage 包含多个 MBConv block ---
        self.block_group_info = []  # 记录每个 stage 包含的 block 索引范围
        blocks = [first_block]
        _block_index = 1

        # 每个 stage 的步长配置（下采样发生在每个 stage 的第一个 block）
        stride_stages = [2, 2, 2, 1, 2, 1]

        if depth_list is None:
            # 默认使用 MobileNetV2 的深度设置
            n_block_list = [2, 3, 4, 3, 3, 1]
            self.depth_list = [4, 4]
            print("Use MobileNetV2 Depth Setting")
        else:
            # 使用 depth_list 中的最大值作为每个 stage 的 block 数量
            # 注意：最后一个 stage 只有 1 个 block
            n_block_list = [max(self.depth_list)] * 5 + [1]

        # 计算每个 stage（除去第一个和最后一个）的宽度列表
        width_list = []
        for base_width in base_stage_width[2:-1]:
            width = [
                make_divisible(base_width * width_mult, 8)
                for width_mult in self.width_mult_list
            ]
            width_list.append(width)

        # 遍历每个 stage，构建 MBConv blocks
        for width, n_block, s in zip(width_list, n_block_list, stride_stages):
            # 记录当前 stage 的 block 索引范围
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s  # 每个 stage 的第一个 block 做下采样
                else:
                    stride = 1  # 其余 block 保持分辨率

                # 创建动态 MBConv 层（支持弹性核大小和扩展比）
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(input_channel, 1),
                    out_channel_list=val2list(output_channel, 1),
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func="relu6",
                )

                # 如果输入输出通道相同且步长为 1，添加恒等快捷连接（残差连接）
                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = MobileInvertedResidualBlock(
                    mobile_inverted_conv, shortcut
                )
                blocks.append(mb_inverted_block)
                input_channel = output_channel

        # --- 4d. 特征混合层和分类器 ---
        # feature_mix_layer: 分类器前的 1x1 卷积，用于特征混合
        # no_mix_layer=True 时可移除以减小模型大小（适合 MCU 部署）
        if no_mix_layer:
            feature_mix_layer = None
            if len(self.width_mult_list) == 1:
                classifier = LinearLayer(
                    max(input_channel), n_classes, dropout_rate=dropout_rate
                )
            else:
                classifier = DynamicLinearLayer(
                    in_features_list=input_channel,
                    out_features=n_classes,
                    bias=True,
                    dropout_rate=dropout_rate,
                )
        else:
            if len(last_channel) == 1:
                feature_mix_layer = ConvLayer(
                    max(input_channel),
                    max(last_channel),
                    kernel_size=1,
                    use_bn=True,
                    act_func="relu6",
                )
                classifier = LinearLayer(
                    max(last_channel), n_classes, dropout_rate=dropout_rate
                )
            else:
                feature_mix_layer = DynamicConvLayer(
                    in_channel_list=input_channel,
                    out_channel_list=last_channel,
                    kernel_size=1,
                    stride=1,
                    act_func="relu6",
                )
                classifier = DynamicLinearLayer(
                    in_features_list=last_channel,
                    out_features=n_classes,
                    bias=True,
                    dropout_rate=dropout_rate,
                )

        # 调用父类 ProxylessNASNets 的构造函数，组装完整网络
        super(OFAProxylessNASNets, self).__init__(
            first_conv, blocks, feature_mix_layer, classifier
        )

        # 设置 BN 参数
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth: 每个 stage 实际使用的 block 数量（可动态调整）
        # 初始化时为每个 stage 使用全部 block
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    """ MyNetwork required methods """

    @staticmethod
    def name():
        """返回网络名称。"""
        return "OFAProxylessNASNets"

    def forward(self, x):
        """
        超网络前向传播。

        与静态网络的关键区别：
          - 使用 self.runtime_depth 控制每个 stage 实际经过的 block 数量
          - 每个 block 内部的动态层会根据当前的 active_* 配置自动调整

        前向流程:
          1. first_conv:   初始卷积下采样
          2. first_block:  第一个 MBConv block
          3. stages:       逐 stage 执行 block，每个 stage 只执行前 depth 个
          4. feature_mix:  1x1 特征混合（如果有）
          5. global pool:  全局平均池化
          6. classifier:   全连接分类
        """
        # 第一个卷积层
        x = self.first_conv(x)
        # 第一个 block
        x = self.blocks[0](x)

        # 主网络：逐 stage 执行
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]  # 当前 stage 使用的深度
            active_idx = block_idx[:depth]  # 只取前 depth 个 block
            for idx in active_idx:
                x = self.blocks[idx](x)

        # 特征混合层（如果存在）
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        # 全局平均池化 (N, C, H, W) -> (N, C)
        x = x.mean(3).mean(2)

        # 分类器
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        """返回网络结构的字符串表示（用于调试和可视化）。"""
        _str = self.first_conv.module_str + "\n"
        _str += self.blocks[0].module_str + "\n"

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"
        if self.feature_mix_layer is not None:
            _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str + "\n"
        return _str

    @property
    def config(self):
        """返回网络配置字典。"""
        return {
            "name": OFAProxylessNASNets.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "feature_mix_layer": None
            if self.feature_mix_layer is None
            else self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        """从配置重建网络（当前未实现）。"""
        raise ValueError("do not support this function")

    def load_weights_from_net(self, proxyless_model_dict):
        """
        从训练好的 ProxylessNAS 静态网络加载权重到 OFA 超网络。

        这是 OFA 训练流程中的一个初始化步骤：
          1. 先训练一个 ProxylessNAS 静态网络（或使用已有 checkpoint）
          2. 将其权重加载到 OFA 超网络中作为初始值
          3. 然后在超网络上进行 OFA 弹性训练

        参数:
          proxyless_model_dict (dict): ProxylessNAS 的 state_dict

        注意:
          - ProxylessNAS 和 OFA 超网络的 key 命名不同
            （OFA 在 conv/bn/linear 外面多包了一层）
          - 所以需要进行 key 的映射转换
        """
        model_dict = self.state_dict()
        for key in proxyless_model_dict:
            if key in model_dict:
                # key 直接匹配（通常不会发生）
                new_key = key
            elif ".bn.bn." in key:
                # OFA 的 BN 命名是 .bn.bn.（外层 DynamicBatchNorm2d.bn，
                # 内层是真实的 nn.BatchNorm2d），
                # 而 ProxylessNAS 只有 .bn.
                new_key = key.replace(".bn.bn.", ".bn.")
            elif ".conv.conv.weight" in key:
                # 类似地，OFA 的 conv 命名是 .conv.conv.weight
                new_key = key.replace(".conv.conv.weight", ".conv.weight")
            elif ".linear.linear." in key:
                new_key = key.replace(".linear.linear.", ".linear.")
            ##############################################################################
            # 反向映射（某些参数需要从 ProxylessNAS 切片到超网络）：
            elif ".linear." in key:
                new_key = key.replace(".linear.", ".linear.linear.")
            elif "bn." in key:
                new_key = key.replace("bn.", "bn.bn.")
            elif "conv.weight" in key:
                new_key = key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = proxyless_model_dict[key]
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, wid=None, ks=None, e=None, d=None):
        """
        手动设置当前激活的子网络配置。

        这是 OFA 框架的核心接口：
          调用后，后续的 forward() 将使用指定的子网络配置进行计算。

        参数:
          wid (int, optional): width_mult 的索引（0 到 len(width_mult_list)-1）
                               子网络使用 width_mult_list[wid] 作为宽度乘子
          ks (int 或 list[int], optional): 每个 block 的卷积核大小
                                            可以是单个值（所有 block 共用）或列表
          e (int 或 list[int], optional): 每个 block 的扩展比
          d (int 或 list[int], optional): 每个 stage 的深度

        实现方式：
          遍历网络中所有模块，找到具有弹性属性的模块并设置其激活值。
        """
        # === 设置宽度（channel 维度）===
        # 遍历所有模块，找到有 out_channel_list 属性的模块（即动态层）
        for m in self.modules():
            if hasattr(m, "out_channel_list"):
                if wid is not None:
                    # 从可选列表中选取指定索引的通道数
                    m.active_out_channel = m.out_channel_list[wid]
                else:
                    # 默认使用最大通道数
                    m.active_out_channel = max(m.out_channel_list)

        # === 设置卷积核大小和扩展比 ===
        # 将参数转换为列表格式（每个 block 对应一个值）
        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))

        # 遍历所有 MBConv block（从 index 1 开始，跳过第一个固定的 block）
        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.mobile_inverted_conv.active_kernel_size = k
            if e is not None:
                block.mobile_inverted_conv.active_expand_ratio = e

        # === 设置深度 ===
        for i, d in enumerate(depth):
            if d is not None:
                # runtime_depth 不能超过该 stage 实际的 block 数
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type="depth"):
        """
        设置约束列表（用于渐进式收缩训练）。

        渐进式收缩策略：
          1. 先训练最大的子网络空间（所有配置可选）
          2. 逐渐缩小搜索空间（减少可选配置）
          3. 最后专注于最小的子网络空间

        每次缩小空间时，通过此方法设置允许的配置值列表。

        参数:
          include_list (list): 允许的配置值列表
          constraint_type (str): 约束类型
            - 'depth':        深度约束
            - 'expand_ratio': 扩展比约束
            - 'kernel_size':  卷积核大小约束
            - 'width_mult':   宽度乘子约束
        """
        if constraint_type == "depth":
            self.__dict__["_depth_include_list"] = include_list.copy()
        elif constraint_type == "expand_ratio":
            self.__dict__["_expand_include_list"] = include_list.copy()
        elif constraint_type == "kernel_size":
            self.__dict__["_ks_include_list"] = include_list.copy()
        elif constraint_type == "width_mult":
            self.__dict__["_widthMult_include_list"] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        """
        清除所有约束列表。
        调用后，子网络采样将在完整的配置空间中进行。
        """
        self.__dict__["_depth_include_list"] = None
        self.__dict__["_expand_include_list"] = None
        self.__dict__["_ks_include_list"] = None
        self.__dict__["_widthMult_include_list"] = None

    def sample_active_subnet(self):
        """
        从超网络中随机采样一个子网络配置。

        这是 OFA 训练的关键步骤：
          每次训练迭代时，随机采样一个子网络，
          只对该子网络对应的权重部分计算梯度并更新。

        采样策略：
          - 支持约束列表（如果有约束，只在约束列表内采样）
          - 每个 block 独立采样 kernel_size 和 expand_ratio
          - 每个 stage 独立采样 depth
          - width_mult 全局只采样一个值

        返回:
          dict: 包含采样结果的字典
            {'wid': int, 'ks': list, 'e': list, 'd': list}
        """
        # 获取约束后的候选值列表（如果没有约束，使用完整列表）
        ks_candidates = (
            self.ks_list
            if self.__dict__.get("_ks_include_list", None) is None
            else self.__dict__["_ks_include_list"]
        )
        expand_candidates = (
            self.expand_ratio_list
            if self.__dict__.get("_expand_include_list", None) is None
            else self.__dict__["_expand_include_list"]
        )
        depth_candidates = (
            self.depth_list
            if self.__dict__.get("_depth_include_list", None) is None
            else self.__dict__["_depth_include_list"]
        )

        # === 采样卷积核大小 ===
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            # 如果每个 block 使用相同的候选集，复制为每个 block 一份
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)  # 每个 block 独立随机采样
            ks_setting.append(k)

        # === 采样扩展比 ===
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # === 采样深度 ===
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [
                depth_candidates for _ in range(len(self.block_group_info))
            ]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        # === 采样宽度乘子 ===
        # width_mult 采样放在最后，以保持随机性的确定性
        width_mult_setting = random.randint(0, len(self.width_mult_list) - 1)

        # 应用采样结果
        self.set_active_subnet(
            width_mult_setting, ks_setting, expand_setting, depth_setting
        )

        return {
            "wid": width_mult_setting,
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        """
        从超网络中提取当前激活的子网络，构建一个独立的静态网络。

        这是 OFA 框架的最后一步：
          训练完成后，选择目标子网络配置，调用此方法
          将其提取为独立的 ProxylessNASNets 静态网络，
          然后可以进一步进行微调、量化或直接部署。

        参数:
          preserve_weight (bool): 是否从超网络复制已训练好的权重

        返回:
          ProxylessNASNets: 一个静态子网络，可直接用于推理
        """

        # 辅助函数：如果模块是动态的，调用 get_active_subnet 提取静态版本
        # 否则直接深度拷贝
        def get_or_copy_subnet(m, **kwargs):
            if hasattr(m, "get_active_subnet"):
                out = m.get_active_subnet(preserve_weight=preserve_weight, **kwargs)
            else:
                out = copy.deepcopy(m)
            return out

        # === 提取第一个卷积层 ===
        first_conv = get_or_copy_subnet(self.first_conv, in_channel=3)
        input_channel = first_conv.out_channels

        # === 提取第一个 block ===
        blocks = [
            MobileInvertedResidualBlock(
                get_or_copy_subnet(
                    self.blocks[0].mobile_inverted_conv, in_channel=input_channel
                ),
                copy.deepcopy(self.blocks[0].shortcut),
            )
        ]
        input_channel = blocks[0].mobile_inverted_conv.out_channels

        # === 提取主网络的每个 stage ===
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]  # 只取当前深度配置下的 block
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    MobileInvertedResidualBlock(
                        self.blocks[idx].mobile_inverted_conv.get_active_subnet(
                            input_channel, preserve_weight
                        ),
                        copy.deepcopy(self.blocks[idx].shortcut),
                    )
                )
                input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
            blocks += stage_blocks

        # === 提取特征混合层和分类器 ===
        feature_mix_layer = get_or_copy_subnet(
            self.feature_mix_layer, input_channel=input_channel
        )
        input_channel = (
            feature_mix_layer.out_channels
            if feature_mix_layer is not None
            else input_channel
        )
        classifier = get_or_copy_subnet(self.classifier, in_features=input_channel)

        # 组装静态网络
        _subnet = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        """
        重新组织所有 block 的中间通道权重顺序。

        在渐进式收缩训练中，当缩小扩展比时调用此方法：
          按照通道重要性重新排列权重，使得被裁剪的通道
          （重要性较低的通道）排在后面。

        参数:
          expand_ratio_stage (int): 当前收缩阶段的索引
        """
        if len(self.width_mult_list) > 1:
            print(
                " * WARNING: sorting is not implemented right for multiple width-mult"
            )

        # 对主网络中的每个 block 进行权重重排
        for block in self.blocks[1:]:
            block.mobile_inverted_conv.re_organize_middle_weights(expand_ratio_stage)
