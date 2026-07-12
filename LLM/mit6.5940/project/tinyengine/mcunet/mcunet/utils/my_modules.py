# ============================================================================
# my_modules.py —— 自定义网络模块和初始化工具
#
# 来源：
#   Once for All: Train One Network and Specialize it for Efficient Deployment
#   Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
#   International Conference on Learning Representations (ICLR), 2020.
#
# 本文件提供了 MCUNet / OFA 框架中核心的自定义模块和工具函数，包括：
#   1. set_bn_param / get_bn_param —— BN/GN 参数的设置与获取
#   2. replace_bn_with_gn          —— 将 BatchNorm 替换为 GroupNorm
#   3. replace_conv2d_with_my_conv2d—— 将 Conv2d 替换为带 Weight Standardization 的版本
#   4. init_models                 —— 模型参数初始化（He 初始化等）
#   5. MyConv2d                    —— 支持 Weight Standardization 的自定义卷积层
#   6. MyModule / MyNetwork        —— 自定义模块和网络的基类
#
# Weight Standardization (WS) 是一种微调技术，通过对卷积权重做标准化来平滑损失
#  landscape，提升训练稳定性。配合 GroupNorm 使用效果尤佳。
# ============================================================================

import math
import torch.nn as nn
import torch.nn.functional as F

from .common_tools import min_divisible_value

# __all__ 控制 from my_modules import * 时暴露的公共接口
__all__ = [
    "MyModule",
    "MyNetwork",
    "init_models",
    "set_bn_param",
    "get_bn_param",
    "replace_bn_with_gn",
    "MyConv2d",
    "replace_conv2d_with_my_conv2d",
]


# ============================================================================
# set_bn_param
# ============================================================================
# 功能：统一设置网络中所有正则化层（BN / GN）的超参数。
#
# 此函数同时完成三件事：
#   1. 可选地将 BN 替换为 GroupNorm（如果 gn_channel_per_group 不为 None）
#   2. 设置所有 BN 层的 momentum 和 eps
#   3. 可选地将普通 Conv2d 替换为带 Weight Standardization 的 MyConv2d
#
# 参数：
#   net                    —— 目标网络模型
#   momentum               —— BN 的动量系数
#   eps                    —— BN/GN 的 epsilon（防止除零的小常数）
#   gn_channel_per_group   —— GroupNorm 每组通道数。如果为 None，不替换 BN
#   ws_eps                 —— Weight Standardization 的 epsilon。如果为 None，不替换 Conv2d
#   **kwargs               —— 预留的未来扩展参数
# ============================================================================
def set_bn_param(net, momentum, eps, gn_channel_per_group=None, ws_eps=None, **kwargs):
    # 如果有 gn_channel_per_group 指定，将 BN 替换为 GroupNorm
    replace_bn_with_gn(net, gn_channel_per_group)

    # 遍历所有模块，设置正则化层参数
    for m in net.modules():
        if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            # BN 层的 momentum（控制 running_mean 更新的平滑程度）
            m.momentum = momentum
            # BN 层的 eps（防止除零）
            m.eps = eps
        elif isinstance(m, nn.GroupNorm):
            # GroupNorm 只有 eps，没有 momentum
            m.eps = eps

    # 如果有 ws_eps 指定，将 Conv2d 替换为带 Weight Standardization 的 MyConv2d
    replace_conv2d_with_my_conv2d(net, ws_eps)
    return


# ============================================================================
# get_bn_param
# ============================================================================
# 功能：获取网络当前使用的正则化层参数。
#
# 返回值：
#   字典，包含 momentum、eps、gn_channel_per_group、ws_eps 等参数。
#   如果网络中没有正则化层，返回 None。
#
# 注意：
#   这个函数会查找网络中的第一个 BN 或 GN 层来读取参数，假设所有层
#   使用相同的参数设置。这在 OFA 框架中通常是成立的。
# ============================================================================
def get_bn_param(net):
    ws_eps = None
    # 先找第一个 MyConv2d 层获取 ws_eps
    for m in net.modules():
        if isinstance(m, MyConv2d):
            ws_eps = m.WS_EPS
            break
    # 再找第一个正则化层获取 BN/GN 参数
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            return {
                "momentum": m.momentum,
                "eps": m.eps,
                "ws_eps": ws_eps,
            }
        elif isinstance(m, nn.GroupNorm):
            # 从 GroupNorm 计算每组通道数
            # num_channels // num_groups = 每组通道数
            return {
                "momentum": None,  # GN 没有 momentum 参数
                "eps": m.eps,
                "gn_channel_per_group": m.num_channels // m.num_groups,
                "ws_eps": ws_eps,
            }
    return None


# ============================================================================
# replace_bn_with_gn
# ============================================================================
# 功能：将模型中的所有 nn.BatchNorm2d 层替换为 nn.GroupNorm 层。
#
# 设计背景：
#   在 MCUNet 这类超轻量级网络中，BatchNorm 在小 batch size 场景下表现不佳
#   （因为 batch 统计量不稳定）。GroupNorm 不依赖于 batch 维度，对 batch size
#   不敏感，更适合 MCUNet 的部署场景（推理时 batch size 通常为 1）。
#
#   此外，Weight Standardization + GroupNorm 的组合已被证明在小 batch 训练
#   中能取得与 BN 相当甚至更好的效果。
#
# 参数：
#   model                —— 目标模型
#   gn_channel_per_group —— GroupNorm 中每组包含的通道数。
#                           如果为 None，则不执行替换。
#                           典型值为 8 或 16。
# ============================================================================
def replace_bn_with_gn(model, gn_channel_per_group):
    if gn_channel_per_group is None:
        return

    # 遍历所有模块，查找子模块中的 BN 层
    for m in model.modules():
        to_replace_dict = {}
        for name, sub_m in m.named_children():
            if isinstance(sub_m, nn.BatchNorm2d):
                # 计算分组数：总通道数 / 每组通道数
                # 使用 min_divisible_value 确保分组数能整除总通道数
                num_groups = sub_m.num_features // min_divisible_value(
                    sub_m.num_features, gn_channel_per_group
                )
                # 创建新的 GroupNorm 层
                gn_m = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=sub_m.num_features,
                    eps=sub_m.eps,
                    affine=True,  # 使用可学习的 scale 和 shift（即 weight 和 bias）
                )

                # 复制 BN 的权重和偏置到 GN
                # 注意：BN 的 weight.shape = (C,)，GN 的 weight.shape 也是 (C,)
                # 可以直接复制
                gn_m.weight.data.copy_(sub_m.weight.data)
                gn_m.bias.data.copy_(sub_m.bias.data)
                # 保持 requires_grad 属性一致
                gn_m.weight.requires_grad = sub_m.weight.requires_grad
                gn_m.bias.requires_grad = sub_m.bias.requires_grad

                # 标记需要替换的层
                to_replace_dict[name] = gn_m

        # 批量替换子模块中的 BN 层为 GN 层
        # m._modules 是 nn.Module 存储子模块的有序字典
        # update 会将 to_replace_dict 中的键值对更新进去
        m._modules.update(to_replace_dict)


# ============================================================================
# replace_conv2d_with_my_conv2d
# ============================================================================
# 功能：将模型中的 nn.Conv2d 替换为支持 Weight Standardization 的 MyConv2d。
#
# 设计背景：
#   Weight Standardization (WS) 对卷积权重做标准化，使梯度分布更均匀，
#   从而允许使用更大的学习率，加速收敛。
#
#   为什么只替换无 bias 的 Conv2d？
#   在典型的 Conv-BN 结构中，Conv2d 通常不带 bias（bias=False），因为
#   BN 层已经包含了可学习的 shift 参数（beta）。替换后 WS 的 epsilon
#   参数由全局统一设置。
#
# 参数：
#   net    —— 目标网络模型
#   ws_eps —— Weight Standardization 的 epsilon。如果为 None，不执行替换。
#             eps 用于在分母中防止除零，通常设置为 1e-5 左右。
# ============================================================================
def replace_conv2d_with_my_conv2d(net, ws_eps=None):
    if ws_eps is None:
        return

    # 第一遍：收集需要替换的模块
    for m in net.modules():
        to_update_dict = {}
        for name, sub_module in m.named_children():
            # 只替换不带 bias 的 Conv2d（因为后面通常会接 BN 或 GN）
            if isinstance(sub_module, nn.Conv2d) and not sub_module.bias:
                to_update_dict[name] = sub_module
        # 执行替换：用 MyConv2d 替代原始的 Conv2d
        for name, sub_module in to_update_dict.items():
            m._modules[name] = MyConv2d(
                sub_module.in_channels,
                sub_module.out_channels,
                sub_module.kernel_size,
                sub_module.stride,
                sub_module.padding,
                sub_module.dilation,
                sub_module.groups,
                sub_module.bias,
            )
            # 保留原始的权重参数
            m._modules[name].load_state_dict(sub_module.state_dict())
            # 保持 requires_grad 属性
            m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
            if sub_module.bias is not None:
                m._modules[name].bias.requires_grad = sub_module.bias.requires_grad

    # 第二遍：统一设置 WS_EPS（确保所有 MyConv2d 使用相同的 epsilon）
    for m in net.modules():
        if isinstance(m, MyConv2d):
            m.WS_EPS = ws_eps


# ============================================================================
# init_models
# ============================================================================
# 功能：初始化模型参数，支持多种初始化策略。
#
# 初始化策略：
#   - "he_fout": He 初始化，基于输出通道数。适合 ReLU 等激活函数。
#     权重从 N(0, sqrt(2 / (kernel_size^2 * out_channels))) 采样。
#   - "he_fin": He 初始化，基于输入通道数。
#     权重从 N(0, sqrt(2 / (kernel_size^2 * in_channels))) 采样。
#
# 对不同类型层的处理：
#   - Conv2d: 用 He 初始化权重，bias 归零
#   - BatchNorm1d/2d, GroupNorm: weight 置 1，bias 置 0
#   - Linear: 均匀分布初始化，bias 置 0
#
# 参数：
#   net       —— 目标网络或网络列表。如果传入列表，递归处理每个网络。
#   model_init—— 初始化方式，支持 "he_fout"（默认）和 "he_fin"。
# ============================================================================
def init_models(net, model_init="he_fout"):
    # 如果传入的是网络列表，递归处理每个子网络
    if isinstance(net, list):
        for sub_net in net:
            init_models(sub_net, model_init)
        return

    # 遍历模型的所有模块
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            # He 初始化（Kaiming Normal）
            if model_init == "he_fout":
                # 基于输出通道数：n = fan_out
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif model_init == "he_fin":
                # 基于输入通道数：n = fan_in
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            else:
                raise NotImplementedError
            # 如果有 bias，初始化为 0
            if m.bias is not None:
                m.bias.data.zero_()

        elif type(m) in [nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm]:
            # 正则化层：weight = 1（不做缩放），bias = 0（不做偏移）
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            # 全连接层：均匀分布初始化
            # stdv = 1 / sqrt(fan_in)
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            # 均匀分布 U(-stdv, stdv)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.zero_()


# ============================================================================
# MyConv2d
# ============================================================================
# 功能：带 Weight Standardization (WS) 的自定义卷积层。
#
# Weight Standardization 的原理：
#   对卷积核的权重做标准化处理：减去均值、除以标准差。
#   WS 可使损失函数更平滑（Lipschitz 常数更小），从而允许使用更大的学习率，
#   是微调（fine-tuning）时的常用技巧。
#
#   计算公式：
#     W_hat = (W - mean(W)) / (std(W) + eps)
#
#   其中 mean 和 std 是在每个输出通道的权重上计算的（即沿输入通道、核高、
#   核宽三个维度求均值和标准差）。
#
# 使用方式：
#   设置 WS_EPS = 1e-5 左右的值即可启用 WS。
#   设置 WS_EPS = None 则不使用 WS，退化为普通 Conv2d。
# ============================================================================
class MyConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(MyConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        # WS_EPS: Weight Standardization 的 epsilon
        # None 表示不使用 Weight Standardization
        self.WS_EPS = None

    def weight_standardization(self, weight):
        """对卷积权重执行标准化

        标准化沿 dim=1,2,3 进行（即输入通道、高度、宽度维度），
        每个输出通道独立标准化。

        参数：
          weight —— 卷积权重形状 (out_channels, in_channels, kh, kw)

        返回值：
          标准化后的权重，形状不变
        """
        if self.WS_EPS is not None:
            # 沿 dim=(1,2,3) 计算均值，keepdim=True 保持维度
            # 结果形状 (out_channels, 1, 1, 1)
            weight_mean = (
                weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .mean(dim=3, keepdim=True)
            )
            # 减均值
            weight = weight - weight_mean
            # 计算标准差：先将 weight 展平为 (out_channels, -1)，再沿 dim=1 求 std
            # .view(-1, 1, 1, 1) 恢复形状以便广播
            # + WS_EPS 防止除零
            std = (
                weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
                + self.WS_EPS
            )
            # 除以标准差
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, x):
        """前向传播

        如果 WS_EPS 为 None，使用标准 Conv2d 前向；
        否则对权重做标准化后再执行卷积。
        """
        if self.WS_EPS is None:
            # 不使用 Weight Standardization，直接使用父类的 Conv2d forward
            return super(MyConv2d, self).forward(x)
        else:
            # 使用 Weight Standardization：先标准化权重，再执行卷积
            return F.conv2d(
                x,
                self.weight_standardization(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        """字符串表示，显示 Conv2d 参数 + ws_eps 值"""
        return super(MyConv2d, self).__repr__()[:-1] + ", ws_eps=%s)" % self.WS_EPS


# ============================================================================
# MyModule
# ============================================================================
# 功能：自定义模块的抽象基类。
#
# 设计意图：
#   在网络结构搜索（NAS）框架中，所有的网络构建块（如卷积块、残差块等）
#   都应该继承自 MyModule，并实现以下四个接口：
#     - forward(x): 前向传播
#     - module_str:  模块的字符串描述（用于架构编码）
#     - config:      模块的配置字典（可序列化，用于重建模块）
#     - build_from_config(config): 从配置字典重建模块的静态方法
#
#   这种设计模式使得网络架构可以被序列化为可配置的字典格式，方便：
#   1. 网络结构搜索中的候选架构存储和变异
#   2. 模型的保存和加载
#   3. 部署时的网络重建
# ============================================================================
class MyModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        """返回模块的字符串表示（用于架构编码）"""
        raise NotImplementedError

    @property
    def config(self):
        """返回模块的配置字典（用于序列化）"""
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        """从配置字典重建模块"""
        raise NotImplementedError


# ============================================================================
# MyNetwork
# ============================================================================
# 功能：自定义网络的抽象基类，继承自 MyModule。
#
# 在 MyModule 的基础上，MyNetwork 额外提供了：
#   - CHANNEL_DIVISIBLE: 通道对齐常数（默认为 8），用于确保通道数能被 8 整除
#   - set_bn_param / get_bn_param: 委托给模块级函数
#   - get_parameters: 按名称关键字过滤参数（支持 include / exclude 模式）
#   - weight_parameters: 返回所有可训练参数
#   - zero_last_gamma: 将最后一层的 gamma 置零（用于残差网络初始化技巧）
#   - grouped_block_index: 返回分组块索引
#
# 通道对齐（CHANNEL_DIVISIBLE=8）的原因：
#   许多硬件（尤其是 MCU）对张量维度对齐到 8 或 16 时计算效率最高。
# ============================================================================
class MyNetwork(MyModule):
    CHANNEL_DIVISIBLE = 8  # 通道数对齐基数

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def zero_last_gamma(self):
        """将最后一层的 gamma（BN 的 weight）置零，用于残差网络的初始化技巧"""
        raise NotImplementedError

    @property
    def grouped_block_index(self):
        """返回网络块的分组索引（用于按阶段分组统计）"""
        raise NotImplementedError

    # ====================================================================
    # 以下方法提供默认实现
    # ====================================================================

    def set_bn_param(self, momentum, eps, gn_channel_per_group=None, **kwargs):
        """设置本网络的正则化层参数，委托给模块级 set_bn_param"""
        set_bn_param(self, momentum, eps, gn_channel_per_group, **kwargs)

    def get_bn_param(self):
        """获取本网络的正则化层参数，委托给模块级 get_bn_param"""
        return get_bn_param(self)

    # ====================================================================
    # get_parameters
    # ====================================================================
    # 功能：按名称关键字过滤可训练参数。
    #
    # 模式说明：
    #   - mode="include": 只返回名称中包含指定关键字的参数
    #   - mode="exclude": 返回名称中不包含指定关键字的参数
    #
    # 使用场景：
    #   在微调（fine-tuning）时，可能只想更新某些层的参数（如只更新分类头），
    #   或冻结某些层的参数。这个函数提供了灵活的参数过滤机制。
    #
    # 参数：
    #   keys —— 关键字列表，如 ["classifier", "fc"]
    #   mode —— 过滤模式，"include"（默认）或 "exclude"
    #
    # 返回值：
    #   满足条件的参数生成器（yield）
    # ====================================================================
    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            # 如果 keys 为 None，返回所有 requires_grad=True 的参数
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            # 只返回名称包含任一关键字的参数
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            # 返回名称不包含任何关键字的参数
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def weight_parameters(self):
        """返回所有可训练参数（等价于 get_parameters()）"""
        return self.get_parameters()
