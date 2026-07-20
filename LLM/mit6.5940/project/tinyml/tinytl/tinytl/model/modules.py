import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ofa.utils.layers import set_layer_from_config, ZeroLayer
from ofa.utils import (
    MyModule,
    MyNetwork,
    MyGlobalAvgPool2d,
    min_divisible_value,
    SEModule,
)
from ofa.utils import get_same_padding, make_divisible, build_activation, init_models

__all__ = ["my_set_layer_from_config", "LiteResidualModule", "ReducedMBConvLayer"]


# 从配置字典构建层对象的工厂函数，扩展了 OFA 的 set_layer_from_config
# 支持 TinyTL 特有的 LiteResidualModule 和 ReducedMBConvLayer，其他层类型回退到 OFA 原生实现
def my_set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    # 注册 TinyTL 自定义层类型，按类名字符串映射
    name2layer = {
        LiteResidualModule.__name__: LiteResidualModule,
        ReducedMBConvLayer.__name__: ReducedMBConvLayer,
    }

    # 从配置中弹出 'name' 字段，确定要构建的层类型
    layer_name = layer_config.pop("name")
    if layer_name in name2layer:
        layer = name2layer[layer_name]
        return layer.build_from_config(layer_config)
    else:
        # 回退到 OFA 原生的层构建函数（如 DynamicMBConvLayer、ResidualBlock 等）
        return set_layer_from_config({"name": layer_name, **layer_config})


# TinyTL 的核心创新：轻量残差模块
# 在冻结的主干网络层之间插入一个极轻量的旁路分支，利用主干的中间激活作为输入，
# 通过 池化 -> 深度可分离卷积 -> 1×1卷积 提供额外的可训练学习能力。
# 参数量很少（主要是 1×1 卷积 + BN），却大幅增加了模型的表达容量。
class LiteResidualModule(MyModule):
    def __init__(
        self,
        main_branch,
        in_channels,
        out_channels,
        expand=1.0,
        kernel_size=3,
        act_func="relu",
        n_groups=2,
        downsample_ratio=2,
        upsample_type="bilinear",
        stride=1,
    ):
        super(LiteResidualModule, self).__init__()

        # 主干分支：原始的冻结网络层（如 MBConv），权重在 TinyTL 训练中不更新
        self.main_branch = main_branch

        # 轻量残差分支的配置参数
        self.lite_residual_config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "expand": expand,  # 中间通道数相对于输入通道数的扩展比例
            "kernel_size": kernel_size,
            "act_func": act_func,
            "n_groups": n_groups,  # 深度可分离卷积的组数（groups=channels 即为 depthwise）
            "downsample_ratio": downsample_ratio,  # 池化下采样倍率，降低轻量分支的计算量
            "upsample_type": upsample_type,  # 上采样方式（bilinear/nearest），恢复空间尺寸
            "stride": stride,
        }

        # 如果不需要下采样（特征图已经是 1×1），用 1×1 卷积代替空间卷积
        kernel_size = 1 if downsample_ratio is None else kernel_size

        padding = get_same_padding(kernel_size)
        if downsample_ratio is None:
            # 特征图已缩到全局，用全局平均池化
            pooling = MyGlobalAvgPool2d()
        else:
            # 空间池化降低分辨率：减少后续卷积的计算量，是 TinyTL 省内存/省计算的关键
            pooling = nn.AvgPool2d(downsample_ratio, downsample_ratio, 0)

        # 确保中间通道数可被 CHANNEL_DIVISIBLE 整除（硬件对齐要求）
        num_mid = make_divisible(
            int(in_channels * expand), divisor=MyNetwork.CHANNEL_DIVISIBLE
        )

        # 轻量残差分支：pooling -> 深度可分离卷积 -> BN -> 激活 -> 1×1卷积 -> BN
        # 设计极简，参数量 << 主分支，确保训练内存增量最小
        self.lite_residual = nn.Sequential(
            OrderedDict(
                {
                    "pooling": pooling,
                    "conv1": nn.Conv2d(
                        in_channels,
                        num_mid,
                        kernel_size,
                        stride,
                        padding,
                        groups=n_groups,  # 深度可分离卷积，大幅减少参数
                        bias=False,
                    ),
                    "bn1": nn.BatchNorm2d(num_mid),
                    "act": build_activation(act_func),
                    "conv2": nn.Conv2d(num_mid, out_channels, 1, 1, 0, bias=False),
                    "final_bn": nn.BatchNorm2d(out_channels),
                }
            )
        )

        # 初始化轻量分支的权重
        # 将最后一个 BN 的 weight 置零 → 残差分支初始输出为零（identity 风格初始化）
        # 这样训练初期主要由主分支贡献输出，残差分支从零开始逐渐学习
        init_models(self.lite_residual)
        self.lite_residual.final_bn.weight.data.zero_()

    def forward(self, x):
        # 主分支前向（冻结的 MBConv 层）
        main_x = self.main_branch(x)
        # 轻量分支前向：池化 → 卷积 → 上采样恢复空间尺寸
        lite_residual_x = self.lite_residual(x)
        if self.lite_residual_config["downsample_ratio"] is not None:
            # 上采样轻量分支输出，使其空间尺寸与主分支输出对齐
            lite_residual_x = F.upsample(
                lite_residual_x,
                main_x.shape[2:],
                mode=self.lite_residual_config["upsample_type"],
            )
        # 残差相加：主分支输出 + 轻量分支输出
        return main_x + lite_residual_x

    @property
    def module_str(self):
        """生成模块的可读字符串描述，用于日志和可视化"""
        return (
            self.main_branch.module_str
            + " + LiteResidual(downsample=%s, n_groups=%s, expand=%s, ks=%s)"
            % (
                self.lite_residual_config["downsample_ratio"],
                self.lite_residual_config["n_groups"],
                self.lite_residual_config["expand"],
                self.lite_residual_config["kernel_size"],
            )
        )

    @property
    def config(self):
        """导出模块的配置字典，用于序列化和从配置重建模块"""
        return {
            "name": LiteResidualModule.__name__,
            "main": self.main_branch.config,
            "lite_residual": self.lite_residual_config,
        }

    @staticmethod
    def build_from_config(config):
        """从配置字典构建 LiteResidualModule 的静态工厂方法"""
        main_branch = my_set_layer_from_config(config["main"])
        lite_residual_module = LiteResidualModule(
            main_branch, **config["lite_residual"]
        )
        return lite_residual_module

    def __repr__(self):
        return (
            "{\n (main branch): "
            + self.main_branch.__repr__()
            + ", "
            + "\n (lite residual): "
            + self.lite_residual.__repr__()
            + "}"
        )

    # 将 LiteResidual 模块注入到 ProxylessNASNets 网络的每一层中
    # 这是 TinyTL 构造可训练模型的关键入口：遍历主干网络的所有 MBConv 层，
    # 用 LiteResidualModule 包裹每一层，使主分支冻结的同时获得额外的可训练旁路
    @staticmethod
    def insert_lite_residual(
        net,
        downsample_ratio=2,
        upsample_type="bilinear",
        expand=1.0,
        max_kernel_size=5,
        act_func="relu",
        n_groups=2,
        **kwargs,
    ):
        # 防止重复插入：如果网络中已有 LiteResidualModule，则跳过
        if LiteResidualModule.has_lite_residual_module(net):
            return
        from ofa.imagenet_classification.networks import ProxylessNASNets

        if isinstance(net, ProxylessNASNets):
            # 保存原始 BN 参数，插入过程中会修改网络结构
            bn_param = net.get_bn_param()

            # 遍历每个 block 组，为每个 MBConv 层包裹 LiteResidualModule
            # 分辨率递减：从 128 开始，遇到 stride=2 就折半
            max_resolution = 128
            stride_stages = [2, 2, 2, 1, 2, 1]
            for block_index_list, stride in zip(net.grouped_block_index, stride_stages):
                for i, idx in enumerate(block_index_list):
                    block = net.blocks[idx].conv
                    # 跳过 ZeroLayer（NAS 搜索中已被裁剪掉的层）
                    if isinstance(block, ZeroLayer):
                        continue
                    # block group 的第一层应用 stride，后续层 stride=1
                    s = stride if i == 0 else 1
                    block_downsample_ratio = downsample_ratio
                    block_resolution = max(1, max_resolution // block_downsample_ratio)
                    max_resolution //= s

                    # 自适应选择轻量分支的卷积核大小
                    # 确保核大小不超过当前特征图分辨率（否则卷积无效）
                    kernel_size = max_kernel_size
                    if block_resolution == 1:
                        # 特征图已缩到 1×1，用 1×1 卷积，不需要空间池化
                        kernel_size = 1
                        block_downsample_ratio = None
                    else:
                        # 逐步减小核大小直到不超过特征图分辨率
                        while block_resolution < kernel_size:
                            kernel_size -= 2

                    # 用 LiteResidualModule 包裹原始 MBConv 层，替换网络中的原层
                    net.blocks[idx].conv = LiteResidualModule(
                        block,
                        block.in_channels,
                        block.out_channels,
                        expand=expand,
                        kernel_size=kernel_size,
                        act_func=act_func,
                        n_groups=n_groups,
                        downsample_ratio=block_downsample_ratio,
                        upsample_type=upsample_type,
                        stride=s,
                    )

            # 恢复 BN 参数（插入过程可能改变了网络的 BN 状态）
            net.set_bn_param(**bn_param)
        else:
            raise NotImplementedError

    # 检查网络中是否已包含 LiteResidualModule（防止重复插入）
    @staticmethod
    def has_lite_residual_module(net):
        for m in net.modules():
            if isinstance(m, LiteResidualModule):
                return True
        return False

    @property
    def in_channels(self):
        return self.lite_residual_config["in_channels"]

    @property
    def out_channels(self):
        return self.lite_residual_config["out_channels"]


# 简化版 MBConv 层，用于 TinyTL 中替换标准 MBConv
# 相比标准 MBConv（expand Conv → depthwise Conv → SE → project Conv），
# 简化为只有 深度可分离expand → 可选SE → 1×1 reduce 两步
class ReducedMBConvLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func="relu6",
        use_se=False,
        groups=1,
    ):
        super(ReducedMBConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.groups = groups

        # 计算中间特征维度：expand_ratio * in_channels
        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        pad = get_same_padding(self.kernel_size)
        # 当 groups=None 时，自动退化为 depthwise convolution（groups=feature_dim）
        groups = (
            feature_dim
            if self.groups is None
            else min_divisible_value(feature_dim, self.groups)
        )

        # expand 阶段：深度可分离卷积（或分组卷积）+ BN + 激活 + 可选SE
        self.expand_conv = nn.Sequential(
            OrderedDict(
                {
                    "conv": nn.Conv2d(
                        in_channels,
                        feature_dim,
                        kernel_size,
                        stride,
                        pad,
                        groups=groups,
                        bias=False,
                    ),
                    "bn": nn.BatchNorm2d(feature_dim),
                    "act": build_activation(self.act_func, inplace=True),
                }
            )
        )
        if self.use_se:
            self.expand_conv.add_module("se", SEModule(feature_dim))

        # reduce 阶段：1×1 卷积投影到输出通道 + BN
        self.reduce_conv = nn.Sequential(
            OrderedDict(
                {
                    "conv": nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False),
                    "bn": nn.BatchNorm2d(out_channels),
                }
            )
        )

    def forward(self, x):
        # expand → reduce，两步完成特征提取和通道投影
        x = self.expand_conv(x)
        x = self.reduce_conv(x)
        return x

    @property
    def module_str(self):
        """生成模块的可读字符串描述"""
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = "%dx%d_ReducedMBConv%.3f_%s" % (
            self.kernel_size,
            self.kernel_size,
            expand_ratio,
            self.act_func.upper(),
        )
        if self.use_se:
            layer_str = "SE_" + layer_str
        layer_str += "_O%d" % self.out_channels
        if self.groups is not None:
            layer_str += "_G%d" % self.groups
        # 区分 GroupNorm 和 BatchNorm
        if isinstance(self.reduce_conv.bn, nn.GroupNorm):
            layer_str += "_GN%d" % self.reduce_conv.bn.num_groups
        elif isinstance(self.reduce_conv.bn, nn.BatchNorm2d):
            layer_str += "_BN"

        return layer_str

    @property
    def config(self):
        """导出模块的配置字典，用于序列化"""
        return {
            "name": ReducedMBConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
            "groups": self.groups,
        }

    @staticmethod
    def build_from_config(config):
        """从配置字典构建 ReducedMBConvLayer"""
        return ReducedMBConvLayer(**config)
