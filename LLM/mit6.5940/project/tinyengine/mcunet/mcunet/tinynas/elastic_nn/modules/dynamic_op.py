# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#
# =============================================================================
# dynamic_op.py — 动态算子（Dynamic Operators）
#
# 本文件定义了 OFA 框架中的核心"动态算子"。
# 在 OFA/MCUNet 框架中，"动态"指的是算子的维度是可变的：
# 同一个算子可以支持多种不同的输入/输出通道数、卷积核大小等，
# 从而允许在推理时灵活地选择子网络结构。
#
# 关键设计原则：
#   1. 超网络（Super-Network）中每个算子都以"最大配置"初始化
#      （例如最大通道数、最大卷积核）。
#   2. 在前向传播时，通过切片（slicing）从最大权重中取出
#      当前子网络所需的部分，实现"权重继承"。
#   3. 这种方式使得所有子网络共享同一组权重参数，
#      无需为每种配置单独训练。
# =============================================================================

import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch

from ....utils import get_same_padding, sub_filter_start_end, make_divisible, SEModule

# 导出符号列表：对外公开的动态算子类
__all__ = [
    "DynamicSeparableConv2d",
    "DynamicPointConv2d",
    "DynamicLinear",
    "DynamicBatchNorm2d",
    "DynamicSE",
]


class DynamicSeparableConv2d(nn.Module):
    """
    ============================================================================
    动态深度可分离卷积（Dynamic Depthwise Separable Convolution）。

    在 MobileNet 风格网络中，深度可分离卷积分为两步：
      1. depthwise conv（深度卷积）：每个输入通道独立做卷积
      2. pointwise conv（逐点卷积）：用 1x1 卷积混合通道信息

    本类实现"动态"的 depthwise 部分，核心特征是：
      - 以最大卷积核大小初始化权重（例如 7x7）
      - 推理时可以根据需要选择较小的卷积核（如 5x5 或 3x3）
      - 通过从大核的中心区域切片来得到小核权重

    KERNEL_TRANSFORM_MODE:
      - None : 直接切片（从大核中心取小核区域）
      - 1    : 使用可学习的线性变换矩阵，
               将大核权重"压缩"为小核权重（更精确但增加参数量）
    ============================================================================
    """

    KERNEL_TRANSFORM_MODE = None  # None 或 1；控制是否使用核变换矩阵

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1):
        """
        初始化动态深度可分离卷积。

        参数说明:
          max_in_channels (int): 最大输入通道数（超网络中的上限）
          kernel_size_list (list[int]): 支持的卷积核大小列表，例如 [3, 5, 7]
          stride (int): 卷积步长
          dilation (int): 膨胀率
        """
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        # 以最大配置初始化真实的卷积层
        # groups = max_in_channels 表示 depthwise conv（每个通道独立卷积）
        # 注意：这里实际创建的是最大核的卷积，小核通过"切片"实现
        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )

        # 去重并排序内核大小列表，例如 [3, 3, 5, 7] -> [3, 5, 7]
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]

        if self.KERNEL_TRANSFORM_MODE is not None:
            # =================================================================
            # 核变换模式（KERNEL_TRANSFORM_MODE = 1）：
            # 每一对相邻的卷积核大小之间，注册一个可学习的变换矩阵。
            # 例如：如果支持 [3, 5, 7]，则创建 7to5_matrix 和 5to3_matrix。
            # 这些矩阵将较大核的权重线性映射为较小核的权重，
            # 相比于直接切片，这种变换可以更好地保留大核训练好的信息。
            # =================================================================
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                # 初始化变换矩阵为单位矩阵，从恒等映射开始学习
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small**2)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        # 当前激活的卷积核大小，默认为最大值
        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        """
        获取当前子网络所需的卷积核权重。

        参数:
          in_channel (int): 实际输入通道数（可能小于 max_in_channels）
          kernel_size (int): 当前选择的卷积核大小

        返回:
          形状为 (out_channel, in_channel, kernel_size, kernel_size) 的权重张量

        实现逻辑:
          1. 从 self.conv.weight 中切片出所需的输入/输出通道
          2. 根据 kernel_size 从大核中心区域切片
          3. 如果启用了 KERNEL_TRANSFORM_MODE，用变换矩阵优化切片权重
        """
        out_channel = in_channel  # depthwise conv 的输出通道等于输入通道
        max_kernel_size = max(self.kernel_size_list)

        # 步骤1: 从大核中心切片出小核区域
        # sub_filter_start_end 计算从大核的哪个位置开始切
        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]

        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            # 步骤2: 如果启用核变换，用可学习的变换矩阵
            # 从大到小逐级变换：7x7 -> 5x5 -> 3x3
            start_filter = self.conv.weight[
                :out_channel, :in_channel, :, :
            ]  # 从最大核开始
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break  # 已经达到目标大小
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                # 应用可学习的变换矩阵
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks**2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        """
        前向传播：使用当前激活的卷积核大小进行 depthwise conv。

        参数:
          x (Tensor): 输入张量，形状为 (N, C, H, W)
          kernel_size (int, optional): 使用的卷积核大小，默认使用 self.active_kernel_size

        动态机制的体现:
          - kernel_size 可以在外部设置（通过 set_active_subnet），
            从而在同一超网络中选择不同大小的卷积核。
          - 每次 forward 时只是从大核权重中"取出一部分"使用，
            所有权重都共享同一个 self.conv.weight。
        """
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        # 获取当前子网络对应的卷积核权重切片
        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        # 使用 functional conv2d 而非模块调用，因为权重是我们手动切片的
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, in_channel)
        return y


class DynamicPointConv2d(nn.Module):
    """
    ============================================================================
    动态逐点卷积（Dynamic Pointwise Convolution），即 1x1 卷积。

    在 MobileNet 中，pointwise conv 负责将 depthwise conv 的输出
    在不同通道之间混合信息，实现通道数的变换。

    动态特性体现：
      - 权重以 max_in_channels → max_out_channels 初始化
      - 推理时可以选择任意 in_channel ≤ max_in_channels 和
        out_channel ≤ max_out_channels
      - 通过切片操作实现通道数的弹性变化
    ============================================================================
    """

    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        """
        参数:
          max_in_channels (int):  最大输入通道数
          max_out_channels (int): 最大输出通道数
          kernel_size (int):      卷积核大小（pointwise 通常是 1）
          stride (int):           步长
          dilation (int):         膨胀率
        """
        super(DynamicPointConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # 以最大配置初始化卷积层
        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        # 当前激活的输出通道数，默认为最大值
        self.active_out_channel = self.max_out_channels

    def forward(self, x, out_channel=None):
        """
        前向传播：使用切片后的权重进行逐点卷积。

        参数:
          x (Tensor): 输入张量
          out_channel (int, optional): 当前选择的输出通道数

        动态机制:
          - 从 self.conv.weight 中切片出前 out_channel 行和前 in_channel 列
          - 这等价于选择了一个"更窄"的子网络
        """
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        # 关键操作：从完整权重中切片出当前子网络所需的部分
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicLinear(nn.Module):
    """
    ============================================================================
    动态全连接层（Dynamic Linear / Fully Connected Layer）。

    用于分类器头部，支持输入/输出特征维度的弹性变化。
    工作原理与 DynamicPointConv2d 类似，通过切片权重实现。
    ============================================================================
    """

    def __init__(self, max_in_features, max_out_features, bias=True):
        """
        参数:
          max_in_features (int):  最大输入特征数
          max_out_features (int): 最大输出特征数（例如类别数）
          bias (bool):            是否包含偏置项
        """
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def forward(self, x, out_features=None):
        """
        前向传播。

        参数:
          x (Tensor): 输入张量，形状为 (N, in_features)
          out_features (int, optional): 当前选择的输出特征数

        动态机制:
          - weight 切片：取前 out_features 行和前 in_features 列
          - bias 切片：取前 out_features 个元素
        """
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.linear.weight[:out_features, :in_features].contiguous()
        bias = self.linear.bias[:out_features] if self.bias else None
        y = F.linear(x, weight, bias)
        return y


class DynamicBatchNorm2d(nn.Module):
    """
    ============================================================================
    动态批归一化（Dynamic Batch Normalization）。

    标准 BatchNorm2d 的 num_features 在初始化后是固定的。
    但 OFA 的子网络可能使用不同数量的通道，
    因此需要动态 BN 来适配不同通道数的子网络。

    实现方式：
      - 以最大特征维度初始化 BN 层
      - 当实际特征维度小于最大值时，从 running_mean / running_var / weight / bias
        中切片出前 feature_dim 个元素来使用
      - SET_RUNNING_STATISTICS 开关控制是否强制使用完整 BN（用于特殊场景）
    ============================================================================
    """

    SET_RUNNING_STATISTICS = False  # 设为 True 时始终使用完整 BN 统计量

    def __init__(self, max_feature_dim):
        """
        参数:
          max_feature_dim (int): 可能的最大通道数
        """
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        # 以最大维度初始化标准 BN
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        """
        动态 BN 的核心前向逻辑（静态方法，方便复用）。

        参数:
          x (Tensor): 输入张量
          bn (nn.BatchNorm2d): 标准 BN 层（以最大维度初始化）
          feature_dim (int): 当前实际使用的通道数

        返回:
          归一化后的张量

        关键设计:
          如果 feature_dim == bn.num_features，直接调用标准 BN forward，
          这样可以利用 PyTorch 的融合 kernel 加速。
          否则，手动调用 F.batch_norm 并传入切片后的统计量。
        """
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                # 以下逻辑来自 PyTorch 源码，用于计算指数平均因子
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # 使用累积移动平均
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # 使用指数移动平均
                        exponential_average_factor = bn.momentum
            # 关键操作：对 running_mean、running_var、weight、bias 全部切片
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        """
        前向传播：从输入 x 的通道数推断 feature_dim 并调用 bn_forward。
        """
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicSE(SEModule):
    """
    ============================================================================
    动态 Squeeze-and-Excitation 模块。

    SE 模块通过"压缩-激发"机制学习通道间的注意力权重：
      1. Squeeze（压缩）：全局平均池化，将每个通道压缩为一个标量
      2. Excitation（激发）：两个全连接层（降维→ReLU→升维→Sigmoid）
      3. Scale（缩放）：将学习到的权重乘以原始特征图

    动态特性：
      - 中间隐藏层维度 num_mid 是根据实际输入通道数动态计算的
      - reduce 和 expand 两个卷积的权重都通过切片获取
      - 继承自 SEModule（定义在 utils/pytorch_modules.py 中）
    ============================================================================
    """

    def __init__(self, max_channel):
        """
        参数:
          max_channel (int): 可能的最大通道数
        """
        super(DynamicSE, self).__init__(max_channel)

    def forward(self, x):
        """
        前向传播：动态计算 SE 注意力权重。

        动态关键点:
          - num_mid 根据 in_channel 自适应计算（通过 make_divisible 对齐到 8 的倍数）
          - reduce_conv 和 expand_conv 的权重均从完整权重中切片
        """
        in_channel = x.size(1)
        # 中间层通道数 = 输入通道 / reduction_ratio，对齐到 8 的倍数
        # 这样在通道数变化时，中间层维度也会自适应调整
        num_mid = make_divisible(in_channel // self.reduction, divisor=8)

        # Squeeze: 全局平均池化，将 (N, C, H, W) -> (N, C, 1, 1)
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)

        # === Excitation 的第一步：降维 ===
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid, :in_channel, :, :].contiguous()
        reduce_bias = (
            reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        )
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)

        # ReLU 激活
        y = self.fc.relu(y)

        # === Excitation 的第二步：升维 ===
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel, :num_mid, :, :].contiguous()
        expand_bias = (
            expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        )
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)

        # Hard Sigmoid 激活函数（比标准 Sigmoid 计算更快）
        y = self.fc.h_sigmoid(y)

        # Scale: 将注意力权重乘以原始特征图
        return x * y
