# ============================================================================
# pytorch_modules.py —— 自定义 PyTorch 网络层和损失函数
#
# 来源：
#   Once for All: Train One Network and Specialize it for Efficient Deployment
#   Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
#   International Conference on Learning Representations (ICLR), 2020.
#
# 本文件提供了一系列在 MCUNet / OFA 框架中常用的自定义网络层和工具函数：
#   1. make_divisible          —— 确保通道数能被指定除数整除（硬件对齐优化）
#   2. build_activation        —— 根据字符串名称构建激活函数层
#   3. ShuffleLayer            —— Channel Shuffle 层（ShuffleNet 风格）
#   4. MyGlobalAvgPool2d       —— 可保持维度的全局平均池化
#   5. Hswish / Hsigmoid      —— Hard Swish / Hard Sigmoid 激活函数
#   6. SEModule                —— Squeeze-and-Excitation 通道注意力模块
#   7. MultiHeadCrossEntropyLoss—— 多头交叉熵损失函数
#
# 这些组件大多在 MobileNetV3 / ShuffleNetV2 / MCUNet 等轻量级网络中使用。
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .my_modules import MyNetwork

# __all__ 控制 from pytorch_modules import * 时暴露的公共接口
__all__ = [
    "make_divisible",
    "build_activation",
    "ShuffleLayer",
    "MyGlobalAvgPool2d",
    "Hswish",
    "Hsigmoid",
    "SEModule",
    "MultiHeadCrossEntropyLoss",
]


# ============================================================================
# make_divisible
# ============================================================================
# 功能：确保通道数能被 divisor 整除，且不低于 min_val。
#
# 设计背景：
#   在许多轻量级网络（MobileNet / MCUNet）中，通道数通常被设计为 8、16 或
#   32 的倍数，这是因为：
#   1. 硬件加速器（如 GPU 的 Tensor Cores、NPU、MCU 的 SIMD 单元）在通道数
#      对齐到特定大小时计算效率最高。
#   2. 分组卷积（group convolution）要求每组通道数能整除总通道数。
#
# 算法：
#   1. 上调 v 到最接近的 divisor 整数倍
#   2. 如果上调后超过原值 10%，则再增加一个 divisor（保证不比原值小太多）
#
# 参数：
#   v       —— 原始通道数
#   divisor —— 对齐基数（通常为 8 或 16）
#   min_val —— 最小值，若为 None 则使用 divisor 本身
#
# 返回值：
#   对齐后的通道数
#
# 举例：
#   make_divisible(13, 8) → 16（因为 13+4=17，17//8*8=16，16 < 0.9*13=11.7 不成立）
#   make_divisible(15, 8) → 16（因为 15+4=19，19//8*8=16，16 >= 0.9*15=13.5）
#   make_divisible(21, 8) → 24（因为 21+4=25，25//8*8=24，24 >= 0.9*21=18.9）
# ============================================================================
def make_divisible(v, divisor, min_val=None):
    if min_val is None:
        min_val = divisor
    # v + divisor/2 实现四舍五入到最近的 divisor 倍数
    # int(v + divisor/2) // divisor * divisor 是经典的上调对齐写法
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # 如果上调后的值小于原值的 90%（即下降太多），再补一个 divisor
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ============================================================================
# build_activation
# ============================================================================
# 功能：根据字符串名称创建对应的激活函数层实例。
#
# 支持的类型：
#   - "relu"      → ReLU
#   - "relu6"     → ReLU6（MobileNetV2 常用）
#   - "tanh"      → Tanh
#   - "sigmoid"   → Sigmoid
#   - "h_swish"   → Hard Swish（MobileNetV3 引入）
#   - "h_sigmoid" → Hard Sigmoid
#   - None / "none" → None（不插入激活函数）
#
# 参数：
#   act_func —— 激活函数名称（字符串或 None）
#   inplace  —— 是否使用 inplace 操作以节省内存（默认 True）
#
# 返回值：
#   对应的 nn.Module 实例，如果 act_func 为 None 则返回 None。
# ============================================================================
def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    elif act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


# ============================================================================
# ShuffleLayer
# ============================================================================
# 功能：Channel Shuffle（通道混洗）层，用于 ShuffleNet 系列网络。
#
# 设计背景：
#   ShuffleNet 使用分组卷积（group convolution）来降低计算量，但分组卷积
#   的问题是不同组之间的信息无法流通。Channel Shuffle 通过将不同组的通道
#   打乱重排，使得后续的分组卷积能接收到来自不同组的特征，促进组间信息
#   交互和特征融合。
#
# 实现原理（以 groups=3, 每组 3 个通道为例）：
#   输入通道: [A1, A2, A3, B1, B2, B3, C1, C2, C3] （3组，组内连续）
#   reshape:  → [[A1,A2,A3], [B1,B2,B3], [C1,C2,C3]]  # (groups, C//groups, H, W)
#   transpose: → [[A1,B1,C1], [A2,B2,C2], [A3,B3,C3]]  # (C//groups, groups, H, W)
#   flatten:  → [A1, B1, C1, A2, B2, C2, A3, B3, C3]  # 组间交错排列
#
# 参数：
#   groups —— 分组卷积的分组数，决定了混洗的粒度
# ============================================================================
class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups  # 保存分组数

    def forward(self, x):
        # x.shape: (batch, num_channels, height, width)
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # 第一步：reshape 为 (batch, groups, C//groups, H, W)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # 第二步：交换维度 1 和 2，将组内通道与组分到同一维度
        # 转置后形状: (batch, C//groups, groups, H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        # .contiguous() 确保转置后张量在内存中是连续存储的
        # 因为 transpose 不改变内存布局，而后续的 view 要求连续内存

        # 第三步：flatten 回 (batch, num_channels, H, W)
        # 现在通道顺序变为: 组1通道0, 组2通道0, 组3通道0, 组1通道1, ...
        x = x.view(batch_size, -1, height, width)
        return x

    def __repr__(self):
        return "ShuffleLayer(groups=%d)" % self.groups


# ============================================================================
# MyGlobalAvgPool2d
# ============================================================================
# 功能：全局平均池化层，可选择是否保持空间维度。
#
# 与 nn.AdaptiveAvgPool2d(1) 的区别：
#   标准的 AdaptiveAvgPool2d(1) 输出形状为 (batch, C, 1, 1)，
#   而 MyGlobalAvgPool2d 可以通过 keep_dim 参数控制：
#     - keep_dim=True:  输出 (batch, C, 1, 1)  —— 保持 4D 张量
#     - keep_dim=False: 输出 (batch, C)        —— 压缩空间维度
#
# 为什么用 mean 而不是 AdaptiveAvgPool2d？
#   对于全局池化（H=W），x.mean(3).mean(2) 与 AdaptiveAvgPool2d(1) 在数学上
#   等价，但前者更轻量，不需要额外的 CUDA kernel 调用。
# ============================================================================
class MyGlobalAvgPool2d(nn.Module):
    def __init__(self, keep_dim=True):
        super(MyGlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        # x.mean(3) 在宽度维上取均值 → (batch, C, H, 1)
        # .mean(2) 在高度维上取均值 → (batch, C, 1, 1) 或 (batch, C)
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return "MyGlobalAvgPool2d(keep_dim=%s)" % self.keep_dim


# ============================================================================
# Hswish —— Hard Swish 激活函数
# ============================================================================
# 公式：H-swish(x) = x * ReLU6(x + 3) / 6
#
# 设计背景：
#   Swish 激活函数（x * sigmoid(x)）在多个任务上表现优于 ReLU，但计算
#   sigmoid 开销较大。Hard Swish 是 Swish 的分段线性近似，用 ReLU6 替代
#   sigmoid，在保持相近效果的同时大幅降低计算开销，特别适合移动端和 MCU
#   部署。MobileNetV3 首次引入了 Hard Swish。
#
#   与标准 Swish 的对比：
#   - Swish:     x * sigmoid(x)
#   - H-Swish:   x * clamp((x+3)/6, 0, 1) ≈ x * ReLU6(x+3) / 6
#
# 参数：
#   inplace —— 是否 inplace 操作（节省显存，默认 True）
# ============================================================================
class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # x + 3 → ReLU6（截断到 [0, 6]）→ 除以 6 → 乘以 x
        # 结果：当 x << -3 时输出趋近 0，当 x >> 3 时输出趋近 x
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hswish()"


# ============================================================================
# Hsigmoid —— Hard Sigmoid 激活函数
# ============================================================================
# 公式：H-sigmoid(x) = ReLU6(x + 3) / 6
#
# 设计背景：
#   Hard Sigmoid 是标准 Sigmoid 的分段线性近似。标准 Sigmoid 计算涉及
#   指数运算，在低端硬件上开销大。Hard Sigmoid 用简单的 ReLU6 + 除法
#   来近似，在保持功能的同时适合轻量级网络部署。
#
#   与标准 Sigmoid 的对比：
#   - Sigmoid:     1 / (1 + exp(-x))
#   - H-Sigmoid:   clamp((x+3)/6, 0, 1) = ReLU6(x+3) / 6
#
# 参数：
#   inplace —— 是否 inplace 操作（默认 True）
# ============================================================================
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # x + 3 → ReLU6（截断到 [0, 6]）→ 除以 6，输出范围 [0, 1]
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hsigmoid()"


# ============================================================================
# SEModule —— Squeeze-and-Excitation 通道注意力模块
# ============================================================================
# 设计背景：
#   SENet（Squeeze-and-Excitation Networks）提出了一种轻量级的通道注意力
#   机制，通过显式建模通道之间的依赖关系，自适应地重新校准通道特征响应。
#
# 工作原理：
#   1. Squeeze（压缩）：全局平均池化，将每个通道的 HxW 特征压缩为一个标量
#   2. Excitation（激励）：两层全连接（用 1x1 卷积实现）：
#      - reduce: 降维（channel → channel/reduction），ReLU 激活
#      - expand: 升维（channel/reduction → channel），H-Sigmoid 激活
#   3. Scale（缩放）：将 Excitation 输出的权重（0~1）乘以原始特征图
#
# 计算量：
#   SE 模块增加的参数量很少（~2 * C^2 / reduction），通常 reduction=4，
#   在 MCUNet 等轻量网络中仍然保持轻量化。
#
# 参数：
#   channel  —— 输入特征图的通道数
#   reduction—— 降维比率（默认 4），越大越节省参数，但可能削弱效果
# ============================================================================
class SEModule(nn.Module):
    REDUCTION = 4  # 默认压缩率

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction

        # 计算中间层的通道数，确保能被 CHANNEL_DIVISIBLE 整除
        # 这是为了硬件对齐优化
        num_mid = make_divisible(
            self.channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE
        )

        # SE 模块的核心：两层 1x1 卷积（等价于全连接层）
        # 使用 1x1 卷积而不是 nn.Linear，因为输入仍然是 4D 特征图
        # (batch, C, 1, 1)，1x1 卷积可以直接处理
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    # 降维层：C → C/r，减少参数量和计算量
                    ("reduce", nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
                    ("relu", nn.ReLU(inplace=True)),  # ReLU 提供非线性
                    # 升维层：C/r → C，恢复到原始通道数
                    ("expand", nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
                    # H-Sigmoid 激活，输出范围 [0, 1]，作为通道权重
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        # Squeeze：全局平均池化，将 (batch, C, H, W) → (batch, C, 1, 1)
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # Excitation：通过两层 1x1 卷积生成通道权重
        y = self.fc(y)
        # Scale：原始特征图乘以通道权重（广播逐通道相乘）
        return x * y

    def __repr__(self):
        return "SE(channel=%d, reduction=%d)" % (self.channel, self.reduction)


# ============================================================================
# MultiHeadCrossEntropyLoss —— 多头交叉熵损失函数
# ============================================================================
# 功能：对多个输出头的交叉熵损失取平均。
#
# 使用场景：
#   在多任务学习或多个分类头的场景中（例如 MCUNet 的 OFA 训练中可能会
#   有多个分类器头），每个头的输出是一个独立的分类，需要对每个头分别
#   计算交叉熵损失然后取平均。
#
# 输入输出形状：
#   outputs.shape = (batch_size, num_heads, num_classes)
#   targets.shape = (batch_size, num_heads)
#   每个 targets[:, k] 对应 outputs[:, k, :] 的真实标签
#
# 计算过程：
#   对每个头 k，计算 F.cross_entropy(outputs[:, k, :], targets[:, k])
#   → 除以 num_heads 取平均
# ============================================================================
class MultiHeadCrossEntropyLoss(nn.Module):
    def forward(self, outputs, targets):
        # 验证维度
        assert outputs.dim() == 3, outputs  # (batch, heads, classes)
        assert targets.dim() == 2, targets  # (batch, heads)
        # 验证 heads 维度一致
        assert outputs.size(1) == targets.size(1), (outputs, targets)

        num_heads = targets.size(1)

        loss = 0
        # 对每个头单独计算交叉熵，然后累加并取平均
        for k in range(num_heads):
            # F.cross_entropy 内部包含 softmax + log + NLLLoss
            loss += F.cross_entropy(outputs[:, k, :], targets[:, k]) / num_heads
        return loss
