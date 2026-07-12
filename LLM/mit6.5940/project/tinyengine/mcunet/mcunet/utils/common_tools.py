# ============================================================================
# common_tools.py —— 通用工具函数集合
#
# 来源：
#   Once for All: Train One Network and Specialize it for Efficient Deployment
#   Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
#   International Conference on Learning Representations (ICLR), 2020.
#
# 本文件提供 MCUNet/OFA 框架中广泛使用的小型工具函数和类，涵盖：
#   - 字典排序、列表统计运算
#   - 卷积 padding 计算、弹性网络中的通道/卷积核大小分配
#   - 文件下载、分类准确率评估
#   - 滑动平均跟踪器（AverageMeter）
#
# 这些函数虽然简单，但在网络结构搜索（NAS）、弹性网络训练和模型评估中
# 被高频调用，统一放在此处避免代码重复。
# ============================================================================

import numpy as np
import os
import sys

# 兼容 Python 2/3 的 urlretrieve 导入
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

# __all__ 控制 from common_tools import * 时暴露的公共接口
__all__ = [
    "sort_dict",
    "get_same_padding",
    "get_split_list",
    "list_sum",
    "list_mean",
    "list_join",
    "subset_mean",
    "sub_filter_start_end",
    "min_divisible_value",
    "val2list",
    "download_url",
    "accuracy",
    "AverageMeter",
]


# ============================================================================
# sort_dict
# ============================================================================
# 功能：按字典的值（value）排序，可选择升序或降序，返回字典或键值对列表。
#
# 参数：
#   src_dict   —— 输入字典，例如 {"conv1": 3, "conv2": 1, "conv3": 2}
#   reverse    —— 是否降序排序（默认 False = 升序）
#   return_dict—— 是否返回字典类型；True 返回 dict，False 返回 list of tuples
#
# 返回值：
#   排序后的字典（默认）或键值对列表。
#
# 使用场景：
#   在网络结构搜索中，可能需要按 FLOPs 或参数量对候选结构排序，此函数可方便地
#   完成这一任务。例如 sort_dict(flops_dict, reverse=True) 可以得到按 FLOPs
#   降序排列的架构字典。
# ============================================================================
def sort_dict(src_dict, reverse=False, return_dict=True):
    # sorted 返回列表，元素为 (key, value) 元组
    # key=lambda x: x[1] 表示按 value 排序
    output = sorted(src_dict.items(), key=lambda x: x[1], reverse=reverse)
    if return_dict:
        return dict(output)
    else:
        return output


# ============================================================================
# get_same_padding
# ============================================================================
# 功能：计算保持输入输出空间尺寸相同的 padding 大小（即 SAME padding）。
#
# 数学原理：
#   对于 stride=1 的卷积，如果 kernel_size 是奇数，那么 padding = kernel_size // 2
#   即可保证输入输出尺寸相同。例如 kernel_size=3 → padding=1，kernel_size=5 → padding=2。
#
# 参数：
#   kernel_size —— 卷积核大小，可以是 int（如 3）或 tuple（如 (3, 5)）
#
# 返回值：
#   对应 kernel_size 的 padding 值（int 或 tuple）。
#
# 为什么只支持奇数 kernel_size？
#   如果 kernel_size 是偶数，SAME padding 需要不对称的 padding（左/右不同），
#   实现更复杂且不常见。现代 CNN 几乎全部使用奇数卷积核（3x3、5x5、7x7），
#   所以这里限定奇数是一个合理的设计选择。
# ============================================================================
# 该函数是为了保证 卷积前后特征图不变，在卷积层叠加时，避免特征图不断缩小，这样才能堆叠深层网络
def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        # 对 tuple 类型递归处理每个维度
        # 例如 kernel_size=(3, 5) → 返回 (1, 2)
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


# ============================================================================
# get_split_list
# ============================================================================
# 功能：将 in_dim 尽量均匀地分成 child_num 份，用于弹性网络中的通道分配。
#
# 使用场景：
#   在 OFA 的弹性通道（Elastic Width）训练中，超网络的每一层有一个最大通道数
#   in_dim，但在训练时会随机选择一个子通道数。这个函数将最大通道数切分成
#   child_num 个子集，每个子集对应一个可选的通道配置。
#
# 举例：
#   in_dim=10, child_num=3 → [4, 3, 3]（尽量均匀，多余的放到前面）
#   accumulate=True → [4, 7, 10]（前缀和形式，用于索引区间计算）
#
# 参数：
#   in_dim     —— 总数（如最大通道数）
#   child_num  —— 要切分的份数
#   accumulate —— 是否返回前缀和。True 时返回 [s1, s1+s2, s1+s2+s3, ...]
#                 用于弹性网络中子卷积核的起止索引计算。
#
# 返回值：
#   列表，每个元素为一份的大小；若 accumulate=True 则为前缀和列表。
# ============================================================================
def get_split_list(in_dim, child_num, accumulate=False):
    # 先均分（向下取整）
    in_dim_list = [in_dim // child_num] * child_num
    # 把余数分配到前面的元素，每个 +1
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    if accumulate:
        # 转换为前缀和：[a, b, c] → [a, a+b, a+b+c]
        for i in range(1, child_num):
            in_dim_list[i] += in_dim_list[i - 1]
    return in_dim_list


# ============================================================================
# list_sum
# ============================================================================
# 功能：递归计算列表所有元素的和。
#
# 为什么用递归？
#   这是一个教学/风格上的选择，展示了函数式编程的风格。
#   也可以用 sum(x) 一行实现，但这里用递归是为了支持嵌套列表的扩展性。
#
# 参数：
#   x —— 数值列表，例如 [1, 2, 3, 4]
#
# 返回值：
#   列表元素的总和（数值）。
# ============================================================================
def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


# ============================================================================
# list_mean
# ============================================================================
# 功能：计算列表元素的算术平均值。
#
# 参数：
#   x —— 数值列表
#
# 返回值：
#   平均值（浮点数）。
# ============================================================================
def list_mean(x):
    return list_sum(x) / len(x)


# ============================================================================
# list_join
# ============================================================================
# 功能：将列表元素用指定分隔符连接成字符串。
#
# 参数：
#   val_list —— 元素列表（元素会被 str() 转换）
#   sep      —— 分隔符，默认制表符 \t
#
# 返回值：
#   连接后的字符串。
#
# 使用场景：
#   日志记录或结果输出时，将多个数值格式化为一行文本。
# ============================================================================
def list_join(val_list, sep="\t"):
    return sep.join([str(val) for val in val_list])


# ============================================================================
# subset_mean
# ============================================================================
# 功能：计算列表中指定索引位置元素的平均值。
#
# 参数：
#   val_list    —— 数值列表
#   sub_indexes —— 要取平均值的索引（可以是单个 int 或列表）
#
# 使用场景：
#   在 NAS 中，可能只关心某些候选架构（由索引标识）的性能指标平均值。
# ============================================================================
def subset_mean(val_list, sub_indexes):
    sub_indexes = val2list(sub_indexes, 1)
    return list_mean([val_list[idx] for idx in sub_indexes])


# ============================================================================
# sub_filter_start_end
# ============================================================================
# 功能：在大卷积核中计算子卷积核的起始和结束索引。
#
# 使用场景：
#   在 OFA 的弹性卷积核大小（Elastic Kernel Size）训练中，超网络使用一个
#   大卷积核（如 7x7），但子网络可能只使用其中的一个子区域（如 5x5 或 3x3）。
#   这个函数计算从大卷积核中心裁剪出子卷积核的起止索引。
#
# 举例：
#   kernel_size=7, sub_kernel_size=3
#   → center=3, dev=1, start=2, end=4
#   即从 7x7 卷积核的索引 [2:5] 范围提取 3x3 的子卷积核。
#
# 参数：
#   kernel_size     —— 大卷积核的大小
#   sub_kernel_size —— 子卷积核的大小
#
# 返回值：
#   (start, end) 起止索引对
# ============================================================================
def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


# ============================================================================
# min_divisible_value
# ============================================================================
# 功能：找到接近 v1 且能整除 n1 的最大值。
#
# 数学定义：
#   返回最大的 v，使得 v <= n1 且 n1 % v == 0，且 v <= v1。
#   实际上是从 v1 开始向下递减，直到找到能整除 n1 的数。
#
# 使用场景：
#   在将 BatchNorm 替换为 GroupNorm 时，需要确定分组数 num_groups。
#   我们希望 num_groups 能整除 num_channels，且 num_channels // num_groups
#   尽量接近 gn_channel_per_group。这个函数用来寻找满足整除条件的实际通道数。
#
# 参数：
#   n1 —— 被除数（通常为通道数 num_features）
#   v1 —— 期望的除数上限（通常为每组通道数 gn_channel_per_group）
#
# 返回值：
#   能整除 n1 且不超过 v1 的最大整数。
# ============================================================================
def min_divisible_value(n1, v1):
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


# ============================================================================
# val2list
# ============================================================================
# 功能：将单个值转换为指定长度的列表。如果输入已经是列表/ndarray/tuple，
#       则直接转为 list 返回。
#
# 参数：
#   val         —— 输入值（可以是 int、float、list、ndarray、tuple）
#   repeat_time —— 当 val 是标量时，重复几次生成列表
#
# 返回值：
#   列表。
#
# 使用场景：
#   在弹性网络中，某些配置参数可能是标量（所有层共享）或列表（每层独立）。
#   此函数统一将标量扩展为列表格式，简化后续处理。
# ============================================================================
def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


# ============================================================================
# download_url
# ============================================================================
# 功能：从指定 URL 下载文件到本地缓存目录，如果文件已存在则直接返回缓存路径。
#
# 设计意图：
#   MCUNet 在初始化时可能需要下载预训练权重或配置文件。此函数提供了一个简单的
#   下载缓存机制，避免重复下载。缓存目录默认为 ~/.torch/mcunet/，与 PyTorch
#   的模型缓存约定保持一致。
#
# 参数：
#   url       —— 文件的远程 URL
#   model_dir —— 本地缓存目录（默认 ~/.torch/mcunet/）
#   overwrite —— 是否覆盖已存在的文件（默认 False）
#
# 返回值：
#   成功时返回本地缓存文件路径；失败时返回 None。
# ============================================================================
def download_url(url, model_dir="~/.torch/mcunet", overwrite=False):
    # 从 URL 中提取文件名作为缓存目标文件名
    target_dir = url.split("/")[-1]
    # 展开用户主目录符号 ~
    model_dir = os.path.expanduser(model_dir)
    try:
        # 确保缓存目录存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        # 如果文件不存在或要求覆盖，则下载
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # 下载失败时尝试删除锁文件，以便下次重试
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
        return None


# ============================================================================
# accuracy
# ============================================================================
# 功能：计算分类任务的 Top-k 准确率。
#
# 参数：
#   output —— 模型输出的 logits 张量，形状 (batch_size, num_classes)
#   target —— 真实标签张量，形状 (batch_size,)
#   topk   —— 要计算的 Top-k 值元组，默认为 (1,) 即 Top-1 准确率。
#             例如 topk=(1, 5) 同时计算 Top-1 和 Top-5 准确率。
#
# 返回值：
#   列表，每个元素是对应 k 的准确率（百分比，0~100 之间的浮点数）。
#
# 计算逻辑：
#   1. 用 output.topk() 获取每个样本预测概率最高的 k 个类别
#   2. 将预测值与真实标签比较，得到布尔张量 correct
#   3. 对每个 k，取 correct 的前 k 行，按列求均值得到准确率
# ============================================================================
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    # output.topk(maxk, 1, True, True) 返回两个张量：
    #   - values: 每个样本最高的 maxk 个概率值
    #   - indices: 对应的类别索引（我们只需要这个）
    # 参数说明：dim=1（在类别维上操作），largest=True（取最大值），sorted=True（排序）
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # 转置为 (maxk, batch_size)

    # target.reshape(1, -1) → (1, batch_size)
    # .expand_as(pred) → (maxk, batch_size)
    # eq: 逐元素比较，得到形状 (maxk, batch_size) 的布尔张量
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct[:k] 取前 k 行 → (k, batch_size)
        # .reshape(-1) → 展平为一维
        # .float().sum(0, keepdim=True) → 求和（正确预测数）
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        # 乘以 100.0 / batch_size 得到百分比形式的准确率
        res.append(correct_k.mul_(100.0 / batch_size).mean())
    return res


# ============================================================================
# AverageMeter
# ============================================================================
# 功能：滑动平均跟踪器，用于在训练/评估过程中记录和计算指标的平均值。
#
# 典型用法：
#   loss_meter = AverageMeter()
#   for batch in dataloader:
#       loss = train_one_batch(batch)
#       loss_meter.update(loss.item(), batch_size)
#   print(f"Average loss: {loss_meter.avg}")
#
# 设计意图：
#   这是一个在 PyTorch 训练脚本中非常常见的工具类模式。它维护了：
#   - val:   最近一次更新的值
#   - sum:   所有更新值的加权和（权重为 n）
#   - count: 累计样本数
#   - avg:   加权平均值（sum / count）
#
#   与简单地维护一个列表然后取平均相比，AverageMeter 的优势在于：
#   1. 不需要存储所有历史值，内存友好
#   2. 支持加权平均（n 参数）
#   3. 可以持续更新，适合在线（online）统计
# ============================================================================
class AverageMeter(object):
    def __init__(self):
        self.val = 0  # 最近一次更新的值
        self.avg = 0  # 从开始到现在的加权平均
        self.sum = 0  # 加权总和
        self.count = 0  # 累计处理的样本数

    def reset(self):
        """重置所有统计量为零"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """更新统计量
        参数：
          val —— 当前批次的值（如 loss）
          n   —— 当前批次的样本数（权重）
        """
        self.val = val
        # sum = sum + val * n，加权累加
        self.sum += val * n
        # count = count + n，总样本数累加
        self.count += n
        # avg = sum / count，重新计算加权平均
        self.avg = self.sum / self.count
