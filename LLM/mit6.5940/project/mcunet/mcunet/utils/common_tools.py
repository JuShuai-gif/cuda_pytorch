# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import numpy as np
import os
import sys

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

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


# 按字典的值排序（默认升序），返回排序后的字典或列表
def sort_dict(src_dict, reverse=False, return_dict=True):
    output = sorted(src_dict.items(), key=lambda x: x[1], reverse=reverse)
    if return_dict:
        return dict(output)
    else:
        return output


# 计算 same padding 的像素数：kernel_size 为奇数时，padding = kernel_size // 2
def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


# 将 in_dim 尽量均匀地切分成 child_num 份（用于弹性网络中通道数的分配）
# accumulate=True 时，返回前缀和列表，用于索引区间
def get_split_list(in_dim, child_num, accumulate=False):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    if accumulate:
        for i in range(1, child_num):
            in_dim_list[i] += in_dim_list[i - 1]
    return in_dim_list


# 递归求和
def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


# 求列表平均值
def list_mean(x):
    return list_sum(x) / len(x)


# 将列表元素用分隔符连接为字符串
def list_join(val_list, sep="\t"):
    return sep.join([str(val) for val in val_list])


# 计算列表中指定索引位置元素的平均值
def subset_mean(val_list, sub_indexes):
    sub_indexes = val2list(sub_indexes, 1)
    return list_mean([val_list[idx] for idx in sub_indexes])


# 计算大卷积核中提取子卷积核的起止索引（用于弹性卷积核）
# 例如 kernel_size=5, sub_kernel_size=3 → start=1, end=4
def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


# 找到 v1 的最大值，使得 v1 ≤ n1 且 n1 能被 v1 整除（用于弹性网络的分组大小计算）
def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


# 将单个值扩展为指定长度的列表（如果已经是 list/ndarray/tuple 则直接返回）
def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


# 下载远程文件，缓存到本地 ~/.torch/mcunet/ 目录（已有则不重复下载）
def download_url(url, model_dir="~/.torch/mcunet", overwrite=False):
    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # 下载失败时删除锁文件，下次可重试
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
        return None


# 计算分类任务的 Top-K 准确率（返回百分比）
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).mean())
    return res


# 滑动平均跟踪器：记录每次更新后的当前值、累计总和与均值
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0  # 最近一次更新的值
        self.avg = 0  # 滑动平均值
        self.sum = 0  # 所有更新值的加权和
        self.count = 0  # 累计样本数

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
