# ============================================================================
# bn_utils.py —— Batch Normalization 工具函数集合
#
# 来源：
#   Once for All: Train One Network and Specialize it for Efficient Deployment
#   Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
#   International Conference on Learning Representations (ICLR), 2020.
#
# 本文件提供三个与 BN（Batch Normalization）层操作密切相关的实用函数：
#   1. set_running_statistics        —— 用校准数据集重新估计 BN 层的 running_mean/var
#   2. adjust_bn_according_to_idx    —— 根据通道索引调整 BN 参数（适配剪枝后的子网络）
#   3. copy_bn                       —— 从一个 BN 层复制参数到另一个 BN 层
#
# 在 Once-for-All (OFA) / MCUNet 这类弹性网络训练框架中，超网络（supernet）训练
# 完成后需要为每个子网络（subnet）重新校准 BN 统计量。因为子网络使用的通道子集
# 与超网络不同，原始的 running_mean/running_var 不再准确，需要用校准数据重新统计。
# ============================================================================

import copy

import torch.nn.functional as F
import torch.nn as nn
import torch

# __all__ 控制 from bn_utils import * 时暴露的公共接口
__all__ = ["set_running_statistics", "adjust_bn_according_to_idx", "copy_bn"]


# ============================================================================
# set_running_statistics
# ============================================================================
# 功能：使用校准数据集（data_loader）重新计算网络中所有 BN 层的
#       running_mean 和 running_var 统计量。
#
# 为什么要重新计算？
#   在弹性网络（如 OFA）中，一个超网络训练完成后，可以从中采样出不同架构的子网络。
#   子网络的每一层可能只用了原始通道的一部分（通道剪枝）。此时原始的 BN 统计量
#   （基于完整通道数计算）不再适用，必须用一小批校准数据重新 forward 来收集新的
#   统计数据。这就是本函数的核心用途。
#
# 参数说明：
#   model       —— 目标 PyTorch 模型（其中的 BN 层将被更新）
#   data_loader —— 校准数据加载器，每次迭代返回 (images, labels)
#                  images 的形状为 (batch_size, channels, height, width)
#   distributed —— 是否使用分布式训练（当前未实现）
#   maximum_iter—— 最多使用多少批数据来校准。默认 -1 表示使用整个 data_loader。
#                  限制迭代次数可以加快校准速度（通常 10~20 批就够了）。
# ============================================================================
def set_running_statistics(model, data_loader, distributed=False, maximum_iter=-1):
    # 延迟导入，避免循环依赖
    # AverageMeter: 滑动平均跟踪器，用于累积各批次的均值和方差
    from .common_tools import AverageMeter

    # get_net_device: 获取模型所在设备（CPU / CUDA）
    from .pytorch_utils import get_net_device

    # DynamicBatchNorm2d: 弹性 BN 层，支持通道数动态变化的场景
    from ..tinynas.elastic_nn.modules import DynamicBatchNorm2d

    # bn_mean / bn_var: 两个字典，以 BN 层名称为键，对应存储该层的
    # running_mean 和 running_var 的滑动平均累积器。后续用这些累积器
    # 对所有校准批次的统计量做加权平均。
    bn_mean = {}
    bn_var = {}

    # ---------------------------------------------------------------
    # 第一步：深拷贝一份模型用于前向统计
    # ---------------------------------------------------------------
    # 为什么要深拷贝？
    #   因为我们会在 forward 过程中劫持（monkey-patch）BN 层的 forward 方法。
    #   这种修改是不可逆的，如果直接在原模型上操作会破坏模型状态。
    #   所以先 copy 一份 forward_model，在它上面进行统计收集，最后再把
    #   统计结果写回原模型的 BN 层。
    forward_model = copy.deepcopy(model)

    # 遍历 forward_model 中的所有命名模块，找到 BatchNorm2d 类型的层
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # ---------------------------------------------------------------
            # 初始化该 BN 层对应的滑动平均跟踪器
            # ---------------------------------------------------------------
            if distributed:
                raise NotImplementedError
            else:
                # AverageMeter 会记录每次 update 传入的值（val）和权重（n），
                # 并维护加权和 sum 与总样本数 count，最终 avg = sum / count。
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

            # ---------------------------------------------------------------
            # 替换 BN 层的 forward 方法 —— 这是整个校准过程的核心
            # ---------------------------------------------------------------
            # 原始的 BN.forward 做两件事：
            #   1. 计算当前批次的 batch_mean / batch_var
            #   2. 用滑动平均更新 self.running_mean / self.running_var
            #      （running_mean = momentum * running_mean + (1 - momentum) * batch_mean）
            #
            # 而我们这里想要的是：累积所有校准批次的无偏统计量，然后手动计算
            # 最终的加权平均。所以我们用 new_forward 闭包替换原始的 forward：
            #   - 仍然计算 batch_mean / batch_var
            #   - 但不更新 running_*，而是通过外层的 AverageMeter 累积
            #   - 使用当前批次的 batch_mean / batch_var 来做 BN 变换
            #     （注意传给 F.batch_norm 的 running_mean/running_var 参数
            #      就是刚算出的 batch_mean/batch_var，并设置 training=False，
            #      这样 F.batch_norm 就不会再去更新 running 值了）
            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    # 计算当前批次的均值：在 H 和 W 维度上取平均
                    # x.shape = (batch, C, H, W)
                    # x.mean(0) 在 batch 维取平均 → (C, H, W)
                    # 再 .mean(2) 在 H 维 → (C, 1, W)，再 .mean(3) 在 W 维 → (C, 1, 1)
                    # 最终形状 (1, C, 1, 1)，保持维度便于广播运算
                    batch_mean = (
                        x.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )  # (1, C, 1, 1)

                    # 计算当前批次的方差：先算差值平方，再在 H/W 维取平均
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = (
                        batch_var.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )

                    # 把 (1, C, 1, 1) 的统计量 squeeze 为 (C,) 向量
                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    # 将当前批次的统计量累加到滑动平均跟踪器中
                    # n = x.size(0) 是该批次的样本数，用于加权
                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # 使用当前批次的 batch_mean/batch_var 做 BN 变换
                    # F.batch_norm 的参数说明：
                    #   - running_mean / running_var: 传入刚算出的 batch_mean/batch_var
                    #   - training=False: 强制使用传入的统计量，不更新 running 值
                    #   - momentum=0.0: 不使用滑动平均更新
                    #   - eps: BN 的 epsilon，防止除零
                    #   - bn.weight[:feature_dim]: 取有效通道数对应的权重
                    #     （弹性网络中 feature_dim <= 原始通道数）
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,  # training=False
                        0.0,  # momentum
                        bn.eps,
                    )

                return lambda_forward

            # 将 BN 层原始的 forward 替换为我们自己定义的闭包
            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # ---------------------------------------------------------------
    # 第二步：用校准数据前向传播，收集统计量
    # ---------------------------------------------------------------
    with torch.no_grad():
        # 设置全局标志，通知 DynamicBatchNorm2d 层正处于统计校准阶段
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True

        for i_iter, (images, _) in enumerate(data_loader):
            if maximum_iter > 0 and i_iter == maximum_iter:
                # 达到最大迭代次数，提前停止收集
                break
            # 如果输入图像是单通道（灰度图），复制三次变为三通道
            # 这是因为模型通常期望 3 通道 RGB 输入
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            # 将图像数据移到模型所在设备
            images = images.to(get_net_device(forward_model))
            # 前向传播，触发所有 BN 层被替换后的 forward 逻辑
            forward_model(images)

        # 校准结束，重置全局标志
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    # ---------------------------------------------------------------
    # 第三步：将累积的统计量写回原模型的 BN 层
    # ---------------------------------------------------------------
    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            # copy_ 将累积的均值/方差的加权平均写入原模型 BN 层的 running_* 中
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

    # 释放深拷贝的 forward_model 占用的显存/内存
    del forward_model


# ============================================================================
# adjust_bn_according_to_idx
# ============================================================================
# 功能：根据索引列表 idx 从 BN 层的 weight/bias/running_mean/running_var 中
#       选取对应的通道子集。
#
# 使用场景：
#   在弹性网络（OFA）中，当我们从超网络中采样出一个子网络时，子网络每个卷积层
#   的输出通道数可能是超网络的子集。相应地，紧跟在卷积层之后的 BN 层的通道数也
#   需要剪裁。这个函数就是做这个剪裁的：用 torch.index_select 按 idx 选出指定
#   的通道。
#
# 参数说明：
#   bn  —— 目标 BN 层（nn.BatchNorm2d 实例），其参数会被就地修改
#   idx —— 一维张量（LongTensor），表示要保留的通道索引，
#           例如 idx = [0, 2, 5, 7, ...] 表示只保留第 0、2、5、7... 个通道
# ============================================================================
def adjust_bn_according_to_idx(bn, idx):
    # torch.index_select 沿第 0 维（通道维）选取 idx 中指定的通道
    # bn.weight.shape = (num_features,)
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
    bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


# ============================================================================
# copy_bn
# ============================================================================
# 功能：将一个 BN 层（src_bn）的参数复制到另一个 BN 层（target_bn）。
#
# 使用场景：
#   在 MCUNet 的部署阶段，当我们从训练好的超网络中 specialize 出一个子网络时，
#   需要创建一个新的（较小的）BN 层，然后将超网络中对应 BN 层的前 feature_dim
#   个通道的参数复制过去。由于子网络可能只用了超网络部分通道，所以切片到
#   feature_dim 即可。
#
# 参数说明：
#   target_bn —— 目标 BN 层（接收参数的一方）
#   src_bn    —— 源 BN 层（提供参数的一方）
#
# 注意：
#   target_bn.num_features 可能小于 src_bn.num_features，此时只复制前
#   feature_dim 个通道的参数。这常用于 OFA 的 subnet specialize 阶段。
# ============================================================================
def copy_bn(target_bn, src_bn):
    feature_dim = target_bn.num_features

    # copy_ 是 PyTorch 张量的就地复制操作
    # [:feature_dim] 切片：只取 src_bn 的前 feature_dim 个通道的参数
    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
    target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])
