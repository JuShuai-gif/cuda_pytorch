# ============================================================================
# eval_torch.py —— 使用 PyTorch 评估 FP32 浮点模型的分类准确率
#
# 用途：
#   对 MCUNet 的 PyTorch 预训练模型在验证集上评测 Top-1 准确率，
#   并统计模型的 FLOPs 和参数量。
#
# 对比 eval_tflite.py（TFLite INT8 评测）：
#   - 本脚本：PyTorch FP32，反映模型的"理论"精度天花板
#   - eval_tflite.py：TFLite INT8，反映量化后部署到 MCU 的实际精度
#   - 两者差异 = 量化带来的精度损失（通常是 0.5%~2%）
#
# 流程：
#   1. 从 model_zoo 下载模型配置 + 预训练权重
#   2. 在 GPU 上构建网络并加载权重
#   3. 统计 FLOPs 和参数量
#   4. 在验证集上逐 batch 前向传播，计算 Top-1 准确率
# ============================================================================

import os
from tqdm import tqdm
import json

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import datasets, transforms

from mcunet.model_zoo import build_model
from mcunet.utils import AverageMeter, accuracy, count_net_flops, count_parameters

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "--net_id", type=str, help="模型标识符，如 mcunet-vww1、mcunet-in0 等"
)
parser.add_argument(
    "--dataset",
    default="imagenet",
    type=str,
    choices=["imagenet", "vww"],
    help="数据集名称：imagenet（ImageNet 分类）/ vww（Visual Wake Words）",
)
parser.add_argument(
    "--data-dir",
    default=os.path.expanduser("/dataset/imagenet/val"),
    help="验证集数据的本地路径",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="推理时的 batch size（GPU 推理，越大吞吐越高）",
)
parser.add_argument(
    "-j", "--workers", default=8, type=int, metavar="N", help="数据加载的线程数"
)

args = parser.parse_args()

# cuDNN 自动调优：根据输入的卷积配置选择最快的卷积算法
# benchmark=True 会在前几次推理中尝试多种算法，选最优的固定下来
torch.backends.cudnn.benchmark = True
device = "cuda"


# ============================================================================
# build_val_data_loader —— 构建验证集的 DataLoader
#
# 参数:
#   resolution (int): 模型的输入分辨率（MCUNet 各模型分辨率不同，
#                     由 model_zoo 中的 JSON 配置指定）
#
# 两种数据集的数据增强策略：
#   - ImageNet：Resize + CenterCrop（标准流程）
#   - VWW：直接 Resize（不 CenterCrop，以免把行人裁掉）
#
# 与 eval_tflite.py 的区别：
#   - PyTorch 评测使用 Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
#     将 [0, 1] 的输入归一化到 [-1, 1]
#   - TFLite 评测不做此归一化（TFLite 的量化和 float 处理方式不同）
# ============================================================================
def build_val_data_loader(resolution):
    # 标准化到 [-1, 1]：x = (x - 0.5) / 0.5
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {"num_workers": args.workers, "pin_memory": True}

    if args.dataset == "imagenet":
        val_transform = transforms.Compose(
            [
                transforms.Resize(int(resolution * 256 / 224)),  # 短边缩放
                transforms.CenterCrop(resolution),  # 中心裁剪
                transforms.ToTensor(),  # [0, 255] → [0, 1]
                normalize,  # [0, 1] → [-1, 1]
            ]
        )
    elif args.dataset == "vww":
        val_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),  # 直接缩放，不裁切
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise NotImplementedError

    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return val_loader


# ============================================================================
# validate —— 在验证集上评测模型的 Top-1 分类准确率
#
# 参数:
#   model: 已加载权重的 PyTorch 模型
#   val_loader: 验证集 DataLoader
#
# 返回:
#   float: Top-1 准确率（百分比）
#
# 注意：
#   - 不计算梯度（torch.no_grad()），节省显存和计算
#   - 使用 AverageMeter 滑动平均跟踪 loss 和 top1
# ============================================================================
def validate(model, val_loader):
    model.eval()  # 切换到 eval 模式（关闭 dropout/BN 统计更新）
    val_loss = AverageMeter()  # 滑动平均跟踪器：记录交叉熵损失
    val_top1 = AverageMeter()  # 滑动平均跟踪器：记录 Top-1 准确率

    with tqdm(total=len(val_loader), desc="Validate") as t:
        with torch.no_grad():  # 推理模式下不计算梯度，省显存
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                # 前向传播（模型在 GPU 上执行）
                output = model(data)

                # 计算交叉熵损失
                val_loss.update(F.cross_entropy(output, target).item())

                # 计算 Top-1 准确率（accuracy 函数返回百分比）
                top1 = accuracy(output, target, topk=(1,))[0]
                val_top1.update(top1.item(), n=data.shape[0])

                # 更新进度条显示
                t.set_postfix({"loss": val_loss.avg, "top1": val_top1.avg})
                t.update(1)

    return val_top1.avg


# ============================================================================
# main —— 主流程
# ============================================================================
def main():
    # 第一步：从 model_zoo 构建模型并加载预训练权重
    # build_model 会：
    #   1. 下载网络结构 JSON 配置
    #   2. 根据配置构建 ProxylessNASNet（MCUNet 的网络结构）
    #   3. 如果 pretrained=True，下载预训练权重并加载
    # 返回: (模型, 输入分辨率, 模型描述)
    model, resolution, description = build_model(args.net_id, pretrained=True)

    # 第二步：将模型移动到 GPU 并切换到 eval 模式
    model = model.to(device)
    model.eval()

    # 第三步：构建验证集 DataLoader
    val_loader = build_val_data_loader(resolution)

    # 第四步：统计模型的计算量和参数量
    #   - FLOPs：浮点运算次数，反映模型的计算复杂度（用 torchprofile 库）
    #   - Params：可训练参数数量，反映模型的存储需求
    total_macs = count_net_flops(model, [1, 3, resolution, resolution])
    total_params = count_parameters(model)
    print(
        " * FLOPs: {:.4}M, param: {:.4}M".format(total_macs / 1e6, total_params / 1e6)
    )

    # 第五步：在验证集上评测准确率
    acc = validate(model, val_loader)
    print(" * Accuracy: {:.2f}%".format(acc))


if __name__ == "__main__":
    main()
