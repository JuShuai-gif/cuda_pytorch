# ============================================================================
# eval_tflite.py —— 使用 TensorFlow Lite 评估量化模型的分类准确率
#
# 用途：
#   将经过量化的 TFLite 模型（INT8）在验证集上评测 Top-1 准确率。
#   这是 MCUNet 部署到 MCU 前的最终精度验证步骤 —— TFLite 的量化推理
#   结果与 TinyEngine 在 MCU 上运行的结果基本一致。
#
# 对比 eval_torch.py（PyTorch FP32 评测）：
#   - FP32 模型精度反映的是浮点模型的"天花板"精度
#   - TFLite INT8 精度反映的是量化后实际部署到 MCU 的精度
#   - 两者之间的差距就是量化带来的精度损失
#
# 加速技巧：
#   1. 先将整个验证集缓存到内存中，避免每次迭代从磁盘读取
#   2. 使用 multiprocessing.Pool 多进程并行推理（单次 TFLite 推理是串行的，
#      多进程可以将多张图片分到多个 CPU 核上推理）
# ============================================================================

import os
import argparse
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
import tensorflow as tf

from mcunet.model_zoo import download_tflite

# 强制 TensorFlow 仅在 CPU 上运行
# TFLite 的 INT8 推理在 CPU 上效率很高（使用了 XNNPACK 等优化），
# 而且 MCU 目标平台（ARM Cortex-M）也是 CPU，用 CPU 评测更真实
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 关闭 TF 的非必要日志输出
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    help="数据集名称：imagenet（ImageNet 分类）或 vww（Visual Wake Words 行人检测分类）",
)
parser.add_argument(
    "--data-dir", default="/dataset/imagenet/val", help="验证集数据的本地路径"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    help="数据加载的 batch size（仅影响数据加载，TFLite 每次推理 1 张图）",
)
parser.add_argument(
    "-j", "--workers", default=16, type=int, metavar="N", help="数据加载的线程数"
)

args = parser.parse_args()


# ============================================================================
# get_val_dataset —— 构建验证集 DataLoader
#
# 参数:
#   resolution (int): 模型的输入分辨率（宽高相等时只传一个整数）
#
# 两种数据集的数据增强策略不同：
#   - ImageNet：先 Resize 到 256/224 比例，再 CenterCrop 到目标分辨率
#   - VWW（Visual Wake Words）：直接 Resize 到目标分辨率，不做 CenterCrop，
#     因为行人在图像边缘时 CenterCrop 可能会把目标裁掉
#
# 注意：TFLite 评测不进行归一化（ToTensor 只将像素缩放到 [0,1]），
# 因为 TFLite 模型的输入量化器会自己处理输入数据到 INT8 的转换。
# ============================================================================
def get_val_dataset(resolution):
    # 数据加载配置：不使用 pin_memory（因为我们在 CPU 上跑 TF Lite 推理）
    kwargs = {"num_workers": args.workers, "pin_memory": False}
    if args.dataset == "imagenet":
        # ImageNet 标准验证流程：Resize + CenterCrop
        val_transform = transforms.Compose(
            [
                transforms.Resize(
                    int(resolution * 256 / 224)
                ),  # 先放大到 256/224 倍，再裁中间
                transforms.CenterCrop(resolution),  # 中心裁剪到目标尺寸
                transforms.ToTensor(),  # [0, 255] → [0, 1]
            ]
        )
    elif args.dataset == "vww":
        # VWW 数据集：直接 Resize，不 CenterCrop（保持行人在画面中的位置）
        val_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),  # [0, 255] → [0, 1]
            ]
        )
    else:
        raise NotImplementedError

    # ImageFolder 自动按子目录名称分配类别标签
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return val_loader


# ============================================================================
# eval_image —— 对单张图像执行 TFLite 推理并判断分类是否正确
#
# 参数:
#   data (tuple): (image_tensor, target_label)
#     - image_tensor: 形状为 (C, H, W) 的 PyTorch Tensor，值范围 [0, 1]
#     - target_label: 真实类别标签
#
# 返回:
#   bool: 模型预测的类别是否与真实标签一致
#
# 数据流:
#   1. 输入 image_tensor: (C, H, W)，值 [0, 1]
#   2. permute 调整维度: (H, W, C) —— TFLite 使用 NHWC 格式
#   3. 转为 NumPy 并量化到 INT8: (pixel * 255 - 128) → [-128, 127]
#   4. 设置到 TFLite 输入张量，调用 invoke 推理
#   5. 读取输出张量中得分最高的类别，与真实标签比较
# ============================================================================
def eval_image(data):
    image, target = data
    # 如果没有 batch 维度，添加一个
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    # PyTorch 使用 NCHW 格式，但 TFLite 使用 NHWC 格式，需要 permute
    image = image.permute(0, 2, 3, 1)  # (1, C, H, W) → (1, H, W, C)
    image_np = image.cpu().numpy()

    # 将 [0, 1] 的 float 图像量化到 INT8 范围 [-128, 127]
    # TFLite 量化模型要求输入为 INT8 类型
    image_np = (image_np * 255 - 128).astype(np.int8)

    # 将输入数据写入 TFLite Interpreter 的输入张量
    interpreter.set_tensor(input_details[0]["index"], image_np.reshape(*input_shape))
    # 执行一次推理（invoke 内部会运行整个计算图）
    interpreter.invoke()
    # 读取分类输出（形状为 (1, num_classes) 的 INT8 数据）
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # 将输出转为 PyTorch 并取 argmax 得到预测类别
    output = torch.from_numpy(output_data).view(1, -1)
    is_correct = torch.argmax(output, dim=1).item() == target.item()
    return is_correct


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    # 第一步：下载指定 net_id 的 TFLite 量化模型
    tflite_path = download_tflite(net_id=args.net_id)

    # 第二步：创建 TFLite Interpreter 并分配张量内存
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()

    # 第三步：获取模型的输入/输出张量信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]

    # 模型的输入形状通常是 (1, H, W, 3)，提取分辨率
    resolution = input_shape[1]

    # 第四步：缓存整个验证集到内存中
    # 直接从磁盘加载每次迭代需要约 20 分钟（磁盘 IO 瓶颈）
    # 缓存到内存后只需要约 2 分钟（完全受 CPU 推理速度限制）
    print(" * start caching the test set...", end="")
    val_loader = get_val_dataset(resolution)  # 值范围 [0, 1]
    val_loader_cache = [v for v in val_loader]  # 将所有 batch 数据加载到内存
    images = torch.cat([v[0] for v in val_loader_cache], dim=0)
    targets = torch.cat([v[1] for v in val_loader_cache], dim=0)
    # 将每张图片和标签打包为 (image, target) 元组列表，方便多进程处理
    val_loader_cache = [[x, y] for x, y in zip(images, targets)]
    print("done.")
    print(" * dataset size:", len(val_loader_cache))

    # 第五步：使用多进程池进行并行推理
    # TFLite 的单次推理在单线程中执行，但多个推理可以并行到不同的 CPU 核上
    # n_thread=32 会根据 CPU 核数自动调整，一般设为物理核数的 2 倍
    n_thread = 32

    p = Pool(n_thread)
    correctness = []  # 存储每个样本的分类是否正确

    # 使用 imap_unordered 实现流水线处理：
    # 只要有一个推理完成就返回结果，无需等待整个 batch
    pbar = tqdm(
        p.imap_unordered(eval_image, val_loader_cache),
        total=len(val_loader_cache),
        desc="Evaluating...",
    )
    for idx, correct in enumerate(pbar):
        correctness.append(correct)
        # 实时更新进度条上的 Top-1 准确率
        pbar.set_postfix(
            {
                "top1": sum(correctness) / len(correctness) * 100,
            }
        )

    # 打印最终准确率
    print(
        "* top1: {:.2f}%".format(
            sum(correctness) / len(correctness) * 100,
        )
    )
