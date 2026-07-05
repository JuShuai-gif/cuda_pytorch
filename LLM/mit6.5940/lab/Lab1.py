import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

from torchprofile import profile_macs

assert torch.cuda.is_available(), (
    "The current runtime does not have CUDA support."
    "Please go to menu bar (Runtime - Change runtime type) and select GPU"
)

"""
目标：
1. 理解剪枝的基本概念
2. 实现并应用细粒度剪枝
3. 实现并应用通道剪枝
4. 理解这些剪枝方法之间的差异和权衡
"""

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def download_url(url, model_dir=".", overwrite=False):
    import os, sys, ssl
    from urllib.request import urlretrieve

    ssl._create_default_https_context = ssl._create_unverified_context
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
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
        return None


class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != "M":
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(kernel_size=2, stride=2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # avgpool: [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    callbacks=None,
) -> None:
    model.train()

    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()

        if callbacks is not None:
            for callback in callbacks:
                callback()


@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, verbose=True) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(
        dataloader, desc="eval", leave=False, disable=not verbose
    ):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()


# 辅助函数 (FLOPs、模型大小计算)
def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def test_fine_grained_prune(
    test_tensor=torch.tensor(
        [
            [-0.46, -0.40, 0.39, 0.19, 0.37],
            [0.00, 0.40, 0.17, -0.15, 0.16],
            [-0.20, -0.23, 0.36, 0.25, 0.03],
            [0.24, 0.41, 0.07, 0.13, -0.15],
            [0.48, -0.09, -0.36, 0.12, 0.45],
        ]
    ),
    test_mask=torch.tensor(
        [
            [True, True, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, True, False, False, False],
            [True, False, False, False, True],
        ]
    ),
    target_sparsity=0.75,
    target_nonzeros=None,
):
    def plot_matrix(tensor, ax, title):
        ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap="tab20c")
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(
                    j,
                    i,
                    f"{tensor[i, j].item():.2f}",
                    ha="center",
                    va="center",
                    color="k",
                )

    test_tensor = test_tensor.clone()
    fig, axes = plt.subplots(1, 2, figsize=(6, 10))
    ax_left, ax_right = axes.ravel()
    plot_matrix(test_tensor, ax_left, "dense tensor")

    sparsity_before_pruning = get_sparsity(test_tensor)
    mask = fine_grained_prune(test_tensor, target_sparsity)
    sparsity_after_pruning = get_sparsity(test_tensor)
    sparsity_of_mask = get_sparsity(mask)

    plot_matrix(test_tensor, ax_right, "sparse tensor")

    fig.tight_layout()
    plt.savefig("pruning_mask.png")
    plt.close(fig)

    print("* Test fine_grained_prune()")
    print(f"    target sparsity: {target_sparsity:.2f}")
    print(f"        sparsity before pruning: {sparsity_before_pruning:.2f}")
    print(f"        sparsity after pruning: {sparsity_after_pruning:.2f}")
    print(f"        sparsity of pruning mask: {sparsity_of_mask:.2f}")

    if target_nonzeros is None:
        if test_mask.equal(mask):
            print("* Test passed.")
        else:
            print("* Test failed.")
    else:
        if mask.count_nonzero() == target_nonzeros:
            print("* Test passed.")
        else:
            print("* Test failed.")


# 从本地加载预训练的 VGG 模型权重
checkpoint_path = "/home/ghr/code/model/vgg.cifar.pretrained.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")  # 先加载到 CPU
model = VGG().cuda()  # 模型移到 GPU
print(f"=> 加载检查点 '{checkpoint_path}'")
model.load_state_dict(checkpoint["state_dict"])  # 载入权重

# 因为这里是保存的最原始的权重
recover_model = lambda: model.load_state_dict(
    checkpoint["state_dict"]
)  # 快速恢复稠密模型的辅助函数

image_size = 32
transforms = {
    "train": Compose(
        [  # 训练时使用数据增强
            RandomCrop(
                image_size, padding=4
            ),  # 随机裁剪：先四周各 pad 4 像素，再随机裁回 32x32
            RandomHorizontalFlip(),  # 随机水平翻转
            ToTensor(),  # 转为 Tensor，并将像素值从 [0,255] 归一化到 [0,1]
        ]
    ),
    "test": ToTensor(),  # 测试时只做归一化，不做增强
}


# 加载 CIFAR-10 数据集（自动识别 train/test 两个子集）
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="/home/ghr/code/data",  # 数据集存放路径，会自动在该目录下找 cifar-10-batches-py/
        train=(split == "train"),  # True 加载训练集，False 加载测试集
        download=False,  # 若 root 下没有数据则自动下载；已有则跳过
        transform=transforms[split],  # 对每张图片应用对应的预处理
    )

# 封装为 DataLoader，提供批量迭代、打乱、多线程加载等功能
dataloader = {}
for split in ["train", "test"]:
    dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,  # 每批 512 张图片
        shuffle=(split == "train"),  # 训练集打乱顺序，测试集保持原序
        num_workers=0,  # 数据加载的子进程数（0 表示主进程加载）
        pin_memory=False,  # 将数据锁页到 CPU 内存，加速向 GPU 传输
    )


# 首先评估稠密模型的准确率和模型大小
"""

"""

dense_model_accuracy = evaluate(model, dataloader["test"])
dense_model_size = get_model_size(model)
print(f"dense model has accuracy={dense_model_accuracy:.2f}%")
print(f"dense model has size={dense_model_size / MiB:.2f} MiB")
exit(0)


# 看看稠密模型中权重值的分布
def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    """绘制模型各层权重的直方图分布"""
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:  # 只绘制卷积层和全连接层的权重（跳过 BN 等一维参数）
            ax = axes[plot_index]
            if count_nonzero_only:
                # 只统计非零权重（用于查看剪枝后剩余权重的分布）
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True, color="blue", alpha=0.5)
            else:
                # 统计所有权重
                ax.hist(
                    param.detach().view(-1).cpu(),
                    bins=bins,
                    density=True,
                    color="blue",
                    alpha=0.5,
                )
            ax.set_xlabel(name)
            ax.set_ylabel("density")
            plot_index += 1
    fig.suptitle("Histogram of Weights")
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    suffix = "_nonzero" if count_nonzero_only else ""
    plt.savefig(f"weight_distribution{suffix}.png")
    plt.close(fig)


plot_weight_distribution(model)

"""
问题1
1. 不同层的权重分布有哪些共同特征？
   均呈零均值对称分布（高斯状），大量权重集中在 0 附近，少量大权重形成长尾；
   不同层方差不同（浅层分布更宽，深层更集中），classifier 层偏态明显。

2. 这些特征如何帮助剪枝？
   大量小权重意味着模型严重过参数化，非常适合基于幅度的非结构化剪枝
   （magnitude pruning）。可按层自适应设置阈值：浅层保守剪、深层激进剪，
   classifier 层轻剪或不剪。剪枝后微调即可恢复精度。
"""

"""
细粒度剪枝：细粒度剪枝移除重要性最低的突触，经过细粒度剪枝后，权重张量 W 会变得稀疏

基于幅度的剪枝：
"""

# 问题 2 ：实现基于幅度的细粒度剪枝函数
"""
step 1: 存储剪枝后零元素数量
step 2: 计算权重张量的重要性
step 3: 计算剪枝 threshold(阈值),使得所有重要性小于 threshold 的突触被移除
step 4: 基于 threshold 计算剪枝 mask
"""


def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    ##################### YOUR CODE STARTS HERE #####################
    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(sparsity * num_elements)
    # Step 2: calculate the importance of weight
    importance = tensor.abs().view(-1)
    # Step 3: calculate the pruning threshold
    threshold = importance.kthvalue(num_zeros).values.item()
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = tensor.abs() > threshold
    ##################### YOUR CODE ENDS HERE #######################

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask


# 虚拟张量上验证细粒度剪枝功能
test_fine_grained_prune()


# 问题 3
# 修改 target_sparsity 的值，使得剪枝后的稀疏张量中仅有 10 个非零元素
target_sparsity = 0.6  # 修改这个
test_fine_grained_prune(target_sparsity=target_sparsity, target_nonzeros=10)


# 细粒度剪枝函数封装到一个类中
class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks


# 灵敏度扫描
"""
不同层对模型性能的贡献不同，决定每层合适的稀疏度具有挑战性。

在灵敏度扫描过程中，每次只剪枝一个层以观察准确率的下降，通过扫描不同的稀疏度，可以
绘制出对应层的灵敏度曲线(即准确率与稀疏度的关系)
"""


# 灵敏度扫描：每次只剪枝一个层，观察准确率随稀疏度的变化曲线
@torch.no_grad()
def sensitivity_scan(
    model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True
):
    sparsities = np.arange(
        start=scan_start, stop=scan_end, step=scan_step
    )  # 稀疏度扫描范围
    accuracies = []
    named_conv_weights = [
        (name, param) for (name, param) in model.named_parameters() if param.dim() > 1
    ]  # 只扫描卷积层和全连接层
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()  # 保存原始权重，每次扫描后恢复
        accuracy = []
        for sparsity in tqdm(
            sparsities,
            desc=f"scanning {i_layer}/{len(named_conv_weights)} weight - {name}",
        ):
            fine_grained_prune(param.detach(), sparsity=sparsity)  # 对当前层剪枝
            acc = evaluate(model, dataloader, verbose=False)  # 评估剪枝后准确率
            if verbose:
                print(f"\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%", end="")
            # 恢复原始权重，确保下一轮扫描从稠密模型开始
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(
                f"\r    sparsity=[{','.join(['{:.2f}'.format(x) for x in sparsities])}]: accuracy=[{', '.join(['{:.2f}%'.format(x) for x in accuracy])}]",
                end="",
            )
        accuracies.append(accuracy)
    return sparsities, accuracies


# 这里比较久
# 执行灵敏度扫描
sparsities, accuracies = sensitivity_scan(
    model, dataloader["test"], scan_step=0.1, scan_start=0.4, scan_end=1.0
)


def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy):
    # 灵敏度下限：允许准确率降至稠密模型的 1.5 倍误差范围内
    # 例如稠密 93% → 误差 7% → 1.5倍误差=10.5% → 下限=82.5%
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    # 创建子图网格，按每层一个子图排列
    fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)), figsize=(15, 8))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:  # 只绘制卷积层和全连接层
            ax = axes[plot_index]
            # 绘制当前层的准确率-稀疏度曲线
            curve = ax.plot(sparsities, accuracies[plot_index])
            # 绘制灵敏下限参考线（低于此线表示该稀疏度不可接受）
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
            ax.set_ylim(80, 95)  # 固定 Y 轴范围便于层间对比
            ax.set_title(name)
            ax.set_xlabel("sparsity")
            ax.set_ylabel("top-1 accuracy")
            ax.legend(
                [
                    "accuracy after pruning",  # 剪枝后准确率曲线
                    f"{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy",  # 灵敏度下限
                ]
            )
            ax.grid(axis="x")
            plot_index += 1
    fig.suptitle("Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity")
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig("sensitivity_curves.png")
    plt.close(fig)


plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy)


# 问题 4
"""
4.1 剪枝稀疏度与模型准确率之间有什么关系？(即，当稀疏度变高时，准确率是上升还是下降？)
下降

4.2 所有层的灵敏度是否相同？
不相同

4.3 那一层对剪枝稀疏度最敏感？

剪枝对模型性能的影响具有明显的分阶段特性：在低稀疏度区间内，剪枝通常不会显著影响精度，甚至可能通过去除冗余参数提升泛化能力；随着稀疏度增加，模型开始进入信息损失阶段，准确率逐渐下降；当稀疏度超过某一临界点后，模型表达能力迅速崩溃。

同时，不同网络层对剪枝的敏感性存在显著差异，整体呈现“浅层鲁棒、深层敏感、分类头最敏感”的规律，这源于各层特征抽象程度、冗余度以及梯度传播路径的不同。
"""


# 每层的参数数量
# 除了准确率之外，每层的参数数量也影响稀疏度的选择，参数较多的层需要更大的稀疏度
def plot_num_parameters_distribution(model):
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis="y")
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title("#Parameter Distribution")
    plt.ylabel("Number of Parameters")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig("param_distribution.png")
    plt.close(fig)


plot_num_parameters_distribution(model)

# 基于灵敏度曲线和参数数量分布选择稀疏度
"""
问题 5：
基于灵敏度曲线和模型中参数数量的分布，为每层选择稀疏度，注意：剪枝模型的整体压缩比主要取决于参数
较多的层，且不同层对剪枝的灵敏度不同，确保剪枝后，系数模型的大小为稠密模型的 25%，并且在微调后验证准确率高于 92.5

提示：
- 参数较多的层应具有较大的稀疏度
- 对剪枝稀疏度敏感的层(即准确率随稀疏度增加而快速下降的层)，应具有较小的稀疏度
"""

recover_model()

sparsity_dict = {
    ##################### YOUR CODE STARTS HERE #####################
    # please modify the sparsity value of each layer
    # please DO NOT modify the key of sparsity_dict
    "backbone.conv0.weight": 0,
    "backbone.conv1.weight": 0,
    "backbone.conv2.weight": 0,
    "backbone.conv3.weight": 0,
    "backbone.conv4.weight": 0,
    "backbone.conv5.weight": 0,
    "backbone.conv6.weight": 0,
    "backbone.conv7.weight": 0,
    "classifier.weight": 0,
    ##################### YOUR CODE ENDS HERE #######################
}


# 根据定义的 sparsity_dict 对模型进行剪枝，并打印稀疏模型的信息
pruner = FineGrainedPruner(model, sparsity_dict)
print(f"After pruning with sparsity dictionary")
for name, sparsity in sparsity_dict.items():
    print(f"  {name}: {sparsity:.2f}")
print(f"The sparsity of each layer becomes")
for name, param in model.named_parameters():
    if name in sparsity_dict:
        print(f"  {name}: {get_sparsity(param):.2f}")

sparse_model_size = get_model_size(model, count_nonzero_only=True)
print(
    f"Sparse model has size={sparse_model_size / MiB:.2f} MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size"
)
sparse_model_accuracy = evaluate(model, dataloader["test"])
print(f"Sparse model has accuracy={sparse_model_accuracy:.2f}% before fintuning")

plot_weight_distribution(model, count_nonzero_only=True)

# 微调细粒度剪枝后的模型
"""
从上面可以看出，尽管细粒度剪枝减少了大部分模型权重，但模型的准确率也下降了
因此，我们必须对稀疏模型进行微调以恢复准确率
"""
num_finetune_epochs = 5
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
criterion = nn.CrossEntropyLoss()

best_sparse_model_checkpoint = dict()
best_accuracy = 0
print(f"Finetuning Fine-grained Pruned Sparse Model")
for epoch in range(num_finetune_epochs):
    # At the end of each train iteration, we have to apply the pruning mask
    #    to keep the model sparse during the training
    train(
        model,
        dataloader["train"],
        criterion,
        optimizer,
        scheduler,
        callbacks=[lambda: pruner.apply(model)],
    )
    accuracy = evaluate(model, dataloader["test"])
    is_best = accuracy > best_accuracy
    if is_best:
        best_sparse_model_checkpoint["state_dict"] = copy.deepcopy(model.state_dict())
        best_accuracy = accuracy
    print(
        f"    Epoch {epoch + 1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%"
    )


# 查看最佳微调后稀疏模型的信息
# load the best sparse model checkpoint to evaluate the final performance
model.load_state_dict(best_sparse_model_checkpoint["state_dict"])
sparse_model_size = get_model_size(model, count_nonzero_only=True)
print(
    f"Sparse model has size={sparse_model_size / MiB:.2f} MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size"
)
sparse_model_accuracy = evaluate(model, dataloader["test"])
print(f"Sparse model has accuracy={sparse_model_accuracy:.2f}% after fintuning")

# 通道剪枝
"""
通道剪枝移除整个通道，同样，我们移除权重幅度较小(通过 Frobenius 范数衡量)的通道

"""
# firstly, let's restore the model weights to the original dense version
#   and check the validation accuracy
recover_model()
dense_model_accuracy = evaluate(model, dataloader["test"])
print(f"dense model has accuracy={dense_model_accuracy:.2f}%")


# 移除通道权重
"""
与细粒度剪枝不同，在通道剪枝中，我们可以完全从张量中移除权重，也就是说，输出通道的数量变少了

通道剪枝后，权重张量 W 仍然是稠密的，可以将 sparsity称为 prune ratio(剪枝比率)

与细粒度剪枝类似，可以对不同层使用不同的剪枝率，目前是对所有层使用统一的剪枝率。
目标是减少 2 倍的计算量，这大约对应 30% 的统一剪枝率

可以将比率列表传递给 channel_prune 函数
"""


# 问题 6
# 朴素地剪枝除前 out_channels_new 个通道之外的所有输出通道
def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio 必须在 [0, 1) 范围内；当前值为 {prune_ratio}")

    ##################### YOUR CODE STARTS HERE #####################
    return max(1, int(round(channels * (1.0 - prune_ratio))))
    ##################### YOUR CODE ENDS HERE #####################


@torch.no_grad()
def channel_prune(model: nn.Module, prune_ratio: Union[List, float]) -> nn.Module:
    """对 backbone 中的每个卷积层进行通道剪枝(channel pruning)
    prune_ratio 可以是:
      - 一个浮点数:所有层使用统一的剪枝比例
      - 一个列表:每一层单独指定剪枝比例

    示例:backbone 有 3 个卷积层,权重形状为 [out, in, kH, kW]:
        conv0: [64, 3, 3, 3]   -> bn0: [64]
        conv1: [128, 64, 3, 3] -> bn1: [128]
        conv2: [256, 128, 3, 3] -> bn2: [256]
    因为有 3 个 conv,所以 prune_ratio 需要 3 - 1 = 2 个比例,设为 [0.5, 0.25]。
    第一次循环 (ratio=0.5):conv0 输出 64->32,bn0 64->32,conv1 输入 64->32
    第二次循环 (ratio=0.25):conv1 输出 128->96,bn1 128->96,conv2 输入 128->96
    最终:
        conv0: [32, 3, 3, 3]   conv1: [96, 32, 3, 3]   conv2: [256, 96, 3, 3]
    关键:前一层砍输出通道,下一层必须同步砍输入通道,否则维度不对齐会报错。
    """
    # 检查 prune_ratio 类型:必须是 float 或 list
    assert isinstance(prune_ratio, (float, list))
    # 统计 backbone 中卷积层的数量
    n_conv = len([m for m in model.backbone if isinstance(m, nn.Conv2d)])
    # 注意:每个剪枝比例同时影响“前一个 conv 的输出”和“后一个 conv 的输入”
    # 即结构为:conv0 - ratio0 - conv1 - ratio1 - ...
    # 所以比例的数量是 n_conv - 1(最后一个 conv 的输出不剪,通常连接分类头)
    if isinstance(prune_ratio, list):
        assert len(prune_ratio) == n_conv - 1
    else:  # 把单个 float 扩展成列表,方便统一处理
        prune_ratio = [prune_ratio] * (n_conv - 1)

    # 深拷贝模型,避免修改原始模型
    model = copy.deepcopy(model)  # prevent overwrite
    # 只对 backbone 的特征提取部分做剪枝
    all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # 这里采用最朴素的策略:直接保留前 k 个通道
    # 前提假设:每个 conv 后面都紧跟一个 bn,所以数量必须相等
    assert len(all_convs) == len(all_bns)
    for i_ratio, p_ratio in enumerate(prune_ratio):
        prev_conv = all_convs[i_ratio]  # 当前(前一个)卷积层
        prev_bn = all_bns[i_ratio]  # 当前卷积层对应的 bn
        next_conv = all_convs[i_ratio + 1]  # 下一个卷积层
        # 前一个 conv 的输出通道数 == 下一个 conv 的输入通道数
        original_channels = prev_conv.out_channels  # same as next_conv.in_channels
        # 根据剪枝比例计算需要保留的通道数量 k
        n_keep = get_num_channels_to_keep(original_channels, p_ratio)

        # 剪掉前一个 conv 的输出通道,只保留前 n_keep 个
        # conv 权重形状 [out, in, kH, kW],对第 0 维(输出通道)切片
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        # bn 的参数都按“通道”对齐,同样只保留前 n_keep 个
        prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])  # 缩放系数 gamma
        prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])  # 偏置 beta
        prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])  # 滑动均值
        prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])  # 滑动方差

        # 剪掉下一个 conv 的输入通道,使其与上面保留的 n_keep 个输出通道对齐
        # next_conv 权重形状 [out, in, kH, kW],对第 1 维(输入通道)切片
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep, :, :])
        ##################### YOUR CODE ENDS HERE #####################

    return model


dummy_input = torch.randn(1, 3, 32, 32).cuda()
pruned_model = channel_prune(model, prune_ratio=0.3)
pruned_macs = get_model_macs(pruned_model, dummy_input)
assert pruned_macs == 305388064
print("* Check passed. Right MACs for the pruned model.")

# 评估采用 30% 统一剪枝率进行通道剪枝后模型的性能
pruned_model_accuracy = evaluate(pruned_model, dataloader["test"])
print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")

# 按重要性对通道排序
"""
移除所有层中前 30% 的通道会导致准确显著下降。
解决该问题的一种潜在方法是找出较不重要的通道权重进行移除

一种流行的重要性准则是使用每个输入通道对应权重的 Frobenius 范数

importance_i = ||W_i||_2

可以将通道权重从更重要到不重要进行排序，然后为每层保留前 k 个通道
"""


# 完成以下基于 Frobenius 范数对权重张量进行排序的函数
# 计算张量的 Frobenius范数，使用 torch.norm
# function to sort the channels from important to non-important
def get_input_channel_importance(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(weight.shape[1]):
        channel_weight = weight.detach()[:, i_c]
        ##################### YOUR CODE STARTS HERE #####################
        importance = torch.norm(channel_weight)
        ##################### YOUR CODE ENDS HERE #####################
        importances.append(importance.view(1))
    return torch.cat(importances)


@torch.no_grad()
def apply_channel_sorting(model):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]
        # note that we always compute the importance according to input channels
        importance = get_input_channel_importance(next_conv.weight)
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)

        # apply to previous conv and its following bn
        prev_conv.weight.copy_(
            torch.index_select(prev_conv.weight.detach(), 0, sort_idx)
        )
        for tensor_name in ["weight", "bias", "running_mean", "running_var"]:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input (hint: one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.copy_(
            torch.index_select(next_conv.weight.detach(), 1, sort_idx)
        )
        ##################### YOUR CODE ENDS HERE #####################

    return model


# 验证是否正确
print("Before sorting...")
dense_model_accuracy = evaluate(model, dataloader["test"])
print(f"dense model has accuracy={dense_model_accuracy:.2f}%")

print("After sorting...")
sorted_model = apply_channel_sorting(model)
sorted_model_accuracy = evaluate(sorted_model, dataloader["test"])
print(f"sorted model has accuracy={sorted_model_accuracy:.2f}%")

# make sure accuracy does not change after sorting, since it is
# equivalent transform
assert abs(sorted_model_accuracy - dense_model_accuracy) < 0.1
print("* Check passed.")

# 比较有无排序的剪枝模型准确率
channel_pruning_ratio = 0.3  # pruned-out ratio

print(" * Without sorting...")
pruned_model = channel_prune(model, channel_pruning_ratio)
pruned_model_accuracy = evaluate(pruned_model, dataloader["test"])
print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")


print(" * With sorting...")
sorted_model = apply_channel_sorting(model)
pruned_model = channel_prune(sorted_model, channel_pruning_ratio)
pruned_model_accuracy = evaluate(pruned_model, dataloader["test"])
print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")


# 通道排序可以略微提升剪枝模型的准确率，但仍存在很大程度的下降，这在通道剪枝中相当常见
# 不过幸运的是，我们可以通过微调来恢复准确率
num_finetune_epochs = 5
optimizer = torch.optim.SGD(
    pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
criterion = nn.CrossEntropyLoss()

best_accuracy = 0
for epoch in range(num_finetune_epochs):
    train(pruned_model, dataloader["train"], criterion, optimizer, scheduler)
    accuracy = evaluate(pruned_model, dataloader["test"])
    is_best = accuracy > best_accuracy
    if is_best:
        best_accuracy = accuracy
    print(
        f"Epoch {epoch + 1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%"
    )


# 衡量剪枝带来的加速
"""
微调后，模型几乎恢复了准确率。

可以清晰的得到：与细粒度剪枝相比，通道剪枝通常更难恢复准确率。
然而，它直接带来了更小的模型大小和更少的计算量，无需专门的模型格式
"""


# 比较剪枝后的模型大小、计算量和延迟
# helper functions to measure latency of a regular PyTorch models.
#   Unlike fine-grained pruning, channel pruning
#   can directly leads to model size reduction and speed up.
@torch.no_grad()
def measure_latency(model, dummy_input, n_warmup=20, n_test=100):
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.time()
    return (t2 - t1) / n_test  # average latency


table_template = "{:<15} {:<15} {:<15} {:<15}"
print(table_template.format("", "Original", "Pruned", "Reduction Ratio"))

# 1. measure the latency of the original model and the pruned model on CPU
#   which simulates inference on an edge device
dummy_input = torch.randn(1, 3, 32, 32).to("cpu")
pruned_model = pruned_model.to("cpu")
model = model.to("cpu")

pruned_latency = measure_latency(pruned_model, dummy_input)
original_latency = measure_latency(model, dummy_input)
print(
    table_template.format(
        "Latency (ms)",
        round(original_latency * 1000, 1),
        round(pruned_latency * 1000, 1),
        round(original_latency / pruned_latency, 1),
    )
)

# 2. measure the computation (MACs)
original_macs = get_model_macs(model, dummy_input)
pruned_macs = get_model_macs(pruned_model, dummy_input)
print(
    table_template.format(
        "MACs (M)",
        round(original_macs / 1e6),
        round(pruned_macs / 1e6),
        round(original_macs / pruned_macs, 1),
    )
)

# 3. measure the model size (params)
original_param = get_num_parameters(model)
pruned_param = get_num_parameters(pruned_model)
print(
    table_template.format(
        "Param (M)",
        round(original_param / 1e6, 2),
        round(pruned_param / 1e6, 2),
        round(original_param / pruned_param, 1),
    )
)

# put model back to cuda
pruned_model = pruned_model.to("cuda")
model = model.to("cuda")


# 使用上面的代码块回答下面的问题
"""
8.1 解释为什么移除 30% 的通道大致能减少 50% 的计算量
8.2 解释为什么延迟减少比例略小于计算量较少比例
"""

"""
比较细粒度剪枝和通道剪枝
9.1 细粒度剪枝和通道剪枝各自的优缺点是什么？可以从压缩率、准确率、延迟、硬件支持(即，是否需要专门的硬件加速器)
等角度进行讨论

9.2 如果你想让模型在智能手机上运行得更快，你会使用那种剪枝方法？为什么？ 

"""
