import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs  # 用于统计模型的 MACs(乘加次数)
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

# 固定所有随机源的种子, 保证实验结果可复现
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 定义训练/测试集的数据预处理(数据增强)
transforms = {
    "train": Compose(
        [
            RandomCrop(
                32, padding=4
            ),  # 随机裁剪(先 padding 4 像素再裁回 32x32), 增加平移不变性
            RandomHorizontalFlip(),  # 随机水平翻转, 数据增强
            ToTensor(),  # PIL 图像 -> Tensor, 并把像素值归一化到 [0, 1]
        ]
    ),
    "test": ToTensor(),  # 测试集不做增强, 只转 Tensor
}

# 加载 CIFAR10 数据集(训练集 + 测试集)
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="/home/ghr/code/data/cifar10",
        train=(split == "train"),  # train=True 取训练集, False 取测试集
        download=False,  # 数据已在本地, 不重新下载
        transform=transforms[split],
    )

# 从测试集中每个类别收集 4 张样本, 用于后面可视化
samples = [[] for _ in range(10)]
for image, label in dataset["test"]:
    if len(samples[label]) < 4:
        samples[label].append(image)

# 可视化每个类别的样本(4 行 x 10 列, 共 40 张)
plt.figure(figsize=(20, 9))
for index in range(40):
    label = index % 10
    image = samples[label][index // 10]

    # 把维度从 CHW 转为 HWC, 以便 matplotlib 显示
    image = image.permute(1, 2, 0)

    # 把类别索引转换为类别名称
    label = dataset["test"].classes[label]

    # 绘制图像
    plt.subplot(4, 10, index + 1)
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")
plt.show()

# 构建数据加载器(DataLoader), 负责按 batch 取数据
dataflow = {}
for split in ["train", "test"]:
    dataflow[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == "train"),  # 训练集打乱, 测试集不打乱
        num_workers=0,  # 加载数据的子进程数(0 表示用主进程)
        pin_memory=True,  # 锁页内存, 加速 CPU->GPU 数据传输
    )

# 取一个 batch 看看数据的 dtype 和 shape
for inputs, targets in dataflow["train"]:
    print("[inputs] dtype: {}, shape: {}".format(inputs.dtype, inputs.shape))
    print("[targets] dtype: {}, shape: {}".format(targets.dtype, targets.shape))
    break


# 定义 VGG 网络(CIFAR10 版本)
class VGG(nn.Module):
    # 网络结构: 数字表示卷积输出通道数, "M" 表示一次最大池化
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)  # 为每种层维护一个自增计数, 用于命名(conv0, conv1...)

        def add(name: str, layer: nn.Module) -> None:
            # 把层加入列表, 名字带递增序号
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3  # 输入是 RGB 三通道
        for x in self.ARCH:
            if x != "M":
                # 卷积块: conv -> bn -> relu
                # bias=False 是因为后面接 BatchNorm, 偏置会被 BN 抵消
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # 最大池化, 特征图尺寸减半
                add("pool", nn.MaxPool2d(2))

        # backbone: 卷积特征提取部分
        self.backbone = nn.Sequential(OrderedDict(layers))
        # classifier: 最后的全连接分类头(512 -> 10 类)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # 全局平均池化: 对 H、W 维度求均值, [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])

        # 分类头: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x


# 实例化模型并搬到 GPU
model = VGG().cuda()

# 打印网络结构
print(model.backbone)

print(model.classifier)

# 统计可训练参数量
num_params = 0
for param in model.parameters():
    if param.requires_grad:
        num_params += param.numel()
print("#Params:", num_params)

# 统计模型的 MACs(用一个全零的伪输入做一次前向来测量)
num_macs = profile_macs(model, torch.zeros(1, 3, 32, 32).cuda())
print("#MACs:", num_macs)

# 损失函数: 交叉熵(多分类)
criterion = nn.CrossEntropyLoss()

# 优化器: 带动量的 SGD + 权重衰减(L2 正则)
optimizer = SGD(
    model.parameters(),
    lr=0.4,
    momentum=0.9,
    weight_decay=5e-4,
)

num_epochs = 20
steps_per_epoch = len(dataflow["train"])  # 每个 epoch 的迭代步数

# 定义分段线性的学习率调度: 前 30% 训练线性升到峰值, 之后线性降到 0(warmup + decay)
lr_lambda = lambda step: np.interp(
    [step / steps_per_epoch], [0, num_epochs * 0.3, num_epochs], [0, 1, 0]
)[0]

# 可视化学习率曲线
steps = np.arange(steps_per_epoch * num_epochs)
plt.plot(steps, [lr_lambda(step) * 0.4 for step in steps])
plt.xlabel("Number of Steps")
plt.ylabel("Learning Rate")
plt.grid("on")
plt.show()

# 用上面的 lambda 构建学习率调度器
scheduler = LambdaLR(optimizer, lr_lambda)


def train(
    model: nn.Module,
    dataflow: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
) -> None:
    model.train()  # 切换到训练模式(启用 dropout/BN 的训练行为)

    for inputs, targets in tqdm(dataflow, desc="train", leave=False):
        # 把数据从 CPU 搬到 GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # 清空上一步累积的梯度
        optimizer.zero_grad()

        # 前向推理 + 计算损失
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播, 计算梯度
        loss.backward()

        # 更新参数, 并推进学习率调度器
        optimizer.step()
        scheduler.step()


@torch.inference_mode()  # 推理模式: 关闭梯度, 比 no_grad 更省内存/更快
def evaluate(model: nn.Module, dataflow: DataLoader) -> float:
    model.eval()  # 切换到评估模式(BN 用全局统计量, 关闭 dropout)

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataflow, desc="eval", leave=False):
        # 把数据从 CPU 搬到 GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # 前向推理
        outputs = model(inputs)

        # 把 logits 转为预测类别(取最大值所在的索引)
        outputs = outputs.argmax(dim=1)

        # 累计样本数与预测正确数
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    # 返回准确率(百分比)
    return (num_correct / num_samples * 100).item()


# 训练主循环: 每个 epoch 先训练再在测试集上评估
for epoch_num in tqdm(range(1, num_epochs + 1)):
    train(model, dataflow["train"], criterion, optimizer, scheduler)
    metric = evaluate(model, dataflow["test"])
    print(f"epoch {epoch_num}:", metric)


# 可视化测试集前 40 张图的预测结果与真实标签
plt.figure(figsize=(20, 10))
for index in range(40):
    image, label = dataset["test"][index]

    # 模型推理
    model.eval()
    with torch.inference_mode():
        pred = model(image.unsqueeze(dim=0).cuda())  # 增加 batch 维度后推理
        pred = pred.argmax(dim=1)

    # 把维度从 CHW 转为 HWC, 以便显示
    image = image.permute(1, 2, 0)

    # 把类别索引转换为类别名称
    pred = dataset["test"].classes[pred]
    label = dataset["test"].classes[label]

    # 绘制图像, 标题显示预测值和真实值
    plt.subplot(4, 10, index + 1)
    plt.imshow(image)
    plt.title(f"pred: {pred}" + "\n" + f"label: {label}")
    plt.axis("off")
plt.show()
