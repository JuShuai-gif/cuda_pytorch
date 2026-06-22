"""
实验 3：神经架构搜索实验 (NAS) - 起始代码
学生需要完成所有标记为 TODO 的部分

本实验实现：
1. 定义 CNN 架构搜索空间
2. 随机搜索 (Random Search)
3. 进化搜索 (Evolutionary Search)
4. 精度预测器 (Accuracy Predictor)
5. Pareto 最优架构搜索
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import copy
import random
import numpy as np
from typing import List, Dict, Tuple
from collections import namedtuple


# ============ 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 架构描述结构体
Architecture = namedtuple(
    "Architecture", ["kernel_sizes", "channels", "num_layers", "use_skip"]
)


# ============ 数据加载 ============
def get_cifar10_dataloaders(batch_size: int = 128):
    """加载 CIFAR-10 数据集"""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


def get_subset_loader(loader, num_batches: int = 20):
    """从数据加载器中取子集用于快速评估"""
    # 简单返回原始 loader 的前 num_batches 批
    return loader


# ============ TODO 1: 定义搜索空间 ============
class CNNBuilder(nn.Module):
    """根据架构参数构建 CNN 模型"""

    def __init__(
        self,
        arch: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        image_size: int = 32,
    ):
        """
        参数:
            arch: 架构描述
            input_channels: 输入通道数
            num_classes: 分类类别数
            image_size: 输入图像大小
        """
        super().__init__()
        # TODO: 根据 arch 参数构建 CNN 网络
        # arch.kernel_sizes: 每层的卷积核大小列表，如 [3, 3, 5, 7]
        # arch.channels: 每层的输出通道数列表，如 [16, 32, 64, 128]
        # arch.num_layers: 卷积层数量
        # arch.use_skip: 是否使用跳跃连接（简单的残差连接）

        self.layers = nn.ModuleList()
        in_channels = input_channels
        current_size = image_size

        # TODO: 构建各层
        # 1. 对每一层创建 Conv2d + BatchNorm2d + ReLU
        # 2. 每隔若干层插入 MaxPool2d 进行下采样
        # 3. 如果 use_skip 为 True，在每两层之间添加残差连接
        # 4. 最后添加全局平均池化和全连接分类头

        # 占位：后续需要填写
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # TODO: 实现前向传播
        pass


# 搜索空间定义
SEARCH_SPACE = {
    "kernel_sizes": [3, 5, 7],
    "channels": [16, 32, 64, 128],
    "num_layers": [2, 3, 4, 5],
    "use_skip": [True, False],
}


def random_sample_architecture() -> Architecture:
    """
    从搜索空间中随机采样一个架构

    返回:
        Architecture 对象
    """
    # TODO: 从 SEARCH_SPACE 中随机选择参数
    # 注意：channels 需要选 num_layers 个，通常通道数随深度递增
    pass


# ============ 评估架构性能 ============
def count_macs(arch: Architecture, input_size: int = 32) -> int:
    """
    估算架构的总 MACs（乘加操作数）

    简化公式：对每层 Conv2d:
        MACs = in_c * out_c * k * k * h_out * w_out

    参数:
        arch: 架构描述
        input_size: 输入图像大小

    返回:
        total_macs: 总 MACs 估算值
    """
    # TODO: 实现 MACs 估算
    # 步骤：
    # 1. 初始化 in_channels = 3, h = w = input_size
    # 2. 对每一层计算 Conv2d 的 MACs
    # 3. 计算全连接层的 MACs
    # 4. 考虑 MaxPool 对空间尺寸的影响
    pass


def evaluate_architecture(
    arch: Architecture, train_loader, test_loader, epochs: int = 3
):
    """
    训练并评估一个架构

    参数:
        arch: 架构描述
        train_loader: 训练数据
        test_loader: 测试数据
        epochs: 训练轮数（为了速度，使用较少轮数）

    返回:
        accuracy: 测试精度
        macs: MACs 估算值
        params: 参数量估算值
    """
    model = CNNBuilder(arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = correct / total
        if acc > best_acc:
            best_acc = acc

    macs = count_macs(arch)
    params = sum(p.numel() for p in model.parameters())
    return best_acc, macs, params


# ============ TODO 2: 实现随机搜索 ============
def random_search(
    num_samples: int = 20, train_loader=None, test_loader=None
) -> List[Tuple[Architecture, float, int]]:
    """
    随机搜索 NAS

    随机采样 num_samples 个架构，训练并评估它们，
    返回所有评估结果

    参数:
        num_samples: 采样数量
        train_loader: 训练数据
        test_loader: 测试数据

    返回:
        results: [(arch, accuracy, macs), ...] 按精度降序排列
    """
    # TODO: 实现随机搜索
    # 1. 循环 num_samples 次
    # 2. 每次随机采样一个架构
    # 3. 训练并评估该架构
    # 4. 收集结果
    # 5. 按精度降序排序并返回
    pass


# ============ TODO 3: 实现进化搜索 ============
def mutate(arch: Architecture, mutation_prob: float = 0.3) -> Architecture:
    """
    对架构进行变异操作

    随机修改架构的一个参数（卷积核大小、通道数等）

    参数:
        arch: 原始架构
        mutation_prob: 每个属性的变异概率

    返回:
        mutated: 变异后的架构
    """
    # TODO: 实现变异操作
    # 1. 以 mutation_prob 的概率对每个属性进行变异
    # 2. 变异 kernel_sizes：随机替换某一层的卷积核大小
    # 3. 变异 channels：随机替换某一层的通道数
    # 4. 变异 use_skip：翻转布尔值
    # 5. 确保变异后的架构仍然有效
    pass


def crossover(parent1: Architecture, parent2: Architecture) -> Architecture:
    """
    对两个架构进行交叉操作

    从两个父代中各取一部分属性组合成子代

    参数:
        parent1: 父代 1
        parent2: 父代 2

    返回:
        child: 子代架构
    """
    # TODO: 实现交叉操作
    # 1. 随机选择一个交叉点
    # 2. 交叉点之前的属性从 parent1 取，之后的从 parent2 取
    # 3. 需要处理 num_layers 可能不同的情况
    pass


def evolutionary_search(
    population_size: int = 10,
    generations: int = 10,
    mutation_prob: float = 0.3,
    train_loader=None,
    test_loader=None,
):
    """
    进化搜索 NAS

    参数:
        population_size: 种群大小
        generations: 进化代数
        mutation_prob: 变异概率
        train_loader: 训练数据
        test_loader: 测试数据

    返回:
        pareto_frontier: Pareto 前沿上的架构列表
        history: 每代的最佳精度
    """
    # TODO: 实现进化搜索
    # 1. 初始化种群：随机生成 population_size 个架构
    # 2. 评估初始种群的适应度
    # 3. 循环 generations 代：
    #    a. 选择：根据适应度选择父代（tournament selection）
    #    b. 交叉：使用 crossover 生成子代
    #    c. 变异：使用 mutate 对子代进行变异
    #    d. 评估子代
    #    e. 替换：用子代替换种群中适应度最低的个体
    # 4. 从最终种群中选出 Pareto 前沿
    pass


# ============ TODO 4: 实现精度预测器 ============
class AccuracyPredictor(nn.Module):
    """
    精度预测器：一个简单的 MLP，根据架构特征预测精度

    这是 OFA (Once-for-All) 论文中精度预测器的简化版本
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        # TODO: 设计 MLP 结构
        # 输入：架构编码（如卷积核大小、通道数、深度等的 one-hot 或均值编码）
        # 输出：预测的精度 (0~1)
        self.net = nn.Sequential(
            # TODO: 添加全连接层
        )

    def forward(self, arch_encoding):
        # TODO: 前向传播
        pass


def encode_architecture(arch: Architecture) -> torch.Tensor:
    """
    将架构编码为固定长度的特征向量

    参数:
        arch: 架构描述

    返回:
        encoding: 特征向量 (input_dim,)
    """
    # TODO: 实现架构编码
    # 建议编码方式：
    # - 平均卷积核大小
    # - 最大/最小卷积核大小
    # - 平均通道数
    # - 最大/最小通道数
    # - 层数
    # - 是否使用跳跃连接 (0/1)
    pass


def train_accuracy_predictor(arch_results: List[Tuple], epochs: int = 100):
    """
    用已知的架构-精度对训练精度预测器

    参数:
        arch_results: [(arch, accuracy, macs), ...] 训练数据
        epochs: 训练轮数

    返回:
        predictor: 训练好的 AccuracyPredictor
    """
    # TODO: 实现预测器训练
    # 1. 准备训练数据：对每个架构进行编码，以精度作为标签
    # 2. 定义损失函数 (MSE) 和优化器
    # 3. 训练循环
    pass


# ============ TODO 5: 寻找 Pareto 最优架构 ============
def find_pareto_frontier(results: List[Tuple[Architecture, float, int]]):
    """
    在精度和 MACs 之间寻找 Pareto 前沿

    参数:
        results: [(arch, accuracy, macs), ...]

    返回:
        pareto_archs: Pareto 最优的架构列表
        pareto_points: [(accuracy, macs), ...] Pareto 前沿点
    """
    # TODO: 实现 Pareto 前沿提取
    # 对于两个目标 (精度 和 -MACs)：
    # 一个架构 A 支配 B，如果 A 的精度 >= B 的精度 且 A 的 MACs <= B 的 MACs
    # 且至少一个严格优于
    # Pareto 前沿 = 所有不被任何其他架构支配的架构
    pass


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 3：神经架构搜索实验")
    print("=" * 60)

    # 1. 加载数据
    print("\n[步骤 1] 加载 CIFAR-10 数据...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    # 2. 测试搜索空间
    print("\n[步骤 2] 测试搜索空间...")
    # TODO: 随机采样几个架构，打印信息，验证搜索空间

    # 3. 运行随机搜索
    print("\n[步骤 3] 运行随机搜索...")
    # TODO: 运行随机搜索并收集结果

    # 4. 运行进化搜索
    print("\n[步骤 4] 运行进化搜索...")
    # TODO: 运行进化搜索，记录每代的进展

    # 5. 训练精度预测器
    print("\n[步骤 5] 训练精度预测器...")
    # TODO: 用随机搜索的结果训练精度预测器

    # 6. 寻找 Pareto 前沿
    print("\n[步骤 6] 寻找 Pareto 前沿...")
    # TODO: 提取 Pareto 前沿并打印

    # 7. 比较随机搜索和进化搜索
    print("\n[步骤 7] 比较搜索策略...")
    # TODO: 对比两种搜索策略的效率

    # 8. 汇总报告
    print("\n[步骤 8] 汇总实验报告...")
    # TODO: 打印完整的对比结果

    print("\n实验完成！请将结果填入 report_template.md。")
