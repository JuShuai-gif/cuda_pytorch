"""
实验 2：量化实验 (Quantization) - 起始代码
学生需要完成所有标记为 TODO 的部分

本实验实现：
1. 线性量化 (Linear Quantization)：int8/int4
2. K-means 量化
3. 激活值校准 (Activation Calibration)
4. 量化推理模块
5. 不同位宽的精度对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import copy
import numpy as np
from typing import Tuple


# ============ 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


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


# ============ 简单 CNN 模型 ============
class SimpleCNN(nn.Module):
    """用于量化实验的简单 CNN 模型"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============ 评估函数 ============
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader):
    """在测试集上评估精度"""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def train_model(model: nn.Module, train_loader, test_loader, epochs: int = 5):
    """训练模型的通用函数"""
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

        acc = evaluate_accuracy(model, test_loader)
        if acc > best_acc:
            best_acc = acc
        print(f"  Epoch {epoch + 1}/{epochs}: Accuracy = {acc:.4f}")

    return model


# ============ TODO 1: 实现线性量化 ============
def linear_quantize(
    tensor: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, float, float]:
    """
    对张量进行线性量化

    参数:
        tensor: 待量化的浮点张量
        bits: 量化位宽（如 8 表示 int8）

    返回:
        q_tensor: 量化后的整数张量（torch.int32 类型）
        scale: 量化缩放因子
        zero_point: 量化零点
    """
    # TODO: 实现线性量化
    # 步骤：
    # 1. 计算量化范围：qmin = -2^(bits-1), qmax = 2^(bits-1) - 1（对称量化）
    # 2. 找到张量的最小值和最大值
    # 3. 计算 scale = (max_val - min_val) / (qmax - qmin)
    # 4. 计算 zero_point（对称量化下为 0）
    # 5. 量化：q = clamp(round(x / scale + zero_point), qmin, qmax)
    # 6. 返回 (q, scale, zero_point)

    # 处理 bits 边界情况
    if bits >= 32:
        return tensor.to(torch.int32), 1.0, 0.0

    pass


def linear_dequantize(
    q_tensor: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    """
    将量化的整数张量反量化为浮点张量

    参数:
        q_tensor: 量化后的整数张量
        scale: 缩放因子
        zero_point: 零点

    返回:
        反量化后的浮点张量
    """
    # TODO: 实现反量化
    # 公式: x_deq = (q - zero_point) * scale
    pass


# ============ TODO 2: 实现 K-means 量化 ============
def kmeans_quantize(
    tensor: torch.Tensor, bits: int, num_iters: int = 10
) -> torch.Tensor:
    """
    使用 K-means 聚类对张量进行非均匀量化

    参数:
        tensor: 待量化的浮点张量
        bits: 量化位宽
        num_iters: K-means 迭代次数

    返回:
        quantized_tensor: 量化后的浮点张量（用聚类中心替代原值）
        centroids: 聚类中心
    """
    # TODO: 实现 K-means 量化
    # 步骤：
    # 1. 将张量展平为一维
    # 2. 确定聚类数量 k = 2^bits
    # 3. 初始化聚类中心（均匀采样或 kmeans++ 初始化）
    # 4. 迭代：
    #    a. 分配：将每个元素分配到最近的聚类中心
    #    b. 更新：计算每个聚类的均值作为新的中心
    # 5. 用最近的聚类中心替换每个元素
    # 6. 恢复原始形状并返回

    flat = tensor.flatten()
    k = 2**bits
    pass


# ============ TODO 3: 实现激活值校准 ============
def calibrate_activation_ranges(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    num_batches: int = 10,
) -> dict:
    """
    通过在校准数据集上运行模型来收集各层的激活值范围

    参数:
        model: 待校准的 FP32 模型
        calibration_loader: 校准数据加载器
        num_batches: 用于校准的批次数

    返回:
        activation_ranges: 字典 {layer_name: (min_val, max_val)}
    """
    # TODO: 实现激活值校准
    # 步骤：
    # 1. 为模型注册前向钩子，收集每层输出的最小值和最大值
    # 2. 对 num_batches 个批次进行前向传播
    # 3. 对每层，使用滑动平均或直接记录 min/max
    # 4. 返回各层的激活值范围
    pass


# ============ TODO 4: 实现量化推理模块 ============
class QuantizedConv2d(nn.Module):
    """模拟量化卷积层：将 FP32 权重和激活值量化为低精度整数进行推理"""

    def __init__(self, conv_layer: nn.Conv2d, weight_bits: int = 8, act_bits: int = 8):
        super().__init__()
        # TODO: 实现量化卷积层
        # 1. 保存原始卷积层
        # 2. 量化权重
        # 3. 在前向传播中量化输入、执行卷积、反量化输出
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.conv = conv_layer

    def forward(self, x):
        # TODO:
        # 1. 量化输入 x
        # 2. 量化卷积权重
        # 3. 用 FP32 卷积模拟（实际硬件上会用整数卷积）
        # 4. 返回结果
        pass


# ============ TODO 5: 比较不同位宽的精度 ============
def compare_bitwidths(
    model: nn.Module, test_loader, bitwidths: list = [2, 4, 8, 16, 32]
):
    """
    比较不同位宽下的模型精度

    参数:
        model: FP32 基线模型
        test_loader: 测试数据加载器
        bitwidths: 要比较的位宽列表

    返回:
        results: 字典 {bitwidth: accuracy}
    """
    # TODO: 实现位宽比较
    # 1. 对每个位宽，量化模型的所有层
    # 2. 评估量化后的精度
    # 3. 计算量化误差
    pass


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 2：量化实验")
    print("=" * 60)

    # 1. 加载数据
    print("\n[步骤 1] 加载 CIFAR-10 数据...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    # 2. 训练 FP32 基线模型
    print("\n[步骤 2] 训练 FP32 基线模型...")
    model = SimpleCNN(num_classes=10).to(device)
    model = train_model(model, train_loader, test_loader, epochs=5)

    # 3. 评估基线精度
    print("\n[步骤 3] 评估基线精度...")
    # TODO: 评估并打印基线精度

    # 4. 测试线性量化
    print("\n[步骤 4] 测试线性量化...")
    # TODO: 使用线性量化对模型权重进行 int8 量化，评估精度
    # TODO: 对不同位宽 (2, 4, 8, 16) 进行量化并比较

    # 5. 测试 K-means 量化
    print("\n[步骤 5] 测试 K-means 量化...")
    # TODO: 使用 K-means 量化对模型权重进行量化，比较与线性量化的差异

    # 6. 校准激活值
    print("\n[步骤 6] 校准激活值范围...")
    # TODO: 运行校准程序，收集各层的激活值范围

    # 7. 构建量化推理模块
    print("\n[步骤 7] 构建量化推理模块...")
    # TODO: 使用 QuantizedConv2d 替换模型的卷积层，评估精度

    # 8. 汇总报告
    print("\n[步骤 8] 汇总实验报告...")
    # TODO: 打印精度-位宽对比表

    print("\n实验完成！请将结果填入 report_template.md。")
