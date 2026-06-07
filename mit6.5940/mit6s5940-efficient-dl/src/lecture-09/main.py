"""
CIFAR-10 上的知识蒸馏 (第 09 讲)

实现知识蒸馏（Knowledge Distillation, Hinton et al., 2015），
其中一个较大的教师 CNN 通过软化的 logits 将知识迁移到较小的学生 CNN。
学生使用组合损失进行训练:

    KD_Loss = alpha * CE(student_logits, targets)
            + (1 - alpha) * T^2 * KL(softmax(teacher_logits/T), softmax(student_logits/T))

我们比较以下情况:
    - 从零开始训练的学生（基线，无蒸馏）
    - 在温度 T = [1, 2, 4, 8, 16] 下使用 KD 训练的学生

所有训练仅使用 CPU 运行。
"""

import copy
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# 设备配置——仅使用 CPU
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# 可复现性设置
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
BATCH_SIZE = 256  # 批次大小
TEACHER_EPOCHS = 30  # 教师网络训练 epoch 数
STUDENT_EPOCHS = 30  # 学生网络从零训练 epoch 数
KD_EPOCHS = 30  # 知识蒸馏训练 epoch 数
LEARNING_RATE = 0.001  # 学习率
ALPHA = 0.5  # KD 中硬标签交叉熵损失的权重
TEMPERATURES = [1, 2, 4, 8, 16]  # 知识蒸馏的温度参数列表
NUM_WORKERS = 2  # DataLoader 工作线程数（CPU 友好）


# ===========================================================================
# 模型定义
# ===========================================================================


class TeacherCNN(nn.Module):
    """较大的教师网络：4 个卷积块 + 2 层全连接。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 卷积块 1: 3 -> 32 通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )
        # 卷积块 2: 32 -> 64 通道
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        # 卷积块 3: 64 -> 128 通道
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        # 卷积块 4: 128 -> 256 通道
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4 -> 2x2
        )
        # 分类头：展平 -> 全连接(256*2*2 -> 256) -> ReLU -> Dropout -> 全连接(256 -> num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()  # 初始化权重

    def _initialize_weights(self) -> None:
        """使用标准方法初始化网络权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用 Kaiming 正态初始化（针对 ReLU 优化）
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化：weight=1, bias=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 全连接层使用小方差正态初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：依次通过 4 个卷积块和分类头。"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


class StudentCNN(nn.Module):
    """较小的学生网络：3 个卷积块 + 1 层全连接。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 卷积块 1: 3 -> 16 通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )
        # 卷积块 2: 16 -> 32 通道
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        # 卷积块 3: 32 -> 64 通道
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        # 分类头：展平 -> 全连接(64*4*4 -> num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, num_classes),
        )

        self._initialize_weights()  # 初始化权重

    def _initialize_weights(self) -> None:
        """使用标准方法初始化网络权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用 Kaiming 正态初始化（针对 ReLU 优化）
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化：weight=1, bias=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 全连接层使用小方差正态初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：依次通过 3 个卷积块和分类头。"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# 数据加载
# ===========================================================================


def get_cifar10_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader]:
    """返回 CIFAR-10 的训练和测试 DataLoader。

    参数:
        batch_size: 批次大小。
        num_workers: DataLoader 工作线程数。

    返回:
        (train_loader, test_loader) 元组。
    """
    # 训练集数据增强：随机裁剪 + 随机水平翻转 + 归一化
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    # 测试集仅做归一化，不做数据增强
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    # 下载并加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    # 创建训练 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # CPU 上不需要 pin_memory
    )
    # 创建测试 DataLoader（不 shuffle）
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


# ===========================================================================
# 训练与评估工具函数
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """训练模型一个 epoch。返回平均损失。

    参数:
        model:     已放置在正确设备上的 PyTorch nn.Module。
        loader:    产生 (images, labels) 批次的 DataLoader。
        optimizer: 优化器实例。
        criterion: 损失函数。

    返回:
        该 epoch 的平均损失。
    """
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item() * images.size(0)  # 累加加权损失
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """评估模型在给定 DataLoader 上的准确率。

    参数:
        model:  已放置在正确设备上的 PyTorch nn.Module。
        loader: 产生 (images, labels) 批次的 DataLoader。

    返回:
        准确率百分比（0-100）。
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)  # 取预测得分最高的类别
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()  # 统计正确预测数
    return 100.0 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float = LEARNING_RATE,
) -> float:
    """使用标准交叉熵训练模型。返回最佳测试准确率。

    参数:
        model:        要训练的模型。
        train_loader: 训练 DataLoader。
        test_loader:  测试 DataLoader。
        epochs:       训练 epoch 数。
        lr:           学习率。

    返回:
        最佳测试准确率（百分比）。
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )  # 余弦退火学习率
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()  # 更新学习率
        acc = evaluate(model, test_loader)  # 每个 epoch 后评估
        if acc > best_acc:
            best_acc = acc  # 跟踪最佳准确率
    return best_acc


# ===========================================================================
# 知识蒸馏训练
# ===========================================================================


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float,
    alpha: float,
) -> torch.Tensor:
    """计算知识蒸馏损失。

    KD_loss = alpha * CE(student_logits, labels)
            + (1-alpha) * T^2 * KL(softmax(teacher_logits/T), softmax(student_logits/T))

    参数:
        student_logits: 学生网络的原始 logits 输出。
        teacher_logits: 教师网络的原始 logits 输出。
        labels:         真实标签。
        T:              温度参数（越高越软化概率分布）。
        alpha:          硬标签交叉熵损失的权重。

    返回:
        知识蒸馏的组合损失值。
    """
    # 硬标签交叉熵损失：标准分类损失
    ce_loss = F.cross_entropy(student_logits, labels)

    # 软化分布之间的 KL 散度。
    # PyTorch F.kl_div(input, target) 计算 KL(target || exp(input)):
    #   target * (log(target) - input)
    # 我们需要 KL(teacher_soft || student_soft)，所以:
    #   target = 教师概率（软化后的 softmax）
    #   input  = 学生对数概率（软化后的 log_softmax）
    teacher_prob = F.softmax(teacher_logits / T, dim=1)  # 教师软化分布
    student_log_prob = F.log_softmax(student_logits / T, dim=1)  # 学生对数软化分布
    kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")

    # 组合损失：CE 损失 + T^2 缩放的 KL 散度（T^2 用于补偿软化后梯度的缩放）
    return alpha * ce_loss + (1.0 - alpha) * (T**2) * kl_loss


def train_kd(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    T: float,
    alpha: float = ALPHA,
    lr: float = LEARNING_RATE,
) -> float:
    """通过知识蒸馏训练学生网络。返回最佳测试准确率。

    参数:
        teacher:      预训练的教师网络（在 KD 期间冻结）。
        student:      学生网络（将被训练）。
        train_loader: 训练 DataLoader。
        test_loader:  测试 DataLoader。
        epochs:       训练 epoch 数。
        T:            蒸馏温度。
        alpha:        CE 损失的权重。
        lr:           学习率。

    返回:
        最佳测试准确率（百分比）。
    """
    # 冻结教师网络：KD 期间教师参数不更新
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )  # 余弦退火学习率

    best_acc = 0.0
    for epoch_idx in range(epochs):
        student.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 教师前向传播：在 torch.no_grad() 下进行（不计算梯度）
            with torch.no_grad():
                teacher_logits = teacher(images)

            optimizer.zero_grad()
            student_logits = student(images)
            # 计算知识蒸馏损失
            loss = kd_loss_fn(student_logits, teacher_logits, labels, T, alpha)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新学生参数
            running_loss += loss.item() * images.size(0)

        scheduler.step()  # 更新学习率
        acc = evaluate(student, test_loader)  # 每个 epoch 后评估
        if acc > best_acc:
            best_acc = acc  # 跟踪最佳准确率

    return best_acc


# ===========================================================================
# 主函数
# ===========================================================================


def print_results_table(
    student_scratch_acc: float,
    kd_results: dict,
) -> None:
    """以整洁的表格格式打印结果。

    参数:
        student_scratch_acc: 从零开始训练的学生准确率。
        kd_results:          字典，键为温度 T，值为 KD 准确率。
    """
    print("\n" + "=" * 72)
    print("  KNOWLEDGE DISTILLATION RESULTS ON CIFAR-10")
    print("=" * 72)
    print(f"  {'Method':<30} {'Temperature':>12} {'Test Accuracy':>16}")
    print("  " + "-" * 60)
    # 打印从零开始训练的学生基线
    print(
        f"  {'Student (no KD, from scratch)':<30} {'---':>12} {student_scratch_acc:>15.2f}%"
    )
    # 按温度从小到大打印各个 KD 结果
    for T in sorted(kd_results.keys()):
        print(f"  {'Student (KD)':<30} {f'T={T}':>12} {kd_results[T]:>15.2f}%")
    print("=" * 72)
    print()


def main() -> None:
    """运行完整的知识蒸馏实验流程。"""
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders()

    # -- 步骤 1: 训练教师网络 ---------------------------------------------------
    print(f"\nTraining TeacherCNN ({TEACHER_EPOCHS} epochs)...")
    teacher = TeacherCNN(num_classes=10).to(DEVICE)
    t_start = time.time()
    teacher_acc = train_model(teacher, train_loader, test_loader, epochs=TEACHER_EPOCHS)
    t_elapsed = time.time() - t_start
    print(f"Teacher test accuracy: {teacher_acc:.2f}%  (took {t_elapsed:.1f}s)")

    # -- 步骤 2: 从零训练学生网络（基线，无 KD）----------------------------------
    print(f"\nTraining StudentCNN from scratch ({STUDENT_EPOCHS} epochs)...")
    student_scratch = StudentCNN(num_classes=10).to(DEVICE)
    s_start = time.time()
    scratch_acc = train_model(
        student_scratch,
        train_loader,
        test_loader,
        epochs=STUDENT_EPOCHS,
    )
    s_elapsed = time.time() - s_start
    print(f"Student (no KD) test accuracy: {scratch_acc:.2f}%  (took {s_elapsed:.1f}s)")

    # -- 步骤 3: 在不同温度下使用 KD 训练学生网络 -------------------------------
    kd_accuracies: dict = {}
    for T in TEMPERATURES:
        print(f"\nTraining StudentCNN with KD at T={T} ({KD_EPOCHS} epochs)...")
        # 每个温度使用一个新的学生网络，保证公平比较
        student_kd = StudentCNN(num_classes=10).to(DEVICE)
        kd_start = time.time()
        acc = train_kd(
            teacher,
            student_kd,
            train_loader,
            test_loader,
            epochs=KD_EPOCHS,
            T=T,
        )
        kd_elapsed = time.time() - kd_start
        kd_accuracies[T] = acc
        print(
            f"Student (KD, T={T}) test accuracy: {acc:.2f}%  (took {kd_elapsed:.1f}s)"
        )

    # -- 步骤 4: 打印最终结果表格 -----------------------------------------------
    print_results_table(scratch_acc, kd_accuracies)


if __name__ == "__main__":
    main()
