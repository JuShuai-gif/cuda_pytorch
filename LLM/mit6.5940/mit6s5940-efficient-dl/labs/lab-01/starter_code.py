"""
实验 1：剪枝实验 (Pruning) - 起始代码
学生需要完成所有标记为 TODO 的部分

本实验实现：
1. 加载预训练 VGG11 并在 CIFAR-10 上评估
2. 幅度剪枝（Magnitude Pruning）
3. 敏感性扫描（Sensitivity Scan）
4. 剪枝后微调（Fine-tuning）
5. 测量精度、参数量、稀疏度和延迟
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

# ============ 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============ 数据加载 ============
def get_cifar10_dataloaders(batch_size: int = 128):
    """
    加载 CIFAR-10 数据集并返回训练和测试数据加载器

    参数:
        batch_size: 批次大小

    返回:
        train_loader, test_loader
    """
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


# ============ 模型加载 ============
def load_pretrained_vgg11(num_classes: int = 10):
    """
    加载预训练的 VGG11 并适配 CIFAR-10

    返回:
        model: 适配后的 VGG11 模型
    """
    model = torchvision.models.vgg11(pretrained=False, num_classes=num_classes)
    # 适配 CIFAR-10 的 32x32 输入
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    # 移除自适应池化，替换为适合 32x32 的池化
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model = model.to(device)
    return model


# ============ 评估函数 ============
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader):
    """
    在测试集上评估模型精度

    参数:
        model: 待评估的模型
        test_loader: 测试数据加载器

    返回:
        accuracy: 精度 (0~1)
    """
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


def measure_latency(
    model: nn.Module, input_shape=(1, 3, 32, 32), num_warmup=10, num_runs=50
):
    """
    测量模型的推理延迟

    参数:
        model: 待测量的模型
        input_shape: 输入张量的形状
        num_warmup: 预热次数
        num_runs: 测量次数

    返回:
        avg_latency_ms: 平均延迟（毫秒）
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # 预热
    for _ in range(num_warmup):
        _ = model(dummy_input)

    # 同步 GPU（如果使用 CUDA）
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    avg_latency_ms = (end_time - start_time) / num_runs * 1000
    return avg_latency_ms


def count_parameters(model: nn.Module):
    """
    统计模型的总参数量和非零参数量

    参数:
        model: 待统计的模型

    返回:
        total_params: 总参数量
        nonzero_params: 非零参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    return total_params, nonzero_params


# ============ TODO 1: 实现幅度剪枝 ============
def magnitude_prune(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    对权重张量执行幅度剪枝，将最小（按绝对值）的权重置零

    参数:
        weight: 权重张量
        sparsity: 目标稀疏度 (0.0 ~ 1.0)，即要置零的权重比例

    返回:
        pruned_weight: 剪枝后的权重张量
    """
    # TODO: 实现幅度剪枝
    # 步骤：
    # 1. 将权重展平为一维
    # 2. 计算需要置零的元素数量 k = int(sparsity * num_elements)
    # 3. 找到第 k 小的绝对值（阈值）
    # 4. 创建掩码：绝对值小于阈值的元素置为 0，其余保持为 1
    # 5. 返回 weight * mask
    pass


def apply_pruning_to_model(model: nn.Module, sparsity: float):
    """
    对整个模型的所有卷积层和全连接层应用幅度剪枝

    参数:
        model: 待剪枝的模型
        sparsity: 目标稀疏度

    注意：只剪枝权重，不剪枝偏置
    """
    # TODO: 遍历 model.named_modules()
    # 对于 nn.Conv2d 和 nn.Linear 层，对其 weight 参数调用 magnitude_prune
    pass


# ============ TODO 2: 实现敏感性扫描 ============
def sensitivity_scan(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    sparsity_levels: list = None,
):
    """
    对模型进行逐层敏感性扫描

    对每一层分别尝试不同的剪枝比例，并测量剪枝后的精度，
    从而确定每层对剪枝的敏感程度

    参数:
        model: 基础模型
        test_loader: 测试数据加载器
        sparsity_levels: 要测试的稀疏度列表

    返回:
        sensitivity: 字典 {layer_name: {sparsity: accuracy}}
    """
    if sparsity_levels is None:
        sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    # TODO: 实现敏感性扫描
    # 1. 先获取基础精度
    # 2. 遍历所有可剪枝的层（Conv2d 和 Linear）
    # 3. 对每一层，独立地应用不同的剪枝比例
    # 4. 测量每次剪枝后的精度
    # 5. 重要：每次测试后将权重恢复到原始状态
    # 6. 将结果存入 sensitivity 字典并返回
    pass


# ============ TODO 3: 实现微调循环 ============
def fine_tune(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    lr: float = 0.001,
):
    """
    对剪枝后的模型进行微调以恢复精度

    参数:
        model: 剪枝后的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 微调轮数
        lr: 学习率

    返回:
        accuracy_history: 每轮微调后的精度列表
    """
    # TODO: 实现微调循环
    # 1. 定义优化器（推荐 SGD 或 Adam）
    # 2. 定义损失函数（交叉熵）
    # 3. 实现训练循环
    # 4. 每轮训练后评估精度
    # 5. 返回精度历史记录
    pass


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 1：剪枝实验")
    print("=" * 60)

    # 1. 加载数据
    print("\n[步骤 1] 加载 CIFAR-10 数据...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    # 2. 加载模型
    print("\n[步骤 2] 加载预训练 VGG11...")
    model = load_pretrained_vgg11(num_classes=10)

    # 3. 训练模型以获得基线（快速训练）
    print("\n[步骤 3] 训练基线模型...")
    # TODO: 在这里训练几个 epoch 获得一个较好的基线
    # 提示：可以使用上面定义的 fine_tune 函数，先训练基础模型

    # 4. 评估基线性能
    print("\n[步骤 4] 评估基线性能...")
    # TODO: 测量并打印基线精度、参数量、延迟
    baseline_accuracy = None  # 替换为实际评估
    total_params, nonzero_params = None, None  # 替换为实际统计
    baseline_latency = None  # 替换为实际测量

    # 5. 执行敏感性扫描
    print("\n[步骤 5] 执行敏感性扫描...")
    # TODO: 调用 sensitivity_scan 并打印结果

    # 6. 对不同稀疏度进行剪枝实验
    print("\n[步骤 6] 对不同稀疏度进行剪枝...")
    sparsity_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    # TODO: 对每个稀疏度执行剪枝并记录精度

    # 7. 微调剪枝后的模型
    print("\n[步骤 7] 微调剪枝后的模型...")
    # TODO: 对剪枝后的模型进行微调，比较微调前后的精度

    # 8. 汇总报告
    print("\n[步骤 8] 汇总实验报告...")
    # TODO: 打印完整的对比表格（精度、参数量、稀疏度、延迟）

    print("\n实验完成！请将结果填入 report_template.md。")
