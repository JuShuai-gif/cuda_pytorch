"""
实验 1：剪枝实验 (Pruning) - 完整参考实现
包含幅度剪枝、敏感性扫描、微调循环和全面的性能评估

所有注释和文档均使用中文
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
import matplotlib.pyplot as plt

# ============ 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置中文字体（如果可用）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 数据加载 ============
def get_cifar10_dataloaders(batch_size: int = 128):
    """加载 CIFAR-10 数据集并返回训练和测试数据加载器"""
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
    """加载预训练的 VGG11 并适配 CIFAR-10 的 32x32 输入"""
    model = torchvision.models.vgg11(pretrained=False, num_classes=num_classes)
    # 适配 CIFAR-10：32x32 输入，保持 3x3 卷积核
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model = model.to(device)
    return model


# ============ 评估函数 ============
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader):
    """在测试集上评估模型精度，返回精度 (0~1)"""
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
    """测量模型的推理延迟，返回平均延迟（毫秒）"""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # 预热阶段：让 CUDA 内核完成初始化
    for _ in range(num_warmup):
        _ = model(dummy_input)

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
    """统计总参数量和非零参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    return total_params, nonzero_params


# ============ 幅度剪枝实现 ============
def magnitude_prune(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    对权重张量执行幅度剪枝

    算法流程：
    1. 计算权重的绝对值
    2. 找到第 k 小的绝对值作为阈值
    3. 将绝对值小于阈值的权重置零

    参数:
        weight: 权重张量
        sparsity: 目标稀疏度（要置零的比例，0.0 ~ 1.0）

    返回:
        pruned_weight: 剪枝后的权重张量
    """
    if sparsity <= 0:
        return weight
    if sparsity >= 1.0:
        return torch.zeros_like(weight)

    # 展平为一维并取绝对值
    flat_weight = weight.abs().view(-1)
    num_elements = flat_weight.numel()

    # 计算需要置零的元素数量
    k = int(sparsity * num_elements)
    if k == 0:
        return weight
    if k >= num_elements:
        return torch.zeros_like(weight)

    # 找到第 k 小的值作为阈值
    threshold = torch.kthvalue(flat_weight, k).values

    # 创建掩码：绝对值大于等于阈值的保留
    mask = (weight.abs() >= threshold).float()

    return weight * mask


def apply_pruning_to_model(model: nn.Module, sparsity: float):
    """
    对模型中所有可剪枝层（Conv2d 和 Linear）的权重应用幅度剪枝

    注意：只剪枝权重参数，不剪枝偏置
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = magnitude_prune(module.weight.data, sparsity)


# ============ 敏感性扫描实现 ============
def get_prunable_layers(model: nn.Module):
    """获取模型中所有可剪枝层的名称和模块"""
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable.append((name, module))
    return prunable


def sensitivity_scan(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    sparsity_levels: list = None,
):
    """
    对模型进行逐层敏感性扫描

    对每一层分别测试不同剪枝比例下的精度变化，
    从而分析各层对剪枝的敏感程度

    参数:
        model: 待分析的基础模型
        test_loader: 测试数据加载器
        sparsity_levels: 要测试的稀疏度列表

    返回:
        sensitivity: 字典 {layer_name: {sparsity: accuracy}}
    """
    if sparsity_levels is None:
        sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    # 测量基础精度
    base_accuracy = evaluate_accuracy(model, test_loader)
    print(f"  基础精度: {base_accuracy:.4f}")

    # 获取所有可剪枝层
    prunable_layers = get_prunable_layers(model)
    sensitivity = {}

    for layer_name, layer_module in prunable_layers:
        sensitivity[layer_name] = {}
        print(f"  扫描层: {layer_name}")

        # 保存原始权重
        original_weight = layer_module.weight.data.clone()

        for sp in sparsity_levels:
            # 仅对当前层应用剪枝
            layer_module.weight.data = magnitude_prune(original_weight.clone(), sp)
            acc = evaluate_accuracy(model, test_loader)
            sensitivity[layer_name][sp] = acc

        # 恢复原始权重
        layer_module.weight.data.copy_(original_weight)

    return sensitivity, base_accuracy


# ============ 微调循环实现 ============
def fine_tune(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    lr: float = 0.001,
    masks: dict = None,
):
    """
    对模型进行微调，可选择性保持剪枝掩码

    参数:
        model: 待微调的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 微调轮数
        lr: 学习率
        masks: 剪枝掩码字典，键为模块名，值为掩码张量

    返回:
        accuracy_history: 每轮微调后的精度列表
    """
    # 微调使用较小的学习率
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    accuracy_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 如果提供了掩码，在每步后重新应用掩码，保持剪枝状态
            if masks is not None:
                for name, module in model.named_modules():
                    if name in masks:
                        module.weight.data *= masks[name]

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = correct / total
        val_acc = evaluate_accuracy(model, test_loader)
        accuracy_history.append(val_acc)

        print(
            f"  Epoch {epoch + 1}/{epochs}: "
            f"Loss={running_loss / len(train_loader):.4f}, "
            f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
        )

    return accuracy_history


# ============ 绘图函数 ============
def plot_sensitivity_curves(sensitivity: dict, base_accuracy: float):
    """绘制敏感性扫描曲线"""
    plt.figure(figsize=(10, 6))

    # 只绘制有显著变化的层
    for layer_name, results in sensitivity.items():
        sparsities = sorted(results.keys())
        accuracies = [results[sp] for sp in sparsities]
        # 计算精度衰减量
        acc_drop = base_accuracy - min(accuracies)
        if acc_drop > 0.01:  # 只绘制精度下降超过 1% 的层
            plt.plot(
                [s * 100 for s in sparsities],
                accuracies,
                marker="o",
                label=layer_name[:30],
            )

    plt.axhline(y=base_accuracy, color="r", linestyle="--", label="基线精度")
    plt.xlabel("稀疏度 (%)")
    plt.ylabel("精度")
    plt.title("逐层敏感性扫描曲线")
    plt.legend(loc="lower left", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sensitivity_curves.png", dpi=150)
    print("敏感性曲线已保存为 sensitivity_curves.png")


def plot_sparsity_vs_accuracy(sparsity_list: list, acc_before: list, acc_after: list):
    """绘制稀疏度 vs 精度曲线"""
    plt.figure(figsize=(8, 5))
    sp_pct = [s * 100 for s in sparsity_list]

    plt.plot(sp_pct, acc_before, marker="s", label="剪枝后（未微调）", linewidth=2)
    plt.plot(sp_pct, acc_after, marker="o", label="剪枝后（微调后）", linewidth=2)

    plt.xlabel("稀疏度 (%)")
    plt.ylabel("精度")
    plt.title("剪枝稀疏度 vs 精度（微调前后对比）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sparsity_vs_accuracy.png", dpi=150)
    print("稀疏度-精度曲线已保存为 sparsity_vs_accuracy.png")


# ============ 训练基线模型 ============
def train_baseline(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
):
    """训练基线模型并返回训练好的模型"""
    print("\n训练基线模型...")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_acc = evaluate_accuracy(model, test_loader)
        print(
            f"  Epoch {epoch + 1}/{epochs}: Loss={running_loss / len(train_loader):.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  最佳精度: {best_acc:.4f}")

    return model


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 1：剪枝实验 - 完整实现")
    print("=" * 60)

    # 1. 加载数据
    print("\n[步骤 1] 加载 CIFAR-10 数据...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    # 2. 加载并训练模型
    print("\n[步骤 2] 加载并训练 VGG11 基线模型...")
    model = load_pretrained_vgg11(num_classes=10)
    model = train_baseline(model, train_loader, test_loader, epochs=5)

    # 3. 评估基线性能
    print("\n[步骤 3] 评估基线性能...")
    baseline_accuracy = evaluate_accuracy(model, test_loader)
    total_params, nonzero_params = count_parameters(model)
    baseline_latency = measure_latency(model)

    print(f"  基线精度:     {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)")
    print(f"  总参数量:     {total_params:,}")
    print(f"  非零参数量:   {nonzero_params:,}")
    print(f"  稀疏度:       {(1 - nonzero_params / total_params) * 100:.2f}%")
    print(f"  推理延迟:     {baseline_latency:.2f} ms")

    # 4. 敏感性扫描
    print("\n[步骤 4] 执行敏感性扫描...")
    model_copy = copy.deepcopy(model)
    sensitivity, _ = sensitivity_scan(model_copy, test_loader)
    plot_sensitivity_curves(sensitivity, baseline_accuracy)

    # 5. 不同稀疏度剪枝实验
    print("\n[步骤 5] 剪枝实验（不同稀疏度）...")
    sparsity_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    acc_before_ft = []
    acc_after_ft = []
    latency_list = []

    for sp in sparsity_list:
        print(f"\n  稀疏度 {sp * 100:.0f}%...")
        pruned_model = copy.deepcopy(model)

        # 应用剪枝
        apply_pruning_to_model(pruned_model, sp)

        # 测量剪枝后精度（未微调）
        acc_before = evaluate_accuracy(pruned_model, test_loader)
        acc_before_ft.append(acc_before)
        print(f"    剪枝后精度（未微调）: {acc_before:.4f}")

        # 微调
        masks = {}
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                masks[name] = (module.weight.data != 0).float()

        fine_tune(
            pruned_model, train_loader, test_loader, epochs=3, lr=0.001, masks=masks
        )

        # 测量微调后精度
        acc_after = evaluate_accuracy(pruned_model, test_loader)
        acc_after_ft.append(acc_after)
        print(f"    微调后精度: {acc_after:.4f}")

        # 测量延迟
        latency = measure_latency(pruned_model)
        latency_list.append(latency)
        print(f"    推理延迟: {latency:.2f} ms")

    # 6. 汇总报告
    print("\n" + "=" * 60)
    print("实验报告汇总")
    print("=" * 60)

    print(f"\n{'指标':<15} {'基线':<12}", end="")
    for sp in sparsity_list:
        print(f"{'稀疏度 ' + str(int(sp * 100)) + '%':<15}", end="")
    print()

    print(f"{'精度':<15} {baseline_accuracy:<12.4f}", end="")
    for i, sp in enumerate(sparsity_list):
        print(f"{acc_before_ft[i]:<15.4f}", end="")
    print()

    print(f"{'精度(微调后)':<15} {'-':<12}", end="")
    for i, sp in enumerate(sparsity_list):
        print(f"{acc_after_ft[i]:<15.4f}", end="")
    print()

    print(f"{'参数量':<15} {total_params:<12,}", end="")
    for sp in sparsity_list:
        nz = int(total_params * (1 - sp))
        print(f"{nz:<15,}", end="")
    print()

    print(f"{'延迟(ms)':<15} {baseline_latency:<12.2f}", end="")
    for lat in latency_list:
        print(f"{lat:<15.2f}", end="")
    print()

    # 7. 绘图
    print("\n[步骤 6] 绘制结果图表...")
    plot_sparsity_vs_accuracy(sparsity_list, acc_before_ft, acc_after_ft)

    print("\n实验完成！")
