"""
实验 2：量化实验 (Quantization) - 完整参考实现
包含线性量化、K-means 量化、激活值校准、量化推理模块和全面对比

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
from typing import Tuple, Dict
import matplotlib.pyplot as plt

# ============ 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


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
    """用于量化实验的轻量 CNN 模型"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============ 评估和训练函数 ============
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, test_loader):
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


def train_model(model, train_loader, test_loader, epochs=5):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

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

        acc = evaluate_accuracy(model, test_loader)
        if acc > best_acc:
            best_acc = acc
        print(
            f"  Epoch {epoch + 1}/{epochs}: Loss={running_loss / len(train_loader):.4f}, Acc={acc:.4f}"
        )

    return model


# ============ 线性量化实现 ============
def linear_quantize(
    tensor: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, float, float]:
    """
    对称线性量化：将 FP32 张量量化为指定比特的整数

    使用对称量化方案（zero_point = 0），适用于权重量化

    参数:
        tensor: 待量化的浮点张量
        bits: 量化位宽

    返回:
        q_tensor: 量化后的整数张量 (int32)
        scale: 缩放因子
        zero_point: 零点（对称量化为 0）
    """
    if bits >= 32:
        return tensor.to(torch.int32), 1.0, 0.0

    # 计算量化范围：对称量化
    qmin = -(1 << (bits - 1))  # 例如 int8: -128
    qmax = (1 << (bits - 1)) - 1  # 例如 int8: 127

    # 计算 scale：使用绝对值的最大值
    max_abs = tensor.abs().max().item()
    if max_abs < 1e-8:
        scale = 1.0
    else:
        scale = max_abs / qmax

    # 量化
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int32)
    return q_tensor, scale, 0.0


def linear_dequantize(
    q_tensor: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    """
    反量化：将整数张量恢复为浮点近似值

    公式: x_approx = (q - zero_point) * scale
    """
    return (q_tensor.float() - zero_point) * scale


def linear_quantize_asymmetric(
    tensor: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, float, int]:
    """
    非对称线性量化：使用 min/max 范围，包含零点

    适用于激活值量化，因为激活值（如 ReLU 输出）是非负的
    """
    if bits >= 32:
        return tensor.to(torch.int32), 1.0, 0

    qmin = 0
    qmax = (1 << bits) - 1

    min_val = tensor.min().item()
    max_val = tensor.max().item()

    range_val = max_val - min_val
    if range_val < 1e-8:
        scale = 1.0
        zero_point = 0
    else:
        scale = range_val / (qmax - qmin)
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))

    q_tensor = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax).to(
        torch.int32
    )
    return q_tensor, scale, zero_point


# ============ K-means 量化实现 ============
def kmeans_quantize(
    tensor: torch.Tensor, bits: int, num_iters: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 K-means 聚类对张量进行非均匀量化

    与均匀量化不同，K-means 量化能找到最优的聚类中心，
    在低位宽下通常比均匀量化表现更好

    参数:
        tensor: 待量化的浮点张量
        bits: 量化位宽
        num_iters: 最大迭代次数

    返回:
        quantized_tensor: 量化后的张量
        centroids: 聚类中心
    """
    original_shape = tensor.shape
    flat = tensor.flatten()
    k = 2**bits
    n = flat.numel()

    if k >= n:
        return tensor.clone(), flat.clone()

    # 初始化：均匀采样 k 个初始聚类中心
    sorted_flat, _ = flat.sort()
    indices = torch.linspace(0, n - 1, k).long()
    centroids = sorted_flat[indices].clone()

    for it in range(num_iters):
        # 分配：计算每个点到各聚类中心的距离
        distances = torch.abs(flat.unsqueeze(1) - centroids.unsqueeze(0))
        assignments = distances.argmin(dim=1)

        # 更新：计算每个聚类的新中心
        new_centroids = torch.zeros_like(centroids)
        empty_count = 0
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = flat[mask].mean()
            else:
                # 空聚类：保留旧中心
                new_centroids[i] = centroids[i]
                empty_count += 1

        # 检查收敛
        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if shift < 1e-6:
            break

    # 用量化值替换原值
    distances = torch.abs(flat.unsqueeze(1) - centroids.unsqueeze(0))
    assignments = distances.argmin(dim=1)
    quantized_flat = centroids[assignments]
    quantized = quantized_flat.reshape(original_shape)

    return quantized, centroids


# ============ 激活值校准实现 ============
def calibrate_activation_ranges(
    model: nn.Module, calib_loader: torch.utils.data.DataLoader, num_batches: int = 20
) -> Dict[str, Tuple[float, float]]:
    """
    通过在校准数据上运行模型来收集各层的激活值范围

    利用前向钩子记录每层输出的最小值和最大值

    参数:
        model: FP32 模型
        calib_loader: 校准数据加载器
        num_batches: 使用的校准批次数

    返回:
        activation_ranges: {layer_name: (min_val, max_val)}
    """
    activation_stats = {}

    def hook_fn(name):
        def fn(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {"min": float("inf"), "max": float("-inf")}
            activation_stats[name]["min"] = min(
                activation_stats[name]["min"], output.min().item()
            )
            activation_stats[name]["max"] = max(
                activation_stats[name]["max"], output.max().item()
            )

        return fn

    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 运行校准
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calib_loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    # 移除钩子
    for h in hooks:
        h.remove()

    # 整理结果
    ranges = {}
    for name, stats in activation_stats.items():
        ranges[name] = (stats["min"], stats["max"])

    return ranges


# ============ 量化推理模块 ============
class QuantizedConv2d(nn.Module):
    """
    模拟量化卷积层

    在实际硬件（如 TensorRT）上，这会被替换为真正的整数卷积；
    此处我们使用浮点模拟来验证精度影响
    """

    def __init__(self, conv_layer: nn.Conv2d, weight_bits: int = 8, act_bits: int = 8):
        super().__init__()
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.conv = conv_layer
        # 预量化权重
        self.register_buffer("q_weight", None)
        self._w_scale = None
        self._quantize_weights()

    def _quantize_weights(self):
        """量化卷积权重"""
        with torch.no_grad():
            q_w, self._w_scale, _ = linear_quantize(
                self.conv.weight.data, self.weight_bits
            )
            self.q_weight = q_w
            self.deq_weight = linear_dequantize(q_w, self._w_scale, 0)

    def forward(self, x):
        # 量化激活值
        q_x, x_scale, _ = linear_quantize(x, self.act_bits)
        deq_x = linear_dequantize(q_x, x_scale, 0)

        # 用反量化的权重和激活进行卷积
        # 实际硬件上这里会用整数卷积：out = conv_i8(q_x, q_weight)
        # 然后反量化：out_fp = out * (x_scale * w_scale)
        with torch.no_grad():
            output = F.conv2d(
                deq_x,
                self.deq_weight,
                self.conv.bias,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
            )
        return output


class QuantizedLinear(nn.Module):
    """模拟量化全连接层"""

    def __init__(
        self, linear_layer: nn.Linear, weight_bits: int = 8, act_bits: int = 8
    ):
        super().__init__()
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.linear = linear_layer
        self._quantize_weights()

    def _quantize_weights(self):
        with torch.no_grad():
            q_w, self._w_scale, _ = linear_quantize(
                self.linear.weight.data, self.weight_bits
            )
            self.deq_weight = linear_dequantize(q_w, self._w_scale, 0)

    def forward(self, x):
        with torch.no_grad():
            q_x, x_scale, _ = linear_quantize(x, self.act_bits)
            deq_x = linear_dequantize(q_x, x_scale, 0)
            output = F.linear(deq_x, self.deq_weight, self.linear.bias)
        return output


def replace_with_quantized(
    model: nn.Module, weight_bits: int = 8, act_bits: int = 8
) -> nn.Module:
    """
    将模型中的 Conv2d 和 Linear 层替换为量化版本

    参数:
        model: 原始 FP32 模型
        weight_bits: 权重量化位宽
        act_bits: 激活量化位宽

    返回:
        量化后的模型
    """
    quantized_model = copy.deepcopy(model)

    def _replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d) and not isinstance(child, QuantizedConv2d):
                setattr(module, name, QuantizedConv2d(child, weight_bits, act_bits))
            elif isinstance(child, nn.Linear) and not isinstance(
                child, QuantizedLinear
            ):
                setattr(module, name, QuantizedLinear(child, weight_bits, act_bits))
            else:
                _replace(child)

    _replace(quantized_model)
    return quantized_model


# ============ 模型级量化函数 ============
def quantize_model_weights(model: nn.Module, bits: int, method: str = "linear"):
    """
    对模型的所有可量化层进行权重量化

    参数:
        model: 原始模型（会被原地修改）
        bits: 量化位宽
        method: "linear" 或 "kmeans"
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if method == "linear":
                q_w, scale, zp = linear_quantize(module.weight.data, bits)
                module.weight.data = linear_dequantize(q_w, scale, zp)
            elif method == "kmeans":
                q_w, _ = kmeans_quantize(module.weight.data, bits)
                module.weight.data = q_w


def compute_quantization_error(
    original_model: nn.Module, quantized_model: nn.Module
) -> Dict[str, float]:
    """计算各层的量化误差 (MSE)"""
    errors = {}
    orig_modules = dict(original_model.named_modules())
    for name, module in quantized_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            orig_weight = orig_modules[name].weight.data
            q_weight = module.weight.data
            mse = F.mse_loss(orig_weight, q_weight).item()
            errors[name] = mse
    return errors


# ============ 比较不同位宽的精度 ============
def compare_bitwidths(
    model: nn.Module, test_loader, bitwidths: list = None, methods: list = None
) -> dict:
    """
    比较不同位宽和量化方法下的模型精度

    参数:
        model: FP32 基线模型
        test_loader: 测试数据加载器
        bitwidths: 要比较的位宽列表
        methods: 要比较的量化方法

    返回:
        results: {method: {bitwidth: accuracy}}
    """
    if bitwidths is None:
        bitwidths = [2, 4, 8, 16, 32]
    if methods is None:
        methods = ["linear", "kmeans"]

    results = {m: {} for m in methods}

    for method in methods:
        for bw in bitwidths:
            if bw >= 32:
                acc = evaluate_accuracy(model, test_loader)
            else:
                q_model = copy.deepcopy(model)
                quantize_model_weights(q_model, bw, method=method)
                acc = evaluate_accuracy(q_model, test_loader)
            results[method][bw] = acc
            print(f"  [{method}] {bw}-bit: Accuracy = {acc:.4f}")

    return results


# ============ QAT (量化感知训练) 模拟实现 ============
class FakeQuantize(nn.Module):
    """
    模拟量化 + 直通估计器 (STE: Straight-Through Estimator)

    前向传播：执行量化
    反向传播：直接将梯度传过量化操作（STE），不做任何修改
    """

    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits

    def forward(self, x):
        if self.bits >= 32:
            return x

        qmin = -(1 << (self.bits - 1))
        qmax = (1 << (self.bits - 1)) - 1

        max_abs = x.abs().max().detach()
        if max_abs < 1e-8:
            scale = torch.tensor(1.0, device=x.device)
        else:
            scale = max_abs / qmax

        # 量化
        q_x = torch.clamp(torch.round(x / scale), qmin, qmax)
        # 反量化 + STE（直通估计器）
        x_deq = q_x * scale
        # STE: 前向用反量化值，反向直接传梯度
        return x + (x_deq - x).detach()


# ============ 绘图函数 ============
def plot_bitwidth_comparison(results: dict):
    """绘制位宽-精度对比图"""
    plt.figure(figsize=(10, 6))

    for method, bw_acc in results.items():
        bitwidths = sorted(bw_acc.keys())
        accuracies = [bw_acc[bw] * 100 for bw in bitwidths]
        plt.plot(bitwidths, accuracies, marker="o", linewidth=2, label=f"{method} 量化")

    plt.xlabel("位宽 (bits)")
    plt.ylabel("精度 (%)")
    plt.title("量化位宽 vs 模型精度")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks([2, 4, 8, 16, 32])
    plt.tight_layout()
    plt.savefig("quantization_bitwidth_accuracy.png", dpi=150)
    print("位宽-精度对比图已保存为 quantization_bitwidth_accuracy.png")


def plot_weight_histogram(
    original: torch.Tensor, quantized: torch.Tensor, bits: int, layer_name: str
):
    """绘制原始权重和量化权重的直方图对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(original.flatten().cpu().numpy(), bins=50, alpha=0.7, color="blue")
    ax1.set_title(f"{layer_name} - FP32 权重分布")
    ax1.set_xlabel("权重值")
    ax1.set_ylabel("频数")

    ax2.hist(quantized.flatten().cpu().numpy(), bins=50, alpha=0.7, color="orange")
    ax2.set_title(f"{layer_name} - INT{bits} 量化权重分布")
    ax2.set_xlabel("权重值")
    ax2.set_ylabel("频数")

    plt.tight_layout()
    plt.savefig(f"weight_hist_{layer_name.replace('.', '_')}_{bits}bit.png", dpi=150)


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 2：量化实验 - 完整实现")
    print("=" * 60)

    # 1. 加载数据
    print("\n[步骤 1] 加载 CIFAR-10 数据...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    # 2. 训练 FP32 基线模型
    print("\n[步骤 2] 训练 FP32 基线模型...")
    model = SimpleCNN(num_classes=10).to(device)
    model = train_model(model, train_loader, test_loader, epochs=5)
    baseline_acc = evaluate_accuracy(model, test_loader)
    print(f"  FP32 基线精度: {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")

    # 3. 比较不同位宽的量化效果
    print("\n[步骤 3] 比较不同位宽的量化精度...")
    bitwidths = [2, 4, 8, 16, 32]
    results = compare_bitwidths(model, test_loader, bitwidths=bitwidths)

    # 4. 计算量化误差
    print("\n[步骤 4] 计算量化误差...")
    for bw in [2, 4, 8]:
        q_model = copy.deepcopy(model)
        quantize_model_weights(q_model, bw, method="linear")
        errors = compute_quantization_error(model, q_model)
        print(f"  INT{bw} 各层量化误差 (MSE):")
        for name, err in errors.items():
            print(f"    {name}: {err:.6f}")

    # 5. 激活值校准
    print("\n[步骤 5] 激活值校准...")
    calib_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        ),
        batch_size=128,
        shuffle=True,
        num_workers=2,
    )
    act_ranges = calibrate_activation_ranges(model, calib_loader, num_batches=10)
    print("  激活值范围:")
    for name, (min_v, max_v) in act_ranges.items():
        print(f"    {name}: [{min_v:.4f}, {max_v:.4f}] (范围: {max_v - min_v:.4f})")

    # 6. 测试量化推理模块 (W8A8)
    print("\n[步骤 6] 测试量化推理模块...")
    q_model_w8a8 = replace_with_quantized(model, weight_bits=8, act_bits=8)
    qa8_acc = evaluate_accuracy(q_model_w8a8, test_loader)
    print(f"  W8A8 精度: {qa8_acc:.4f}")

    q_model_w4a8 = replace_with_quantized(model, weight_bits=4, act_bits=8)
    q4a8_acc = evaluate_accuracy(q_model_w4a8, test_loader)
    print(f"  W4A8 精度: {q4a8_acc:.4f}")

    # 7. 汇总报告
    print("\n" + "=" * 60)
    print("实验报告汇总")
    print("=" * 60)

    print(f"\n{'方法':<15} {'位宽':<8} {'精度':<10} {'相对精度':<10}")
    print("-" * 45)
    print(f"{'FP32 基线':<15} {'32':<8} {baseline_acc:<10.4f} {'100.00%':<10}")

    for bw in bitwidths:
        for method in results:
            acc = results[method][bw]
            rel = acc / baseline_acc * 100
            print(f"{method + ' 量化':<15} {str(bw):<8} {acc:<10.4f} {rel:<10.2f}%")

    # 8. 绘图
    print("\n[步骤 7] 绘制结果图表...")
    plot_bitwidth_comparison(results)

    # 绘制权重直方图示例
    orig_weight = model.conv1.weight.data
    q_model_4 = copy.deepcopy(model)
    quantize_model_weights(q_model_4, 4, "linear")
    q_weight_4 = q_model_4.conv1.weight.data
    plot_weight_histogram(orig_weight, q_weight_4, 4, "conv1")

    print("\n实验完成！")
