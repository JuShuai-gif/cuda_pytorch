"""
实验 3：神经架构搜索实验 (NAS) - 完整参考实现
包含随机搜索、进化搜索、精度预测器和 Pareto 前沿分析

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
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import namedtuple
import matplotlib.pyplot as plt


# ============ 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 架构描述结构体
Architecture = namedtuple(
    "Architecture", ["kernel_sizes", "channels", "num_layers", "use_skip"]
)

# ============ 搜索空间定义 ============
SEARCH_SPACE = {
    "kernel_sizes": [3, 5, 7],
    "channels": [16, 32, 64, 128],
    "num_layers": [2, 3, 4, 5],
    "use_skip": [True, False],
}


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


# ============ 随机采样架构 ============
def random_sample_architecture() -> Architecture:
    """从搜索空间中随机采样一个架构"""
    num_layers = random.choice(SEARCH_SPACE["num_layers"])
    # 通道数通常随深度递增
    available_channels = sorted(SEARCH_SPACE["channels"])
    channels = []
    for i in range(num_layers):
        # 逐层递增或保持通道数
        if i == 0:
            ch = random.choice(available_channels[:2])  # 前两层用较小通道
        else:
            ch = random.choice(available_channels)
        channels.append(ch)

    kernel_sizes = [
        random.choice(SEARCH_SPACE["kernel_sizes"]) for _ in range(num_layers)
    ]
    use_skip = random.choice(SEARCH_SPACE["use_skip"])

    return Architecture(kernel_sizes, channels, num_layers, use_skip)


# ============ CNN 构建器 ============
class CNNBuilder(nn.Module):
    """根据架构参数动态构建 CNN 模型"""

    def __init__(
        self,
        arch: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        image_size: int = 32,
    ):
        super().__init__()
        self.arch = arch
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.use_pool = []

        in_channels = input_channels
        size = image_size

        for i in range(arch.num_layers):
            out_channels = arch.channels[i]
            k = arch.kernel_sizes[i]
            padding = k // 2

            self.convs.append(nn.Conv2d(in_channels, out_channels, k, padding=padding))
            self.bns.append(nn.BatchNorm2d(out_channels))

            # 跳跃连接投影层
            if arch.use_skip and in_channels != out_channels:
                self.skip_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            else:
                self.skip_convs.append(nn.Identity())

            # 每 2 层插入一次池化
            if (i + 1) % 2 == 0 and size > 4:
                self.use_pool.append(True)
                size //= 2
            else:
                self.use_pool.append(False)

            in_channels = out_channels

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        for i in range(self.arch.num_layers):
            residual = self.skip_convs[i](x)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = F.relu(x)

            if self.arch.use_skip:
                x = x + residual
                x = F.relu(x)

            if self.use_pool[i]:
                x = self.pool(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============ MACs 估算 ============
def count_macs(arch: Architecture, input_size: int = 32) -> int:
    """
    估算架构的总 MACs（乘加操作数）

    对 Conv2d: MACs = in_c * out_c * k * k * h_out * w_out
    对 Linear: MACs = in_features * out_features
    """
    in_channels = 3
    h, w = input_size, input_size
    total_macs = 0

    for i in range(arch.num_layers):
        out_channels = arch.channels[i]
        k = arch.kernel_sizes[i]
        padding = k // 2

        # 卷积输出尺寸（stride=1）
        h_out = (h + 2 * padding - k) // 1 + 1
        w_out = (w + 2 * padding - k) // 1 + 1

        # 卷积 MACs
        conv_macs = in_channels * out_channels * k * k * h_out * w_out
        total_macs += conv_macs

        # 每 2 层池化一次
        if (i + 1) % 2 == 0 and h_out > 4:
            h = h_out // 2
            w = w_out // 2
        else:
            h, w = h_out, w_out

        in_channels = out_channels

    # 全连接层 MACs
    fc_macs = in_channels * 10  # 假设 10 类
    total_macs += fc_macs

    return total_macs


# ============ 架构评估 ============
def evaluate_architecture(
    arch: Architecture,
    train_loader,
    test_loader,
    epochs: int = 3,
    verbose: bool = False,
):
    """训练并评估一个架构，返回 (精度, MACs, 参数量)"""
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

        # 快速评估
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

    if verbose:
        ks = arch.kernel_sizes
        ch = arch.channels
        print(
            f"  架构: ks={ks}, ch={ch}, L={arch.num_layers}, "
            f"skip={arch.use_skip} -> Acc={best_acc:.4f}, "
            f"MACs={macs:.0f}, Params={params:,}"
        )

    return best_acc, macs, params


# ============ 随机搜索 ============
def random_search(
    num_samples: int = 20, train_loader=None, test_loader=None
) -> List[Tuple[Architecture, float, int, int]]:
    """
    随机搜索 NAS

    随机采样 num_samples 个架构，逐一训练评估，返回结果列表
    """
    results = []
    for i in range(num_samples):
        print(f"  随机搜索 {i + 1}/{num_samples}...")
        arch = random_sample_architecture()
        acc, macs, params = evaluate_architecture(
            arch, train_loader, test_loader, epochs=3
        )
        results.append((arch, acc, macs, params))
        ks = arch.kernel_sizes
        ch = arch.channels
        print(
            f"    ks={ks}, ch={ch}, L={arch.num_layers}, Acc={acc:.4f}, MACs={macs:.0f}"
        )

    # 按精度降序排列
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ============ 进化搜索 ============
def mutate(arch: Architecture, mutation_prob: float = 0.3) -> Architecture:
    """对架构进行变异"""
    kernel_sizes = list(arch.kernel_sizes)
    channels = list(arch.channels)
    num_layers = arch.num_layers
    use_skip = arch.use_skip

    # 变异卷积核大小
    for i in range(num_layers):
        if random.random() < mutation_prob:
            kernel_sizes[i] = random.choice(SEARCH_SPACE["kernel_sizes"])

    # 变异通道数
    for i in range(num_layers):
        if random.random() < mutation_prob:
            channels[i] = random.choice(SEARCH_SPACE["channels"])

    # 变异跳跃连接
    if random.random() < mutation_prob:
        use_skip = not use_skip

    # 偶尔变异层数（谨慎操作）
    if random.random() < mutation_prob * 0.3:
        if random.random() < 0.5 and num_layers < max(SEARCH_SPACE["num_layers"]):
            num_layers += 1
            kernel_sizes.append(random.choice(SEARCH_SPACE["kernel_sizes"]))
            channels.append(random.choice(SEARCH_SPACE["channels"]))
        elif num_layers > min(SEARCH_SPACE["num_layers"]):
            num_layers -= 1
            kernel_sizes.pop()
            channels.pop()

    return Architecture(kernel_sizes, channels, num_layers, use_skip)


def crossover(parent1: Architecture, parent2: Architecture) -> Architecture:
    """对两个父代架构进行交叉，生成子代"""
    # 取两个父代的最小层数
    min_layers = min(parent1.num_layers, parent2.num_layers)

    if min_layers < 2:
        return random.choice([parent1, parent2])

    # 随机选择交叉点
    crossover_point = random.randint(1, min_layers - 1)

    # 前半部分来自 parent1，后半部分来自 parent2
    child_kernels = list(parent1.kernel_sizes[:crossover_point]) + list(
        parent2.kernel_sizes[crossover_point:min_layers]
    )
    child_channels = list(parent1.channels[:crossover_point]) + list(
        parent2.channels[crossover_point:min_layers]
    )

    child_num_layers = min_layers
    child_use_skip = random.choice([parent1.use_skip, parent2.use_skip])

    return Architecture(child_kernels, child_channels, child_num_layers, child_use_skip)


def tournament_select(population, fitnesses, tournament_size=3):
    """锦标赛选择"""
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx]


def evolutionary_search(
    population_size: int = 10,
    generations: int = 10,
    mutation_prob: float = 0.3,
    train_loader=None,
    test_loader=None,
):
    """
    进化搜索 NAS

    流程：
    1. 初始化种群
    2. 评估适应度（精度）
    3. 选择、交叉、变异生成新个体
    4. 精英保留：保留最佳个体
    5. 重复直到收敛或达到代数上限
    """
    # 1. 初始化种群
    print("  初始化种群...")
    population = [random_sample_architecture() for _ in range(population_size)]
    fitnesses = []
    macs_list = []

    for i, arch in enumerate(population):
        acc, macs, _ = evaluate_architecture(arch, train_loader, test_loader, epochs=3)
        fitnesses.append(acc)
        macs_list.append(macs)
        ks = arch.kernel_sizes
        ch = arch.channels
        print(f"    个体 {i + 1}: ks={ks}, ch={ch}, Acc={acc:.4f}")

    history = {"best_acc": [], "avg_acc": [], "best_macs": []}

    # 2. 进化循环
    for gen in range(generations):
        print(f"  第 {gen + 1}/{generations} 代...")

        # 精英保留：保留最佳的 20%
        elite_count = max(1, int(population_size * 0.2))
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        elite_archs = [population[i] for i in elite_indices]
        elite_fitnesses = [fitnesses[i] for i in elite_indices]
        elite_macs = [macs_list[i] for i in elite_indices]

        new_population = list(elite_archs)
        new_fitnesses = list(elite_fitnesses)
        new_macs = list(elite_macs)

        # 生成新个体
        while len(new_population) < population_size:
            parent1 = tournament_select(population, fitnesses, 3)
            parent2 = tournament_select(population, fitnesses, 3)

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_prob)

            child_acc, child_macs, _ = evaluate_architecture(
                child, train_loader, test_loader, epochs=2
            )

            new_population.append(child)
            new_fitnesses.append(child_acc)
            new_macs.append(child_macs)

        population = new_population
        fitnesses = new_fitnesses
        macs_list = new_macs

        best_idx = np.argmax(fitnesses)
        history["best_acc"].append(fitnesses[best_idx])
        history["avg_acc"].append(np.mean(fitnesses))
        history["best_macs"].append(macs_list[best_idx])

        print(
            f"    最佳精度: {fitnesses[best_idx]:.4f}, "
            f"平均精度: {np.mean(fitnesses):.4f}"
        )

    return population, fitnesses, macs_list, history


# ============ 精度预测器 ============
def encode_architecture(arch: Architecture) -> torch.Tensor:
    """
    将架构编码为固定长度特征向量

    编码特征包括统计信息：均值、最大值、最小值等
    """
    features = [
        np.mean(arch.kernel_sizes),
        np.max(arch.kernel_sizes),
        np.min(arch.kernel_sizes),
        np.std(arch.kernel_sizes) if len(arch.kernel_sizes) > 1 else 0,
        np.mean(arch.channels),
        np.max(arch.channels),
        np.min(arch.channels),
        np.std(arch.channels) if len(arch.channels) > 1 else 0,
        float(arch.num_layers),
        float(arch.num_layers) / max(SEARCH_SPACE["num_layers"]),  # 归一化层数
        float(arch.use_skip),
        np.sum(arch.kernel_sizes),  # 总卷积核感受野
        np.sum(arch.channels),  # 总通道数
    ]
    # 填充到固定长度
    while len(features) < 16:
        features.append(0.0)
    return torch.tensor(features[:16], dtype=torch.float32)


class AccuracyPredictor(nn.Module):
    """精度预测器：用 MLP 从架构特征预测精度"""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_accuracy_predictor(arch_results, epochs: int = 200):
    """训练精度预测器"""
    X = torch.stack([encode_architecture(arch) for arch, _, _, _ in arch_results])
    y = torch.tensor([acc for _, acc, _, _ in arch_results], dtype=torch.float32)

    # 划分训练/测试集
    n = len(X)
    indices = torch.randperm(n)
    split = int(n * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    predictor = AccuracyPredictor()
    optimizer = optim.Adam(predictor.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        predictor.train()
        optimizer.zero_grad()
        pred = predictor(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred_test = predictor(X_test)
                test_loss = criterion(pred_test, y_test).item()
                print(
                    f"    Epoch {epoch + 1}: Train Loss={loss.item():.6f}, "
                    f"Test Loss={test_loss:.6f}"
                )

    with torch.no_grad():
        pred_test = predictor(X_test)
        final_mse = criterion(pred_test, y_test).item()
    print(f"  最终测试 MSE: {final_mse:.6f}")

    return predictor


# ============ Pareto 前沿 ============
def find_pareto_frontier(results):
    """
    在精度和 MACs 之间寻找 Pareto 前沿

    精度越大越好，MACs 越小越好
    """
    points = np.array([(acc, macs) for _, acc, macs, _ in results])
    n = len(points)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i 被 j 支配：j.acc >= i.acc AND j.macs <= i.macs，且至少一个严格优于
            if points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1]:
                if points[j, 0] > points[i, 0] or points[j, 1] < points[i, 1]:
                    dominated[i] = True
                    break

    pareto_archs = [results[i] for i in range(n) if not dominated[i]]
    pareto_points = points[~dominated]

    # 按 MACs 排序
    order = np.argsort(pareto_points[:, 1])
    pareto_archs = [pareto_archs[i] for i in order]
    pareto_points = pareto_points[order]

    return pareto_archs, pareto_points


# ============ 绘图函数 ============
def plot_search_comparison(random_results, evo_history):
    """绘制随机搜索和进化搜索的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：精度 vs MACs 散点图
    ax = axes[0]
    random_acc = [r[1] for r in random_results]
    random_macs = [r[2] for r in random_results]
    ax.scatter(
        random_macs, random_acc, alpha=0.6, label="随机搜索", c="blue", marker="o", s=60
    )

    ax.set_xlabel("MACs")
    ax.set_ylabel("精度")
    ax.set_title("随机搜索：精度 vs MACs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：进化搜索收敛曲线
    ax = axes[1]
    gens = range(1, len(evo_history["best_acc"]) + 1)
    ax.plot(gens, evo_history["best_acc"], marker="o", label="最佳精度", linewidth=2)
    ax.plot(
        gens,
        evo_history["avg_acc"],
        marker="s",
        label="平均精度",
        linewidth=2,
        linestyle="--",
    )

    ax.set_xlabel("代数")
    ax.set_ylabel("精度")
    ax.set_title("进化搜索收敛曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("nas_search_comparison.png", dpi=150)
    print("搜索对比图已保存为 nas_search_comparison.png")


def plot_pareto_frontier(all_results, pareto_archs, pareto_points):
    """绘制 Pareto 前沿"""
    plt.figure(figsize=(8, 6))

    all_acc = [r[1] for r in all_results]
    all_macs = [r[2] for r in all_results]
    plt.scatter(all_macs, all_acc, alpha=0.4, label="所有架构", c="gray", s=50)

    plt.scatter(
        pareto_points[:, 1],
        pareto_points[:, 0],
        alpha=1.0,
        label="Pareto 前沿",
        c="red",
        s=100,
        edgecolors="darkred",
        linewidths=1.5,
    )

    plt.plot(pareto_points[:, 1], pareto_points[:, 0], "--", color="red", alpha=0.5)

    plt.xlabel("MACs")
    plt.ylabel("精度")
    plt.title("Pareto 前沿：精度 vs MACs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_frontier.png", dpi=150)
    print("Pareto 前沿图已保存为 pareto_frontier.png")


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 3：神经架构搜索实验 - 完整实现")
    print("=" * 60)

    # 1. 加载数据
    print("\n[步骤 1] 加载 CIFAR-10 数据...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    # 2. 测试搜索空间
    print("\n[步骤 2] 测试搜索空间...")
    test_arch = random_sample_architecture()
    print(
        f"  随机采样架构: ks={test_arch.kernel_sizes}, "
        f"ch={test_arch.channels}, L={test_arch.num_layers}, "
        f"skip={test_arch.use_skip}"
    )
    model = CNNBuilder(test_arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")
    print(f"  估算 MACs: {count_macs(test_arch):,}")

    # 3. 运行随机搜索（使用较少的采样数以加快速度）
    print("\n[步骤 3] 运行随机搜索...")
    num_random = 12
    random_results = random_search(
        num_samples=num_random, train_loader=train_loader, test_loader=test_loader
    )

    print(f"\n  随机搜索结果（前 5）:")
    for i, (arch, acc, macs, params) in enumerate(random_results[:5]):
        ks = arch.kernel_sizes
        ch = arch.channels
        print(
            f"    {i + 1}. ks={ks}, ch={ch}, L={arch.num_layers}, "
            f"Acc={acc:.4f}, MACs={macs:.0f}"
        )

    # 4. 运行进化搜索
    print("\n[步骤 4] 运行进化搜索...")
    evo_pop, evo_fit, evo_macs, evo_history = evolutionary_search(
        population_size=8,
        generations=6,
        mutation_prob=0.3,
        train_loader=train_loader,
        test_loader=train_loader,
    )
    # 注意：为加速，进化搜索中也使用 train_loader 做快速评估

    # 5. 训练精度预测器
    print("\n[步骤 5] 训练精度预测器...")
    predictor = train_accuracy_predictor(random_results, epochs=200)

    # 测试预测器
    print("  预测器测试（随机搜索的最佳 3 个架构）:")
    for arch, acc, macs, _ in random_results[:3]:
        encoded = encode_architecture(arch)
        with torch.no_grad():
            pred_acc = predictor(encoded.unsqueeze(0)).item()
        print(
            f"    实际精度: {acc:.4f}, 预测精度: {pred_acc:.4f}, "
            f"误差: {abs(acc - pred_acc):.4f}"
        )

    # 6. 寻找 Pareto 前沿
    print("\n[步骤 6] 寻找 Pareto 前沿...")
    pareto_archs, pareto_points = find_pareto_frontier(random_results)

    print(f"  Pareto 前沿包含 {len(pareto_archs)} 个架构:")
    for arch, acc, macs, _ in pareto_archs:
        ks = arch.kernel_sizes
        ch = arch.channels
        print(
            f"    ks={ks}, ch={ch}, L={arch.num_layers}, Acc={acc:.4f}, MACs={macs:.0f}"
        )

    # 7. 汇总报告
    print("\n" + "=" * 60)
    print("实验报告汇总")
    print("=" * 60)

    print(
        f"\n  搜索空间大小: {len(SEARCH_SPACE['kernel_sizes'])} × "
        f"(channels) × {len(SEARCH_SPACE['num_layers'])} × "
        f"{len(SEARCH_SPACE['use_skip'])}"
    )

    print(f"\n  随机搜索:")
    random_accs = [r[1] for r in random_results]
    print(f"    采样数: {num_random}")
    print(f"    最佳精度: {max(random_accs):.4f}")
    print(f"    平均精度: {np.mean(random_accs):.4f}")
    print(f"    标准差: {np.std(random_accs):.4f}")

    print(f"\n  进化搜索:")
    print(f"    最佳精度: {evo_history['best_acc'][-1]:.4f}")
    print(f"    平均精度: {evo_history['avg_acc'][-1]:.4f}")

    # 8. 绘图
    print("\n[步骤 7] 绘制结果图表...")
    plot_search_comparison(random_results, evo_history)
    plot_pareto_frontier(random_results, pareto_archs, pareto_points)

    print("\n实验完成！")
