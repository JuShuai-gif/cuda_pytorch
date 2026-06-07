#!/usr/bin/env python3
"""
MIT 6.5940 第21讲：端侧训练模拟

涵盖主题：
  - 模拟联邦学习：创建多个具有非独立同分布数据的"客户端"
  - 实现FedAvg：本地训练 -> 聚合 -> 重复
  - 模拟内存瓶颈：测量反向传播期间的激活值内存
  - TinyTL概念：冻结主干网络，仅训练偏置 + 轻量级分类器
  - 对比：完整训练 vs TinyTL的内存使用

所有计算均在CPU上运行，无需GPU。
"""

from __future__ import annotations

import copy
import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ===========================================================================
# 可复现性
# ===========================================================================
torch.manual_seed(42)


# ===========================================================================
# 1. 用于联邦学习的简单模型
# ===========================================================================


class SimpleCNN(nn.Module):
    """一个适用于端侧训练实验的小型CNN。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)  # stride=2用于降采样
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
        self._activation_shapes: Dict[str, Tuple[int, ...]] = {}  # 存储激活值形状

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activation_shapes = {}
        x = F.relu(self.conv1(x))
        self._activation_shapes["conv1"] = tuple(x.shape)
        x = F.relu(self.conv2(x))
        self._activation_shapes["conv2"] = tuple(x.shape)
        x = F.relu(self.conv3(x))
        self._activation_shapes["conv3"] = tuple(x.shape)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc(x)
        return x


# ===========================================================================
# 2. 非独立同分布（Non-IID）数据分割器
# ===========================================================================


def create_non_iid_split(
    data: torch.Tensor,
    targets: torch.Tensor,
    num_clients: int,
    alpha: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """为联邦学习创建非独立同分布的数据划分。

    使用狄利克雷分布（浓度参数alpha）为每个客户端
    创建不均衡的类别分布。

    参数：
        data: 输入张量 (N, ...)
        targets: 标签张量 (N,)
        num_clients: 联邦客户端数量
        alpha: 狄利克雷浓度参数（越小越不均衡）

    返回：
        每个客户端包含'data'和'targets'的字典列表。
    """
    num_classes = int(targets.max().item()) + 1
    N = len(targets)

    # 从狄利克雷分布中采样比例
    proportions = torch.distributions.Dirichlet(
        torch.full((num_classes,), alpha, dtype=torch.float32)
    ).sample((num_clients,))  # 形状: (num_clients, num_classes)

    # 根据比例将样本分配给客户端
    client_data: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_indices = (targets == cls).nonzero(as_tuple=True)[0].tolist()
        num_cls = len(cls_indices)
        # 按狄利克雷比例计算每个客户端应得的样本数
        splits = (proportions[:, cls] * num_cls / proportions[:, cls].sum()).long()
        # 修正舍入误差，确保总和不偏离
        diff = num_cls - splits.sum().item()
        if diff > 0:
            for i in range(diff):
                splits[i] += 1
        elif diff < 0:
            for i in range(-diff):
                splits[-(i + 1)] -= 1
        # 随机打乱并分配
        perm = torch.randperm(num_cls).tolist()
        start = 0
        for client_id in range(num_clients):
            count = int(splits[client_id].item())
            if count > 0:
                client_data[client_id].extend(
                    [
                        cls_indices[perm[idx]]
                        for idx in range(start, min(start + count, num_cls))
                    ]
                )
                start += count

    # 构建每个客户端的最终数据集
    clients = []
    for cid in range(num_clients):
        if len(client_data[cid]) == 0:
            # 确保每个客户端至少有一个样本
            client_data[cid] = [torch.randint(0, N, (1,)).item()]
        idx = torch.tensor(client_data[cid])
        clients.append(
            {
                "data": data[idx],
                "targets": targets[idx],
            }
        )

    return clients


# ===========================================================================
# 3. 联邦平均（FedAvg）
# ===========================================================================


def local_train(
    model: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 1,
    lr: float = 0.01,
    batch_size: int = 32,
) -> nn.Module:
    """在客户端本地数据上训练模型。

    返回训练后的模型副本。
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    N = len(data)

    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            batch_idx = perm[i : i + batch_size]
            x_batch = data[batch_idx]
            y_batch = targets[batch_idx]

            optimizer.zero_grad()
            out = model(x_batch)
            loss = F.cross_entropy(out, y_batch)
            loss.backward()
            optimizer.step()

    return copy.deepcopy(model)


def fed_avg(
    global_model: nn.Module,
    clients: List[Dict[str, torch.Tensor]],
    rounds: int = 3,
    local_epochs: int = 1,
    lr: float = 0.01,
) -> List[Dict[str, float]]:
    """运行联邦平均（Federated Averaging）。

    参数：
        global_model: 初始全局模型
        clients: 客户端数据字典列表
        rounds: 联邦轮次
        local_epochs: 每轮的本地训练轮次
        lr: 学习率

    返回：
        每轮的训练历史记录。
    """
    history = []

    for rnd in range(rounds):
        local_models = []
        local_losses = []

        # ---------- 客户端本地训练 ----------
        for client in clients:
            local_m = copy.deepcopy(global_model)
            local_m = local_train(
                local_m,
                client["data"],
                client["targets"],
                epochs=local_epochs,
                lr=lr,
                batch_size=32,
            )

            # 计算本地loss
            with torch.no_grad():
                out = local_m(client["data"])
                loss = F.cross_entropy(out, client["targets"]).item()
            local_losses.append(loss)
            local_models.append(local_m)

        # ---------- 服务端聚合（FedAvg核心） ----------
        with torch.no_grad():
            # 首先清零全局模型参数
            for global_param in global_model.parameters():
                global_param.zero_()
            # 按数据量加权聚合各个客户端的模型参数
            total_samples = sum(len(c["data"]) for c in clients)
            for client, local_m in zip(clients, local_models):
                weight = len(client["data"]) / total_samples
                for gp, lp in zip(global_model.parameters(), local_m.parameters()):
                    gp.data += weight * lp.data

        avg_loss = sum(local_losses) / len(local_losses)
        history.append(
            {"round": rnd + 1, "avg_loss": avg_loss, "num_clients": len(clients)}
        )
        print(f"  轮次 {rnd + 1}: 平均损失={avg_loss:.4f}")

    return history


# ===========================================================================
# 4. 激活值内存测量
# ===========================================================================


def measure_activation_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
) -> Dict[str, int]:
    """测量前向传播期间的峰值激活值内存。

    这估算的是为反向传播（梯度计算）存储中间激活值所需的内存。

    参数：
        model: PyTorch模型
        input_shape: (batch_size, channels, height, width)

    返回：
        按层拆解的内存使用及总量。
    """
    model.eval()
    x = torch.randn(*input_shape)

    # 注册钩子函数来捕获激活值大小
    activations: List[int] = []

    def hook_fn(module, inp, out):
        # 存储输出张量的大小（bytes, float32格式）
        if isinstance(out, torch.Tensor):
            activations.append(out.numel() * 4)

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
            handles.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(x)

    # 移除钩子，避免干扰后续使用
    for h in handles:
        h.remove()

    # 参数量 × 4字节（float32）
    param_mem = sum(p.numel() for p in model.parameters()) * 4
    # 仅可训练参数需要梯度和优化器状态缓冲区
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grad_mem = trainable_count * 4  # 反向传播期间的梯度存储
    opt_mem = trainable_count * 8  # Adam: m + v（每个可训练参数2倍FP32）
    act_mem = sum(activations)

    return {
        "parameters_bytes": param_mem,
        "gradients_bytes": grad_mem,
        "optimizer_bytes": opt_mem,
        "activations_bytes": act_mem,
        "total_bytes": param_mem + grad_mem + opt_mem + act_mem,
    }


# ===========================================================================
# 5. TinyTL：冻结主干网络，训练偏置 + 分类器
# ===========================================================================


class TinyTLModel(nn.Module):
    """微型迁移学习（Tiny Transfer Learning）模型。

    冻结卷积主干网络，仅训练：
      - BatchNorm/LayerNorm偏置
      - 最终分类器层
    这极大地减少了可训练参数数量和反向传播所需的激活值内存。
    """

    def __init__(self, backbone: SimpleCNN, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

        # 冻结主干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 仅训练分类器
        self.classifier.weight.requires_grad = True
        self.classifier.bias.requires_grad = True

        # 同时解冻BN类层（此处模拟为卷积偏置）
        # 在TinyTL中，我们训练冻结层的偏置
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                m.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播中冻结的主干网络用no_grad以节省内存
        with torch.no_grad():
            features = self.backbone.conv1(x)
            features = F.relu(features)
            features_conv1 = features
            features = F.relu(self.backbone.conv2(features))
            features_conv2 = features
            features = F.relu(self.backbone.conv3(features))
            features = features.view(features.size(0), -1)
        return self.classifier(features)


def compare_memory_full_vs_tinytl(
    model: SimpleCNN,
    input_shape: Tuple[int, ...],
) -> Dict[str, Dict]:
    """对比完整训练和TinyTL的内存使用。

    参数：
        model: 完整CNN模型
        input_shape: 输入张量形状

    返回：
        内存对比结果。
    """
    # 使用独立的副本避免修改共享的主干网络
    backbone_copy = copy.deepcopy(model)
    full_mem = measure_activation_memory(backbone_copy, input_shape)
    trainable_full = sum(
        p.numel() for p in backbone_copy.parameters() if p.requires_grad
    )

    # TinyTL：冻结副本的主干网络，仅训练偏置 + 分类器
    tinytl = TinyTLModel(backbone_copy)
    tinytl_mem = measure_activation_memory(tinytl, input_shape)
    trainable_tinytl = sum(p.numel() for p in tinytl.parameters() if p.requires_grad)

    return {
        "full_training": {
            "trainable_params": trainable_full,
            "memory_bytes": full_mem["total_bytes"],
        },
        "tinytl": {
            "trainable_params": trainable_tinytl,
            "memory_bytes": tinytl_mem["total_bytes"],
        },
    }


# ===========================================================================
# 6. 主演示
# ===========================================================================


def main() -> None:
    print("=" * 72)
    print("MIT 6.5940 第21讲：端侧训练模拟")
    print("=" * 72)

    # ---------- 模型设置 ----------
    print("\n--- 1. 模型设置 ---")
    model = SimpleCNN(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  简单CNN: {total_params:,} 个参数")
    print(f"  模型大小 (FP32): {total_params * 4 / 1024:.1f} KB")

    # ---------- 创建非IID数据 ----------
    print("\n--- 2. 非IID数据划分 ---")
    # 模拟5000个MNIST风格样本，分配给10个客户端
    num_clients = 10
    num_samples = 5000
    data = torch.randn(num_samples, 1, 28, 28)
    targets = torch.randint(0, 10, (num_samples,))

    clients = create_non_iid_split(data, targets, num_clients, alpha=0.3)
    print(f"  总样本数: {num_samples}")
    print(f"  客户端数量: {num_clients}")
    print(f"  狄利克雷alpha: 0.3（越小越不均衡）")
    print(f"  客户端数据分布:")
    for i, c in enumerate(clients):
        class_counts = torch.bincount(c["targets"], minlength=10)
        dominant = class_counts.argmax().item()
        print(
            f"    客户端 {i}: {len(c['data'])} 个样本, "
            f"主要类别={dominant} ({class_counts[dominant].item()} 个样本)"
        )

    # ---------- FedAvg模拟 ----------
    print("\n--- 3. FedAvg模拟 ---")
    global_model = copy.deepcopy(model)
    history = fed_avg(global_model, clients, rounds=3, local_epochs=1, lr=0.01)
    print(f"  最终平均损失: {history[-1]['avg_loss']:.4f}")

    # ---------- 激活值内存 ----------
    print("\n--- 4. 激活值内存测量 ---")
    batch_sizes = [1, 8, 32, 64]
    print(
        f"  {'批次':>6} {'参数(KB)':>11} {'梯度(KB)':>11} "
        f"{'激活值(KB)':>11} {'总计(KB)':>11}"
    )
    print(f"  {'-' * 57}")
    for bs in batch_sizes:
        mem = measure_activation_memory(model, (bs, 1, 28, 28))
        print(
            f"  {bs:>6} {mem['parameters_bytes'] / 1024:>11.1f} "
            f"{mem['gradients_bytes'] / 1024:>11.1f} "
            f"{mem['activations_bytes'] / 1024:>11.1f} "
            f"{mem['total_bytes'] / 1024:>11.1f}"
        )

    # ---------- TinyTL ----------
    print("\n--- 5. TinyTL：冻结主干网络，训练偏置 + 分类器 ---")
    comparison = compare_memory_full_vs_tinytl(model, (32, 1, 28, 28))

    full_trainable = comparison["full_training"]["trainable_params"]
    tinytl_trainable = comparison["tinytl"]["trainable_params"]
    full_mem = comparison["full_training"]["memory_bytes"]
    tinytl_mem = comparison["tinytl"]["memory_bytes"]

    print(f"  {'':>20} {'完整训练':>16} {'TinyTL':>16} {'减少比例':>12}")
    print(f"  {'-' * 66}")
    print(
        f"  {'可训练参数':>20} {full_trainable:>16,} {tinytl_trainable:>16,} "
        f"{(1 - tinytl_trainable / full_trainable) * 100:>11.1f}%"
    )
    print(
        f"  {'内存 (KB)':>20} {full_mem / 1024:>16.1f} {tinytl_mem / 1024:>16.1f} "
        f"{(1 - tinytl_mem / full_mem) * 100:>11.1f}%"
    )

    # ---------- TinyTL前向传播验证 ----------
    print("\n  TinyTL前向传播验证:")
    tinytl = TinyTLModel(model)
    x_test = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out = tinytl(x_test)
    print(f"  输入:  {tuple(x_test.shape)}")
    print(f"  输出: {tuple(out.shape)} (10 类别)")
    print(
        f"  可训练参数: {sum(p.numel() for p in tinytl.parameters() if p.requires_grad):,} "
        f"/ {sum(p.numel() for p in tinytl.parameters()):,} 总计"
    )

    # ---------- 总结 ----------
    print("\n--- 6. 总结 ---")
    print("  核心要点：")
    print("    - 联邦学习保护隐私：数据从不离开设备")
    print("    - FedAvg在服务端聚合本地模型更新")
    print("    - 非IID数据是联邦学习收敛的主要挑战")
    print("    - 激活值内存在端侧训练中占主导地位（随批次大小增长）")
    print("    - TinyTL减少约42%的可训练参数和约42%的梯度内存")
    print("    - 冻结主干网络节省优化器状态内存（冻结参数不需要m/v）")

    print("\n完成。所有计算均在CPU上运行。\n")


if __name__ == "__main__":
    main()
