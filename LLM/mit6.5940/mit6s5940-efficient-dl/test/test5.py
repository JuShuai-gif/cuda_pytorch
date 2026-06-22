"""End-to-end pruning and fine-tuning demo.

这个脚本演示一个完整小流程：
1. 构造 TinyCNN 和 synthetic dataset。
2. 评估剪枝前 accuracy/latency。
3. 调用 test4.py 中的全局幅度剪枝。
4. 用 mask 约束进行微调，避免被剪权重复活。
5. 输出剪枝前、剪枝后、微调后的指标对比。

运行方式：
    python test/test5.py
"""

from __future__ import annotations

import copy  # 深拷贝模型：在不破坏原始 baseline 的前提下得到一份用于剪枝的副本

import torch
import torch.nn as nn

# 直接复用 test4.py 里已经写好的工具，避免重复造轮子。
from test4 import (
    TinyCNN,  # 小型 CNN 模型
    apply_masks,  # 把 mask 重新乘回权重，固化稀疏结构
    benchmark_latency_ms,  # 延迟基准测试
    evaluate_accuracy,  # 精度评估
    global_magnitude_prune,  # 全局幅度剪枝
    make_synthetic_loader,  # 生成随机数据的 DataLoader
    prunable_modules,  # 遍历可剪枝层
    sparsity_of_prunable_weights,  # 统计实际稀疏度
)


# -----------------------------------------------------------------------------
# Demo：完整剪枝 + 微调流水线。
# 作用：把 test4.py 里的工具串起来，形成一个可运行的端到端小实验。
# -----------------------------------------------------------------------------
def finetune_with_masks(
    model: nn.Module,
    train_loader,
    masks: dict[str, torch.Tensor],
    epochs: int = 2,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """在微调过程中持续应用 mask，确保被剪权重保持为 0。

    标准训练会更新所有权重，从而让“已被剪掉的权重”重新变成非零，破坏稀疏结构。
    这里通过两道保险来维持稀疏：
    (1) 反传后把被剪位置的梯度清零；
    (2) 每个 step 更新完再乘一遍 mask 兜底。
    """
    model.to(device)
    # 带动量的 SGD：动量项是“权重复活”的主要来源，所以下面要专门处理梯度。
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失

    for epoch in range(epochs):
        model.train()  # 训练模式
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()  # 清空上一步累积的梯度
            loss = criterion(model(x), y)
            loss.backward()  # 反向传播计算梯度

            # 关键：清掉被剪权重的梯度，避免 momentum 把它们推回非零。
            for name, module in prunable_modules(model):
                if name in masks and module.weight.grad is not None:
                    # mask 中为 0 的位置，对应梯度也被乘成 0。
                    module.weight.grad.mul_(
                        masks[name].to(device, module.weight.grad.dtype)
                    )

            optimizer.step()  # 按（已屏蔽的）梯度更新权重
            apply_masks(model, masks)  # 兜底：再把被剪权重强制归零
            total_loss += float(loss)
        print(f"finetune epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}")
    return model


def prune_and_finetune_demo(sparsity: float = 0.5, device: str = "cpu") -> dict:
    """完整 demo：baseline -> prune -> finetune -> report。"""
    # 准备训练集与验证集（均为随机数据，仅用于跑通流程）。
    train_loader = make_synthetic_loader(n=256, batch_size=32)
    val_loader = make_synthetic_loader(n=128, batch_size=32)
    model = TinyCNN().to(device)
    # 取 1 张样本作为延迟测试的固定输入。
    example_x, _ = next(iter(val_loader))
    example_x = example_x[:1].to(device)

    # 阶段 1：未剪枝基线指标。
    baseline_acc = evaluate_accuracy(model, val_loader, device=device)
    baseline_latency = benchmark_latency_ms(model, example_x)

    # 阶段 2：在模型副本上做全局剪枝并测指标（此时未微调，精度通常会下降）。
    pruned_model = copy.deepcopy(model).to(device)
    masks = global_magnitude_prune(pruned_model, sparsity=sparsity)
    pruned_acc = evaluate_accuracy(pruned_model, val_loader, device=device)
    pruned_latency = benchmark_latency_ms(pruned_model, example_x)

    # 阶段 3：带 mask 微调，让剩余权重补偿被剪掉的部分。
    finetune_with_masks(
        pruned_model, train_loader, masks, epochs=2, lr=1e-3, device=device
    )
    finetuned_acc = evaluate_accuracy(pruned_model, val_loader, device=device)
    finetuned_latency = benchmark_latency_ms(pruned_model, example_x)

    # 汇总三个阶段的精度、实际稀疏度与延迟，供对比。
    return {
        "baseline_acc": baseline_acc,
        "pruned_acc": pruned_acc,
        "finetuned_acc": finetuned_acc,
        "actual_sparsity": sparsity_of_prunable_weights(pruned_model),
        "baseline_latency": baseline_latency,
        "pruned_latency": pruned_latency,
        "finetuned_latency": finetuned_latency,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    report = prune_and_finetune_demo(sparsity=0.5)
    print("\n[Demo] 完整剪枝 + 微调报告")
    for key, value in report.items():
        print(f"{key}: {value}")
