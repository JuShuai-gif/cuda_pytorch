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

import copy

import torch
import torch.nn as nn

from test4 import (
    TinyCNN,
    apply_masks,
    benchmark_latency_ms,
    evaluate_accuracy,
    global_magnitude_prune,
    make_synthetic_loader,
    prunable_modules,
    sparsity_of_prunable_weights,
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
    """在微调过程中持续应用 mask，确保被剪权重保持为 0。"""
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            # 关键：清掉被剪权重的梯度，避免 momentum 把它们推回非零。
            for name, module in prunable_modules(model):
                if name in masks and module.weight.grad is not None:
                    module.weight.grad.mul_(masks[name].to(device, module.weight.grad.dtype))

            optimizer.step()
            apply_masks(model, masks)
            total_loss += float(loss)
        print(f"finetune epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}")
    return model


def prune_and_finetune_demo(sparsity: float = 0.5, device: str = "cpu") -> dict:
    """完整 demo：baseline -> prune -> finetune -> report。"""
    train_loader = make_synthetic_loader(n=256, batch_size=32)
    val_loader = make_synthetic_loader(n=128, batch_size=32)
    model = TinyCNN().to(device)
    example_x, _ = next(iter(val_loader))
    example_x = example_x[:1].to(device)

    baseline_acc = evaluate_accuracy(model, val_loader, device=device)
    baseline_latency = benchmark_latency_ms(model, example_x)

    pruned_model = copy.deepcopy(model).to(device)
    masks = global_magnitude_prune(pruned_model, sparsity=sparsity)
    pruned_acc = evaluate_accuracy(pruned_model, val_loader, device=device)
    pruned_latency = benchmark_latency_ms(pruned_model, example_x)

    finetune_with_masks(pruned_model, train_loader, masks, epochs=2, lr=1e-3, device=device)
    finetuned_acc = evaluate_accuracy(pruned_model, val_loader, device=device)
    finetuned_latency = benchmark_latency_ms(pruned_model, example_x)

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
