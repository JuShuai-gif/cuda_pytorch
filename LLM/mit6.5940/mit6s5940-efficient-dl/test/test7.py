"""Demo: 训练过程中的渐进式剪枝。

这个脚本演示 gradual pruning：训练初期不剪或少剪，随后逐步提高稀疏度。
这种做法比“一次性剪到目标稀疏度”更稳定，因为模型有时间适应稀疏结构。

核心组件：
1. gradual_pruning_schedule: 根据当前 step 计算目标稀疏度；
2. GradualPruner: 包装 PyTorch prune API，在训练过程中更新 mask；
3. remove_masks: 训练结束后把 mask 固化到 weight 中。
"""

import torch
import torch.nn as nn
from torch.nn.utils import (
    prune,
)  # PyTorch 内置剪枝工具：自动管理 weight_mask/weight_orig


def gradual_pruning_schedule(
    current_step,
    total_steps,
    initial_sparsity=0.0,
    final_sparsity=0.9,
):
    """计算当前训练 step 的目标稀疏度。

    这里使用三次多项式 schedule：
    - 前期剪得慢，避免刚开始训练就破坏模型；
    - 后期逐渐接近 final_sparsity；
    - 是很多 pruning 论文和工程实现中的常见策略。
    """
    t = current_step / total_steps  # 训练进度，范围约 [0, 1]
    # 当 t=0 时返回 initial_sparsity；t=1 时返回 final_sparsity；中间按 (1-t)^3 平滑过渡。
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - t) ** 3


class GradualPruner:
    """训练时逐步更新剪枝 mask 的简单封装。"""

    def __init__(
        self,
        model,
        initial_sparsity=0.0,
        final_sparsity=0.9,
        total_steps=10000,
    ):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.total_steps = total_steps
        self.current_step = 0  # 训练步计数器

        # 初始化所有可剪枝参数的 mask。
        # prune.identity 不会立即剪权重，只是注册 weight_mask 和 weight_orig。
        self.prune_params = []  # 保存 (module, 参数名) 对，后续统一更新
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.identity(module, "weight")
                self.prune_params.append((module, "weight"))

    def step(self):
        """每个训练 step 后调用一次，按 schedule 把稀疏度提升到当前目标值。"""
        # 1) 根据当前进度算出本步应达到的“绝对目标稀疏度”。
        target_sparsity = gradual_pruning_schedule(
            self.current_step,
            self.total_steps,
            self.initial_sparsity,
            self.final_sparsity,
        )

        # 2) 关键：PyTorch 的 prune 是“累积式”的——对同一参数反复调用
        #    l1_unstructured 会在已有 mask 的基础上继续往下剪，稀疏度会叠加暴涨
        #    （实测几步就冲到 100%），而不是停在 target_sparsity。
        #    所以要先 prune.remove 撤销上一步的 reparametrization（把当前权重固化
        #    回普通 weight），再基于当前权重重新按 L1 幅度剪到绝对目标稀疏度。
        for module, name in self.prune_params:
            if prune.is_pruned(module):
                prune.remove(module, name)
            prune.l1_unstructured(module, name=name, amount=target_sparsity)

        self.current_step += 1  # 步数自增

    def remove_masks(self):
        """训练完成后固化稀疏结构。

        prune.remove 会把 weight_orig * weight_mask 写回 weight，
        同时移除 reparameterization，便于保存和部署。
        """
        for module, name in self.prune_params:
            prune.remove(module, name)


# -----------------------------------------------------------------------------
# 完整可运行示例：在一个小网络上做渐进式剪枝训练。
# -----------------------------------------------------------------------------
class TinyNet(nn.Module):
    """演示用的小型 CNN：2 个卷积 + 全局池化 + 全连接分类头。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@torch.no_grad()
def current_sparsity(model: nn.Module) -> float:
    """统计模型当前实际稀疏度。

    被 prune 接管后，module.weight 是由 forward hook 实时算出的
    (weight_orig * weight_mask)，直接读它统计 0 的比例即可反映当前稀疏度。
    """
    total = zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            total += w.numel()
            zeros += int((w == 0).sum())
    return zeros / max(total, 1)


def train_with_gradual_pruning():
    """完整流程：建模 -> 注册剪枝 -> 边训练边升稀疏度 -> 固化。"""
    torch.manual_seed(0)
    model = TinyNet()

    # 合成数据：8x8 单通道随机图 + 随机标签（仅用于跑通流程，不追求精度）。
    x = torch.randn(64, 1, 8, 8)
    y = torch.randint(0, 10, (64,))

    total_steps = 20
    # 关键顺序：先注册剪枝（prune.identity 把 weight 变成 weight_orig + weight_mask），
    # 再创建 optimizer。这样 model.parameters() 拿到的是 weight_orig，训练才会更新它。
    pruner = GradualPruner(
        model, initial_sparsity=0.0, final_sparsity=0.9, total_steps=total_steps
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print(f"step -1: sparsity={current_sparsity(model):.2%} (初始，未剪)")
    for step in range(total_steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        pruner.step()  # 每个训练 step 之后按 schedule 提升稀疏度
        if step % 4 == 0 or step == total_steps - 1:
            print(
                f"step {step:2d}: loss={loss.item():.4f} "
                f"sparsity={current_sparsity(model):.2%}"
            )

    print(f"\n固化前: sparsity={current_sparsity(model):.2%}")
    print(
        "固化前 conv1 含 reparam 缓存(weight_orig):",
        hasattr(model.conv1, "weight_orig"),
    )

    pruner.remove_masks()  # 训练结束，把 mask 固化进 weight、移除 reparametrization

    print(f"固化后: sparsity={current_sparsity(model):.2%}")
    print("固化后 conv1 仍有 weight_orig:", hasattr(model.conv1, "weight_orig"))
    return model


if __name__ == "__main__":
    train_with_gradual_pruning()
