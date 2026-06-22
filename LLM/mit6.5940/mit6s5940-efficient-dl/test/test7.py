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
from torch.nn.utils import prune


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
    t = current_step / total_steps
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
        self.current_step = 0

        # 初始化所有可剪枝参数的 mask。
        # prune.identity 不会立即剪权重，只是注册 weight_mask 和 weight_orig。
        self.prune_params = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.identity(module, 'weight')
                self.prune_params.append((module, 'weight'))

    def step(self):
        """每个训练 step 后调用一次，按 schedule 更新当前稀疏度。"""
        target_sparsity = gradual_pruning_schedule(
            self.current_step,
            self.total_steps,
            self.initial_sparsity,
            self.final_sparsity,
        )

        # 对每个可剪层按 L1 magnitude 更新非结构化 mask。
        for module, name in self.prune_params:
            prune.l1_unstructured(module, name=name, amount=target_sparsity)

        self.current_step += 1

    def remove_masks(self):
        """训练完成后固化稀疏结构。

        prune.remove 会把 weight_orig * weight_mask 写回 weight，
        同时移除 reparameterization，便于保存和部署。
        """
        for module, name in self.prune_params:
            prune.remove(module, name)
