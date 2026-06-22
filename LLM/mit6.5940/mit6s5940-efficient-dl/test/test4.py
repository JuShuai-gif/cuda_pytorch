"""Lecture 03 pruning mini demos.

这个脚本把 lecture-03 中的剪枝代码拆成多个小 demo：
1. demo_01_tensor_pruning: 单个权重张量的幅度剪枝。
2. demo_02_global_pruning: 对整个模型做全局非结构化剪枝。
3. demo_03_sensitivity_scan: 逐层敏感度扫描，观察哪一层更怕剪。
4. demo_04_channel_importance: 计算 Conv2d 输出通道重要性，这是结构化剪枝的第一步。

运行方式：
    python test/test4.py
"""

# from __future__ import annotations 必须放在文件最前面（除文档字符串外）。
# 作用：把所有类型注解延迟成字符串求值，这样可以在 Python 3.9 里直接用
# tuple[str, nn.Module]、dict[str, torch.Tensor] 这类“新式泛型写法”而不报错。
from __future__ import annotations

import copy  # 深拷贝模型，用于敏感度扫描时互不干扰地试剪每一层
import time  # 测量推理延迟（perf_counter 高精度计时）
from dataclasses import dataclass  # 用极简语法定义“只装数据”的结构体
from typing import Iterable  # 类型注解：表示“可迭代对象”

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset  # 把张量包成可迭代的批数据


# -----------------------------------------------------------------------------
# 公共工具：构造一个很小的 CNN 和 synthetic dataloader，保证 demo 不依赖外部数据集。
# -----------------------------------------------------------------------------
class TinyCNN(nn.Module):
    """用于剪枝演示的小型 CNN。

    结构很浅，只为让 demo 跑得快：两层卷积 + 全局平均池化 + 一个全连接分类头。
    输入约定为 (N, 3, 32, 32) 的图片，输出为 (N, 10) 的 logits（10 类）。
    """

    def __init__(self) -> None:
        super().__init__()
        # features：负责把原始像素抽成 32 维特征向量。
        self.features = nn.Sequential(
            nn.Conv2d(
                3, 16, 3, padding=1
            ),  # 3 通道 -> 16 通道，3x3 卷积，padding=1 保持空间尺寸不变
            nn.ReLU(),  # 非线性激活：把负值截断为 0
            nn.Conv2d(16, 32, 3, padding=1),  # 16 通道 -> 32 通道
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 不论输入多大，都池化成 1x1 -> 每通道压成一个标量
        )
        # classifier：把 32 维特征映射到 10 个类别的得分。
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features 输出形状为 (N, 32, 1, 1)，flatten(1) 把后三维拉平成 (N, 32)。
        x = self.features(x).flatten(1)
        return self.classifier(x)


def make_synthetic_loader(n: int = 128, batch_size: int = 32) -> DataLoader:
    """生成随机图片和随机标签，用于快速 smoke test。

    注意：这里的数据完全是噪声，模型在上面的 accuracy 约等于随机猜（~10%）。
    目的不是训练出好模型，而是验证“剪枝/评估代码流程”能跑通。
    """
    x = torch.randn(n, 3, 32, 32)  # n 张 3 通道、32x32 的随机图片
    y = torch.randint(0, 10, (n,))  # n 个 [0, 10) 的随机整数标签
    # shuffle=True：每个 epoch 打乱顺序，更接近真实训练场景。
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


# -----------------------------------------------------------------------------
# 公共工具：剪枝、稀疏度、评估、延迟测量。
# -----------------------------------------------------------------------------
def magnitude_prune_tensor(weight: torch.Tensor, sparsity: float):
    """对单个权重张量做幅度剪枝，返回剪枝后的权重和二值 mask。

    核心思想（magnitude pruning）：绝对值越小的权重，对输出影响越小，越值得删掉。
    sparsity 表示“要置零的比例”，例如 0.5 表示把幅值最小的 50% 权重清零。

    返回值：
        pruned_weight：与输入同形状，被剪掉的位置为 0。
        mask：布尔张量，True 表示保留、False 表示剪掉。
    """
    # 校验稀疏度参数：必须落在 [0, 1] 闭区间内。
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

    # 边界情况：稀疏度为 0，不剪任何权重，mask 全 True。
    if sparsity == 0.0:
        mask = torch.ones_like(weight, dtype=torch.bool)
        return weight.clone(), mask

    # 边界情况：稀疏度为 1，剪掉全部权重，mask 全 False。
    if sparsity == 1.0:
        mask = torch.zeros_like(weight, dtype=torch.bool)
        return torch.zeros_like(weight), mask

    # 1) 展平成一维并取绝对值，得到每个权重的“重要性分数”。
    #    detach() 切断梯度，避免阈值计算被记录进计算图。
    flat = weight.detach().abs().flatten()
    # 2) 计算需要置零的权重个数 k。
    #    max(k, 1)：防止小张量 + 小稀疏度时算出 k=0，导致 kthvalue(k=0) 报错。
    k = max(int(sparsity * flat.numel()), 1)  # 需要置零的权重个数（至少 1 个）
    # 3) kthvalue 返回“第 k 小”的值，作为剪枝阈值。
    threshold = torch.kthvalue(flat, k).values  # 第 k 小的幅值

    # 4) 保留严格大于阈值的权重；其余位置乘 0 被清零。
    mask = weight.detach().abs() > threshold
    return weight * mask.to(weight.dtype), mask


def prunable_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """遍历模型中可剪枝的 Conv2d/Linear 层。

    named_modules() 会递归列出所有子模块（含容器本身），这里只筛出带权重矩阵、
    适合做幅度剪枝的卷积层和全连接层；ReLU、池化层等没有权重，直接跳过。
    用 yield 实现惰性遍历，调用方可以边遍历边处理。
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield name, module


@torch.no_grad()
def global_magnitude_prune(
    model: nn.Module, sparsity: float
) -> dict[str, torch.Tensor]:
    """对整个模型做全局非结构化幅度剪枝，原地修改模型并返回 mask。

    举例：假设有两层 conv1.weight=[0.1, 0.2, 0.5, 0.05], fc1.weight=[0.8, 0.01, 0.3, 0.4]
    sparsity=0.5 时：
        all_scores = [0.1, 0.2, 0.5, 0.05, 0.8, 0.01, 0.3, 0.4]  （8个数）
        k = int(0.5*8) = 4，排序找第4小 -> 0.2，所以 threshold=0.2
        conv1: [0.1,0.2,0.5,0.05] > 0.2? -> [F,F,T,F] -> 保留 0.5，其余置零
        fc1:   [0.8,0.01,0.3,0.4] > 0.2? -> [T,F,T,T] -> 保留 0.8/0.3/0.4
        全局恰好 50% 为零，但 conv1 被剪了 75%，fc1 只剪了 25%。
    """
    # 1) 获取所有可剪枝层；没有可剪层时直接返回空字典。
    named_modules = list(prunable_modules(model))
    print(
        f"Found {len(named_modules)} prunable layers: {[name for name, _ in named_modules]}"
    )
    if not named_modules:
        return {}

    # 2) 把所有层的权重绝对值拼成一个长向量，做“全局统一排序”。
    #    这正是全局剪枝与逐层剪枝的本质区别：阈值是全局唯一的一个数。
    all_scores = torch.cat(
        [m.weight.detach().abs().flatten() for _, m in named_modules]
    )
    k = int(sparsity * all_scores.numel())  # 全局需要置零的权重个数
    # 3) 找到第 k 小的幅值作为全局阈值；
    #    k 超出总个数时（sparsity 接近 1）设为无穷大，等价于全部剪掉；
    #    max(k, 1) 避免 kthvalue(k=0) 报错。
    threshold = (
        torch.inf
        if k >= all_scores.numel()
        else torch.kthvalue(all_scores, max(k, 1)).values
    )

    # 4) 逐层应用同一个阈值：mul_ 原地把不达标的权重清零，并记录 mask 供后续恢复。
    masks: dict[str, torch.Tensor] = {}
    for name, module in named_modules:
        mask = module.weight.detach().abs() > threshold
        module.weight.mul_(mask.to(module.weight.dtype))
        masks[name] = mask
    return masks


@torch.no_grad()
def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """微调后重新应用 mask，防止被剪权重恢复成非零。

    微调时即使把梯度清零，优化器的动量(momentum)/权重衰减仍可能让被剪权重
    偏离 0。每个 step 后再乘一遍 mask，是保证“稀疏结构不被破坏”的兜底手段。
    """
    # 先建立 名字 -> 模块 的映射，方便按 mask 的 key 快速定位对应层。
    module_dict = dict(model.named_modules())
    for name, mask in masks.items():
        module = module_dict[name]
        # 把 mask 对齐到权重所在的设备和 dtype，再原地相乘清零。
        module.weight.mul_(mask.to(module.weight.device, module.weight.dtype))


@torch.no_grad()
def sparsity_of_prunable_weights(model: nn.Module) -> float:
    """统计 Conv2d/Linear 权重中的实际稀疏度（零值占比）。"""
    total = 0  # 权重总个数
    zeros = 0  # 值为 0 的权重个数
    for _, module in prunable_modules(model):
        w = module.weight.detach()
        total += w.numel()
        zeros += int((w == 0).sum())
    # max(total, 1) 防止模型没有可剪层时出现除零。
    return zeros / max(total, 1)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> float:
    """在 synthetic loader 上计算 top-1 accuracy。随机数据只用于验证代码流程。"""
    model.eval()  # 切到评估模式：关闭 dropout、固定 BN 统计量
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)  # 取每行最大 logit 的下标作为预测类别
        correct += int((pred == y).sum())  # 累加预测正确的样本数
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def benchmark_latency_ms(
    model: nn.Module, example_input: torch.Tensor, warmup: int = 5, runs: int = 20
):
    """CPU 简易延迟测试，返回 mean/p50/p95（单位毫秒）。"""
    model.eval()
    # 预热：前几次推理会触发各种缓存/惰性初始化，不计入正式测量。
    for _ in range(warmup):
        model(example_input)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()  # 高精度计时起点
        model(example_input)
        times.append((time.perf_counter() - t0) * 1000)  # 秒转毫秒
    t = torch.tensor(times)
    return {
        "mean_ms": float(t.mean()),  # 平均延迟
        "p50_ms": float(t.quantile(0.50)),  # 中位数延迟
        "p95_ms": float(t.quantile(0.95)),  # 95 分位延迟（反映长尾）
    }


# -----------------------------------------------------------------------------
# Demo 1：单个 tensor 的幅度剪枝。
# 作用：理解 magnitude pruning 的最小单元：小权重被置零，大权重保留。
# -----------------------------------------------------------------------------
def demo_01_tensor_pruning() -> None:
    w = torch.randn(64, 64)  # 随机构造一个 64x64 权重矩阵
    w_pruned, mask = magnitude_prune_tensor(w, sparsity=0.5)  # 剪掉幅值最小的 50%
    print("\n[Demo 1] 单 tensor 幅度剪枝")
    # 实际稀疏度可能与 50% 略有偏差（取决于是否有重复幅值）。
    print(f"target sparsity=50%, actual sparsity={(w_pruned == 0).float().mean():.2%}")
    print(f"mask dtype={mask.dtype}, kept weights={int(mask.sum())}/{mask.numel()}")


# -----------------------------------------------------------------------------
# Demo 2：全局非结构化剪枝。
# 作用：理解“全局剪 50%”不是“每层各剪 50%”，而是所有权重一起排序。
# -----------------------------------------------------------------------------
def demo_02_global_pruning() -> None:
    model = TinyCNN()
    before = sparsity_of_prunable_weights(model)  # 剪枝前稀疏度（应接近 0）
    masks = global_magnitude_prune(model, sparsity=0.5)  # 全局剪 50%
    after = sparsity_of_prunable_weights(model)  # 剪枝后稀疏度（应接近 50%）
    print("\n[Demo 2] 全局非结构化剪枝")
    print(f"sparsity before={before:.2%}, after={after:.2%}")
    print("mask layers:", list(masks.keys()))


# -----------------------------------------------------------------------------
# Demo 3：逐层敏感度扫描。
# 作用：每次只剪一层，观察 accuracy drop，用于决定不同层剪多少。
# -----------------------------------------------------------------------------
@dataclass
class SensitivityPoint:
    """敏感度扫描的一条记录：某层在某稀疏度下剪枝后得到的精度。"""

    layer: str  # 层名
    sparsity: float  # 该层试剪的稀疏度
    accuracy: float  # 仅剪这一层后模型的精度


def layerwise_sensitivity_scan(
    model: nn.Module,
    val_loader: DataLoader,
    sparsities: tuple[float, ...] = (0.3, 0.6),
    device: str = "cpu",
) -> list[SensitivityPoint]:
    """逐层敏感度扫描：每次只剪一层，记录精度下降幅度。

    思路：固定其它层不动，单独把某层剪到某稀疏度，看精度掉多少。
    掉得多 => 该层“敏感”，应少剪；掉得少 => 该层“冗余”，可多剪。
    """
    results: list[SensitivityPoint] = []
    # 先测未剪枝时的基线精度，作为后续 drop 的参照。
    baseline_acc = evaluate_accuracy(model, val_loader, device=device)
    for layer_name, _ in prunable_modules(model):
        for sparsity in sparsities:
            # 深拷贝一份模型，保证每次试验互不影响（不污染原模型）。
            trial = copy.deepcopy(model).to(device)
            module = dict(trial.named_modules())[layer_name]
            with torch.no_grad():
                # 只对当前这一层做幅度剪枝并写回。
                pruned_w, _ = magnitude_prune_tensor(module.weight, sparsity)
                module.weight.copy_(pruned_w)
            acc = evaluate_accuracy(trial, val_loader, device=device)
            results.append(SensitivityPoint(layer_name, sparsity, acc))
            print(
                f"layer={layer_name:20s} sparsity={sparsity:.1f} "
                f"acc={acc:.4f} drop={baseline_acc - acc:.4f}"  # drop = 基线 - 当前
            )
    return results


def demo_03_sensitivity_scan() -> None:
    print("\n[Demo 3] 逐层敏感度扫描")
    model = TinyCNN()
    val_loader = make_synthetic_loader(n=96, batch_size=32)
    layerwise_sensitivity_scan(model, val_loader)


# -----------------------------------------------------------------------------
# Demo 4：通道重要性排序。
# 作用：结构化剪枝前先找出哪些输出通道更重要；这里只排序，不真正改模型结构。
# -----------------------------------------------------------------------------
@torch.no_grad()
def conv_out_channel_importance(conv: nn.Conv2d) -> torch.Tensor:
    """计算 Conv2d 每个输出通道(filter)的重要性分数，返回形状 [out_channels]。

    原理（出自 Pruning Filters for Efficient ConvNets, Li et al. 2017）：
    - 权重形状为 [out_channels, in_channels, kh, kw]，第 0 维的每个切片
      weight[o]（形状 [in_channels, kh, kw]）就是一个独立的 filter，
      它单独负责生成第 o 个输出特征图(feature map)。
    - L2 范数衡量该 filter 权重的整体“能量/幅度”：
        范数大  -> 权重普遍较大 -> 输出响应强 -> 对后续层影响大 -> 重要，保留；
        范数≈0  -> 权重几乎全是小值 -> 输出≈0 -> 冗余，可整条通道剪掉。
    - 这是结构化剪枝的核心：以“整个输出通道”为单位评估，剪掉后能直接缩小
      out_channels，得到真正更小更快的稠密卷积层（区别于只置零的非结构化剪枝）。
    - 局限：只看权重大小，未考虑真实数据下的激活分布；更精细的方法会用 BN 的
      缩放系数 γ、激活统计或 Taylor 展开来估计删除通道对 loss 的实际影响。
    """
    # 形状变化：[out, in, kh, kw] --flatten(1)--> [out, in*kh*kw]
    #          每一行 = 一个输出通道的全部参数；
    #          --norm(p=2, dim=1)--> [out]，逐行求 L2 范数 sqrt(Σ wᵢ²)。
    # detach() 切断梯度，纯数值计算。
    return conv.weight.detach().flatten(1).norm(p=2, dim=1)


@torch.no_grad()
def select_conv_out_channels(conv: nn.Conv2d, keep_ratio: float):
    scores = conv_out_channel_importance(conv)  # 每个输出通道的重要性
    keep = max(1, int(scores.numel() * keep_ratio))  # 要保留的通道数（至少 1）
    # topk 取分数最高的 keep 个通道，再排序使索引按从小到大排列，便于阅读。
    keep_idx = torch.topk(scores, keep).indices.sort().values
    # 剩下未被保留的通道索引即为待剪通道。
    prune_idx = torch.tensor(
        [i for i in range(scores.numel()) if i not in set(keep_idx.tolist())]
    )
    return keep_idx, prune_idx, scores


def demo_04_channel_importance() -> None:
    print("\n[Demo 4] Conv2d 输出通道重要性排序")
    conv = nn.Conv2d(3, 16, 3, padding=1)  # 16 个输出通道
    keep_idx, prune_idx, scores = select_conv_out_channels(conv, keep_ratio=0.5)
    print(
        "importance scores:", [round(float(x), 4) for x in scores[:5]]
    )  # 仅展示前 5 个
    print("keep channels:", keep_idx.tolist())
    print("prune channels:", prune_idx.tolist())


if __name__ == "__main__":
    torch.manual_seed(0)  # 固定随机种子，保证每次运行结果可复现
    demo_01_tensor_pruning()
    demo_02_global_pruning()
    demo_03_sensitivity_scan()
    demo_04_channel_importance()
