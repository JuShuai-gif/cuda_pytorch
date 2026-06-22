"""
PTQ 流水线 + 使用 FakeQuantize 与 STE 的 QAT（第 06 讲）
========================================================

实现训练后量化（PTQ）、使用 FakeQuantize 与直通估计器（STE）的
量化感知训练（QAT）、FP32 基线对比以及混合精度模拟（不同层使用
不同位宽）。

核心概念：
  - PTQ: 校准激活值范围，然后量化权重与激活值
  - QAT: 插入 FakeQuantize 节点；使用 STE 进行训练以恢复精度
  - STE: 梯度在通过取整操作时保持不变（反向传播为恒等映射）
  - 混合精度: 为不同层分配不同的位宽
  - 对比 FP32 基线 vs PTQ vs QAT 的准确率

所有计算均在 CPU 上运行；无需 GPU。
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 数据集
NUM_CLASSES: int = 10  # 分类类别数
INPUT_CHANNELS: int = 1  # 类 MNIST 灰度图，单通道
IMAGE_SIZE: int = 28  # 输入图像的空间尺寸
NUM_TRAIN: int = 4096  # 训练样本数
NUM_TEST: int = 1024  # 测试样本数
CALIB_SAMPLES: int = 512  # 用于校准的样本数

# 训练
BATCH_SIZE: int = 128  # 批量大小
FP32_EPOCHS: int = 8  # FP32 模型训练 epoch 数
QAT_EPOCHS: int = 5  # QAT 微调 epoch 数
LR: float = 0.01  # 学习率

# 量化
DEFAULT_BITS: int = 8  # PTQ/QAT 的默认位宽
SEED: int = 42  # 随机种子，保证可复现性

# 混合精度的逐层位宽配置
MIXED_PRECISION_CONFIG: Dict[str, int] = {
    "conv1": 8,
    "conv2": 4,
    "fc1": 4,
    "fc2": 8,
}

# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


def _create_synthetic_dataset(
    n: int,
    c: int = INPUT_CHANNELS,
    h: int = IMAGE_SIZE,
    w: int = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成可复现的合成图像和随机标签。

    参数:
        n:           样本数量。
        c:           输入通道数（灰度图为 1）。
        h, w:        空间维度。
        num_classes: 标签类别数。
        seed:        随机种子，保证可复现性。

    返回:
        (images, labels) 元组。
    """
    g = torch.Generator()
    g.manual_seed(seed)
    # 生成范围约为 [0, 1] 的图像
    images = torch.randn(n, c, h, w, generator=g) * 0.5 + 0.5
    images = images.clamp(0.0, 1.0)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    return images, labels


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """用于类 MNIST 28x28 灰度分类的紧凑 CNN。

    网络结构:
        Conv2d(1,  16, 3, padding=1) -> ReLU -> MaxPool2d(2)   # 14x14
        Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)   #  7x7
        Flatten -> Linear(32*7*7, 128) -> ReLU
        Linear(128, 10)

    参数:
        num_classes: 输出类别数（默认为 10）。
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    @property
    def prunable_layers(self) -> List[nn.Module]:
        """返回权重可以被量化的层（Conv2d, Linear）。"""
        return [self.conv1, self.conv2, self.fc1, self.fc2]


# ---------------------------------------------------------------------------
# 训练与评估
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    lr: float = LR,
) -> float:
    """训练模型一个 epoch。

    参数:
        model:  PyTorch nn.Module 模型。
        loader: 产生 (images, labels) 批次的 DataLoader。
        lr:     学习率。

    返回:
        该 epoch 的平均训练损失。
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """在给定 DataLoader 上评估 top-1 准确率。

    参数:
        model:  PyTorch nn.Module 模型。
        loader: 产生 (images, labels) 批次的 DataLoader。

    返回:
        [0.0, 1.0] 范围内的准确率浮点数。
    """
    model.eval()
    correct = 0
    total = 0

    for xb, yb in loader:
        logits = model(xb)
        # 取最大 logit 对应的索引作为预测类别
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return correct / max(total, 1)


def count_params(model: nn.Module) -> int:
    """返回模型的总参数数量。

    参数:
        model: PyTorch nn.Module 模型。

    返回:
        总参数数量。
    """
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 量化原语
# ---------------------------------------------------------------------------


def compute_scale_zp(
    tensor: torch.Tensor,
    bits: int,
) -> Tuple[float, int, int, int]:
    """为张量计算非对称仿射量化参数。

    映射公式为:
        scale  = (x_max - x_min) / (2^bits - 1)
        zp     = round(-x_min / scale)   [截断到 [0, 2^bits-1]]

    参数:
        tensor: 任意形状的 float32 张量。
        bits:   位宽（例如 8, 4, 2）。

    返回:
        (scale, zero_point, qmin, qmax) 元组。
    """
    if bits <= 0:
        raise ValueError(f"bits 必须为正数；当前值为 {bits}")

    qmin: int = 0  # 量化范围最小值
    qmax: int = int(2**bits - 1)  # 量化范围最大值

    x_min = tensor.min().item()
    x_max = tensor.max().item()

    # 当所有值相同时，返回默认参数
    if x_max == x_min:
        return 1.0, 0, qmin, qmax

    # 计算缩放因子: 将浮点范围映射到整数范围
    scale = (x_max - x_min) / (qmax - qmin)
    # 零点: 浮点 0.0 对应的量化值
    zp_f = round(-x_min / scale)
    zp = max(qmin, min(qmax, int(zp_f)))

    return float(scale), zp, qmin, qmax


def fake_quantize(
    x: torch.Tensor,
    scale: float,
    zp: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """将浮点张量量化后再立即反量化。

    这模拟了真实整数量化的效果，但并没有将张量真正转换为整数数据类型——
    返回的张量仍然是 float32，但带有量化噪声。

    参数:
        x:     float32 输入张量。
        scale: 量化缩放因子。
        zp:    零点。
        qmin:  最小量化值。
        qmax:  最大量化值。

    返回:
        与 x 形状相同的伪量化 float32 张量。
    """
    # 量化到整数
    x_q = torch.round(x / scale + zp)
    x_q = torch.clamp(x_q, qmin, qmax)
    # 反量化回浮点
    x_dq = (x_q - zp) * scale
    return x_dq


def fake_quantize_ste(
    x: torch.Tensor,
    scale: float,
    zp: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """使用直通估计器（STE）进行伪量化，解决梯度无法通过取整操作的问题。

    前向:  x_fq = fake_quantize(x, scale, zp, qmin, qmax)
    反向:  梯度通过取整操作时不发生变化（恒等映射）。

    实现使用 ``.detach()`` 技巧:
        output = x + (x_fq - x).detach()
    这意味着前向 = x_fq，反向 = dL/d(x_fq) * 1 = dL/d(x_fq)。

    参数:
        x:     float32 输入张量。
        scale: 量化缩放因子。
        zp:    零点。
        qmin:  最小量化值。
        qmax:  最大量化值。

    返回:
        带 STE 梯度的伪量化张量（形状与 x 相同）。
    """
    x_fq = fake_quantize(x, scale, zp, qmin, qmax)
    # STE: 前向路径使用 x_fq，反向路径使用 x（恒等梯度）
    return x + (x_fq - x).detach()


# 同时提供一个 torch.autograd.Function 版本用于教学说明
class _FakeQuantizeSTE(torch.autograd.Function):
    """自定义 autograd Function，实现带 STE 的 FakeQuantize。

    这与上述 ``.detach()`` 技巧版本在语义上等价，但演示了 STE 如何在
    autograd 层面实现。
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        scale: float,
        zp: int,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        """前向传播: 将输入量化后再反量化。"""
        # 量化到整数
        x_q = torch.round(x / scale + zp)
        x_q = torch.clamp(x_q, qmin, qmax)
        # 反量化
        x_dq = (x_q - zp) * scale
        return x_dq

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        """反向传播: STE——梯度保持不变地通过。"""
        return grad_output, None, None, None, None


def _ste_autograd_fn(
    x: torch.Tensor,
    scale: float,
    zp: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """STE 的 autograd.Function 版本的便捷封装。"""
    # 将标量作为张量传入，使 autograd.Function 能够正确接收
    return _FakeQuantizeSTE.apply(x, scale, zp, qmin, qmax)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# FakeQuantize 模块 (nn.Module)
# ---------------------------------------------------------------------------


class FakeQuantize(nn.Module):
    """用于 QAT 的可训练 FakeQuantize 模块。

    当 ``observer_enabled=True`` 时，模块会收集最小/最大值统计信息，
    并在每次前向传播时重新计算 scale/zp。一旦校准完成，应将
    ``observer_enabled`` 设置为 False 以冻结量化参数。

    参数:
        bits: 量化位宽（默认为 8）。
    """

    def __init__(self, bits: int = DEFAULT_BITS) -> None:
        super().__init__()
        self.bits = bits
        self.qmin: int = 0
        self.qmax: int = int(2**bits - 1)

        # 可学习/可冻结的量化参数（存储为 buffer，不参与梯度更新）
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0))

        self.observer_enabled: bool = True  # 观察者模式: 是否更新 scale/zp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            # 在校准阶段，根据输入更新 scale 和 zero_point
            self._observe(x)

        # 使用 STE 进行伪量化
        return fake_quantize_ste(
            x,
            self.scale.item(),
            int(self.zero_point.item()),
            self.qmin,
            self.qmax,
        )

    def _observe(self, x: torch.Tensor) -> None:
        """根据观测到的张量范围更新 scale 和 zero_point。"""
        x_min = x.min().item()
        x_max = x.max().item()
        if x_max == x_min:
            return
        new_scale = (x_max - x_min) / (self.qmax - self.qmin)
        new_zp_f = round(-x_min / new_scale)
        new_zp = max(self.qmin, min(self.qmax, int(new_zp_f)))
        self.scale.fill_(new_scale)
        self.zero_point.fill_(new_zp)

    def freeze(self) -> None:
        """禁用观察者；量化参数此后被冻结，不再更新。"""
        self.observer_enabled = False


# ---------------------------------------------------------------------------
# QAT 层封装
# ---------------------------------------------------------------------------


class QATConv2d(nn.Module):
    """在 Conv2d 前后分别用 FakeQuantize 封装权重和激活值，用于 QAT。

    权重 FakeQuantize 应用于卷积之前。
    激活值 FakeQuantize 应用于卷积之后。

    参数:
        conv: 要封装的已有 Conv2d 层。
        bits: 权重和激活值量化的位宽。
    """

    def __init__(self, conv: nn.Conv2d, bits: int = DEFAULT_BITS) -> None:
        super().__init__()
        self.conv = conv
        # 权重量化器（作用于权重张量）
        self.weight_fq = FakeQuantize(bits)
        # 激活值量化器（作用于卷积的输出）
        self.act_fq = FakeQuantize(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在卷积前对权重进行伪量化
        w_q = self.weight_fq(self.conv.weight)
        # 使用伪量化后的权重进行函数式 conv2d
        out = nn.functional.conv2d(
            x,
            w_q,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        # 对输出激活值进行伪量化
        out = self.act_fq(out)
        return out


class QATLinear(nn.Module):
    """在 Linear 前后分别用 FakeQuantize 封装权重和激活值，用于 QAT。

    权重 FakeQuantize 应用于线性变换之前。
    激活值 FakeQuantize 应用于线性变换之后。

    参数:
        linear: 要封装的已有 Linear 层。
        bits:   权重和激活值量化的位宽。
    """

    def __init__(self, linear: nn.Linear, bits: int = DEFAULT_BITS) -> None:
        super().__init__()
        self.linear = linear
        self.weight_fq = FakeQuantize(bits)
        self.act_fq = FakeQuantize(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在矩阵乘法前对权重进行伪量化
        w_q = self.weight_fq(self.linear.weight)
        out = nn.functional.linear(x, w_q, self.linear.bias)
        # 对输出激活值进行伪量化
        out = self.act_fq(out)
        return out


# ---------------------------------------------------------------------------
# PTQ 流水线
# ---------------------------------------------------------------------------


def calibrate_activation_ranges(
    model: nn.Module,
    loader: DataLoader,
    bits: int = DEFAULT_BITS,
    num_batches: int = 8,
) -> Dict[str, Tuple[float, int, int, int]]:
    """通过前向传播校准激活值的量化参数。

    对于每个可量化层（Conv2d, Linear），前向钩子捕获输出张量，
    并根据在多个批次中观测到的最小/最大值范围计算 (scale, zp, qmin, qmax)。

    参数:
        model:       要校准的 FP32 模型。
        loader:      校准数据的 DataLoader。
        bits:        目标位宽。
        num_batches: 用于校准的最大批次数。

    返回:
        以层名为键的字典，映射到 (scale, zp, qmin, qmax)。
    """
    model.eval()
    # 累积每层的全局最小/最大值
    layer_min: Dict[str, float] = {}
    layer_max: Dict[str, float] = {}

    # 建立模块到层名的映射
    module_name_map: Dict[nn.Module, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            module_name_map[mod] = name

    def _hook(
        mod: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        """前向钩子: 记录每个可量化层输出的最小/最大值。"""
        name = module_name_map.get(mod)
        if name is None:
            return
        v_min = out.min().item()
        v_max = out.max().item()
        if name not in layer_min:
            layer_min[name] = v_min
            layer_max[name] = v_max
        else:
            layer_min[name] = min(layer_min[name], v_min)
            layer_max[name] = max(layer_max[name], v_max)

    # 注册钩子
    handles = []
    for mod, name in module_name_map.items():
        handles.append(mod.register_forward_hook(_hook))

    # 校准前向传播
    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            _ = model(xb)

    # 移除钩子
    for h in handles:
        h.remove()

    # 构建参数字典
    param_map: Dict[str, Tuple[float, int, int, int]] = {}
    qmin, qmax = 0, 2**bits - 1
    for name in module_name_map.values():
        v_min = layer_min.get(name, 0.0)
        v_max = layer_max.get(name, 0.0)
        if v_max == v_min:
            param_map[name] = (1.0, 0, qmin, qmax)
        else:
            scale = (v_max - v_min) / (qmax - qmin)
            zp = int(round(-v_min / scale))
            zp = max(qmin, min(qmax, zp))
            param_map[name] = (float(scale), zp, qmin, qmax)

    return param_map


@torch.no_grad()
def ptq_quantize_model(
    model: nn.Module,
    act_params: Dict[str, Tuple[float, int, int, int]],
    bits: int = DEFAULT_BITS,
) -> nn.Module:
    """对模型权重应用训练后量化，并插入激活值量化钩子。

    此函数原地量化权重（将浮点权重替换为其量化-反量化后的版本），
    并在每个可量化层的输出端附加激活值量化器。

    参数:
        model:      已训练的 FP32 模型。
        act_params: 校准得到的逐层激活量化参数。
        bits:       权重量化的位宽。

    返回:
        同一模型（原地修改），权重已量化且已附加激活值量化器。
    """
    model.eval()

    # 收集所有可量化层
    layer_map: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            layer_map[name] = mod

    for name, mod in layer_map.items():
        # ---- 量化权重 ----
        w = mod.weight.data
        scale_w, zp_w, qmin_w, qmax_w = compute_scale_zp(w, bits)
        # 将量化后反量化的权重写回原张量
        w_q = fake_quantize(w, scale_w, zp_w, qmin_w, qmax_w)
        mod.weight.data.copy_(w_q)

    # 通过前向钩子附加激活值量化器
    _attach_activation_quantizers(model, act_params)

    return model


def _attach_activation_quantizers(
    model: nn.Module,
    act_params: Dict[str, Tuple[float, int, int, int]],
) -> None:
    """附加对激活值输出进行伪量化的前向钩子。

    钩子存储在模型上，以便后续可以移除。

    参数:
        model:      要插桩的模型（原地修改）。
        act_params: 逐层的 (scale, zp, qmin, qmax) 元组。
    """
    if not hasattr(model, "_quant_hooks"):
        model._quant_hooks = []  # type: ignore[attr-defined]

    layer_map: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            layer_map[name] = mod

    for name, mod in layer_map.items():
        if name not in act_params:
            continue
        scale, zp, qmin, qmax = act_params[name]

        def _make_hook(s: float, z: int, qn: int, qx: int):
            """创建闭包，为特定层生成固定的量化钩子。"""

            def _hook(
                mod: nn.Module,
                inp: Tuple[torch.Tensor, ...],
                out: torch.Tensor,
            ) -> torch.Tensor:
                return fake_quantize(out, s, z, qn, qx)

            return _hook

        hook = mod.register_forward_hook(_make_hook(scale, zp, qmin, qmax))
        model._quant_hooks.append(hook)  # type: ignore[attr-defined]


def remove_quant_hooks(model: nn.Module) -> None:
    """从模型中移除所有激活值量化钩子。

    参数:
        model: 已附加量化钩子的模型。
    """
    if hasattr(model, "_quant_hooks"):
        for h in model._quant_hooks:  # type: ignore[attr-defined]
            h.remove()
        model._quant_hooks.clear()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# QAT 模型构建器
# ---------------------------------------------------------------------------


def build_qat_model(
    fp32_model: nn.Module,
    bits: int = DEFAULT_BITS,
) -> nn.Module:
    """将 FP32 模型转换为 QAT 就绪模型，用 FakeQuantize 封装 Conv2d/Linear 层。

    这会创建一个新的 ``nn.Sequential``，结构与原模型镜像，
    但使用 ``QATConv2d`` / ``QATLinear`` 封装。所有原始训练权重
    会复制到 QAT 层中。

    参数:
        fp32_model: 已训练的 FP32 SimpleCNN 模型。
        bits:       所有量化器的位宽。

    返回:
        新的 QAT 模型 (nn.Module)，可进行量化感知训练。
    """
    qat = SimpleCNN(num_classes=NUM_CLASSES)

    with torch.no_grad():
        # 从 fp32_model 复制权重到 qat 模型
        qat.load_state_dict(fp32_model.state_dict(), strict=False)

    # 将 Conv2d / Linear 替换为 QAT 封装版本
    _replace_with_qat(qat, bits)

    return qat


def _replace_with_qat(model: nn.Module, bits: int) -> None:
    """原地将 Conv2d 和 Linear 替换为 QAT 封装版本。

    参数:
        model: 要原地修改的模型。
        bits:  量化器的位宽。
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(model, name, QATConv2d(child, bits))
        elif isinstance(child, nn.Linear):
            setattr(model, name, QATLinear(child, bits))
        else:
            # 递归处理子模块
            _replace_with_qat(child, bits)


def freeze_qat_observers(model: nn.Module) -> None:
    """冻结 QAT 模型中所有 FakeQuantize 观察者。

    调用此函数后，量化参数被固定，不再根据输入统计信息更新。

    参数:
        model: 要冻结的 QAT 模型（原地修改）。
    """
    for mod in model.modules():
        if isinstance(mod, FakeQuantize):
            mod.freeze()


def calibrate_qat(
    model: nn.Module,
    loader: DataLoader,
    num_batches: int = 8,
) -> None:
    """通过少量前向传播校准 QAT FakeQuantize 观察者。

    在校准期间，FakeQuantize.observer_enabled 为 True，因此每个模块
    会记录最小/最大激活值范围并更新其 scale/zp。
    校准完成后，所有观察者被冻结。

    参数:
        model:       包含 FakeQuantize 模块的 QAT 模型。
        loader:      校准数据的 DataLoader。
        num_batches: 校准批次数。
    """
    model.train()  # 观察者需要在训练模式下才能更新
    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            _ = model(xb)

    # 冻结所有观察者
    freeze_qat_observers(model)


# ---------------------------------------------------------------------------
# 混合精度模拟
# ---------------------------------------------------------------------------


def build_mixed_precision_model(
    fp32_model: nn.Module,
    bit_config: Dict[str, int],
) -> nn.Module:
    """构建一个按层分配位宽的 QAT 模型。

    ``bit_config`` 中列出的每层使用指定的位宽；
    未列出的层默认使用 ``DEFAULT_BITS``。

    配置示例::

        bit_config = {"conv1": 8, "conv2": 4, "fc1": 4, "fc2": 8}

    参数:
        fp32_model: 已训练的 FP32 模型。
        bit_config: 层名到位宽的映射字典。

    返回:
        具有逐层位宽 FakeQuantize 封装的 QAT 模型。
    """
    qat = SimpleCNN(num_classes=NUM_CLASSES)
    with torch.no_grad():
        qat.load_state_dict(fp32_model.state_dict(), strict=False)

    _replace_with_qat_custom_bits(qat, bit_config)
    return qat


def _replace_with_qat_custom_bits(
    model: nn.Module,
    bit_config: Dict[str, int],
) -> None:
    """使用自定义位宽将 Conv2d/Linear 替换为 QAT 封装。

    参数:
        model:      要原地修改的模型。
        bit_config: 层名 -> 位宽映射。
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            bits = bit_config.get(name, DEFAULT_BITS)
            setattr(model, name, QATConv2d(child, bits))
        elif isinstance(child, nn.Linear):
            bits = bit_config.get(name, DEFAULT_BITS)
            setattr(model, name, QATLinear(child, bits))
        else:
            _replace_with_qat_custom_bits(child, bit_config)


# ---------------------------------------------------------------------------
# 对比与打印
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    """打印章节标题。"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------


def main() -> None:
    """运行完整的 PTQ + QAT + 混合精度流水线。"""
    torch.manual_seed(SEED)

    print_header("第 06 讲: PTQ 流水线 + 使用 FakeQuantize 与 STE 的 QAT")

    # ---- 1. 创建合成数据 ------------------------------------------------
    print("\n[1] 正在生成合成数据集 ...")
    train_images, train_labels = _create_synthetic_dataset(NUM_TRAIN)
    test_images, test_labels = _create_synthetic_dataset(NUM_TEST, seed=SEED + 1)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 用于校准的较小 loader（使用训练集的子集）
    calib_images = train_images[:CALIB_SAMPLES]
    calib_labels = train_labels[:CALIB_SAMPLES]
    calib_dataset = TensorDataset(calib_images, calib_labels)
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练集: {train_images.shape}, 测试集: {test_images.shape}")
    print(f"  校准子集: {CALIB_SAMPLES} 个样本")

    # ---- 2. 训练 FP32 基线 ---------------------------------------------
    print(f"\n[2] 正在训练 FP32 基线 ({FP32_EPOCHS} 个 epoch) ...")
    fp32_model = SimpleCNN(num_classes=NUM_CLASSES)
    fp32_params = count_params(fp32_model)
    print(f"  模型参数: {fp32_params:,}")

    for epoch in range(1, FP32_EPOCHS + 1):
        loss = train_one_epoch(fp32_model, train_loader)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2d}  loss={loss:.4f}")

    fp32_acc = evaluate_accuracy(fp32_model, test_loader)
    print(f"  FP32 基线准确率: {fp32_acc:.4f}")

    # ---- 3. PTQ: 校准 + 量化 ------------------------------------------
    print(f"\n[3] PTQ ({DEFAULT_BITS} 比特): 正在校准激活值范围 ...")
    # 深拷贝已训练的 FP32 模型用于 PTQ
    ptq_model = copy.deepcopy(fp32_model)

    act_params = calibrate_activation_ranges(
        ptq_model, calib_loader, bits=DEFAULT_BITS, num_batches=8
    )
    print(f"  已校准 {len(act_params)} 层")

    for name, (scale, zp, qmin, qmax) in act_params.items():
        print(f"    {name:<25}  scale={scale:.6f}  zp={zp:>4d}")

    print("\n  正在量化权重并插入激活值量化器 ...")
    ptq_quantize_model(ptq_model, act_params, bits=DEFAULT_BITS)

    ptq_acc = evaluate_accuracy(ptq_model, test_loader)
    print(f"  PTQ 准确率 ({DEFAULT_BITS}-bit): {ptq_acc:.4f}")

    # ---- 4. 使用 FakeQuantize + STE 的 QAT -----------------------------
    print(
        f"\n[4] 使用 FakeQuantize + STE 的 QAT ({DEFAULT_BITS}-bit, {QAT_EPOCHS} 个 epoch) ..."
    )
    qat_model = build_qat_model(fp32_model, bits=DEFAULT_BITS)

    # 校准 QAT 观察者
    print("  正在校准 QAT 观察者 ...")
    calibrate_qat(qat_model, calib_loader, num_batches=8)
    qat_acc_before = evaluate_accuracy(qat_model, test_loader)
    print(f"  QAT 微调前准确率: {qat_acc_before:.4f}")

    # QAT 微调
    print(f"  正在进行 QAT 微调 ({QAT_EPOCHS} 个 epoch) ...")
    for epoch in range(1, QAT_EPOCHS + 1):
        # 使用较小的学习率进行微调
        loss = train_one_epoch(qat_model, train_loader, lr=LR * 0.5)
        if epoch % 2 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>2d}  loss={loss:.4f}")

    qat_acc = evaluate_accuracy(qat_model, test_loader)
    print(f"  QAT 微调后准确率: {qat_acc:.4f}")

    # ---- 5. 混合精度模拟（加分项） ------------------------------------
    print("\n[5] 混合精度模拟（不同层使用不同位宽） ...")
    mp_model = build_mixed_precision_model(fp32_model, MIXED_PRECISION_CONFIG)

    # 打印每层的位宽分配情况
    print("  各层位宽分配:")
    for name, child in mp_model.named_modules():
        if isinstance(child, (QATConv2d, QATLinear)):
            w_bits = child.weight_fq.bits
            a_bits = child.act_fq.bits
            print(f"    {name:<25}  weight={w_bits}-bit, act={a_bits}-bit")

    calibrate_qat(mp_model, calib_loader, num_batches=8)
    mp_acc_before = evaluate_accuracy(mp_model, test_loader)
    print(f"  混合精度微调前准确率: {mp_acc_before:.4f}")

    print(f"  正在进行混合精度微调 ({QAT_EPOCHS} 个 epoch) ...")
    for epoch in range(1, QAT_EPOCHS + 1):
        loss = train_one_epoch(mp_model, train_loader, lr=LR * 0.5)
        if epoch % 2 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>2d}  loss={loss:.4f}")

    mp_acc = evaluate_accuracy(mp_model, test_loader)
    print(f"  混合精度微调后准确率: {mp_acc:.4f}")

    # ---- 6. STE 梯度正确性检查 ----------------------------------------
    print("\n[6] STE 梯度正确性检查 ...")
    x = torch.tensor([0.3, 1.7, -0.5], requires_grad=True)
    scale_s, zp_s, qmin_s, qmax_s = compute_scale_zp(x.detach(), 4)
    # 使用 STE 进行伪量化
    y = fake_quantize_ste(x, scale_s, zp_s, qmin_s, qmax_s)
    loss = y.sum()
    loss.backward()
    print(f"  输入:            {x.detach().tolist()}")
    print(f"  伪量化后:        {y.detach().tolist()}")
    print(f"  梯度 (STE):      {x.grad.tolist()}")
    # 使用 STE 时，梯度应全部为 1（等同于取整操作为恒等映射时的梯度）
    grad_ok = torch.allclose(x.grad, torch.ones_like(x.grad))
    print(f"  梯度与恒等映射一致: {grad_ok}")

    # ---- 7. 准确率对比摘要 --------------------------------------------
    print("\n[7] 准确率对比摘要")
    print(f"  {'':-<60}")
    print(f"  {'方法':<30} {'准确率':>10} {'相对于 FP32 的变化':>15}")
    print(f"  {'':-<60}")
    print(f"  {'FP32 基线':<30} {fp32_acc:>10.4f} {'---':>15}")
    print(
        f"  {'PTQ (' + str(DEFAULT_BITS) + '-bit)':<30} "
        f"{ptq_acc:>10.4f} "
        f"{(ptq_acc - fp32_acc) * 100:>+14.2f}%"
    )
    print(
        f"  {'QAT (' + str(DEFAULT_BITS) + '-bit)':<30} "
        f"{qat_acc:>10.4f} "
        f"{(qat_acc - fp32_acc) * 100:>+14.2f}%"
    )
    print(f"  {'混合精度':<30} {mp_acc:>10.4f} {(mp_acc - fp32_acc) * 100:>+14.2f}%")
    print(f"  {'':-<60}")

    # ---- 8. 完成 -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    print(f"  模型:       SimpleCNN ({fp32_params:,} 个参数)")
    print(f"  数据:       合成类 MNIST ({NUM_TRAIN} 训练 / {NUM_TEST} 测试)")
    print(f"  PTQ 位宽:   {DEFAULT_BITS}")
    print(f"  QAT 位宽:   {DEFAULT_BITS}")
    print(f"  混合精度位宽: {MIXED_PRECISION_CONFIG}")
    print(f"  FP32 准确率: {fp32_acc:.4f}")
    print(f"  PTQ 准确率:  {ptq_acc:.4f}  ({(ptq_acc - fp32_acc) * 100:+.2f}%)")
    print(f"  QAT 准确率:  {qat_acc:.4f}  ({(qat_acc - fp32_acc) * 100:+.2f}%)")
    print(f"  混合准确率:  {mp_acc:.4f}  ({(mp_acc - fp32_acc) * 100:+.2f}%)")
    print("=" * 70)

    print("\n第 06 讲完成。")


if __name__ == "__main__":
    main()
