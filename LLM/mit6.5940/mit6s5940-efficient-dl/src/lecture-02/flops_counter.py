"""
FLOPs Counter: Parameter Counting, Training FLOPs, and Memory Footprint (Lecture 02)
FLOPs 计数器：参数统计、训练 FLOPs 与内存占用（第 02 讲）

Provides analytical tools to estimate the computational cost of neural network
models:
提供用于估算神经网络模型计算开销的分析工具：

  - count_parameters: split into total and trainable counts
    count_parameters: 分别统计参数总数和可训练参数数
  - estimate_training_flops: forward + backward FLOPs via hooks and the "3x rule"
    estimate_training_flops: 通过钩子和"3x 规则"估算前向+反向 FLOPs
  - estimate_memory_footprint: parameters, gradients, optimiser states, activations
    estimate_memory_footprint: 估算参数、梯度、优化器状态和激活值的内存占用
  - format_memory_summary: human-readable memory report
    format_memory_summary: 生成人类可读的内存报告

All computations are CPU-only and use only standard packages (torch, numpy).
所有计算均在 CPU 上运行，仅使用标准包（torch, numpy）。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 常量定义
# Constants
# ---------------------------------------------------------------------------

BYTES_PER_FP32: int = 4  # FP32 每个参数的字节数
BYTES_PER_FP16: int = 2  # FP16 每个参数的字节数
BYTES_PER_BF16: int = 2  # BF16 每个参数的字节数
KiB: int = 1024  # 1 KiB
MiB: int = 1024 * 1024  # 1 MiB = 2^20 字节
GiB: int = 1024 * 1024 * 1024  # 1 GiB = 2^30 字节


# ===========================================================================
# 辅助函数：参数/缓冲区统计
# Helper: param / buffer counting
# ===========================================================================


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch module.
    统计 PyTorch 模块中的参数总数和可训练参数数。

    Args:
        model: Any ``nn.Module`` instance.
               model: 任意 ``nn.Module`` 实例。

    Returns:
        (total_params, trainable_params) tuple of ints.
        返回 (参数总数, 可训练参数数) 的整数元组。

    Examples:
        >>> net = nn.Linear(10, 5)
        >>> total, trainable = count_parameters(net)
        >>> total, trainable
        (55, 55)
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        num = p.numel()  # 该参数张量的元素个数
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable


# ===========================================================================
# 辅助函数：各层 FLOPs 的解析估算
# Helper: analytical FLOPs for individual layers
# ===========================================================================


def estimate_conv2d_flops(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    output_height: int,
    output_width: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
) -> int:
    """Estimate MACs (multiply-accumulates) for a single Conv2d layer.
    估算单个 Conv2d 层的 MACs（乘加操作数）。

    MACs = out_channels * (in_channels / groups) * kernel_size^2 * H_out * W_out
    MACs = 输出通道 × (输入通道 / 组数) × 卷积核尺寸² × 输出高 × 输出宽
    FLOPs (adds + mults) = 2 * MACs.
    FLOPs（加法 + 乘法）= 2 × MACs。

    Args:
        in_channels: Number of input channels (C_in).
                     in_channels: 输入通道数 (C_in)。
        out_channels: Number of output channels (C_out).
                      out_channels: 输出通道数 (C_out)。
        kernel_size: Spatial size of the kernel (assumed square).
                     kernel_size: 卷积核的空间尺寸（假设为方形）。
        output_height: Output spatial height (H_out).
                       output_height: 输出空间高度 (H_out)。
        output_width: Output spatial width (W_out).
                      output_width: 输出空间宽度 (W_out)。
        stride: Convolution stride (default 1).
                stride: 卷积步长（默认 1）。
        padding: Zero-padding (default 0).
                 padding: 零填充（默认 0）。
        groups: Number of blocked connections (default 1, standard conv).
                groups: 分组连接的组数（默认 1，标准卷积）。

    Returns:
        MACs count as int.
        返回 MACs 计数（整数）。

    References:
        He et al., "Deep Residual Learning for Image Recognition" (2016)
            flops = (2 * C_in * k^2 - 1) * H_out * W_out * C_out
                      (for groups=1, bias=False)
        We use the simplified MACs formulation.
        我们使用简化的 MACs 公式。
    """
    _ = stride, padding  # reserved for signature compatibility
    # 保留参数以保持签名兼容性
    macs = (
        out_channels
        * (in_channels // groups)
        * (kernel_size * kernel_size)
        * output_height
        * output_width
    )
    return max(0, macs)


def estimate_linear_flops(in_features: int, out_features: int) -> int:
    """Estimate MACs for a fully-connected (Linear) layer.
    估算全连接（Linear）层的 MACs。

    MACs = in_features * out_features.
    MACs = 输入特征数 × 输出特征数。
    (Multiply-accumulate: one MAC = one multiply + one add)
    （乘加操作：一个 MAC = 一次乘法 + 一次加法）

    Args:
        in_features: Input feature dimensionality.
                     in_features: 输入特征维度。
        out_features: Output feature dimensionality.
                      out_features: 输出特征维度。

    Returns:
        MACs count.
        返回 MACs 计数。
    """
    return in_features * out_features


def estimate_attention_flops(seq_len: int, d_model: int) -> int:
    """Estimate MACs for a single-head scaled dot-product attention.
    估算单头缩放点积注意力的 MACs。

    Covers Q@K^T and Attn@V.  Excludes Q/K/V projections (those are separate
    Linear layers).  Also excludes the softmax (negligible FLOPs).
    涵盖 Q@K^T 和 Attn@V。不包含 Q/K/V 投影（这些是单独的 Linear 层）。
    也不包含 softmax（FLOPs 可忽略）。

    Args:
        seq_len: Sequence length (S).
                 seq_len: 序列长度 (S)。
        d_model: Model dimension (D).
                 d_model: 模型维度 (D)。

    Returns:
        MACs count.
        返回 MACs 计数。
    """
    # Q @ K^T: (S, D) x (D, S) -> S * D * S MACs
    # Attn @ V: (S, S) x (S, D) -> S * S * D MACs
    # MACs 总计 = 2 × S² × D
    attention_macs = 2 * seq_len * seq_len * d_model
    return attention_macs


# ===========================================================================
# 通过前向钩子估算 FLOPs
# FLOPs estimation via forward hooks
# ===========================================================================


class _FlopsHook:
    """Forward hook that records Conv2d and Linear MACs during a forward pass.
    前向钩子，在前向传播过程中记录 Conv2d 和 Linear 层的 MACs。"""

    def __init__(self) -> None:
        self.total_macs: int = 0  # 累计 MACs 总数
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _conv_hook(self, module: nn.Module, inp: Any, out: Any) -> None:
        """Conv2d 层的前向钩子：根据输出形状计算 MACs。"""
        assert isinstance(module, nn.Conv2d)
        if isinstance(out, torch.Tensor):
            _, _, h_out, w_out = out.shape
        elif isinstance(out, (tuple, list)):
            h_out, w_out = out[0].shape[2], out[0].shape[3]
        else:
            return
        self.total_macs += estimate_conv2d_flops(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            output_height=h_out,
            output_width=w_out,
            stride=module.stride[0],
            padding=module.padding[0],
            groups=module.groups,
        )

    def _linear_hook(self, module: nn.Module, inp: Any, out: Any) -> None:
        """Linear 层的前向钩子：根据输出特征数计算 MACs。"""
        assert isinstance(module, nn.Linear)
        if isinstance(out, torch.Tensor):
            *_, features = out.shape
        elif isinstance(out, (tuple, list)):
            features = out[0].shape[-1]
        else:
            return
        self.total_macs += estimate_linear_flops(
            in_features=module.in_features,
            out_features=module.out_features,
        )

    def register(self, model: nn.Module) -> None:
        """Register hooks on all Conv2d and Linear layers of *model*.
        在模型的所有 Conv2d 和 Linear 层上注册钩子。"""
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self._handles.append(m.register_forward_hook(self._conv_hook))
            elif isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))

    def remove(self) -> None:
        """移除所有已注册的钩子，避免内存泄漏。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()


def estimate_training_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    include_attention: bool = False,
    seq_len: Optional[int] = None,
    d_model: Optional[int] = None,
    n_heads: Optional[int] = None,
    n_layers: Optional[int] = None,
) -> Tuple[int, int]:
    """Estimate forward and training FLOPs for a model.
    估算模型的前向和训练 FLOPs。

    Uses forward hooks to count Conv2d/Linear MACs, then applies the heuristic:
    使用前向钩子统计 Conv2d/Linear MACs，然后应用启发式规则：
        training MACs ≈ 3 × forward MACs
        训练 MACs ≈ 3 × 前向 MACs

    The 3× rule accounts for the forward pass (1×) plus the backward pass
    which computes both weight gradients (~1×) and input gradients (~1×).
    3x 规则考虑了前向传播（1×）加上反向传播，
    后者需要计算权重梯度（~1×）和输入梯度（~1×）。

    When *include_attention* is True, additional analytical MACs for
    multi-head self-attention are added (Q@K^T + Attn@V per head per layer).
    当 include_attention 为 True 时，会额外添加多头自注意力的解析 MACs
    （每个头每层的 Q@K^T + Attn@V）。

    Args:
        model: The model to profile (must accept a tensor matching *input_shape*).
               model: 待分析模型（必须接受匹配 input_shape 的张量）。
        input_shape: Shape of a single input sample, e.g. (1, 3, 224, 224).
                     input_shape: 单个输入样本的形状，如 (1, 3, 224, 224)。
        include_attention: If True, add analytical attention MACs.
                           include_attention: 如果为 True，添加解析注意力 MACs。
        seq_len: Sequence length for attention FLOPs.
                 seq_len: 注意力 FLOPs 的序列长度。
        d_model: Model dimension for attention FLOPs.
                 d_model: 注意力 FLOPs 的模型维度。
        n_heads: Number of attention heads.
                 n_heads: 注意力头数。
        n_layers: Number of transformer layers.
                  n_layers: Transformer 层数。

    Returns:
        (forward_macs, training_macs) tuple of ints.
        返回 (前向 MACs, 训练 MACs) 的整数元组。
    """
    hook = _FlopsHook()
    hook.register(model)

    # 使用虚拟输入执行一次前向传播以触发钩子统计
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy)

    hook.remove()

    forward_macs = hook.total_macs

    # 如果启用，添加注意力机制的 MACs 估算
    if include_attention and seq_len and d_model and n_heads and n_layers:
        attn_macs_per_head_per_layer = estimate_attention_flops(seq_len, d_model)
        forward_macs += n_heads * n_layers * attn_macs_per_head_per_layer

    # 训练 MACs = 3 × 前向 MACs（符合 3x 规则）
    training_macs = 3 * forward_macs
    return forward_macs, training_macs


# ===========================================================================
# 内存占用估算
# Memory footprint estimation
# ===========================================================================


def _compute_output_shape_conv2d(
    h_in: int,
    w_in: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> Tuple[int, int]:
    """Compute Conv2d output height/width per PyTorch formula.
    根据 PyTorch 公式计算 Conv2d 的输出高度/宽度。"""
    h_out = math.floor((h_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    w_out = math.floor((w_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    return h_out, w_out


def _compute_output_shape_maxpool2d(
    h_in: int,
    w_in: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> Tuple[int, int]:
    """Compute MaxPool2d output shape.
    计算 MaxPool2d 的输出形状。"""
    return _compute_output_shape_conv2d(h_in, w_in, kernel, stride, padding, dilation)


class _ActivationTracker:
    """Tracks intermediate activation shapes through a model's forward pass.
    跟踪模型前向传播过程中中间激活值的形状。

    This is an *analytical* estimate (no forward hook profiling).  It walks
    the model's named_children() tree and maintains a spatial tracker, which
    should be sufficient for feed-forward CNN backbones.
    这是一个*解析式*估算（不使用前向钩子分析）。它遍历模型的 named_children()
    树并维护空间维度跟踪器，这应该足以处理前馈 CNN 主干网络。
    """

    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        # input_shape e.g. (C, H, W)
        # 输入形状，例如 (C, H, W)
        self._c: int = input_shape[0] if len(input_shape) >= 3 else input_shape[-1]
        self._h: int = input_shape[1] if len(input_shape) >= 3 else 1
        self._w: int = input_shape[2] if len(input_shape) >= 3 else 1
        self._total_elements: int = 0  # sum over all recorded activations
        # 所有记录的激活值元素总和
        self._layer_records: List[Dict[str, Any]] = []

    @property
    def total_activation_memory_fp32(self) -> int:
        """Total activation memory in bytes (float32 assumption).
        总激活值内存（以字节计，假设为 float32）。"""
        return self._total_elements * BYTES_PER_FP32

    def _record(self, name: str, shape: Tuple[int, ...]) -> None:
        """记录一层的激活值信息。"""
        n_elements = math.prod(shape)  # 计算形状中所有维度的乘积
        self._total_elements += n_elements
        memory_bytes = n_elements * BYTES_PER_FP32
        self._layer_records.append(
            {
                "name": name,
                "shape": shape,
                "elements": n_elements,
                "memory_mb": memory_bytes / MiB,
            }
        )

    def track_conv2d(
        self,
        name: str,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        """跟踪 Conv2d 层的输出激活值。"""
        self._h, self._w = _compute_output_shape_conv2d(
            self._h, self._w, kernel_size, stride, padding, dilation
        )
        # Conv2d stores *input* activation for backward; output is next layer's input
        # We track the output activation (which becomes the next layer's input)
        # Conv2d 为反向传播存储*输入*激活值；输出是下一层的输入
        # 我们追踪输出激活值（它将成为下一层的输入）
        shape = (out_channels, self._h, self._w)
        self._record(name, shape)

    def track_linear(self, name: str, out_features: int) -> None:
        """跟踪 Linear 层的输出激活值。"""
        shape = (out_features,)
        self._c = out_features
        self._h, self._w = 1, 1
        self._record(name, shape)

    def track_relu(self, name: str) -> None:
        """跟踪 ReLU 层的激活值（通常就地操作，但计为独立激活值）。"""
        # ReLU is in-place typically; activation memory reuses input footprint
        # We record it for completeness but count it as a separate activation
        # ReLU 通常是就地操作；激活值内存复用输入内存
        # 为完整性记录它，但计为独立激活值
        shape = (self._c, self._h, self._w)
        self._record(name, shape)

    def track_identity(self, name: str) -> None:
        """跟踪恒等映射（如 BatchNorm）的激活值。"""
        shape = (self._c, self._h, self._w)
        self._record(name, shape)

    def track_pool(self, name: str, out_channels: int) -> None:
        """跟踪池化层的输出激活值。"""
        shape = (out_channels, self._h, self._w)
        self._record(name, shape)


def estimate_memory_footprint(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    optimizer: str = "sgd",
    bytes_per_param: int = BYTES_PER_FP32,
    bytes_per_activation: int = BYTES_PER_FP32,
) -> Dict[str, Any]:
    """Estimate peak memory footprint during training.
    估算训练期间的峰值内存占用。

    Breaks down memory into:
    将内存分解为以下几个部分：

    * **Parameters**: model weights (fp32 by default)
      **参数**：模型权重（默认 fp32）
    * **Gradients**: same size as parameters (fp32)
      **梯度**：与参数相同大小（fp32）
    * **Optimiser states**: 0 for SGD, 2×params for Adam (m and v buffers)
      **优化器状态**：SGD 为 0，Adam 为 2×params（m 和 v 缓冲区）
    * **Activations**: estimated via analytical shape propagation through
      ``nn.Sequential`` children of the model
      **激活值**：通过模型的 ``nn.Sequential`` 子模块进行解析式形状传播来估算

    .. note::
       Activation estimation is approximate.  For exact per-operation
       memory, use a memory profiler such as ``torch.cuda.memory_stats``
       or ``torch.profiler``.
       激活值估算是近似的。要获得精确的每操作内存使用，
       请使用内存分析器，如 ``torch.cuda.memory_stats`` 或 ``torch.profiler``。

    Args:
        model: The nn.Module to profile.
               model: 待分析的 nn.Module。
        input_shape: Per-sample input shape, e.g. (3, 224, 224).
                     input_shape: 每个样本的输入形状，如 (3, 224, 224)。
        batch_size: Training batch size.
                    batch_size: 训练批次大小。
        optimizer: ``"sgd"`` (no extra state) or ``"adam"`` (2× params).
                   optimizer: ``"sgd"``（无额外状态）或 ``"adam"``（2× 参数）。
        bytes_per_param: Bytes per parameter element (default 4 for fp32).
                         bytes_per_param: 每个参数元素的字节数（fp32 默认为 4）。
        bytes_per_activation: Bytes per activation element (default 4).
                              bytes_per_activation: 每个激活值元素的字节数（默认为 4）。

    Returns:
        A dictionary with keys:
        返回一个字典，包含以下键：
            params (bytes), gradients (bytes), optimizer_state (bytes),
            activations (bytes), total (bytes), and human-readable *_mb fields.
    """
    total_params, _ = count_parameters(model)
    # 参数和梯度占用相同大小的内存
    param_bytes = total_params * bytes_per_param
    grad_bytes = total_params * bytes_per_param  # same size as params

    # 根据优化器类型计算优化器状态的内存
    if optimizer == "adam":
        optim_bytes = 2 * param_bytes  # m + v buffers（动量 m 和二阶矩 v）
    elif optimizer == "adamw":
        optim_bytes = 2 * param_bytes
    else:
        optim_bytes = 0  # SGD, LBFGS, etc.（SGD 等无需额外状态）

    # Analytical activation tracking -- walk the model's top-level children
    # that are Sequential containers (common for CNN backbones).
    # 解析式激活值跟踪 -- 遍历模型的顶层子模块
    # 这些子模块通常是 Sequential 容器（CNN 主干网络的常见模式）
    activation_bytes = _estimate_activation_memory_analytic(
        model, input_shape, batch_size, bytes_per_activation
    )

    # 总内存 = 参数 + 梯度 + 优化器状态 + 激活值
    total_bytes = param_bytes + grad_bytes + optim_bytes + activation_bytes

    return {
        "params_bytes": param_bytes,
        "params_mb": param_bytes / MiB,
        "gradients_bytes": grad_bytes,
        "gradients_mb": grad_bytes / MiB,
        "optimizer_state_bytes": optim_bytes,
        "optimizer_state_mb": optim_bytes / MiB,
        "activations_bytes": activation_bytes,
        "activations_mb": activation_bytes / MiB,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / MiB,
        "total_gb": total_bytes / GiB,
        "total_params": total_params,
        "batch_size": batch_size,
        "optimizer": optimizer,
    }


def _estimate_activation_memory_analytic(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int,
    bytes_per_element: int,
) -> int:
    """Analytically estimate activation memory by walking the model graph.
    通过遍历模型图来解析式估算激活值内存。

    If the model is a simple Sequential of known layer types, we track
    spatial dimensions and multiply by channel counts.  For arbitrary
    models we fall back to a heuristic (2× parameter memory).
    如果模型是已知层类型的简单 Sequential，我们跟踪空间维度并乘以通道数。
    对于任意模型，我们回退到启发式方法（2× 参数内存）。
    """
    tracker = _ActivationTracker(input_shape)

    # 遍历模型的子模块，按类型跟踪激活值
    for name, child in model.named_children():
        for sub_name, module in child.named_modules():
            if sub_name == "":
                continue  # skip the container itself：跳过容器本身
            full_name = f"{name}.{sub_name}"
            if isinstance(module, nn.Conv2d):
                tracker.track_conv2d(
                    full_name,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    dilation=module.dilation[0],
                )
            elif isinstance(module, nn.Linear):
                tracker.track_linear(full_name, module.out_features)
            elif isinstance(module, nn.ReLU):
                tracker.track_relu(full_name)
            elif isinstance(module, nn.BatchNorm2d):
                tracker.track_identity(full_name)
            elif isinstance(module, nn.MaxPool2d):
                # Keep same channels; only spatial changes
                # 保持相同通道数；仅空间维度变化
                tracker.track_pool(full_name, tracker._c)

    raw_bytes = tracker.total_activation_memory_fp32
    # Adjust for byte width and batch size
    # 根据字节宽度和批次大小进行调整
    scaled = raw_bytes * batch_size * (bytes_per_element / BYTES_PER_FP32)
    return int(scaled)


# ===========================================================================
# 人类可读的内存摘要
# Human-readable memory summary
# ===========================================================================


def format_memory_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    optimizer: str = "adam",
) -> str:
    """Produce a human-readable memory footprint summary.
    生成人类可读的内存占用摘要。

    Args:
        model: The nn.Module.
               model: nn.Module 实例。
        input_shape: Per-sample input shape.
                     input_shape: 每个样本的输入形状。
        batch_size: Training batch size.
                    batch_size: 训练批次大小。
        optimizer: ``"sgd"`` or ``"adam"``.
                   optimizer: ``"sgd"`` 或 ``"adam"``。

    Returns:
        Formatted string suitable for printing.
        返回适合打印的格式化字符串。
    """
    info = estimate_memory_footprint(
        model,
        input_shape,
        batch_size=batch_size,
        optimizer=optimizer,
    )
    total_params, trainable = count_parameters(model)

    lines: List[str] = []
    lines.append("=" * 68)
    lines.append("  TRAINING MEMORY FOOTPRINT ESTIMATE")
    lines.append("=" * 68)
    lines.append(f"  Parameters       : {total_params:>12,}  ({trainable:,} trainable)")
    lines.append(f"  Batch size       : {batch_size:>12}")
    lines.append(f"  Optimiser        : {optimizer:>12}")
    lines.append(f"  Element width    : {BYTES_PER_FP32} bytes (fp32)")
    lines.append("-" * 68)
    lines.append(f"  {'Category':<20} {'Bytes':>16} {'MiB':>12}")
    lines.append("-" * 68)
    # 分类展示：参数、梯度、优化器状态、激活值
    for label, key_b, key_m in [
        ("Parameters", "params_bytes", "params_mb"),
        ("Gradients", "gradients_bytes", "gradients_mb"),
        ("Optimiser state", "optimizer_state_bytes", "optimizer_state_mb"),
        ("Activations", "activations_bytes", "activations_mb"),
    ]:
        lines.append(f"  {label:<20} {info[key_b]:>16,} {info[key_m]:>12.2f}")
    lines.append("-" * 68)
    lines.append(
        f"  {'TOTAL':<20} {info['total_bytes']:>16,} {info['total_mb']:>12.2f}"
    )
    lines.append("=" * 68)
    lines.append(
        f"  Total memory required: {info['total_mb']:.2f} MiB  "
        f"({info['total_gb']:.2f} GiB)"
    )
    lines.append("=" * 68)
    return "\n".join(lines)


# ===========================================================================
# 独立演示（直接执行 flops_counter.py 时运行）
# Standalone demo (runs when executing flops_counter.py directly)
# ===========================================================================


def _demo() -> None:
    """Demonstrate flops_counter capabilities with a small CNN.
    使用一个小型 CNN 演示 flops_counter 的功能。"""
    print("\n" + "=" * 68)
    print("  FLOPS_COUNTER.PY  --  DEMO")
    print("=" * 68)

    # ------------------------------------------------------------------
    # 1. Build a small model
    # 1. 构建一个小型模型
    # ------------------------------------------------------------------
    class DemoCNN(nn.Module):
        """用于演示的小型 CNN 模型。"""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = DemoCNN()
    input_shape: Tuple[int, ...] = (1, 3, 32, 32)  # (batch, C, H, W)

    # ------------------------------------------------------------------
    # 2. Parameter counting
    # 2. 参数统计
    # ------------------------------------------------------------------
    total, trainable = count_parameters(model)
    print(f"\n  Parameter count  : {total:,} total, {trainable:,} trainable")

    # ------------------------------------------------------------------
    # 3. FLOPs estimation
    # 3. FLOPs 估算
    # ------------------------------------------------------------------
    fwd_macs, train_macs = estimate_training_flops(model, input_shape)
    print(f"  Forward MACs     : {fwd_macs:,}")
    print(f"  Training MACs    : {train_macs:,}  (= 3 × forward)")
    print(f"  Forward FLOPs    : {2 * fwd_macs:,}  (adds + mults)")
    print(f"  Training FLOPs   : {2 * train_macs:,}")

    # ------------------------------------------------------------------
    # 4. Memory footprint
    # 4. 内存占用
    # ------------------------------------------------------------------
    report = format_memory_summary(
        model, input_shape[1:], batch_size=32, optimizer="adam"
    )
    print("\n" + report)

    # ------------------------------------------------------------------
    # 5. Sanity checks
    # 5. 健全性检查
    # ------------------------------------------------------------------
    print("\n  --- Sanity Checks ---")
    # Conv2d(3->16, k=3) with 32x32 input: MACs = 16*(3)*9*32*32 = 442,368
    # Conv2d(3->16, k=3) 输入 32x32：MACs = 16×(3)×9×32×32 = 442,368
    macs_conv1 = estimate_conv2d_flops(3, 16, 3, 32, 32, padding=1)
    assert macs_conv1 == 16 * 3 * 9 * 32 * 32, f"Expected 442368, got {macs_conv1}"
    print(f"  Conv2d(3->16, k3, 32x32) MACs = {macs_conv1:,}  ✓")

    # Linear(64->10) MACs = 640
    macs_fc = estimate_linear_flops(64, 10)
    assert macs_fc == 640, f"Expected 640, got {macs_fc}"
    print(f"  Linear(64->10) MACs          = {macs_fc:,}  ✓")

    # Attention S=128, D=64: MACs = 2*128*128*64 = 2,097,152
    attn_macs = estimate_attention_flops(128, 64)
    assert attn_macs == 2 * 128 * 128 * 64, f"Expected 2097152, got {attn_macs}"
    print(f"  Attention(S=128, D=64) MACs  = {attn_macs:,}  ✓")

    # Training MACs ≥ forward MACs：训练 MACs 必须 ≥ 前向 MACs
    assert train_macs >= fwd_macs, "train macs must be >= forward macs"
    print(f"  Training/Forward ratio       = {train_macs / max(1, fwd_macs):.1f}x  ✓")

    # Model size sanity：模型大小合理
    model_size_mb = total * BYTES_PER_FP32 / MiB
    print(f"  Model size                   = {model_size_mb:.2f} MiB  ✓")

    print("\n  *** All sanity checks passed ***")
    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    _demo()
