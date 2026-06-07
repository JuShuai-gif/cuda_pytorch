"""
PTQ Pipeline + QAT with FakeQuantize & STE (Lecture 06)
========================================================
Implements Post-Training Quantization (PTQ), Quantization-Aware Training (QAT)
using FakeQuantize with Straight-Through Estimator (STE), FP32 baseline
comparison, and mixed-precision simulation (different bits per layer).

Key concepts:
  - PTQ: calibrate activation ranges, then quantize weights & activations
  - QAT: insert FakeQuantize nodes; train with STE to recover accuracy
  - STE: gradient passes through rounding unchanged (identity backward)
  - Mixed precision: assign different bit-widths to different layers
  - Compare FP32 baseline vs PTQ vs QAT accuracy

All computations run on CPU; no GPU required.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dataset
NUM_CLASSES: int = 10
INPUT_CHANNELS: int = 1  # MNIST-like grayscale
IMAGE_SIZE: int = 28
NUM_TRAIN: int = 4096
NUM_TEST: int = 1024
CALIB_SAMPLES: int = 512  # number of samples for calibration

# Training
BATCH_SIZE: int = 128
FP32_EPOCHS: int = 8
QAT_EPOCHS: int = 5
LR: float = 0.01

# Quantization
DEFAULT_BITS: int = 8  # default bit width for PTQ / QAT
SEED: int = 42

# Mixed-precision layer -> bits configuration
MIXED_PRECISION_CONFIG: Dict[str, int] = {
    "conv1": 8,
    "conv2": 4,
    "fc1": 4,
    "fc2": 8,
}

# ---------------------------------------------------------------------------
# Synthetic Data
# ---------------------------------------------------------------------------


def _create_synthetic_dataset(
    n: int,
    c: int = INPUT_CHANNELS,
    h: int = IMAGE_SIZE,
    w: int = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate reproducible synthetic images and random labels.

    Args:
        n:           Number of samples.
        c:           Number of input channels (1 for grayscale).
        h, w:        Spatial dimensions.
        num_classes: Number of label classes.
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (images, labels).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    images = torch.randn(n, c, h, w, generator=g) * 0.5 + 0.5
    images = images.clamp(0.0, 1.0)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    return images, labels


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """A compact CNN for MNIST-like 28x28 grayscale classification.

    Architecture:
        Conv2d(1,  16, 3, padding=1) -> ReLU -> MaxPool2d(2)   # 14x14
        Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)   #  7x7
        Flatten -> Linear(32*7*7, 128) -> ReLU
        Linear(128, 10)

    Args:
        num_classes: Number of output classes (default 10).
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
        """Return layers whose weights can be quantized (Conv2d, Linear)."""
        return [self.conv1, self.conv2, self.fc1, self.fc2]


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    lr: float = LR,
) -> float:
    """Train the model for one epoch.

    Args:
        model:  A PyTorch nn.Module.
        loader: DataLoader yielding (images, labels) batches.
        lr:     Learning rate.

    Returns:
        Average training loss over the epoch.
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
    """Evaluate top-1 accuracy on a given DataLoader.

    Args:
        model:  A PyTorch nn.Module.
        loader: DataLoader yielding (images, labels) batches.

    Returns:
        Accuracy as a float in [0.0, 1.0].
    """
    model.eval()
    correct = 0
    total = 0

    for xb, yb in loader:
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return correct / max(total, 1)


def count_params(model: nn.Module) -> int:
    """Return total number of parameters in the model.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        Total parameter count.
    """
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Quantization Primitives
# ---------------------------------------------------------------------------


def compute_scale_zp(
    tensor: torch.Tensor,
    bits: int,
) -> Tuple[float, int, int, int]:
    """Compute asymmetric affine quantisation parameters for a tensor.

    The mapping is:
        scale  = (x_max - x_min) / (2^bits - 1)
        zp     = round(-x_min / scale)   [clamped to [0, 2^bits-1]]

    Args:
        tensor: Float32 tensor of any shape.
        bits:   Bit width (e.g. 8, 4, 2).

    Returns:
        Tuple of (scale, zero_point, qmin, qmax).
    """
    if bits <= 0:
        raise ValueError(f"bits must be positive; got {bits}")

    qmin: int = 0
    qmax: int = int(2**bits - 1)

    x_min = tensor.min().item()
    x_max = tensor.max().item()

    if x_max == x_min:
        return 1.0, 0, qmin, qmax

    scale = (x_max - x_min) / (qmax - qmin)
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
    """Quantize a float tensor, then immediately dequantize it.

    This simulates the effect of real integer quantisation without actually
    converting the tensor to an integer dtype -- the returned tensor is
    still float32 but has quantisation noise.

    Args:
        x:    Float32 input tensor.
        scale: Quantisation scale.
        zp:   Zero point.
        qmin: Minimum quantized value.
        qmax: Maximum quantized value.

    Returns:
        Fake-quantized float32 tensor with the same shape as x.
    """
    # Quantize to integers
    x_q = torch.round(x / scale + zp)
    x_q = torch.clamp(x_q, qmin, qmax)
    # Dequantize back to float
    x_dq = (x_q - zp) * scale
    return x_dq


def fake_quantize_ste(
    x: torch.Tensor,
    scale: float,
    zp: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Fake-quantize with Straight-Through Estimator (STE) gradient.

    Forward:  x_fq = fake_quantize(x, scale, zp, qmin, qmax)
    Backward: gradient passes through rounding unchanged (identity).

    Implementation uses the ``.detach()`` trick:
        output = x + (x_fq - x).detach()
    which means forward = x_fq, backward = dL/d(x_fq) * 1 = dL/d(x_fq).

    Args:
        x:    Float32 input tensor.
        scale: Quantisation scale.
        zp:   Zero point.
        qmin: Minimum quantized value.
        qmax: Maximum quantized value.

    Returns:
        Fake-quantized tensor with STE gradient (same shape as x).
    """
    x_fq = fake_quantize(x, scale, zp, qmin, qmax)
    # STE: forward path uses x_fq, backward path uses x (identity gradient)
    return x + (x_fq - x).detach()


# Also provide a torch.autograd.Function version for educational clarity
class _FakeQuantizeSTE(torch.autograd.Function):
    """Custom autograd Function implementing FakeQuantize with STE.

    This is semantically equivalent to the ``.detach()``-trick version above,
    but demonstrates how STE is implemented at the autograd level.
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
        # Quantize to integers
        x_q = torch.round(x / scale + zp)
        x_q = torch.clamp(x_q, qmin, qmax)
        # Dequantize
        x_dq = (x_q - zp) * scale
        return x_dq

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        # STE: pass gradient through unchanged
        return grad_output, None, None, None, None


def _ste_autograd_fn(
    x: torch.Tensor,
    scale: float,
    zp: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Convenience wrapper for the autograd.Function version of STE."""
    # Convert scalars to tensors so autograd.Function sees them
    return _FakeQuantizeSTE.apply(x, scale, zp, qmin, qmax)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# FakeQuantize Module (nn.Module)
# ---------------------------------------------------------------------------


class FakeQuantize(nn.Module):
    """A trainable FakeQuantize module for QAT.

    When ``observer_enabled=True``, the module collects min/max statistics
    and recomputes scale/zp on each forward pass.  Once calibration is
    complete, ``observer_enabled`` should be set to False to freeze the
    quantisation parameters.

    Args:
        bits: Bit width for quantisation (default 8).
    """

    def __init__(self, bits: int = DEFAULT_BITS) -> None:
        super().__init__()
        self.bits = bits
        self.qmin: int = 0
        self.qmax: int = int(2**bits - 1)

        # Learnable / freezable quantisation parameters
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0))

        self.observer_enabled: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            self._observe(x)

        # Fake-quantize with STE
        return fake_quantize_ste(
            x,
            self.scale.item(),
            int(self.zero_point.item()),
            self.qmin,
            self.qmax,
        )

    def _observe(self, x: torch.Tensor) -> None:
        """Update scale and zero_point from the observed tensor range."""
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
        """Disable observer; quantisation parameters are now frozen."""
        self.observer_enabled = False


# ---------------------------------------------------------------------------
# QAT Layer Wrappers
# ---------------------------------------------------------------------------


class QATConv2d(nn.Module):
    """Conv2d wrapped with weight and activation FakeQuantize for QAT.

    Weight FakeQuantize is applied before the convolution.
    Activation FakeQuantize is applied after the convolution.

    Args:
        conv: Pre-existing Conv2d layer to wrap.
        bits: Bit width for weight and activation quantisation.
    """

    def __init__(self, conv: nn.Conv2d, bits: int = DEFAULT_BITS) -> None:
        super().__init__()
        self.conv = conv
        # Quantiser for weights (operates on the weight tensor)
        self.weight_fq = FakeQuantize(bits)
        # Quantiser for activations (operates on the output of conv)
        self.act_fq = FakeQuantize(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fake-quantize weights before convolution
        w_q = self.weight_fq(self.conv.weight)
        # Use functional conv2d with fake-quantized weights
        out = nn.functional.conv2d(
            x,
            w_q,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        # Fake-quantize output activations
        out = self.act_fq(out)
        return out


class QATLinear(nn.Module):
    """Linear wrapped with weight and activation FakeQuantize for QAT.

    Weight FakeQuantize is applied before the linear transformation.
    Activation FakeQuantize is applied after the linear transformation.

    Args:
        linear: Pre-existing Linear layer to wrap.
        bits:   Bit width for weight and activation quantisation.
    """

    def __init__(self, linear: nn.Linear, bits: int = DEFAULT_BITS) -> None:
        super().__init__()
        self.linear = linear
        self.weight_fq = FakeQuantize(bits)
        self.act_fq = FakeQuantize(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self.weight_fq(self.linear.weight)
        out = nn.functional.linear(x, w_q, self.linear.bias)
        out = self.act_fq(out)
        return out


# ---------------------------------------------------------------------------
# PTQ Pipeline
# ---------------------------------------------------------------------------


def calibrate_activation_ranges(
    model: nn.Module,
    loader: DataLoader,
    bits: int = DEFAULT_BITS,
    num_batches: int = 8,
) -> Dict[str, Tuple[float, int, int, int]]:
    """Calibrate activation quantisation parameters via forward passes.

    For each prunable layer (Conv2d, Linear), a forward hook captures the
    output tensor and computes (scale, zp, qmin, qmax) based on the observed
    min/max range across multiple batches.

    Args:
        model:       FP32 model to calibrate.
        loader:      DataLoader for calibration data.
        bits:        Target bit width.
        num_batches: Maximum number of batches to use for calibration.

    Returns:
        Dictionary mapping layer name -> (scale, zp, qmin, qmax).
    """
    model.eval()
    # Accumulate global min / max per layer
    layer_min: Dict[str, float] = {}
    layer_max: Dict[str, float] = {}

    # Map module -> layer name
    module_name_map: Dict[nn.Module, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            module_name_map[mod] = name

    def _hook(
        mod: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
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

    # Register hooks
    handles = []
    for mod, name in module_name_map.items():
        handles.append(mod.register_forward_hook(_hook))

    # Calibration forward passes
    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            _ = model(xb)

    # Remove hooks
    for h in handles:
        h.remove()

    # Build parameter dict
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
    """Apply post-training quantisation to model weights and insert activation
    quantisation hooks.

    This function quantizes weights in-place (replacing float weights with
    their quantized-then-dequantized versions) and wraps each quantizable
    layer with an activation quantiser at its output.

    Args:
        model:      Trained FP32 model.
        act_params: Per-layer activation quantisation parameters from calibration.
        bits:       Bit width for weight quantisation.

    Returns:
        The same model, modified in-place with quantized weights and
        activation quantisers attached.
    """
    model.eval()

    layer_map: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            layer_map[name] = mod

    for name, mod in layer_map.items():
        # ---- quantize weights ----
        w = mod.weight.data
        scale_w, zp_w, qmin_w, qmax_w = compute_scale_zp(w, bits)
        w_q = fake_quantize(w, scale_w, zp_w, qmin_w, qmax_w)
        mod.weight.data.copy_(w_q)

    # Attach activation quantisers via forward hooks
    _attach_activation_quantizers(model, act_params)

    return model


def _attach_activation_quantizers(
    model: nn.Module,
    act_params: Dict[str, Tuple[float, int, int, int]],
) -> None:
    """Attach forward hooks that fake-quantize activation outputs.

    The hooks are stored on the model so they can be removed later.

    Args:
        model:      Model to instrument (modified in-place).
        act_params: Per-layer (scale, zp, qmin, qmax) tuples.
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
    """Remove all activation quantisation hooks from the model.

    Args:
        model: Model with attached quantisation hooks.
    """
    if hasattr(model, "_quant_hooks"):
        for h in model._quant_hooks:  # type: ignore[attr-defined]
            h.remove()
        model._quant_hooks.clear()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# QAT Model Builder
# ---------------------------------------------------------------------------


def build_qat_model(
    fp32_model: nn.Module,
    bits: int = DEFAULT_BITS,
) -> nn.Module:
    """Convert an FP32 model into a QAT-ready model by wrapping Conv2d/Linear
    layers with FakeQuantize.

    This creates a new ``nn.Sequential`` that mirrors the original model's
    architecture but uses ``QATConv2d`` / ``QATLinear`` wrappers.  All
    original trained weights are copied into the QAT layers.

    Args:
        fp32_model: Trained FP32 SimpleCNN model.
        bits:       Bit width for all quantisers.

    Returns:
        A new QAT model (nn.Module) ready for quantization-aware training.
    """
    qat = SimpleCNN(num_classes=NUM_CLASSES)

    with torch.no_grad():
        # Copy weights from fp32_model to qat
        qat.load_state_dict(fp32_model.state_dict(), strict=False)

    # Replace Conv2d / Linear with their QAT wrappers
    _replace_with_qat(qat, bits)

    return qat


def _replace_with_qat(model: nn.Module, bits: int) -> None:
    """In-place replacement of Conv2d and Linear with QAT wrappers.

    Args:
        model: Model to modify in-place.
        bits:  Bit width for quantisers.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(model, name, QATConv2d(child, bits))
        elif isinstance(child, nn.Linear):
            setattr(model, name, QATLinear(child, bits))
        else:
            _replace_with_qat(child, bits)


def freeze_qat_observers(model: nn.Module) -> None:
    """Freeze all FakeQuantize observers in the QAT model.

    After this call, quantisation parameters are fixed and no longer updated
    from input statistics.

    Args:
        model: QAT model to freeze (modified in-place).
    """
    for mod in model.modules():
        if isinstance(mod, FakeQuantize):
            mod.freeze()


def calibrate_qat(
    model: nn.Module,
    loader: DataLoader,
    num_batches: int = 8,
) -> None:
    """Calibrate QAT FakeQuantize observers using a few forward passes.

    During calibration, FakeQuantize.observer_enabled is True so each module
    records min/max activation ranges and updates its scale/zp.
    After calibration, observers are frozen.

    Args:
        model:       QAT model with FakeQuantize modules.
        loader:      DataLoader for calibration data.
        num_batches: Number of calibration batches.
    """
    model.train()  # observers need training mode to update
    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            _ = model(xb)

    # Freeze all observers
    freeze_qat_observers(model)


# ---------------------------------------------------------------------------
# Mixed-Precision Simulation
# ---------------------------------------------------------------------------


def build_mixed_precision_model(
    fp32_model: nn.Module,
    bit_config: Dict[str, int],
) -> nn.Module:
    """Build a QAT model with per-layer bit-width assignments.

    Each layer listed in ``bit_config`` uses the specified bit width;
    layers not listed default to ``DEFAULT_BITS``.

    Example configuration::

        bit_config = {"conv1": 8, "conv2": 4, "fc1": 4, "fc2": 8}

    Args:
        fp32_model: Trained FP32 model.
        bit_config: Dictionary mapping layer names to bit widths.

    Returns:
        A QAT model with per-layer bit-width FakeQuantize wrappers.
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
    """Replace Conv2d/Linear with QAT wrappers using custom bit widths.

    Args:
        model:      Model to modify in-place.
        bit_config: Layer name -> bits mapping.
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
# Comparison & Printing
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full PTQ + QAT + Mixed Precision pipeline."""
    torch.manual_seed(SEED)

    print_header("LECTURE 06: PTQ Pipeline + QAT with FakeQuantize & STE")

    # ---- 1. Create synthetic data ------------------------------------------
    print("\n[1] Generating synthetic dataset ...")
    train_images, train_labels = _create_synthetic_dataset(NUM_TRAIN)
    test_images, test_labels = _create_synthetic_dataset(NUM_TEST, seed=SEED + 1)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Smaller loader for calibration (use a subset)
    calib_images = train_images[:CALIB_SAMPLES]
    calib_labels = train_labels[:CALIB_SAMPLES]
    calib_dataset = TensorDataset(calib_images, calib_labels)
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    print(f"  Calibration subset: {CALIB_SAMPLES} samples")

    # ---- 2. Train FP32 baseline -------------------------------------------
    print(f"\n[2] Training FP32 baseline ({FP32_EPOCHS} epochs) ...")
    fp32_model = SimpleCNN(num_classes=NUM_CLASSES)
    fp32_params = count_params(fp32_model)
    print(f"  Model parameters: {fp32_params:,}")

    for epoch in range(1, FP32_EPOCHS + 1):
        loss = train_one_epoch(fp32_model, train_loader)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2d}  loss={loss:.4f}")

    fp32_acc = evaluate_accuracy(fp32_model, test_loader)
    print(f"  FP32 baseline accuracy: {fp32_acc:.4f}")

    # ---- 3. PTQ: Calibrate + Quantize -------------------------------------
    print(f"\n[3] PTQ ({DEFAULT_BITS}-bit): Calibrating activation ranges ...")
    # Deep-copy the trained FP32 model before PTQ
    ptq_model = copy.deepcopy(fp32_model)

    act_params = calibrate_activation_ranges(
        ptq_model, calib_loader, bits=DEFAULT_BITS, num_batches=8
    )
    print(f"  Calibrated {len(act_params)} layers")

    for name, (scale, zp, qmin, qmax) in act_params.items():
        print(f"    {name:<25}  scale={scale:.6f}  zp={zp:>4d}")

    print("\n  Quantizing weights and inserting activation quantizers ...")
    ptq_quantize_model(ptq_model, act_params, bits=DEFAULT_BITS)

    ptq_acc = evaluate_accuracy(ptq_model, test_loader)
    print(f"  PTQ accuracy ({DEFAULT_BITS}-bit): {ptq_acc:.4f}")

    # ---- 4. QAT with FakeQuantize + STE -----------------------------------
    print(
        f"\n[4] QAT with FakeQuantize + STE ({DEFAULT_BITS}-bit, {QAT_EPOCHS} epochs) ..."
    )
    qat_model = build_qat_model(fp32_model, bits=DEFAULT_BITS)

    # Calibrate the QAT observers
    print("  Calibrating QAT observers ...")
    calibrate_qat(qat_model, calib_loader, num_batches=8)
    qat_acc_before = evaluate_accuracy(qat_model, test_loader)
    print(f"  QAT accuracy BEFORE fine-tuning: {qat_acc_before:.4f}")

    # QAT fine-tuning
    print(f"  QAT fine-tuning for {QAT_EPOCHS} epochs ...")
    for epoch in range(1, QAT_EPOCHS + 1):
        loss = train_one_epoch(qat_model, train_loader, lr=LR * 0.5)
        if epoch % 2 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>2d}  loss={loss:.4f}")

    qat_acc = evaluate_accuracy(qat_model, test_loader)
    print(f"  QAT accuracy AFTER fine-tuning: {qat_acc:.4f}")

    # ---- 5. Mixed Precision Simulation (BONUS) ----------------------------
    print("\n[5] Mixed Precision Simulation (different bits per layer) ...")
    mp_model = build_mixed_precision_model(fp32_model, MIXED_PRECISION_CONFIG)

    # Print the bit assignment per layer
    print("  Layer bit-width assignments:")
    for name, child in mp_model.named_modules():
        if isinstance(child, (QATConv2d, QATLinear)):
            w_bits = child.weight_fq.bits
            a_bits = child.act_fq.bits
            print(f"    {name:<25}  weight={w_bits}-bit, act={a_bits}-bit")

    calibrate_qat(mp_model, calib_loader, num_batches=8)
    mp_acc_before = evaluate_accuracy(mp_model, test_loader)
    print(f"  Mixed-precision accuracy BEFORE fine-tuning: {mp_acc_before:.4f}")

    print(f"  Mixed-precision fine-tuning for {QAT_EPOCHS} epochs ...")
    for epoch in range(1, QAT_EPOCHS + 1):
        loss = train_one_epoch(mp_model, train_loader, lr=LR * 0.5)
        if epoch % 2 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>2d}  loss={loss:.4f}")

    mp_acc = evaluate_accuracy(mp_model, test_loader)
    print(f"  Mixed-precision accuracy AFTER fine-tuning: {mp_acc:.4f}")

    # ---- 6. STE Gradient Sanity Check -------------------------------------
    print("\n[6] STE gradient sanity check ...")
    x = torch.tensor([0.3, 1.7, -0.5], requires_grad=True)
    scale_s, zp_s, qmin_s, qmax_s = compute_scale_zp(x.detach(), 4)
    y = fake_quantize_ste(x, scale_s, zp_s, qmin_s, qmax_s)
    loss = y.sum()
    loss.backward()
    print(f"  Input:            {x.detach().tolist()}")
    print(f"  Fake-quantized:   {y.detach().tolist()}")
    print(f"  Gradient (STE):   {x.grad.tolist()}")
    # With STE, the gradient should be all ones (same as if rounding were identity)
    grad_ok = torch.allclose(x.grad, torch.ones_like(x.grad))
    print(f"  Gradient matches identity: {grad_ok}")

    # ---- 7. Comparison Summary --------------------------------------------
    print("\n[7] Accuracy Comparison Summary")
    print(f"  {'':-<60}")
    print(f"  {'Method':<30} {'Accuracy':>10} {'Delta vs FP32':>15}")
    print(f"  {'':-<60}")
    print(f"  {'FP32 Baseline':<30} {fp32_acc:>10.4f} {'---':>15}")
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
    print(
        f"  {'Mixed Precision':<30} "
        f"{mp_acc:>10.4f} "
        f"{(mp_acc - fp32_acc) * 100:>+14.2f}%"
    )
    print(f"  {'':-<60}")

    # ---- 8. Done ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model:       SimpleCNN ({fp32_params:,} params)")
    print(f"  Data:        Synthetic MNIST-like ({NUM_TRAIN} train / {NUM_TEST} test)")
    print(f"  PTQ bits:    {DEFAULT_BITS}")
    print(f"  QAT bits:    {DEFAULT_BITS}")
    print(f"  Mixed-precision bits: {MIXED_PRECISION_CONFIG}")
    print(f"  FP32 acc:    {fp32_acc:.4f}")
    print(f"  PTQ acc:     {ptq_acc:.4f}  ({(ptq_acc - fp32_acc) * 100:+.2f}%)")
    print(f"  QAT acc:     {qat_acc:.4f}  ({(qat_acc - fp32_acc) * 100:+.2f}%)")
    print(f"  Mixed acc:   {mp_acc:.4f}  ({(mp_acc - fp32_acc) * 100:+.2f}%)")
    print("=" * 70)

    print("\nLecture 06 complete.")


if __name__ == "__main__":
    main()
