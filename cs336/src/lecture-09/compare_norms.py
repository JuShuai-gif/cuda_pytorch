"""
对比 LayerNorm 和 RMSNorm 的速度与数值稳定性。

测试：
  - 前向传播速度对比
  - 反向传播速度对比
  - 极端输入下的数值稳定性
  - 梯度范数保持能力
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from .normalization import LayerNorm, RMSNorm


def compare_speed() -> None:
    """对比 LayerNorm 和 RMSNorm 的前向和反向速度。"""
    print("=" * 70)
    print("Speed Comparison: LayerNorm vs RMSNorm")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    configs: list[tuple[int, int, int]] = [
        (1, 128, 768),  # 小
        (1, 512, 1024),  # 中
        (1, 1024, 2048),  # 大
        (1, 2048, 4096),  # 超大
        (4, 2048, 4096),  # 批次
    ]

    print(
        f"\n  {'Config (B, S, H)':<22} {'LN Fwd (ms)':<14} {'RMS Fwd (ms)':<14} {'Speedup':<10} {'LN Bwd (ms)':<14} {'RMS Bwd (ms)':<14} {'Bwd Speedup':<12}"
    )
    print("  " + "-" * 100)

    def _benchmark(fn: Any, x: torch.Tensor, num_iters: int = 100) -> float:
        # 预热
        for _ in range(10):
            y = fn(x)
            y.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            y = fn(x)
            y.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (
            (time.perf_counter() - start) / num_iters * 1000 / 2
        )  # /2 因为前向+反向分开计时

    for batch, seq_len, hidden in configs:
        x = torch.randn(batch, seq_len, hidden, device=device, requires_grad=True)

        ln = LayerNorm(hidden).to(device)
        rms = RMSNorm(hidden).to(device)

        # 分别计时前向和反向
        def _time_fwd_bwd(norm_fn: Any) -> tuple[float, float]:
            # 仅前向
            for _ in range(5):
                y = norm_fn(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                y = norm_fn(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - start) / 50 * 1000

            # 前向 + 反向
            for _ in range(5):
                y = norm_fn(x.clone().requires_grad_(True))
                y.sum().backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                y = norm_fn(x.clone().requires_grad_(True))
                y.sum().backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_bwd_time = (time.perf_counter() - start) / 50 * 1000

            return fwd_time, fwd_bwd_time - fwd_time

        with torch.no_grad():
            ln_fwd, ln_bwd = _time_fwd_bwd(ln)
            rms_fwd, rms_bwd = _time_fwd_bwd(rms)

        fwd_speedup = ln_fwd / rms_fwd if rms_fwd > 0 else 0
        bwd_speedup = ln_bwd / rms_bwd if rms_bwd > 0 else 0

        label = f"({batch}, {seq_len}, {hidden})"
        print(
            f"  {label:<22} {ln_fwd:>10.4f}   {rms_fwd:>10.4f}   "
            f"{fwd_speedup:>6.2f}x  {ln_bwd:>10.4f}   {rms_bwd:>10.4f}   {bwd_speedup:>6.2f}x"
        )

    print("\n  RMSNorm 通常快 5-15%，因为它跳过了均值减法步骤。")


def compare_numerical_stability() -> None:
    """对比极端输入下 LayerNorm 和 RMSNorm 的数值稳定性。"""
    print("\n" + "=" * 70)
    print("Numerical Stability Comparison")
    print("=" * 70)

    hidden_size = 256
    torch.manual_seed(42)

    ln = LayerNorm(hidden_size)
    rms = RMSNorm(hidden_size)

    scenarios = [
        ("Normal", torch.randn(4, 16, hidden_size)),
        ("Large values", torch.randn(4, 16, hidden_size) * 100),
        ("Small values", torch.randn(4, 16, hidden_size) * 1e-8),
        ("Mixed NaN", torch.randn(4, 16, hidden_size)),
        ("All zeros", torch.zeros(4, 16, hidden_size)),
        ("Near overflow", torch.randn(4, 16, hidden_size) * 1e15),
    ]

    # 向 "Mixed NaN" 场景注入 NaN
    scenarios[3][1][0, 0, 0] = float("nan")

    print(
        f"\n  {'Scenario':<18} {'LN mean':>10} {'LN std':>10} {'RMS mean':>10} {'RMS std':>10} {'Has NaN':>10} {'Has Inf':>10}"
    )
    print("  " + "-" * 80)

    for name, x in scenarios:
        with torch.no_grad():
            ln_out = ln(x)
            rms_out = rms(x)

        ln_mean = ln_out.mean().item()
        ln_std = ln_out.std().item()
        rms_mean = rms_out.mean().item()
        rms_std = rms_out.std().item()

        ln_nan = torch.isnan(ln_out).any().item()
        rms_nan = torch.isnan(rms_out).any().item()
        ln_inf = torch.isinf(ln_out).any().item()
        rms_inf = torch.isinf(rms_out).any().item()

        print(
            f"  {name:<18} {ln_mean:>10.4f} {ln_std:>10.4f} {rms_mean:>10.4f} {rms_std:>10.4f} "
            f"{'LN:' + str(ln_nan) + '/RMS:' + str(rms_nan):>10} "
            f"{'LN:' + str(ln_inf) + '/RMS:' + str(rms_inf):>10}"
        )


def compare_gradient_behavior() -> None:
    """对比 LayerNorm 和 RMSNorm 如何影响梯度流动。"""
    print("\n" + "=" * 70)
    print("Gradient Behavior Comparison")
    print("=" * 70)

    hidden_size = 256
    num_layers = 20
    batch_size = 2
    seq_len = 16

    ln = LayerNorm(hidden_size)
    rms = RMSNorm(hidden_size)

    def _run_chain(norm: nn.Module, x: torch.Tensor) -> float:
        """通过多层 norm 运行并测量输入梯度范数。"""
        for _ in range(num_layers):
            x = norm(x)
        loss = x.sum()
        loss.backward()
        return x.grad.norm().item() if x.grad is not None else 0.0

    x_ln = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_rms = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

    ln_grad = _run_chain(ln, x_ln)
    rms_grad = _run_chain(rms, x_rms)

    print(f"\n  经过 {num_layers} 层 norm 后：")
    print(f"  LayerNorm 输入梯度范数：{ln_grad:.6f}")
    print(f"  RMSNorm 输入梯度范数：  {rms_grad:.6f}")
    print(f"\n  LayerNorm 对数据做中心化（mean=0），这可能比 RMSNorm")
    print(f"  稍微更削弱梯度。总体两者都非常稳定。")


def compare_with_pytorch_builtin() -> None:
    """将我们的实现与 PyTorch 内置版本进行对比。"""
    print("\n" + "=" * 70)
    print("Correctness: Custom vs PyTorch Built-in")
    print("=" * 70)

    torch.manual_seed(123)
    hidden = 512
    x = torch.randn(4, 32, hidden)

    # LayerNorm
    ln_custom = LayerNorm(hidden)
    ln_torch = nn.LayerNorm(hidden)
    with torch.no_grad():
        ln_torch.weight.copy_(ln_custom.weight)
        ln_torch.bias.copy_(ln_custom.bias)

    with torch.no_grad():
        diff_ln = (ln_custom(x) - ln_torch(x)).abs().max().item()
    print(f"\n  LayerNorm max diff: {diff_ln:.8f}")

    # RMSNorm（与 PyTorch >= 2.1 对比）
    rms_custom = RMSNorm(hidden)
    if hasattr(nn, "RMSNorm"):
        rms_torch = nn.RMSNorm(hidden)
        with torch.no_grad():
            rms_torch.weight.copy_(rms_custom.weight)
        with torch.no_grad():
            diff_rms = (rms_custom(x) - rms_torch(x)).abs().max().item()
        print(f"  RMSNorm max diff:   {diff_rms:.8f}")
    else:
        print(f"  RMSNorm: PyTorch 版本太旧，无可对比的内置实现。")


def main() -> None:
    compare_speed()
    compare_numerical_stability()
    compare_gradient_behavior()
    compare_with_pytorch_builtin()


if __name__ == "__main__":
    main()
