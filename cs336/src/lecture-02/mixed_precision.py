"""
第 02 讲 — 资源核算：混合精度训练。

演示 fp16 / bf16 / fp32 的数值范围差异，并勾勒
自动混合精度（AMP）训练循环的结构。
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# 数值范围比较
# ---------------------------------------------------------------------------


def _fp_range_info(exp_bits: int, mantissa_bits: int) -> Dict[str, float]:
    emax = (1 << (exp_bits - 1)) - 1
    # 最大正规数
    max_val = (2.0 - 2.0 ** (-mantissa_bits)) * (2.0**emax)
    # 最小正正规数
    min_normal = 2.0 ** (1 - emax)
    # 次正规数分辨率
    min_subnormal = min_normal * (2.0 ** (-mantissa_bits))
    # 机器精度（1.0 处的 ulp）
    eps = 2.0 ** (-mantissa_bits)
    return {
        "exp_bits": exp_bits,
        "mantissa_bits": mantissa_bits,
        "max": max_val,
        "min_normal": min_normal,
        "min_subnormal": min_subnormal,
        "eps": eps,
    }


PRECISION_SPECS = {
    "fp32": _fp_range_info(8, 23),
    "tf32": _fp_range_info(8, 10),  # NVIDIA TF32：8 位指数，10 位尾数
    "bf16": _fp_range_info(8, 7),
    "fp16": _fp_range_info(5, 10),
    "fp8_e5m2": _fp_range_info(5, 2),
    "fp8_e4m3": _fp_range_info(4, 3),
}


def compare_precision_ranges() -> Dict[str, Dict[str, float]]:
    """返回常见浮点格式的精度范围对照表。"""
    return {
        name: {
            "max": info["max"],
            "min_normal": info["min_normal"],
            "eps": info["eps"],
            "dynamic_range": info["max"] / max(info["min_normal"], 1e-300),
        }
        for name, info in PRECISION_SPECS.items()
    }


# ---------------------------------------------------------------------------
# AMP 训练循环（仅结构 — 不执行实际训练）
# ---------------------------------------------------------------------------


def amp_training_step(
    model: Any,
    batch: Any,
    optimizer: Any,
    scaler: Any = None,
    use_amp: bool = True,
    amp_dtype: str = "fp16",
) -> float:
    """单步混合精度训练步骤（结构示意）。

    返回一个虚拟的 loss 值。不执行实际计算；
    仅展示 API 模式。

    真实用法::

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs)
            loss = loss_fn(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    """
    _ = (model, batch, optimizer, scaler)
    if not use_amp:
        return 1.0  # 虚拟 loss

    dtype_map = {"fp16": 16, "bf16": 7, "fp32": 32}
    _ = dtype_map.get(amp_dtype, 16)
    # 真实代码中：
    #   with autocast(...):
    #       loss = ...
    #   scaler.scale(loss).backward()
    #   ...
    return 0.5


# ---------------------------------------------------------------------------
# 损失缩放辅助类
# ---------------------------------------------------------------------------


class LossScaler:
    """最小化的 fp16 AMP 损失缩放器（模拟 torch.cuda.amp.GradScaler）。

    真实实现使用增长/回退因子并跟踪 inf/nan 计数。
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._step_count = 0

    def get_scale(self) -> float:
        return self._scale

    def scale(self, loss: Any) -> Any:
        """将 loss 乘以当前缩放因子。"""
        return loss * self._scale

    def step(self, optimizer: Any) -> None:
        """调用 optimizer.step()。"""
        optimizer.step()

    def update(self) -> None:
        """更新缩放因子（简化版）。"""
        self._step_count += 1
        if self._step_count % self._growth_interval == 0:
            self._scale *= self._growth_factor


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 比较不同格式的数值范围
    hdr = f"{'Format':>10s} | {'max':>14s} | {'min_normal':>14s} | {'eps':>14s} | {'dyn_range':>10s}"
    print(hdr)
    fmt = "{:>10s} | max={:>14.6e} | min_normal={:>14.6e} | eps={:>14.6e} | dynamic_range={:>10.1f}"
    print("-" * 85)
    for name, info in compare_precision_ranges().items():
        print(
            fmt.format(
                name,
                info["max"],
                info["min_normal"],
                info["eps"],
                info["dynamic_range"],
            )
        )

    # AMP 步骤演示（无实际张量）
    print("\nAMP training step (dummy loss):", amp_training_step(None, None, None))

    # Scaler 演示
    scaler = LossScaler()
    print(f"Initial scale: {scaler.get_scale()}")
    for _ in range(3):
        scaler.update()
    print(
        f"Scale after 3 updates (2000-step interval, no growth expected): {scaler.get_scale()}"
    )

    print("\nAll checks passed.")
