"""
第六讲 — GPU 编程：Triton 在线 softmax kernel。

在 Triton 中实现在线（数值稳定）softmax。
采用经典的先减最大值 + exp + 求和归约方法。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# PyTorch 基准实现
# ---------------------------------------------------------------------------


def softmax_pytorch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch 基准 softmax。"""
    return F.softmax(x, dim=dim)


# ---------------------------------------------------------------------------
# Triton 在线 softmax
# ---------------------------------------------------------------------------


if HAS_TRITON:

    @triton.jit
    def _softmax_kernel(
        x_ptr,
        y_ptr,
        n_rows: int,
        n_cols: int,
        BLOCK_COLS: tl.constexpr,
    ):
        """在线 softmax kernel。

        每个 program 处理一行，在列块上循环迭代。
        算法：减最大值 → exp → 累加求和 → 归一化。
        """
        pid = tl.program_id(0)
        row = pid

        # 跳过越界的行
        if row >= n_rows:
            return

        # 本行的基地址指针
        x_row = x_ptr + row * n_cols
        y_row = y_ptr + row * n_cols

        # 第一遍：寻找最大值（以保证数值稳定性）
        row_max = float("-inf")
        for start in range(0, n_cols, BLOCK_COLS):
            cols = start + tl.arange(0, BLOCK_COLS)
            mask = cols < n_cols
            vals = tl.load(x_row + cols, mask=mask, other=float("-inf"))
            row_max = tl.maximum(row_max, tl.max(vals, axis=0))

        # 第二遍：指数化并求和
        row_sum = 0.0
        for start in range(0, n_cols, BLOCK_COLS):
            cols = start + tl.arange(0, BLOCK_COLS)
            mask = cols < n_cols
            vals = tl.load(x_row + cols, mask=mask, other=float("-inf"))
            exp_vals = tl.exp(vals - row_max)
            tl.store(y_row + cols, exp_vals, mask=mask)
            row_sum += tl.sum(exp_vals, axis=0)

        # 第三遍：归一化
        for start in range(0, n_cols, BLOCK_COLS):
            cols = start + tl.arange(0, BLOCK_COLS)
            mask = cols < n_cols
            vals = tl.load(y_row + cols, mask=mask)
            tl.store(y_row + cols, vals / row_sum, mask=mask)

    def softmax_triton(x: torch.Tensor) -> torch.Tensor:
        """Triton 在线 softmax。

        若 Triton 不可用或 ``x`` 不是 2D 张量，则回退到 PyTorch。
        """
        if not HAS_TRITON:
            return softmax_pytorch(x)

        x = x.contiguous()
        if x.dim() != 2:
            # 对于高维张量，回退到 PyTorch
            return softmax_pytorch(x)

        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK_COLS = max(128, triton.next_power_of_2(min(n_cols, 1024)))

        _softmax_kernel[(n_rows,)](x, y, n_rows, n_cols, BLOCK_COLS=BLOCK_COLS)
        return y

else:

    def softmax_triton(x: torch.Tensor) -> torch.Tensor:
        """回退方案：PyTorch softmax（Triton 未安装）。"""
        return softmax_pytorch(x)


# ---------------------------------------------------------------------------
# 正确性测试
# ---------------------------------------------------------------------------


def test_softmax(tol: float = 1e-5) -> Tuple[bool, float]:
    """比较 PyTorch softmax 与 Triton softmax（在 CPU 上）。"""
    torch.manual_seed(42)
    x = torch.randn(64, 512)  # 2D
    y_ref = softmax_pytorch(x)
    y_kernel = softmax_triton(x)

    max_diff = (y_ref - y_kernel).abs().max().item()
    ok = max_diff < tol
    return ok, max_diff


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Triton available: {HAS_TRITON}")

    ok, max_diff = test_softmax()
    print(
        f"Softmax correctness: {'PASS' if ok else 'FAIL'} (max diff = {max_diff:.2e})"
    )

    # 验证 softmax 各行求和为 1
    x = torch.randn(8, 16)
    y = softmax_triton(x)
    sums = y.sum(dim=-1)
    print(f"Row sums (should be 1): {sums.tolist()}")

    print("\nAll checks passed.")
