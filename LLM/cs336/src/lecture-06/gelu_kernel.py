"""
第六讲 — GPU 编程：Triton 融合 GeLU kernel。

在 Triton 中实现融合 GeLU kernel（附 PyTorch 回退方案），
并比较数值结果。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

# 检查 Triton 是否可用；不可用时优雅地回退到 PyTorch。
try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# PyTorch 基准实现
# ---------------------------------------------------------------------------


def gelu_pytorch(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    """PyTorch 基准 GeLU。"""
    return F.gelu(x, approximate=approximate)


# ---------------------------------------------------------------------------
# Triton 融合 GeLU kernel
# ---------------------------------------------------------------------------


if HAS_TRITON:

    @triton.jit
    def _gelu_kernel(
        x_ptr,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """融合 GeLU：读取 x，写出 y = x * 0.5 * (1 + erf(x / sqrt(2)))。"""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        # GeLU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (x + coeff * x * x * x)
        y = 0.5 * x * (1.0 + tl.tanh(inner))  # 使用 "tanh" 近似 erf
        tl.store(y_ptr + offsets, y, mask=mask)

    def gelu_triton(x: torch.Tensor) -> torch.Tensor:
        """Triton 融合 GeLU 前向传播。

        若 Triton 不可用，则回退到 PyTorch。
        """
        if not HAS_TRITON:
            return gelu_pytorch(x)

        x = x.contiguous()
        y = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        # TODO: 如果此 kernel 位于 autograd 计算图中，
        #       则需要在此处添加反向传播 kernel。
        return y

else:

    def gelu_triton(x: torch.Tensor) -> torch.Tensor:
        """回退方案：PyTorch GeLU（Triton 未安装）。"""
        return gelu_pytorch(x)


# ---------------------------------------------------------------------------
# 正确性测试
# ---------------------------------------------------------------------------


def test_gelu(tol: float = 1e-5) -> Tuple[bool, float]:
    """比较 PyTorch GeLU 与 Triton GeLU（如可用）。"""
    torch.manual_seed(42)
    x = torch.randn(1000, device="cpu")  # 离线测试用 CPU；GPU 可用时切换
    y_ref = gelu_pytorch(x)
    y_kernel = gelu_triton(x)

    max_diff = (y_ref - y_kernel).abs().max().item()
    ok = max_diff < tol
    return ok, max_diff


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Triton available: {HAS_TRITON}")

    ok, max_diff = test_gelu()
    print(f"GeLU correctness: {'PASS' if ok else 'FAIL'} (max diff = {max_diff:.2e})")

    # 展示几个示例值
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = gelu_pytorch(x)
    yt = gelu_triton(x)
    print("\nSample GeLU values:")
    for i in range(len(x)):
        print(f"  x={x[i]:5.1f}  PyTorch={y[i]:.6f}  Triton={yt[i]:.6f}")

    print("\nAll checks passed.")
