"""
第04讲 — 训练：gradient clipping。

从零实现 ``clip_grad_norm_`` 并与 PyTorch 内置版本进行对比。
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch


# ---------------------------------------------------------------------------
# 梯度范数计算
# ---------------------------------------------------------------------------


def compute_grad_norm(
    parameters: Iterable[torch.Tensor],
    norm_type: float = 2.0,
) -> torch.Tensor:
    """计算所有参数梯度的总范数。

    参数
    ----------
    parameters : 具有 ``.grad`` 属性的张量的可迭代对象。
    norm_type : float
        范数类型（2.0 = L2，float('inf') = 最大范数）。

    返回
    -------
    total_norm : 标量张量。
    """
    grads = [p.grad for p in parameters if p.grad is not None]

    if norm_type == float("inf"):
        total_norm = torch.tensor(
            max(g.detach().abs().max().item() for g in grads),
            device=grads[0].device if grads else torch.device("cpu"),
        )
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
            norm_type,
        )
    return total_norm


# ---------------------------------------------------------------------------
# clip_grad_norm_
# ---------------------------------------------------------------------------


def clip_grad_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """对参数可迭代对象的梯度范数进行裁剪。

    范数是所有梯度拼接在一起视为单个向量后计算的。梯度在原地被修改。

    参数
    ----------
    parameters : 具有 ``.grad`` 属性的张量的可迭代对象。
    max_norm : float
        允许的最大范数。
    norm_type : float
        使用的范数类型。
    error_if_nonfinite : bool
        如果为 True，当总范数为 NaN 或 Inf 时抛出错误。

    返回
    -------
    total_norm : 标量张量（裁剪前）。
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(grads) == 0:
        return torch.tensor(0.0)

    device = grads[0].device

    if norm_type == float("inf"):
        total_norm = torch.tensor(
            max(g.detach().abs().max().item() for g in grads),
            device=device,
        )
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
            norm_type,
        )

    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError(f"Gradient norm is non-finite: {total_norm}")

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.detach().mul_(clip_coef)

    return total_norm


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def verify_clip_grad_norm() -> None:
    """将自定义 clip_grad_norm_ 与 torch.nn.utils.clip_grad_norm_ 进行比较。"""
    # 使用已知梯度创建参数
    p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    p2 = torch.nn.Parameter(torch.tensor([4.0, 5.0, 6.0]))
    p1c = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    p2c = torch.nn.Parameter(torch.tensor([4.0, 5.0, 6.0]))

    p1.grad = torch.tensor([1.0, 0.0, 0.0])
    p2.grad = torch.tensor([0.0, 0.0, 1.0])
    p1c.grad = torch.tensor([1.0, 0.0, 0.0])
    p2c.grad = torch.tensor([0.0, 0.0, 1.0])

    # 计算范数
    norm_custom = compute_grad_norm([p1, p2])
    norm_torch = torch.nn.utils.clip_grad_norm_([p1c, p2c], max_norm=1000.0)
    assert torch.allclose(norm_custom, norm_torch, atol=1e-6), (
        f"{norm_custom} vs {norm_torch}"
    )
    print(f"Grad norm match: {norm_custom:.6f} == {norm_torch:.6f}  ✓")

    # 裁剪
    max_norm = 0.5
    # 重置梯度
    p1.grad = torch.tensor([1.0, 0.0, 0.0])
    p2.grad = torch.tensor([0.0, 0.0, 1.0])
    p1c.grad = torch.tensor([1.0, 0.0, 0.0])
    p2c.grad = torch.tensor([0.0, 0.0, 1.0])

    custom_norm = clip_grad_norm_([p1, p2], max_norm=max_norm)
    torch_norm = torch.nn.utils.clip_grad_norm_([p1c, p2c], max_norm=max_norm)

    print(f"Clipped norm — custom: {custom_norm:.6f}, torch: {torch_norm:.6f}  ✓")
    print(f"p1 grad after clip: {p1.grad}")
    print(f"p1c grad after clip: {p1c.grad}")
    assert torch.allclose(p1.grad, p1c.grad, atol=1e-6), (
        "Gradient mismatch after clipping"
    )
    assert torch.allclose(p2.grad, p2c.grad, atol=1e-6), (
        "Gradient mismatch after clipping"
    )
    print("Gradient values match after clipping  ✓")

    # 无操作裁剪（max_norm > 范数）
    p1.grad = torch.tensor([1.0, 0.0, 0.0])
    p2.grad = torch.tensor([0.0, 0.0, 1.0])
    original_p1 = p1.grad.clone()
    original_p2 = p2.grad.clone()
    _ = clip_grad_norm_([p1, p2], max_norm=100.0)
    assert torch.allclose(p1.grad, original_p1), "Grads should be unchanged"
    assert torch.allclose(p2.grad, original_p2), "Grads should be unchanged"
    print("No-op clipping (max_norm >> norm) preserves grads  ✓")


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    verify_clip_grad_norm()
    print("\nAll checks passed.")
