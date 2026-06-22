"""
mup_demo.py — muP (Maximal Update Parameterization) 演示

实现一个小型 MLP，比较不同网络宽度下
Standard Parameterization (SP) 与 muP 的行为。展示:

  1. 随宽度增加时的 activation RMS 稳定性。
  2. 随宽度增加时的 gradient RMS 稳定性。
  3. 输出层 activation 缩放。
  4. 窄网络 vs 宽网络的训练动态（SP vs muP）。

关键 muP 规则:
  - 隐藏层权重: 与 SP 相同的初始化方差，但 LR ∝ 1 / width。
  - 输出权重: 以 1/d² 方差初始化（非常小），constant LR。
  - 这使得在 d → ∞ 时，所有 activations、gradients 和 feature updates 保持 O(1)。

图表保存为 PNG 文件。
用法:
    python mup_demo.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Dict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# MLP 实现: Standard vs muP
# ---------------------------------------------------------------------------


class StandardMLP:
    """3层 MLP，采用标准（He-like）参数化。

    所有权重以 N(0, 1/fan_in) 初始化，所有学习率相等。
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1: NDArray[np.float64] = rng.normal(
            0, np.sqrt(1.0 / d_in), (d_in, d_hidden)
        )
        self.W2: NDArray[np.float64] = rng.normal(
            0, np.sqrt(1.0 / d_hidden), (d_hidden, d_hidden)
        )
        self.W3: NDArray[np.float64] = rng.normal(
            0, np.sqrt(1.0 / d_hidden), (d_hidden, d_out)
        )
        self.b1: NDArray[np.float64] = np.zeros(d_hidden)
        self.b2: NDArray[np.float64] = np.zeros(d_hidden)
        self.b3: NDArray[np.float64] = np.zeros(d_out)

    def forward(
        self, x: NDArray[np.float64]
    ) -> Tuple[
        NDArray[np.float64], List[NDArray[np.float64]], List[NDArray[np.float64]]
    ]:
        """前向传播，返回 (output, pre_acts, post_acts)。"""
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        z3 = a2 @ self.W3 + self.b3  # 输出层无激活函数
        return z3, [z1, z2, z3], [x, a1, a2, z3]

    def backward(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        lr: float,
    ) -> None:
        """使用 MSE loss 的 SGD 步骤（批量梯度下降）。"""
        batch = x.shape[0]
        out, pre, post = self.forward(x)

        # Loss = MSE
        d_out = 2.0 * (out - y) / batch

        # 第 3 层（输出层）
        dW3 = post[2].T @ d_out
        db3 = d_out.sum(axis=0)
        d_pre2 = (d_out @ self.W3.T) * (pre[1] > 0)  # ReLU 反向传播

        # 第 2 层
        dW2 = post[1].T @ d_pre2
        db2 = d_pre2.sum(axis=0)
        d_pre1 = (d_pre2 @ self.W2.T) * (pre[0] > 0)

        # 第 1 层
        dW1 = post[0].T @ d_pre1
        db1 = d_pre1.sum(axis=0)

        # 所有层使用相同 LR 更新
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3


class MuPMLP:
    """3层 MLP，采用 muP (Maximal Update Parameterization)。

    与 StandardMLP 的关键区别:
      - 输出权重以 1/d² 方差初始化。
      - 隐藏层学习率按 base_width / width 缩放。
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        seed: int = 42,
        base_width: int = 128,
    ):
        rng = np.random.default_rng(seed)
        # 输入和隐藏层权重: 与 SP 相同
        self.W1: NDArray[np.float64] = rng.normal(
            0, np.sqrt(1.0 / d_in), (d_in, d_hidden)
        )
        self.W2: NDArray[np.float64] = rng.normal(
            0, np.sqrt(1.0 / d_hidden), (d_hidden, d_hidden)
        )

        # 输出权重: muP 使用 1/d² 方差（非常小的初始化）
        self.W3: NDArray[np.float64] = rng.normal(
            0, np.sqrt(1.0 / (d_hidden * d_hidden)), (d_hidden, d_out)
        )

        self.b1: NDArray[np.float64] = np.zeros(d_hidden)
        self.b2: NDArray[np.float64] = np.zeros(d_hidden)
        self.b3: NDArray[np.float64] = np.zeros(d_out)

        self._base_width = base_width
        self._d_hidden = d_hidden

    def forward(
        self, x: NDArray[np.float64]
    ) -> Tuple[
        NDArray[np.float64], List[NDArray[np.float64]], List[NDArray[np.float64]]
    ]:
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ self.W3 + self.b3
        return z3, [z1, z2, z3], [x, a1, a2, z3]

    def backward(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        lr: float,
    ) -> None:
        """使用 muP 学习率缩放的 SGD 步骤。"""
        batch = x.shape[0]
        out, pre, post = self.forward(x)

        d_out = 2.0 * (out - y) / batch

        # 第 3 层（输出层）— 恒定 LR
        dW3 = post[2].T @ d_out
        db3 = d_out.sum(axis=0)
        d_pre2 = (d_out @ self.W3.T) * (pre[1] > 0)

        # 第 2 层（隐藏层）— LR 按 base_width / width 缩放
        dW2 = post[1].T @ d_pre2
        db2 = d_pre2.sum(axis=0)
        d_pre1 = (d_pre2 @ self.W2.T) * (pre[0] > 0)

        # 第 1 层（输入层）— 恒定 LR
        dW1 = post[0].T @ d_pre1
        db1 = d_pre1.sum(axis=0)

        # muP LR 缩放
        hidden_lr_scale = float(self._base_width) / float(self._d_hidden)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1  # 恒定 LR
        self.W2 -= lr * hidden_lr_scale * dW2
        self.b2 -= lr * hidden_lr_scale * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3  # 恒定 LR


# ---------------------------------------------------------------------------
# 合成回归数据
# ---------------------------------------------------------------------------


def make_regression_data(
    n_samples: int,
    d_in: int,
    d_out: int = 1,
    seed: int = 123,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """生成一个简单的非线性回归数据集。

    y = sin(2 * W_true @ x)，其中 W_true 是一个随机投影。
    """
    rng = np.random.default_rng(seed)
    W_true = rng.normal(0, 1.0 / np.sqrt(d_in), (d_in, d_out))
    X = rng.normal(0, 1, (n_samples, d_in))
    Y = np.sin(2.0 * X @ W_true)
    return X, Y


# ---------------------------------------------------------------------------
# 分析: activation 和 gradient RMS 随宽度的变化
# ---------------------------------------------------------------------------


def compute_rms_activations(
    mlp_class: type,
    widths: List[int],
    d_in: int = 64,
    d_out: int = 1,
    n_samples: int = 256,
) -> Dict[str, List[float]]:
    """计算不同宽度下各层 activations 的 RMS。"""
    X, _ = make_regression_data(n_samples, d_in, d_out, seed=99)
    rms: Dict[str, List[float]] = {
        "pre1": [],
        "post1": [],
        "pre2": [],
        "post2": [],
        "pre3": [],
        "post3": [],
    }

    for d in widths:
        net = mlp_class(d_in, d, d_out, seed=42)
        out, pre, post = net.forward(X)
        rms["pre1"].append(float(np.sqrt(np.mean(pre[0] ** 2))))
        rms["post1"].append(float(np.sqrt(np.mean(post[1] ** 2))))
        rms["pre2"].append(float(np.sqrt(np.mean(pre[1] ** 2))))
        rms["post2"].append(float(np.sqrt(np.mean(post[2] ** 2))))
        rms["pre3"].append(float(np.sqrt(np.mean(pre[2] ** 2))))
        rms["post3"].append(float(np.sqrt(np.mean(post[3] ** 2))))
    return rms


def compute_rms_gradients(
    mlp_class: type,
    widths: List[int],
    d_in: int = 64,
    d_out: int = 1,
    n_samples: int = 256,
) -> Dict[str, List[float]]:
    """计算不同宽度下各层权重梯度的 RMS。"""
    X, Y = make_regression_data(n_samples, d_in, d_out, seed=99)
    grad_rms: Dict[str, List[float]] = {"dW1": [], "dW2": [], "dW3": []}

    for d in widths:
        net = mlp_class(d_in, d, d_out, seed=42)
        out, pre, post = net.forward(X)

        batch = X.shape[0]
        d_out_val = 2.0 * (out - Y) / batch

        d_pre2 = (d_out_val @ net.W3.T) * (pre[1] > 0)
        dW3 = post[2].T @ d_out_val

        d_pre1 = (d_pre2 @ net.W2.T) * (pre[0] > 0)
        dW2 = post[1].T @ d_pre2

        dW1 = post[0].T @ d_pre1

        grad_rms["dW1"].append(float(np.sqrt(np.mean(dW1**2))))
        grad_rms["dW2"].append(float(np.sqrt(np.mean(dW2**2))))
        grad_rms["dW3"].append(float(np.sqrt(np.mean(dW3**2))))
    return grad_rms


# ---------------------------------------------------------------------------
# 训练动态对比
# ---------------------------------------------------------------------------


def train_one_model(
    mlp_class: type,
    d_hidden: int,
    lr: float,
    d_in: int = 64,
    d_out: int = 1,
    n_steps: int = 500,
    seed: int = 42,
) -> List[float]:
    """训练一个 MLP 并返回 loss 历史。"""
    X, Y = make_regression_data(256, d_in, d_out, seed=123)
    net = mlp_class(d_in, d_hidden, d_out, seed=seed)
    losses: List[float] = []
    for step in range(n_steps):
        out, _, _ = net.forward(X)
        loss = float(np.mean((out - Y) ** 2))
        losses.append(loss)
        net.backward(X, Y, lr)
    return losses


# ---------------------------------------------------------------------------
# 绘图辅助函数
# ---------------------------------------------------------------------------


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 8,
            "figure.figsize": (8, 5.5),
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
        }
    )


# ---------------------------------------------------------------------------
# 主演示
# ---------------------------------------------------------------------------


def main() -> None:
    set_style()
    print("=" * 70)
    print("muP (Maximal Update Parameterization) Demonstration")
    print("=" * 70)

    d_in, d_out = 64, 1
    widths = [32, 64, 128, 256, 512, 1024]

    # ---- 1. Activation RMS 随宽度的变化 ----
    print("\n[1] Activation RMS across widths (output-layer pre-activation) ...")
    sp_act = compute_rms_activations(StandardMLP, widths, d_in, d_out)
    mup_act = compute_rms_activations(MuPMLP, widths, d_in, d_out)

    print(f"    {'Width':>6s}  {'SP(out)':>10s}  {'muP(out)':>10s}")
    for i, d in enumerate(widths):
        print(f"    {d:6d}  {sp_act['post3'][i]:10.4f}  {mup_act['post3'][i]:10.4f}")

    # ---- 2. Gradient RMS 随宽度的变化 ----
    print("\n[2] Gradient RMS across widths ...")
    sp_grad = compute_rms_gradients(StandardMLP, widths, d_in, d_out)
    mup_grad = compute_rms_gradients(MuPMLP, widths, d_in, d_out)

    print(
        f"    {'Width':>6s}  {'SP(dW1)':>10s}  {'muP(dW1)':>10s}  {'SP(dW3)':>10s}  {'muP(dW3)':>10s}"
    )
    for i, d in enumerate(widths):
        print(
            f"    {d:6d}  {sp_grad['dW1'][i]:10.6f}  {mup_grad['dW1'][i]:10.6f}  "
            f"{sp_grad['dW3'][i]:10.6f}  {mup_grad['dW3'][i]:10.6f}"
        )

    # ---- 3. 训练动态 ----
    print("\n[3] Training dynamics (narrow=64 vs wide=1024, SP vs muP) ...")
    lr_sp, lr_mup = 0.01, 0.01
    steps = 200

    loss_sp_narrow = train_one_model(StandardMLP, 64, lr_sp, n_steps=steps)
    loss_sp_wide = train_one_model(StandardMLP, 1024, lr_sp, n_steps=steps)
    loss_mup_narrow = train_one_model(MuPMLP, 64, lr_mup, n_steps=steps)
    loss_mup_wide = train_one_model(MuPMLP, 1024, lr_mup, n_steps=steps)

    print(
        f"    Final losses: SP narrow={loss_sp_narrow[-1]:.5f}, "
        f"SP wide={loss_sp_wide[-1]:.5f}, "
        f"muP narrow={loss_mup_narrow[-1]:.5f}, "
        f"muP wide={loss_mup_wide[-1]:.5f}"
    )

    # ---- 4. 绘图 ----

    # 4a: Activation RMS（输出层） vs 宽度
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(widths, sp_act["post3"], "o-", color="crimson", label="SP output RMS")
    ax.plot(widths, mup_act["post3"], "s-", color="steelblue", label="muP output RMS")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Hidden width  d")
    ax.set_ylabel("Activation RMS (output logits)")
    ax.set_title("Output Activation Scale vs Width: SP vs muP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("mup_activation_rms.png", bbox_inches="tight")
    print("[Saved] mup_activation_rms.png")
    plt.close(fig)

    # 4b: Gradient RMS（输出权重） vs 宽度
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(widths, sp_grad["dW3"], "o-", color="crimson", label="SP dW3 RMS")
    ax.plot(widths, mup_grad["dW3"], "s-", color="steelblue", label="muP dW3 RMS")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Hidden width  d")
    ax.set_ylabel("Gradient RMS (output weights)")
    ax.set_title("Output-Weight Gradient Scale vs Width: SP vs muP")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("mup_gradient_rms.png", bbox_inches="tight")
    print("[Saved] mup_gradient_rms.png")
    plt.close(fig)

    # 4c: 训练 loss 曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, param, narrow_loss, wide_loss, color_n, color_w in [
        (axes[0], "SP", loss_sp_narrow, loss_sp_wide, "darkred", "lightcoral"),
        (axes[1], "muP", loss_mup_narrow, loss_mup_wide, "darkblue", "lightblue"),
    ]:
        ax.plot(narrow_loss, color=color_n, linewidth=1.5, label=f"d=64 (narrow)")
        ax.plot(wide_loss, color=color_w, linewidth=1.5, label=f"d=1024 (wide)")
        ax.set_xlabel("SGD steps")
        ax.set_title(f"{param}: narrow vs wide")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    axes[0].set_ylabel("MSE Loss (log scale)")

    fig.suptitle(
        "Training Dynamics: Standard vs muP across widths", fontsize=13, y=1.02
    )
    fig.tight_layout()
    fig.savefig("mup_training_dynamics.png", bbox_inches="tight")
    print("[Saved] mup_training_dynamics.png")
    plt.close(fig)

    # 4d: 汇总柱状图 — 最终 loss 比率 (wide / narrow)
    fig, ax = plt.subplots(figsize=(5, 4))
    sp_ratio = loss_sp_wide[-1] / max(loss_sp_narrow[-1], 1e-8)
    mup_ratio = loss_mup_wide[-1] / max(loss_mup_narrow[-1], 1e-8)
    ax.bar(
        ["SP", "muP"],
        [sp_ratio, mup_ratio],
        color=["crimson", "steelblue"],
        alpha=0.85,
        width=0.45,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Final loss ratio  (wide / narrow)")
    ax.set_title("Width Transfer: SP breaks, muP preserves")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("mup_width_transfer.png", bbox_inches="tight")
    print("[Saved] mup_width_transfer.png")
    plt.close(fig)

    print("\nAll muP plots saved.")


if __name__ == "__main__":
    main()
