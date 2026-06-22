"""
第04讲 — 训练：简单的训练 / 评估循环。

提供一个轻量级的 PyTorch 模型训练循环框架。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., torch.Tensor],
    device: torch.device | str = "cpu",
    clip_grad_norm: Optional[float] = None,
    log_interval: int = 10,
) -> Dict[str, float]:
    """运行一个训练 epoch。

    返回包含 ``loss``、``lr`` 和 ``time`` 的字典。
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    start = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader):
        # 解包 batch — 假设为 (input_ids, targets) 或类似格式
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs = targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        bs = inputs.size(0) if hasattr(inputs, "size") else len(inputs)  # type: ignore[arg-type]
        total_loss += loss.item() * bs
        total_samples += bs

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / max(total_samples, 1)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [train] batch {batch_idx + 1:4d} | loss={avg_loss:.4f} | lr={lr:.2e}"
            )

    elapsed = time.perf_counter() - start
    return {
        "loss": total_loss / max(total_samples, 1),
        "lr": optimizer.param_groups[0]["lr"],
        "time": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[..., torch.Tensor],
    device: torch.device | str = "cpu",
) -> Dict[str, float]:
    """在验证 / 测试集上评估模型。

    返回包含 ``loss`` 和 ``perplexity`` 的字典。
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs = targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        bs = inputs.size(0) if hasattr(inputs, "size") else len(inputs)  # type: ignore[arg-type]
        total_loss += loss.item() * bs
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return {"loss": avg_loss, "perplexity": perplexity}


# ---------------------------------------------------------------------------
# 完整训练流程封装
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., torch.Tensor],
    num_epochs: int = 3,
    device: torch.device | str = "cpu",
    clip_grad_norm: Optional[float] = 1.0,
    log_interval: int = 10,
) -> List[Dict[str, Any]]:
    """跨多个 epoch 的完整训练循环。

    返回每个 epoch 的指标列表。
    """
    history: List[Dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            clip_grad_norm=clip_grad_norm,
            log_interval=log_interval,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device=device)
        print(
            f"  train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_ppl={val_metrics['perplexity']:.2f}"
        )
        history.append({**train_metrics, **val_metrics, "epoch": epoch})

    return history


# ---------------------------------------------------------------------------
# 演示（需要一个简单模型 + 随机数据）
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    device = torch.device("cpu")
    vocab_size = 512
    dim = 64
    B, S = 4, 16

    # 虚拟模型
    class _DummyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.lin = nn.Linear(dim, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(self.embed(x))

    model = _DummyLM().to(device)

    # 随机数据
    x = torch.randint(0, vocab_size, (B * 20, S))
    y = torch.randint(0, vocab_size, (B * 20, S))
    train_ds = TensorDataset(x[: B * 10], y[: B * 10])
    val_ds = TensorDataset(x[B * 10 :], y[B * 10 :])

    train_loader = DataLoader(train_ds, batch_size=B, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=B)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 使用一个 cross-entropy 包装函数，处理 (B, S, V) → (B*S, V) 的 reshape
    def _loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, S, V) → CrossEntropyLoss 需要 (N, C) 或 (N, C, d1...) 格式
        # 将其 permute 为 (B*S, V) 作为常用的批处理模式
        return nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )

    loss_fn = _loss_fn

    history = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        num_epochs=2,
        device=device,
        clip_grad_norm=1.0,
        log_interval=5,
    )

    print(f"\nTraining history: {len(history)} epochs")
    for entry in history:
        print(
            f"  epoch {entry['epoch']}: train_loss={entry['loss']:.4f}, val_loss={entry.get('loss', 0):.4f}"
        )
    print("\nTraining loop completed successfully.")
