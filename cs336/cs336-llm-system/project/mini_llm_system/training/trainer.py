"""
MiniLLM 的完整训练循环。

支持以下功能：
- 梯度累积（Gradient accumulation），实现更大的有效批次大小。
- 混合精度训练（autocast + GradScaler），提高内存效率。
- checkpoint 保存与加载，支持恢复训练。
- 记录 loss、learning rate、每秒 token 数。
- 周期性评估。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# 允许直接运行此文件，也可作为包的一部分导入
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.optimizer import AdamW
from training.lr_scheduler import CosineWarmupScheduler
from training.loss import cross_entropy_loss


class Trainer:
    """
    MiniLLM 模型的训练器（Trainer）。

    负责完整的训练循环，包括前向传播、反向传播、
    优化器更新、日志记录、checkpoint 保存和评估。

    Args:
        model: MiniLLM 模型实例。
        train_dataloader: 训练数据的 DataLoader。
        val_dataloader: 验证数据的 DataLoader（可选）。
        optimizer: AdamW 优化器。
        lr_scheduler: 余弦预热调度器（Cosine warmup scheduler）。
        config: 训练配置字典。
        device: 训练设备（例如 'cuda'、'cpu'）。
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: AdamW,
        lr_scheduler: CosineWarmupScheduler,
        config: dict[str, Any],
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> None:
        self.model: nn.Module = model
        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloader: Optional[DataLoader] = val_dataloader
        self.optimizer: AdamW = optimizer
        self.lr_scheduler: CosineWarmupScheduler = lr_scheduler
        self.device: str = device
        self.config: dict[str, Any] = config

        # 配置参数
        self.max_steps: int = config.get("max_steps", 10000)
        self.gradient_accumulation_steps: int = config.get(
            "gradient_accumulation_steps", 1
        )
        self.log_interval: int = config.get("log_interval", 10)
        self.eval_interval: int = config.get("eval_interval", 500)
        self.save_interval: int = config.get("save_interval", 1000)
        self.checkpoint_dir: Path = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.mixed_precision: bool = config.get("mixed_precision", True)
        self.max_grad_norm: float = config.get("max_grad_norm", 1.0)

        # 混合精度
        self.scaler: Optional[torch.amp.GradScaler] = None
        if self.mixed_precision and device.startswith("cuda"):
            self.scaler = torch.amp.GradScaler("cuda")

        # 训练状态
        self.global_step: int = 0
        self.epoch: int = 0
        self.best_val_loss: float = float("inf")

        # 指标追踪
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.lr_history: list[float] = []

        # 将模型移至目标设备
        self.model.to(device)

        # 创建 checkpoint 目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """
        运行完整的训练循环。

        遍历 epochs，处理批次数据，执行梯度累积、
        混合精度训练，并定期进行验证评估和 checkpoint 保存。
        """
        self.model.train()

        # 用于吞吐量测量的 CUDA event 计时
        if self.device.startswith("cuda"):
            start_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
            end_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

        trainable_params: int = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Starting training...")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Device: {self.device}")

        start_time: float = time.time()
        tokens_processed: int = 0

        while self.global_step < self.max_steps:
            self.epoch += 1

            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.global_step >= self.max_steps:
                    break

                input_ids: torch.Tensor = batch["input_ids"].to(self.device)
                labels: torch.Tensor = batch["labels"].to(self.device)

                if self.device.startswith("cuda"):
                    start_event.record()

                # 混合精度前向传播
                with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                    logits, _ = self.model(input_ids)
                    loss: torch.Tensor = cross_entropy_loss(logits, labels)
                    # 对 loss 进行缩放以支持梯度累积
                    loss = loss / self.gradient_accumulation_steps

                # 反向传播
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.device.startswith("cuda"):
                    end_event.record()

                # 梯度累积：仅在累积足够梯度后才执行优化器步骤
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # 追踪指标
                batch_tokens: int = input_ids.numel()
                tokens_processed += batch_tokens

                # 日志记录
                if self.global_step % self.log_interval == 0:
                    current_lr: float = self.optimizer.param_groups[0]["lr"]
                    elapsed: float = time.time() - start_time
                    tokens_per_sec: float = tokens_processed / max(elapsed, 1e-6)

                    # 使用 CUDA 计时获取每步耗时
                    if self.device.startswith("cuda"):
                        torch.cuda.synchronize()
                        step_time_ms: float = start_event.elapsed_time(end_event)

                    self.train_losses.append(
                        loss.item() * self.gradient_accumulation_steps
                    )
                    self.lr_history.append(current_lr)

                    print(
                        f"Step {self.global_step:6d}/{self.max_steps} | "
                        f"Loss: {self.train_losses[-1]:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Tokens/s: {tokens_per_sec:.0f} | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

                # 验证评估
                if (
                    self.val_dataloader is not None
                    and self.global_step % self.eval_interval == 0
                    and self.global_step > 0
                ):
                    val_loss: float = self.evaluate()
                    self.val_losses.append(val_loss)
                    print(f"  Validation loss: {val_loss:.4f}")
                    self.model.train()

                # 保存 checkpoint
                if self.global_step % self.save_interval == 0 and self.global_step > 0:
                    self.save_checkpoint()

                if self.global_step >= self.max_steps:
                    break

        # 最终验证评估和 checkpoint 保存
        if self.val_dataloader is not None:
            val_loss = self.evaluate()
            self.val_losses.append(val_loss)
            print(f"Final validation loss: {val_loss:.4f}")

        self.save_checkpoint(final=True)
        total_time: float = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time:.1f}s")
        print(f"Total steps: {self.global_step}")
        print(f"Total tokens processed: {tokens_processed:,}")

    def evaluate(self) -> float:
        """
        在验证集上评估模型。

        Returns:
            平均验证 loss。
        """
        self.model.eval()
        total_loss: float = 0.0
        num_batches: int = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids: torch.Tensor = batch["input_ids"].to(self.device)
                labels: torch.Tensor = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                    logits, _ = self.model(input_ids)
                    loss: torch.Tensor = cross_entropy_loss(logits, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss: float = total_loss / max(num_batches, 1)
        return avg_loss

    def save_checkpoint(self, final: bool = False) -> None:
        """
        保存训练 checkpoint。

        Args:
            final: 若为 True，则保存为最终 checkpoint。
        """
        suffix: str = "final" if final else f"step_{self.global_step}"
        checkpoint_path: Path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"

        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        加载训练 checkpoint，以恢复训练。

        Args:
            checkpoint_path: checkpoint 文件路径。
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint.get("val_losses", [])

        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")


# 快速测试
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformer.config import MiniLLMConfig
    from transformer.layers import MiniLLM

    print("Testing Trainer setup...")

    # 创建一个迷你模型
    config = MiniLLMConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        max_seq_len=64,
    )
    model = MiniLLM(config)

    # 创建虚拟数据
    class DummyDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 100

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.randint(0, 100, (16,)),
                "labels": torch.randint(0, 100, (16,)),
            }

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 创建优化器和调度器
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=10, total_steps=100, min_lr_ratio=0.1
    )

    # 创建 trainer 配置
    trainer_config = {
        "max_steps": 50,
        "gradient_accumulation_steps": 2,
        "log_interval": 10,
        "eval_interval": 500,
        "save_interval": 1000,
        "checkpoint_dir": "/tmp/mini_llm_test_checkpoints",
        "mixed_precision": False,  # 使用兼容 CPU 的设置进行测试
        "max_grad_norm": 1.0,
    }

    # 创建 trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=trainer_config,
        device="cpu",
    )

    # 运行少量训练步骤
    trainer.train()

    # 清理
    import shutil

    shutil.rmtree("/tmp/mini_llm_test_checkpoints", ignore_errors=True)
    print("Trainer test passed!")
