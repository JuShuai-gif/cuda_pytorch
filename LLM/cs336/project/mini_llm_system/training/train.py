"""
MiniLLM 训练入口脚本。

本脚本提供用于训练模型的命令行接口。
用法：
    python train.py --config config.json
    python train.py --resume checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# 将父目录添加到路径中以供导入使用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.config import MiniLLMConfig
from transformer.layers import MiniLLM
from training.optimizer import AdamW
from training.lr_scheduler import CosineWarmupScheduler
from training.dataset import TextDataset, create_dataloader
from training.trainer import Trainer
from tokenizer.bpe_tokenizer import BPETokenizer


def get_default_config() -> dict[str, Any]:
    """返回默认训练配置。"""
    return {
        # 模型
        "model": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "num_kv_heads": 4,
            "intermediate_size": 3072,
            "max_seq_len": 2048,
            "norm_eps": 1e-5,
            "rope_theta": 10000.0,
        },
        # 训练
        "training": {
            "max_steps": 100000,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "warmup_steps": 1000,
            "min_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
            "mixed_precision": True,
            "seq_len": 2048,
            "log_interval": 10,
            "eval_interval": 500,
            "save_interval": 5000,
        },
        # 数据
        "data": {
            "train_file": "data/train.txt",
            "val_file": "data/val.txt",
            "tokenizer_path": "tokenizer.json",
        },
        # 系统
        "system": {
            "checkpoint_dir": "./checkpoints",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_workers": 4,
            "seed": 42,
        },
    }


def build_model(config: dict[str, Any]) -> MiniLLM:
    """根据配置构建 MiniLLM 模型。"""
    model_cfg: dict[str, Any] = config["model"]
    mini_cfg = MiniLLMConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        num_kv_heads=model_cfg["num_kv_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_seq_len=model_cfg["max_seq_len"],
        norm_eps=model_cfg.get("norm_eps", 1e-5),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
    )
    return MiniLLM(mini_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniLLM")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Path to training text file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Path to validation text file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to pre-trained tokenizer JSON file",
    )
    args = parser.parse_args()

    # 加载配置
    cfg: dict[str, Any] = get_default_config()
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, "r") as f:
                user_cfg = json.load(f)
            # 深度合并（简单的顶层合并，用于演示）
            for key in user_cfg:
                if key in cfg:
                    cfg[key].update(user_cfg[key])
                else:
                    cfg[key] = user_cfg[key]

    # 通过命令行参数覆盖
    if args.train_file:
        cfg["data"]["train_file"] = args.train_file
    if args.val_file:
        cfg["data"]["val_file"] = args.val_file
    if args.tokenizer:
        cfg["data"]["tokenizer_path"] = args.tokenizer

    # 设置随机种子
    seed: int = cfg["system"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device: str = cfg["system"]["device"]
    train_cfg: dict[str, Any] = cfg["training"]
    data_cfg: dict[str, Any] = cfg["data"]
    sys_cfg: dict[str, Any] = cfg["system"]

    print("=" * 60)
    print("MiniLLM Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # 加载或训练分词器
    tokenizer: BPETokenizer
    tokenizer_path = Path(data_cfg["tokenizer_path"])
    if tokenizer_path.exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print("Training new tokenizer...")
        tokenizer = BPETokenizer()
        # 在训练数据上训练
        train_file = Path(data_cfg["train_file"])
        if train_file.exists():
            with open(train_file, "r") as f:
                texts: list[str] = [line.strip() for line in f if line.strip()]
            tokenizer.train(texts, vocab_size=cfg["model"]["vocab_size"])
            tokenizer.save(tokenizer_path)
            print(f"Tokenizer trained and saved to {tokenizer_path}")
        else:
            print(f"Warning: Training file not found: {train_file}")

    # 构建模型
    print("Building model...")
    model: MiniLLM = build_model(cfg)
    print(f"Model parameters: {model.get_num_params():,}")

    # 创建数据集
    train_file = Path(data_cfg["train_file"])
    val_file = Path(data_cfg["val_file"])

    train_dataset = TextDataset(
        train_file,
        tokenizer,
        seq_len=train_cfg["seq_len"],
    )

    val_dataset = None
    val_dataloader = None
    if val_file.exists():
        val_dataset = TextDataset(val_file, tokenizer, seq_len=train_cfg["seq_len"])
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=sys_cfg["num_workers"],
        )

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=sys_cfg["num_workers"],
    )

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation samples: {len(val_dataset)}")

    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # 创建学习率调度器
    lr_scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=train_cfg["max_steps"],
        min_lr_ratio=train_cfg["min_lr_ratio"],
    )

    # 创建训练器配置
    trainer_config: dict[str, Any] = {
        "max_steps": train_cfg["max_steps"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "log_interval": train_cfg["log_interval"],
        "eval_interval": train_cfg["eval_interval"],
        "save_interval": train_cfg["save_interval"],
        "checkpoint_dir": sys_cfg["checkpoint_dir"],
        "mixed_precision": train_cfg["mixed_precision"],
        "max_grad_norm": train_cfg["max_grad_norm"],
    }

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=trainer_config,
        device=device,
    )

    # 如果指定了检查点，则从中恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
