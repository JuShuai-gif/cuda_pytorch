"""
语言模型训练的简单 dataset 与 dataloader。

支持读取文本文件、进行 tokenize，并创建 (input, target) 对，
其中 target 是 input 向后偏移一个位置的结果。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class TextDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Map-style dataset，加载已 tokenize 的文本并产生 (input, target) 对。

    每个样本是一个连续的 token ID 块。input 为 tokens[0:n-1]，
    target 为 tokens[1:n]（向后偏移一位）。

    Args:
        file_path: 文本文件路径（每行一个样本或原始文本）。
        tokenizer: 具有 `encode` 方法的 tokenizer，返回 list[int]。
        seq_len: 每个输入序列的长度。
        stride: 连续样本之间的步长（默认：seq_len，即无重叠）。
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer,  # duck-typed: 具有 encode(text) -> list[int] 方法
        seq_len: int = 2048,
        stride: int | None = None,
    ) -> None:
        self.file_path: Path = Path(file_path)
        self.seq_len: int = seq_len
        self.stride: int = stride if stride is not None else seq_len

        # 读取并 tokenize 全部文本
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_text: str = f.read()

        # 对整个语料进行 tokenize
        self.tokens: list[int] = tokenizer.encode(raw_text, add_special_tokens=False)

        # 计算样本数量
        self.num_samples: int = max(0, (len(self.tokens) - seq_len) // self.stride)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start: int = idx * self.stride
        end: int = start + self.seq_len

        # Input 为 tokens[start:end]，Target 为 tokens[start+1:end+1]
        input_ids: list[int] = self.tokens[start:end]
        target_ids: list[int] = self.tokens[start + 1 : end + 1]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
        }


class StreamingTextDataset(IterableDataset[dict[str, torch.Tensor]]):
    """
    用于流式处理大型文本语料的 Iterable-style dataset。

    逐行读取文件，进行 tokenize，并 yield (input, target) 对。
    适用于无法全部加载到内存中的超大数据集。

    Args:
        file_path: 文本文件路径。
        tokenizer: 具有 `encode` 方法的 tokenizer，返回 list[int]。
        seq_len: 每个序列的长度（input + target 合计）。
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer,  # duck-typed
        seq_len: int = 2048,
    ) -> None:
        self.file_path: Path = Path(file_path)
        self.seq_len: int = seq_len
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_ids: list[int] = self.tokenizer.encode(
                    line, add_special_tokens=False
                )
                buffer.extend(token_ids)

                # 当 buffer 足够大时 yield chunk
                while len(buffer) >= self.seq_len + 1:
                    chunk: list[int] = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len :]

                    yield {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    创建用于训练的 DataLoader。

    Args:
        dataset: PyTorch Dataset 实例。
        batch_size: 每个 batch 的样本数。
        shuffle: 是否在每个 epoch 打乱数据。
        num_workers: 数据加载的子进程数。
        pin_memory: 是否 pin memory 以加速 GPU 传输。

    Returns:
        配置好的 DataLoader。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 丢弃不完整的 batch 以保持 batch size 一致
    )


# 快速测试
if __name__ == "__main__":
    import tempfile
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tokenizer.bpe_tokenizer import BPETokenizer

    # 创建测试数据 - 使用简单的重复文本以加快 BPE 训练
    test_text: str = "the quick brown fox jumps over the lazy dog. " * 200
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(test_text)
        tmp_path: str = f.name

    try:
        # 训练一个小型 tokenizer - 使用较高的 min_frequency 以限制 merge 次数（更快）
        tokenizer = BPETokenizer()
        tokenizer.train([test_text], vocab_size=280, min_frequency=5)

        # 使用较小的 seq_len 创建 dataset
        seq_len: int = 32
        dataset = TextDataset(tmp_path, tokenizer, seq_len=seq_len, stride=seq_len)

        print(f"Dataset size: {len(dataset)} samples")
        assert len(dataset) > 0, "Dataset should have at least one sample"

        # 检查一个样本
        sample = dataset[0]
        assert sample["input_ids"].shape == (seq_len,), (
            f"Input shape: {sample['input_ids'].shape}, expected ({seq_len},)"
        )
        assert sample["labels"].shape == (seq_len,), (
            f"Target shape: {sample['labels'].shape}, expected ({seq_len},)"
        )
        print(f"Sample input shape: {sample['input_ids'].shape}")
        print(f"Sample labels shape: {sample['labels'].shape}")

        # 创建 dataloader
        dataloader = create_dataloader(dataset, batch_size=4)
        batch = next(iter(dataloader))
        assert batch["input_ids"].shape == (4, seq_len)
        print(f"Batch input shape: {batch['input_ids'].shape}")
        print("TextDataset test passed!")
    finally:
        os.unlink(tmp_path)
