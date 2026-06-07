"""
Megatron 风格的 MLP 层张量并行。
通过手动拆分权重矩阵来实现列并行和行并行线性层。
无需实际分布式通信即可理解相关概念。

参考文献：Megatron-LM: Training Multi-Billion Parameter Language Models
           Using Model Parallelism（Shoeybi et al., 2019）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层。

    权重矩阵 W 沿列维度在设备之间拆分。
    输入在所有设备上完全复制（相同）。
    输出沿最后一维分区。

    前向：
        y_i = x @ W_i  （无需通信）

    在 Transformer 中，通常用于 FFN 的第一个线性层，
    或注意力的 QKV 投影。如果其后接非列并行层（如 GeLU），
    则输出需要对激活进行 all-reduce。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_partitions: int = 2,
        partition_idx: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx

        # 每个分区有 out_features // num_partitions 列
        assert out_features % num_partitions == 0, (
            f"out_features（{out_features}）必须能被 num_partitions（{num_partitions}）整除"
        )
        self.partition_out_features = out_features // num_partitions

        # 本地权重：(in_features, out_features // num_partitions)
        self.weight = nn.Parameter(
            torch.randn(in_features, self.partition_out_features) * 0.02
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.partition_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x：(batch, seq_len, in_features) - 复制的输入
        返回：(batch, seq_len, partition_out_features) - 分区的输出
        """
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def gather_output(self, local_output: torch.Tensor) -> torch.Tensor:
        """
        模拟从所有设备收集分区输出。
        在实际分布式环境中，这将是 all-gather 操作。
        """
        # 在实际实现中，每个设备在此处执行 all-gather。
        # 由于我们只是模拟，这里返回完整输出的占位值。
        full_weight = self.get_full_weight()
        full_bias = self.get_full_bias()
        return local_output @ torch.eye(self.partition_out_features)  # 占位

    def get_full_weight(self) -> torch.Tensor:
        """返回概念上的完整权重矩阵（仅用于说明）。"""
        # 实践中，这个矩阵永远不会存在于单个 GPU 上
        return torch.randn(self.in_features, self.out_features)

    def get_full_bias(self) -> torch.Tensor | None:
        """返回概念上的完整偏置向量。"""
        if self.bias is not None:
            return torch.randn(self.out_features)
        return None


class RowParallelLinear(nn.Module):
    """
    行并行线性层。

    权重矩阵 W 沿行维度在设备之间拆分。
    输入沿最后一维分区。
    输出在 all-reduce 后在所有设备上完全复制（相同）。

    前向：
        y_i = x_i @ W_i（部分和）
        y = all-reduce(y_i)（跨分区求和）

    在 Transformer FFN 中，通常用在激活函数之后：
        [ColumnParallel FC] → GeLU → [RowParallel FC]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_partitions: int = 2,
        partition_idx: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx

        assert in_features % num_partitions == 0, (
            f"in_features（{in_features}）必须能被 num_partitions（{num_partitions}）整除"
        )
        self.partition_in_features = in_features // num_partitions

        # 本地权重：(in_features // num_partitions, out_features)
        self.weight = nn.Parameter(
            torch.randn(self.partition_in_features, out_features) * 0.02
        )
        if bias:
            # 偏置被复制（每个分区都有完整偏置）。也可以拆分。
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x：(batch, seq_len, partition_in_features) - 分区的输入
        返回：(batch, seq_len, out_features) - 部分和（需要 all-reduce）
        """
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def simulate_all_reduce(self, partial_outputs: list[torch.Tensor]) -> torch.Tensor:
        """模拟跨分区的 all-reduce。"""
        return torch.stack(partial_outputs).sum(dim=0)


class TensorParallelMLP(nn.Module):
    """
    完整的张量并行 MLP，使用 Megatron 风格的列 + 行并行。

    架构：
        输入（复制）→ ColumnParallelLinear → ReLU → RowParallelLinear → 输出（复制）

    通信分析：
        - ColumnParallel：f（前向无需通信，反向无需通信）
        - ReLU：f（逐元素操作，无需通信）
        - RowParallel：f（前向需要 all-reduce，反向无需通信）
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_partitions: int = 2,
        partition_idx: int = 0,
    ):
        super().__init__()
        # 列并行：in_features→intermediate，拆分输出列
        self.fc1 = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_partitions=num_partitions,
            partition_idx=partition_idx,
        )
        # 行并行：intermediate→hidden，拆分输入行
        self.fc2 = RowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            num_partitions=num_partitions,
            partition_idx=partition_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x：复制的输入（在所有分区上相同）
        返回需要 all-reduce 才能复制的部分输出
        """
        # 列并行：无需通信
        h = F.relu(self.fc1(x))
        # 行并行：每个分区计算部分和
        # 在实际分布式环境中，此处需要 all-reduce 来求和部分输出
        y = self.fc2(h)
        return y  # 需要来自所有分区的 all-reduce


def demo_tensor_parallel() -> None:
    """通过模拟多个设备来演示张量并行。"""
    print("=" * 60)
    print("张量并行演示（Megatron 风格）")
    print("=" * 60)

    hidden_size = 64
    intermediate_size = 128
    num_partitions = 4
    batch_size = 2
    seq_len = 8

    # 创建张量并行 MLP 分区
    print("\n创建具有 4 个分区的张量并行 MLP……")
    partitions = [
        TensorParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_partitions=num_partitions,
            partition_idx=i,
        )
        for i in range(num_partitions)
    ]

    # 模拟前向传播
    x = torch.randn(batch_size, seq_len, hidden_size)

    print(f"\n输入 shape：{x.shape}（在所有设备上复制）")
    print(f"\n列并行 FC1：{hidden_size} → {intermediate_size}")
    print(f"  每分区权重：({hidden_size}, {intermediate_size // num_partitions})")
    print(
        f"  每分区输出：({batch_size}, {seq_len}, {intermediate_size // num_partitions})"
    )
    print(f"  通信：无（输入已复制）")

    # 列并行前向传播
    h_partitions = [p.fc1(x) for p in partitions]
    for i, h in enumerate(h_partitions):
        print(f"  分区 {i} 输出 shape：{h.shape}")

    print(f"\nReLU 激活：逐元素操作，无需通信")

    h_activated = [F.relu(h) for h in h_partitions]

    print(f"\n行并行 FC2：{intermediate_size} → {hidden_size}")
    print(f"  每分区权重：({intermediate_size // num_partitions}, {hidden_size})")
    print(f"  每分区输出：({batch_size}, {seq_len}, {hidden_size}) [部分和]")
    print(f"  通信：需要 all-reduce 来求和部分输出")

    y_partitions = [p.fc2(h_act) for p, h_act in zip(partitions, h_activated)]
    y_reduced = torch.stack(y_partitions).sum(dim=0)

    print(f"\nall-reduce（求和）之后：{y_reduced.shape}")
    print(f"输出 shape：{y_reduced.shape}（在所有设备上复制）")

    # 通信分析
    print(f"\n通信分析：")
    print(f"  FC1（列并行）：f = 0 bytes（无需通信）")
    print(f"  ReLU：              f = 0 bytes")
    print(f"  FC2（行并行）：f = batch*seq*hidden*每元素字节数（all-reduce）")
    print(f"  每个 Transformer 块的前向通信总量：1 次 all-reduce")
    print(f"  每个 Transformer 块的反向通信总量：1 次 all-reduce（针对列并行梯度）")

    # 每设备内存分析
    print(f"\n每设备内存：")
    fc1_params = hidden_size * (intermediate_size // num_partitions)
    fc2_params = (intermediate_size // num_partitions) * hidden_size
    total_params = fc1_params + fc2_params
    print(f"  FC1 参数：{fc1_params:,}")
    print(f"  FC2 参数：{fc2_params:,}")
    print(
        f"  每设备总计：{total_params:,}（不使用 TP 时为 {hidden_size * intermediate_size * 2:,}）"
    )
    print(f"  内存减少：{1 - total_params / (hidden_size * intermediate_size * 2):.0%}")


def main() -> None:
    demo_tensor_parallel()


if __name__ == "__main__":
    main()
