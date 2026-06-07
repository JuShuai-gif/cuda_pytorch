"""
通过代码注释解释 FSDP（Fully Sharded Data Parallel）/ ZeRO 各阶段。
不涉及实际分布式执行；本文档通过带注释的伪代码说明每个 ZeRO 阶段的内存节省策略。
"""

from __future__ import annotations

import torch


# =====================================================================
# ZeRO 阶段概览
# =====================================================================
#
# 训练模型需要存储：
#   1. 模型参数（P）
#   2. 梯度（G）       - 与参数量相同
#   3. 优化器状态（O）  - 例如 Adam 中的动量 + 方差 = 2P
#
# 不使用并行时，每个 GPU 存储：P + G + O = P + P + 2P = 4P（针对 Adam）
#
# 数据并行（DP）：每个 GPU 有完整的 P、G、O → 每 GPU 4P
# DDP：每个 GPU 有完整的 P、G、O → 每 GPU 4P（内存相同，通信方式不同）
#
# ZeRO-1（优化器状态分片）：
#   - 将 O 在 GPU 之间分区：每个 GPU 存储 O/N 而非 O
#   - 内存：P + G + O/N = P + P + 2P/N
#   - 节省：优化器状态在 Adam 中是最大的内存消耗项
#
# ZeRO-2（梯度分片）：
#   - 额外将 G 在 GPU 之间分区
#   - 内存：P + G/N + O/N = P + P/N + 2P/N
#   - 反向传播后，梯度使用 reduce-scatter（而非 all-reduce）
#
# ZeRO-3（参数分片）：
#   - 额外将 P 在 GPU 之间分区
#   - 内存：P/N + G/N + O/N = (P + P + 2P) / N = 4P/N
#   - 在每层的前向/反向传播前 all-gather 参数
#   - 该层计算完成后丢弃参数
#
# PyTorch 中的 FSDP 实现了 ZeRO-3 的语义。
# =====================================================================


def print_zeero_stages() -> None:
    """打印各 ZeRO 阶段的内存对比。"""
    print("=" * 70)
    print("ZeRO 阶段内存分析（针对 Adam 优化器）")
    print("=" * 70)

    # 示例：1B 参数模型，4 块 GPU
    P = 1e9  # 参数量
    N = 4  # GPU 数量
    bytes_per_param = 4  # fp32

    param_mem = P * bytes_per_param / 1e9  # GB
    grad_mem = P * bytes_per_param / 1e9  # GB
    opt_mem = 2 * P * bytes_per_param / 1e9  # GB（Adam：m + v）

    print(f"\n模型：{P / 1e9:.1f}B 参数，{N} 块 GPU，fp32")
    print(f"  参数内存： {param_mem:.1f} GB")
    print(f"  梯度内存： {grad_mem:.1f} GB")
    print(f"  优化器内存：{opt_mem:.1f} GB（Adam：动量 + 方差）")
    print()

    strategies = {
        "朴素 DP / DDP": (1.0, 1.0, 1.0),
        "ZeRO-1 (OS)   ": (1.0, 1.0, 1.0 / N),
        "ZeRO-2 (OS+G) ": (1.0, 1.0 / N, 1.0 / N),
        "ZeRO-3 (OS+G+P)": (1.0 / N, 1.0 / N, 1.0 / N),
    }

    print(
        f"{'策略':<18} {'参数 (GB)':<12} {'梯度 (GB)':<12} {'优化器 (GB)':<12} {'总计 (GB)':<12}"
    )
    print("-" * 66)
    for name, (pf, gf, of_) in strategies.items():
        p_mem = param_mem * pf
        g_mem = grad_mem * gf
        o_mem = opt_mem * of_
        total = p_mem + g_mem + o_mem
        print(
            f"  {name:<16} {p_mem:<12.2f} {g_mem:<12.2f} {o_mem:<12.2f} {total:<12.2f}"
        )


def print_zeero3_workflow() -> None:
    """通过注释解释 ZeRO-3（FSDP）的前向/反向工作流。"""
    print("\n" + "=" * 70)
    print("ZeRO-3 / FSDP 工作流（逐层执行）")
    print("=" * 70)
    print("""
    对模型中的每个 Transformer 块 i：

    1. 为块 i all-gather 参数：
       - 根进程从所有 GPU 收集分片后的参数
       - 重建块 i 的完整权重张量
       - 通信：all-gather（体积 = 完整块参数量）

    2. 通过块 i 进行前向传播：
       - 使用完整参数计算
       - 丢弃收集到的参数（释放内存）
       - 保留激活用于反向传播

    3.（所有块完成后）计算损失并开始反向传播：

    4. 对每个块 i 按逆序执行：
       a. 再次为块 i all-gather 参数
       b. 通过块 i 执行反向传播
       c. 对块 i 的梯度进行 reduce-scatter
          （每个 GPU 只保留其梯度分片）
       d. 丢弃完整参数

    5. 使用分片后的梯度更新分片后的参数（优化器步骤）：
       - 每个 GPU 仅更新自己的参数分片
       - 优化器状态也被分片 → 每 GPU 内存为 O/N

    关键洞察：参数一次只"物化"（完整收集）一层。
    在任何时刻，只有一层的完整参数驻留在内存中，
    从而大幅降低峰值内存。
    """)


def print_communication_patterns() -> None:
    """展示每个 ZeRO 阶段使用的通信模式。"""
    print("\n" + "=" * 70)
    print("每步的通信模式")
    print("=" * 70)
    print("""
    ZeRO-1：
      - 前向：无（参数已在本地）
      - 反向：Reduce-scatter 梯度（通信量与 all-reduce 相同）
      - 优化器：每个 GPU 更新自己的优化器分区

    ZeRO-2：
      - 前向：无
      - 反向：Reduce-scatter 梯度（通信量与 ZeRO-1 相同）
      - 优化器：每个 GPU 更新自己的优化器分区
      - 注意：通信量与 DDP 相同，只是重新排列了顺序

    ZeRO-3：
      - 前向：All-gather 参数（一次一层）
               → 增加通信量：每层 P * 每层字节数
               → 总计：每步发送 P * 每参数字节数
      - 反向：All-gather 参数 + reduce-scatter 梯度
               → 总计：每步发送 2P * 每参数字节数
               → 为 DDP 通信量的 1.5 倍

    权衡：更多通信换取更少内存。
          ZeRO-3：1.5x 通信量，1/N 内存
          DDP：   1.0x 通信量，1.0x 内存
    """)


def main() -> None:
    print_zeero_stages()
    print_zeero3_workflow()
    print_communication_patterns()


if __name__ == "__main__":
    main()
