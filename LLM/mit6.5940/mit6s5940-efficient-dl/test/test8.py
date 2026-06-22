"""Demo: 对比稠密矩阵乘法和 CSR 稀疏矩阵乘法。

这个脚本用于说明一个重要工业事实：
“稀疏度高”不必然意味着“推理更快”。CSR 格式会引入索引读取和不规则访存，
在某些硬件/shape/batch size 下可能比稠密计算更慢。
"""

import time  # 用 perf_counter 做高精度计时

import torch


def benchmark_sparse_matmul(size=1024, sparsity=0.9):
    """对比 dense matmul 和 CSR sparse matmul 的平均耗时。

    参数：
        size: 方阵大小。
        sparsity: 目标稀疏度，例如 0.9 表示 90% 元素为 0。
    """
    # 生成一个稠密矩阵。
    dense = torch.randn(size, size)

    # 生成随机 mask；True 的位置保留，False 的位置置零。
    # rand > sparsity 的概率约等于 (1 - sparsity)，即保留比例。
    mask = torch.rand(size, size) > sparsity

    # 注意：sparse_dense 虽然有很多 0，但底层仍然是 dense tensor 存储。
    sparse_dense = dense * mask

    # 转换为 CSR 格式后，才真正使用稀疏存储结构（只存非零值 + 索引）。
    sparse_csr = sparse_dense.to_sparse_csr()

    vec = torch.randn(size)  # 用于矩阵-向量乘法的右乘向量

    # 稠密矩阵乘法计时：重复 100 次取平均，减少单次噪声。
    t0 = time.perf_counter()
    for _ in range(100):
        _ = dense @ vec
    t1 = time.perf_counter()

    # CSR 稀疏矩阵乘法计时。
    t2 = time.perf_counter()
    for _ in range(100):
        _ = sparse_csr @ vec
    t3 = time.perf_counter()

    # 输出两者的单次平均耗时（毫秒）。
    print(
        f"Dense: {(t1 - t0) / 100 * 1000:.3f}ms | CSR: {(t3 - t2) / 100 * 1000:.3f}ms"
    )
    print(f"Sparsity: {sparsity:.0%}")


# 分别测试中等、高、极高稀疏度。
# 通常只有稀疏度非常高、且稀疏 kernel 足够好时，CSR 才可能明显受益。
benchmark_sparse_matmul(sparsity=0.5)  # 50% 稀疏：CSR 往往比 dense 还慢
benchmark_sparse_matmul(sparsity=0.9)  # 90% 稀疏
benchmark_sparse_matmul(sparsity=0.99)  # 99% 稀疏：此时 CSR 才更可能有优势
