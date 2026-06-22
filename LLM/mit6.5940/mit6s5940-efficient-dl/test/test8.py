"""Demo: 对比稠密矩阵乘法和 CSR 稀疏矩阵乘法。

这个脚本用于说明一个重要工业事实：
“稀疏度高”不必然意味着“推理更快”。CSR 格式会引入索引读取和不规则访存，
在某些硬件/shape/batch size 下可能比稠密计算更慢。
"""

import time

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
    mask = torch.rand(size, size) > sparsity

    # 注意：sparse_dense 虽然有很多 0，但底层仍然是 dense tensor 存储。
    sparse_dense = dense * mask

    # 转换为 CSR 格式后，才真正使用稀疏存储结构。
    sparse_csr = sparse_dense.to_sparse_csr()

    vec = torch.randn(size)

    # 稠密矩阵乘法计时。
    t0 = time.perf_counter()
    for _ in range(100):
        _ = dense @ vec
    t1 = time.perf_counter()

    # CSR 稀疏矩阵乘法计时。
    t2 = time.perf_counter()
    for _ in range(100):
        _ = sparse_csr @ vec
    t3 = time.perf_counter()

    print(f"Dense: {(t1 - t0) / 100 * 1000:.3f}ms | "
          f"CSR: {(t3 - t2) / 100 * 1000:.3f}ms")
    print(f"Sparsity: {sparsity:.0%}")


# 分别测试中等、高、极高稀疏度。
# 通常只有稀疏度非常高、且稀疏 kernel 足够好时，CSR 才可能明显受益。
benchmark_sparse_matmul(sparsity=0.5)
benchmark_sparse_matmul(sparsity=0.9)
benchmark_sparse_matmul(sparsity=0.99)
