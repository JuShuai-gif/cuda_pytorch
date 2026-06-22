"""Demo: 2:4 稀疏权重准备与 CSR 存储开销估算。

这个脚本说明生产部署中的关键点：
1. 2:4 稀疏是 NVIDIA Ampere+ Sparse Tensor Core 支持的结构化稀疏模式；
2. CSR 是通用稀疏格式，但在 50% 稀疏度时索引开销可能抵消存储收益；
3. 真正部署时应在目标硬件、目标 batch size 和目标 runtime 上 benchmark。
"""

import torch


def production_sparse_inference_setup(
    dense_weight: torch.Tensor,
    sparsity: float = 0.5,
):
    """准备 2:4 稀疏权重，并估算 CSR 存储是否比 dense 更省。

    关键结论：CSR 格式通常需要在高稀疏度下才节省内存。
    在 50% 稀疏度附近，values + col_idx + row_ptr 的索引开销可能很高。
    这也是为什么 2:4 稀疏更依赖硬件原生支持，而不是普通 CSR。
    """
    original_size = dense_weight.numel() * dense_weight.element_size()

    # 生成 2:4 稀疏 mask：每连续 4 个元素中保留幅度最大的 2 个。
    dense_reshaped = dense_weight.view(-1, 4)

    # 对每组 4 个元素按绝对值排序，第二小作为阈值：小的 2 个被剪掉。
    sorted_mag, _ = dense_reshaped.abs().sort(dim=1)
    threshold = sorted_mag[:, 1:2]
    mask_2_4 = (dense_reshaped.abs() >= threshold).float()
    sparse_weight = (dense_reshaped * mask_2_4).view_as(dense_weight)

    # 转成 CSR 只是为了估算通用稀疏存储开销；TensorRT 的 2:4 路径不是这样部署。
    sparse_csr = sparse_weight.to_sparse_csr()

    csr_size = (
        sparse_csr.values().numel() * sparse_csr.values().element_size()
        + sparse_csr.col_indices().numel() * sparse_csr.col_indices().element_size()
        + sparse_csr.crow_indices().numel() * sparse_csr.crow_indices().element_size()
    )

    return {
        'dense_size_mb': original_size / (1024**2),
        'csr_size_mb': csr_size / (1024**2),
        'sparsity': (sparse_weight == 0).float().mean().item(),
        'break_even': csr_size < original_size,
    }


# 关键教训：必须在目标硬件和真实 batch size 上 benchmark。
# 纸面上快 3x 的 CSR matmul，在真实 GPU batch=1 场景下可能因为不规则访存变慢。
if __name__ == "__main__":
    weight = torch.randn(1024, 1024)
    report = production_sparse_inference_setup(weight)
    print(report)
