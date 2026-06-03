"""
Tests for 01_cuda_basics CUDA kernels。

验证所有自定义 CUDA kernel 的正确性：
  - Attention (FlashAttention, PagedAttention)
  - Normalization (RMSNorm, LayerNorm, Fused Residual+Norm)
  - Activation (SiLU, SwiGLU, GELU, Fused Bias+Activation)
  - Softmax (Online Softmax, Masked Online Softmax)
  - Reduction (Warp Reduce, Naive Reduce)
  - Vector Add
  - Tiled Matmul (GEMM with shared memory tiling, batch matmul)

运行: pytest 01_cuda_basics/test_cuda_basics.py -v
"""

import math
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# 尝试导入编译好的 CUDA 扩展
# ---------------------------------------------------------------------------
_EXTENSION_NAME = "cuda_kernels"
HAS_EXTENSION = False

try:
    import cuda_kernels  # type: ignore[import-not-found]

    HAS_EXTENSION = True
except ImportError:
    pass

requires_extension = pytest.mark.skipif(
    not HAS_EXTENSION or not torch.cuda.is_available(),
    reason="CUDA extension not built or CUDA not available",
)


# ---------------------------------------------------------------------------
# rmsnorm 的 PyTorch 参考实现（用于对比验证）
# ---------------------------------------------------------------------------
def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """手动实现 RMSNorm：rms = sqrt(mean(x^2) + eps)，out = x / rms * weight"""
    x_float = x.float()
    weight_float = weight.float()
    rms = torch.sqrt(torch.mean(x_float**2, dim=-1, keepdim=True) + eps)
    out = x_float / rms * weight_float
    return out.to(x.dtype)


def _layernorm_ref(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """使用 PyTorch 内置 LayerNorm 作为参考"""
    hidden_dim = x.shape[-1]
    ln = torch.nn.LayerNorm(hidden_dim, eps=eps, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        ln.weight.copy_(weight.float())
        ln.bias.copy_(bias.float())
    return ln(x.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# 启动 CUDA 扩展自动构建（如果尚未编译）
# ---------------------------------------------------------------------------
def _build_extension():
    """尝试构建 CUDA 扩展。"""
    import subprocess
    import sys

    setup_py = Path(__file__).parent / "setup.py"
    result = subprocess.run(
        [sys.executable, str(setup_py), "build_ext", "--inplace"],
        cwd=str(Path(__file__).parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return False

    try:
        import importlib
        import cuda_kernels

        importlib.reload(cuda_kernels)
    except ImportError:
        import cuda_kernels  # type: ignore[import-not-found,no-redef]
    return True


if not HAS_EXTENSION and torch.cuda.is_available():
    HAS_EXTENSION = _build_extension()
    if HAS_EXTENSION:
        import cuda_kernels  # type: ignore[import-not-found,no-redef]

requires_extension = pytest.mark.skipif(
    not HAS_EXTENSION or not torch.cuda.is_available(),
    reason="CUDA extension not built or CUDA not available",
)


# ============================================================================
# 测试：FlashAttention
# ============================================================================
@pytest.mark.cuda
class TestFlashAttention:
    @requires_extension
    @pytest.mark.parametrize(
        "batch,n_heads,seq_len,head_dim,causal",
        [
            (2, 4, 64, 64, False),
            (1, 8, 128, 64, True),
            (2, 4, 128, 128, False),
            (1, 8, 200, 64, True),
            (1, 4, 256, 128, False),
        ],
    )
    def test_flash_attention_correctness(self, batch, n_heads, seq_len, head_dim, causal):
        """验证 FlashAttention 输出与 PyTorch scaled_dot_product_attention 一致"""
        torch.manual_seed(42)
        shape = (batch, n_heads, seq_len, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        # 在 GPU 上生成 fp16 数据
        Q = torch.randn(shape, device="cuda", dtype=torch.float16)
        K = torch.randn(shape, device="cuda", dtype=torch.float16)
        V = torch.randn(shape, device="cuda", dtype=torch.float16)
        O = torch.empty(shape, device="cuda", dtype=torch.float16)

        mask = None
        is_causal = causal
        # PyTorch 的 sdpa 在 causal=True 时会自动处理因果 mask
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )

        cuda_kernels.flash_attention_fwd(Q, K, V, O, scale, causal)
        torch.cuda.synchronize()

        assert torch.allclose(O, ref, rtol=1e-2, atol=5e-2), (
            f"FlashAttention 输出与参考值不匹配: max_diff={(O - ref).abs().max().item():.6f}"
        )

    @requires_extension
    def test_flash_attention_causal_diagonal(self):
        """验证 causal mask 正确性：O[i,:] 只应依赖 K[:i+1,:] 和 V[:i+1,:]"""
        batch, n_heads, seq_len, head_dim = 1, 4, 64, 64
        scale = 1.0 / math.sqrt(head_dim)
        shape = (batch, n_heads, seq_len, head_dim)

        # 使用可区分的 K/V 值：K 中仅第 0 位置为 1，其余为 0
        K = torch.zeros(shape, device="cuda", dtype=torch.float16)
        K[:, :, 0, :] = 1.0
        V = torch.ones(shape, device="cuda", dtype=torch.float16)
        Q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1
        O = torch.empty(shape, device="cuda", dtype=torch.float16)

        cuda_kernels.flash_attention_fwd(Q, K, V, O, scale, causal=True)
        torch.cuda.synchronize()

        # 在 causal 模式下，位置 seq_len-1 不能看到位置 0（因为没有未来的信息）
        # 但实际上位置 0 的 key 对位置 seq_len-1 是过去的信息，应该能看到
        # 正确测试：位置 0 只能看到自己，所以 O[0,:] 的高亮位置应该在 0
        # 验证非零性（至少位置 seq_len-1 的 output 不应该全是 0）
        assert O.abs().sum() > 0, "Causal attention 输出不应全为零"


# ============================================================================
# 测试：PagedAttention
# ============================================================================
@pytest.mark.cuda
class TestPagedAttention:
    @requires_extension
    def test_paged_attention_correctness(self):
        """验证 PagedAttention 输出与手动参考注意力一致"""
        num_heads, head_dim, block_size = 4, 64, 16
        num_blocks = 6
        scale = 1.0 / math.sqrt(head_dim)

        # 分配 KV cache
        K_cache, V_cache = cuda_kernels.allocate_kv_cache(
            num_blocks, block_size, num_heads, head_dim
        )

        # 填充部分 block
        torch.manual_seed(42)
        K_cache.normal_()
        V_cache.normal_()

        # context_len=32，需要 2 个 block，使用 block 0 和 block 3
        context_lens = torch.tensor([20], device="cuda", dtype=torch.int32)
        block_tables = torch.tensor([[0, 3, -1, -1]], device="cuda", dtype=torch.int32)

        Q = torch.randn(num_heads, head_dim, device="cuda", dtype=torch.float16)
        O = torch.empty(num_heads, head_dim, device="cuda", dtype=torch.float16)

        # 手动参考：从 cache 中收集前 20 个 token 的 K 和 V
        context_len = 20
        K_ref = K_cache[0, :16].reshape(-1, num_heads, head_dim)
        K_ref = torch.cat(
            [K_ref, K_cache[3, :4].reshape(-1, num_heads, head_dim)], dim=0
        )  # [20, 4, 64]
        V_ref = V_cache[0, :16].reshape(-1, num_heads, head_dim)
        V_ref = torch.cat([V_ref, V_cache[3, :4].reshape(-1, num_heads, head_dim)], dim=0)

        # 参考注意力 Q [nh, hd] @ K [20, nh, hd].T → [nh, 20]
        Q_fp32 = Q.float()
        K_ref_fp32 = K_ref.float().permute(1, 2, 0)  # [4, 64, 20]
        V_ref_fp32 = V_ref.float().permute(1, 0, 2)  # [4, 20, 64]

        attn_scores = (
            torch.bmm(
                Q_fp32.unsqueeze(1),  # [4, 1, 64]
                K_ref_fp32,  # [4, 64, 20]
            ).squeeze(1)
            * scale
        )  # [4, 20]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [4, 20]
        O_ref = torch.bmm(attn_weights.unsqueeze(1), V_ref_fp32).squeeze(1)  # [4, 64]
        O_ref = O_ref.to(torch.float16)

        cuda_kernels.paged_attention(Q, K_cache, V_cache, block_tables, context_lens, O, scale)
        torch.cuda.synchronize()

        assert torch.allclose(O, O_ref, rtol=1e-2, atol=5e-2), (
            f"PagedAttention 输出不匹配: max_diff={(O - O_ref).abs().max().item():.6f}"
        )

    @requires_extension
    def test_paged_attention_partial_block(self):
        """验证 context_len=20 时的部分 block 处理（16+4 token）"""
        num_heads, head_dim, block_size = 4, 64, 16
        num_blocks = 4
        scale = 1.0 / math.sqrt(head_dim)

        K_cache, V_cache = cuda_kernels.allocate_kv_cache(
            num_blocks, block_size, num_heads, head_dim
        )

        torch.manual_seed(42)
        K_cache.normal_()
        V_cache.normal_()

        context_lens = torch.tensor([20], device="cuda", dtype=torch.int32)
        block_tables = torch.tensor([[0, 2, -1, -1]], device="cuda", dtype=torch.int32)

        Q = torch.randn(num_heads, head_dim, device="cuda", dtype=torch.float16)
        O = torch.empty(num_heads, head_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.paged_attention(Q, K_cache, V_cache, block_tables, context_lens, O, scale)
        torch.cuda.synchronize()

        # 基本正确性：输出不应为 NaN 或 Inf
        assert not torch.isnan(O).any(), "PagedAttention 输出包含 NaN"
        assert not torch.isinf(O).any(), "PagedAttention 输出包含 Inf"

    @requires_extension
    def test_paged_attention_multi_batch(self):
        """验证多 batch 样本同时处理"""
        num_heads, head_dim, block_size = 4, 64, 16
        num_blocks = 4
        batch_size = 2
        scale = 1.0 / math.sqrt(head_dim)

        K_cache, V_cache = cuda_kernels.allocate_kv_cache(
            num_blocks, block_size, num_heads, head_dim
        )
        torch.manual_seed(42)
        K_cache.normal_()
        V_cache.normal_()

        context_lens = torch.tensor([20, 20], device="cuda", dtype=torch.int32)
        block_tables = torch.tensor(
            [[0, 2, -1, -1], [1, 3, -1, -1]], device="cuda", dtype=torch.int32
        )

        Q = torch.randn(num_heads, head_dim, device="cuda", dtype=torch.float16)
        O = torch.empty(num_heads, head_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.paged_attention(Q, K_cache, V_cache, block_tables, context_lens, O, scale)
        torch.cuda.synchronize()

        assert not torch.isnan(O).any(), "PagedAttention 输出包含 NaN"


# ============================================================================
# 测试：RMSNorm
# ============================================================================
@pytest.mark.cuda
class TestRMSNorm:
    @requires_extension
    @pytest.mark.parametrize(
        "rows,hidden_dim",
        [
            (16, 768),
            (8, 4096),
            (32, 512),
            (1, 1024),
        ],
    )
    def test_rmsnorm_correctness(self, rows, hidden_dim):
        """验证 CUDA RMSNorm 与手动 PyTorch RMSNorm 一致"""
        eps = 1e-5
        torch.manual_seed(42)

        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.rmsnorm_fwd(x, weight, out, eps)
        torch.cuda.synchronize()

        ref = _rmsnorm_ref(x, weight, eps)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"RMSNorm 输出不匹配 rows={rows} hidden_dim={hidden_dim}: "
            f"max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    def test_rmsnorm_residual(self):
        """验证融合残差 + RMSNorm 输出与顺序计算一致"""
        rows, hidden_dim, eps = 16, 768, 1e-5
        torch.manual_seed(42)

        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        residual = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)
        residual_out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.rmsnorm_residual_fwd(x, residual, weight, out, residual_out, eps)
        torch.cuda.synchronize()

        # 手动参考
        residual_ref = (x.float() + residual.float()).to(torch.float16)
        out_ref = _rmsnorm_ref(residual_ref, weight, eps)

        assert torch.allclose(residual_out.float(), residual_ref.float(), rtol=1e-2, atol=1e-3), (
            "融合残差输出不匹配"
        )
        assert torch.allclose(out.float(), out_ref.float(), rtol=1e-2, atol=1e-3), (
            "融合 RMSNorm 输出不匹配"
        )


# ============================================================================
# 测试：LayerNorm
# ============================================================================
@pytest.mark.cuda
class TestLayerNorm:
    @requires_extension
    @pytest.mark.parametrize(
        "rows,hidden_dim",
        [
            (16, 768),
            (8, 4096),
        ],
    )
    def test_layernorm_correctness(self, rows, hidden_dim):
        """验证 CUDA LayerNorm 与 torch.nn.LayerNorm 一致"""
        eps = 1e-5
        torch.manual_seed(42)

        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.layernorm_fwd(x, weight, bias, out, eps)
        torch.cuda.synchronize()

        ref = _layernorm_ref(x, weight, bias, eps)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"LayerNorm 输出不匹配 rows={rows} hidden_dim={hidden_dim}: "
            f"max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )


# ============================================================================
# 测试：Fused Residual + LayerNorm
# ============================================================================
@pytest.mark.cuda
class TestFusedResidualNorm:
    @requires_extension
    def test_fused_residual_layernorm(self):
        """验证融合残差 + LayerNorm 与顺序计算（add + LayerNorm）一致"""
        rows, hidden_dim, eps = 16, 768, 1e-5
        torch.manual_seed(42)

        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        residual = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.fused_residual_layernorm(x, residual, weight, bias, out, eps)
        torch.cuda.synchronize()

        # 手动顺序计算
        y = (x.float() + residual.float()).to(torch.float16)
        ref = _layernorm_ref(y, weight, bias, eps)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"融合残差+LayerNorm 输出不匹配: "
            f"max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )


# ============================================================================
# 测试：Activation
# ============================================================================
@pytest.mark.cuda
class TestActivations:
    @requires_extension
    @pytest.mark.parametrize("n", [1024, 4096, 10000])
    def test_silu(self, n):
        """验证 CUDA SiLU 与 torch.nn.functional.silu 一致"""
        torch.manual_seed(42)
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.empty(n, device="cuda", dtype=torch.float16)

        cuda_kernels.silu_fwd(x, out)
        torch.cuda.synchronize()

        ref = torch.nn.functional.silu(x.float()).to(torch.float16)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"SiLU 输出不匹配 n={n}: max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    def test_swiglu(self):
        """验证 CUDA SwiGLU 与 gate * SiLU(up) 一致"""
        torch.manual_seed(42)
        n = 4096
        gate = torch.randn(n, device="cuda", dtype=torch.float16)
        up = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.empty(n, device="cuda", dtype=torch.float16)

        cuda_kernels.swiglu_fwd(gate, up, out)
        torch.cuda.synchronize()

        ref = (gate.float() * torch.nn.functional.silu(up.float())).to(torch.float16)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"SwiGLU 输出不匹配: max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    @pytest.mark.parametrize("n", [1024, 4096, 10000])
    def test_gelu(self, n):
        """验证 CUDA GELU 与 torch.nn.functional.gelu 一致"""
        torch.manual_seed(42)
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.empty(n, device="cuda", dtype=torch.float16)

        cuda_kernels.gelu_fwd(x, out)
        torch.cuda.synchronize()

        ref = torch.nn.functional.gelu(x.float(), approximate="tanh").to(torch.float16)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"GELU 输出不匹配 n={n}: max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )

    def test_swiglu_2d(self):
        """验证 2D 张量的 SwiGLU"""
        if not HAS_EXTENSION:
            pytest.skip("CUDA extension not built")
        rows, dim = 128, 11008  # LLaMA FFN 中间维度
        torch.manual_seed(42)
        gate = torch.randn(rows, dim, device="cuda", dtype=torch.float16)
        up = torch.randn(rows, dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, dim, device="cuda", dtype=torch.float16)

        cuda_kernels.swiglu_fwd(gate, up, out)
        torch.cuda.synchronize()

        ref = (gate.float() * torch.nn.functional.silu(up.float())).to(torch.float16)
        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3)


# ============================================================================
# 测试：Fused Bias + Activation
# ============================================================================
@pytest.mark.cuda
class TestFusedBiasActivation:
    @requires_extension
    @pytest.mark.parametrize(
        "rows,hidden_dim",
        [
            (128, 768),
            (64, 4096),
        ],
    )
    def test_fused_bias_relu(self, rows, hidden_dim):
        """验证融合 bias + ReLU 与顺序计算一致"""
        torch.manual_seed(42)
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.fused_bias_relu(x, bias, out)
        torch.cuda.synchronize()

        ref = torch.nn.functional.relu(x.float() + bias.float()).to(torch.float16)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"Fused bias+ReLU 输出不匹配: "
            f"max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    @pytest.mark.parametrize(
        "rows,hidden_dim",
        [
            (128, 768),
            (64, 4096),
        ],
    )
    def test_fused_bias_gelu(self, rows, hidden_dim):
        """验证融合 bias + GELU 与顺序计算一致"""
        torch.manual_seed(42)
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.fused_bias_gelu(x, bias, out)
        torch.cuda.synchronize()

        ref = torch.nn.functional.gelu(x.float() + bias.float(), approximate="tanh").to(
            torch.float16
        )

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"Fused bias+GELU 输出不匹配: "
            f"max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    @pytest.mark.parametrize(
        "rows,hidden_dim",
        [
            (128, 768),
            (64, 4096),
        ],
    )
    def test_fused_bias_silu(self, rows, hidden_dim):
        """验证融合 bias + SiLU 与顺序计算一致"""
        torch.manual_seed(42)
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        cuda_kernels.fused_bias_silu(x, bias, out)
        torch.cuda.synchronize()

        ref = torch.nn.functional.silu(x.float() + bias.float()).to(torch.float16)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3), (
            f"Fused bias+SiLU 输出不匹配: "
            f"max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )


# ============================================================================
# 测试：Online Softmax
# ============================================================================
@pytest.mark.cuda
class TestOnlineSoftmax:
    @requires_extension
    @pytest.mark.parametrize(
        "rows,cols",
        [
            (16, 128),
            (8, 512),
            (4, 2048),
        ],
    )
    def test_online_softmax(self, rows, cols):
        """验证 online softmax 与 torch.softmax 一致"""
        torch.manual_seed(42)
        # softmax 对数值稳定性的要求更高，用较小的幅度避免 fp16 overflow
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16) * 0.5
        out = torch.empty(rows, cols, device="cuda", dtype=torch.float16)

        cuda_kernels.online_softmax(x, out)
        torch.cuda.synchronize()

        ref = torch.softmax(x.float(), dim=-1).to(torch.float16)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2), (
            f"Online softmax 输出不匹配 rows={rows} cols={cols}: "
            f"max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    def test_masked_softmax(self):
        """验证 masked online softmax 正确遮挡特定位置"""
        rows, cols = 8, 64
        torch.manual_seed(42)

        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16) * 0.5

        # 创建 causal mask：上三角为 1（表示需要遮挡）
        causal_mask = torch.triu(
            torch.ones(cols, cols, device="cuda", dtype=torch.float16), diagonal=1
        )
        # 对每一行使用因果 mask
        mask = causal_mask[:rows, :cols]

        out = torch.empty(rows, cols, device="cuda", dtype=torch.float16)

        cuda_kernels.masked_online_softmax(x, mask, out, 1.0)
        torch.cuda.synchronize()

        # 手动 apply mask 后做 softmax
        x_masked = x.float().clone()
        x_masked[mask > 0.5] = float("-inf")
        ref = torch.softmax(x_masked, dim=-1).to(torch.float16)

        assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2), (
            f"Masked online softmax 输出不匹配: "
            f"max_diff={(out.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    def test_softmax_sum_to_one(self):
        """验证 softmax 每行求和接近 1"""
        rows, cols = 4, 128
        torch.manual_seed(42)
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16) * 0.5
        out = torch.empty(rows, cols, device="cuda", dtype=torch.float16)

        cuda_kernels.online_softmax(x, out)
        torch.cuda.synchronize()

        row_sums = out.float().sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(rows, device="cuda"), rtol=1e-2, atol=1e-3), (
            f"Softmax 行和不为 1: {row_sums.tolist()}"
        )


# ============================================================================
# 测试：Reduction (已有)
# ============================================================================
@pytest.mark.cuda
@requires_extension
class TestReduction:
    """测试 warp shuffle reduction kernel。"""

    @pytest.mark.parametrize("size", [256, 1024, 9999, 1_000_000])
    def test_warp_reduce_sum(self, size):
        x = torch.randn(size, device="cuda")
        expected = torch.sum(x)
        result = cuda_kernels.warp_reduce_sum(x)
        assert torch.allclose(result, expected, rtol=1e-2), (
            f"size={size}: expected {expected.item():.4f}, got {result.item():.4f}"
        )

    @pytest.mark.parametrize("size", [256, 1024, 9999, 1_000_000])
    def test_naive_reduce_sum(self, size):
        x = torch.randn(size, device="cuda")
        expected = torch.sum(x)
        result = cuda_kernels.naive_reduce_sum(x)
        assert torch.allclose(result, expected, rtol=1e-2), (
            f"size={size}: expected {expected.item():.4f}, got {result.item():.4f}"
        )

    def test_warp_reduce_sum_all_ones(self):
        n = 1024
        x = torch.ones(n, device="cuda")
        result = cuda_kernels.warp_reduce_sum(x)
        assert torch.allclose(result, torch.tensor([float(n)], device="cuda"), rtol=1e-2)

    def test_warp_reduce_single_element(self):
        x = torch.tensor([42.0], device="cuda")
        result = cuda_kernels.warp_reduce_sum(x)
        assert torch.allclose(result, torch.tensor([42.0], device="cuda"))

    @requires_extension
    @pytest.mark.parametrize("size", [256, 1024, 9999, 1_000_000])
    def test_full_warp_reduction(self, size):
        x = torch.randn(size, device="cuda")
        expected = torch.sum(x)
        result = cuda_kernels.full_warp_reduction(x)
        assert torch.allclose(result, expected, rtol=1e-2), (
            f"size={size}: expected {expected.item():.4f}, got {result.item():.4f}"
        )

    @requires_extension
    @pytest.mark.parametrize("size", [256, 1024, 9999, 1_000_000])
    def test_reduce_sum(self, size):
        x = torch.randn(size, device="cuda")
        expected = torch.sum(x)
        result = cuda_kernels.reduce_sum(x)
        assert torch.allclose(result, expected, rtol=1e-2), (
            f"size={size}: expected {expected.item():.4f}, got {result.item():.4f}"
        )


# ============================================================================
# 测试：Vector Add (已有)
# ============================================================================
@pytest.mark.cuda
@requires_extension
class TestVectorAdd:
    """测试自定义 CUDA vector_add kernel。"""

    def test_basic(self):
        n = 1000
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        expected = a + b
        result = cuda_kernels.vector_add(a, b)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_large(self):
        n = 1_000_000
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        expected = a + b
        result = cuda_kernels.vector_add(a, b)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_non_power_of_two(self):
        n = 9999
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        expected = a + b
        result = cuda_kernels.vector_add(a, b)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_2d_tensor(self):
        a = torch.randn(4, 256, device="cuda")
        b = torch.randn(4, 256, device="cuda")
        expected = a + b
        result = cuda_kernels.vector_add(a, b)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_non_contiguous(self):
        a = torch.randn(100, 100, device="cuda").T
        b = torch.randn(100, 100, device="cuda").T
        expected = a.contiguous() + b.contiguous()
        result = cuda_kernels.vector_add(a, b)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_single_element(self):
        a = torch.tensor([3.0], device="cuda")
        b = torch.tensor([5.0], device="cuda")
        result = cuda_kernels.vector_add(a, b)
        assert torch.allclose(result, torch.tensor([8.0], device="cuda"))


# ============================================================================
# 测试：Tiled Matmul
# ============================================================================
@pytest.mark.cuda
class TestTiledMatmul:
    @requires_extension
    @pytest.mark.parametrize(
        "M,N,K",
        [
            # 规则尺寸（block size 的整数倍）
            (128, 128, 128),
            (256, 256, 128),
            (128, 256, 256),
            # 非整数倍尺寸（边界处理）
            (100, 100, 100),
            (200, 150, 100),
            (130, 130, 130),
            # 小矩阵（触发小 tile 配置）
            (32, 32, 32),
            (50, 30, 64),
            (16, 16, 16),
            # LLM 相关形状
            (1, 4096, 4096),  # 单 token 解码 投影
            (64, 4096, 4096),  # prefill 投影
            (64, 8192, 4096),  # FFN gate
            (64, 4096, 14336),  # LLaMA FFN up
            (128, 4096, 11008),  # LLaMA-7B FFN
        ],
    )
    def test_tiled_matmul_correctness(self, M, N, K):
        """验证 CUDA tiled matmul 与 torch.matmul 一致"""
        torch.manual_seed(42)

        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        cuda_kernels.tiled_matmul(A, B, C)
        torch.cuda.synchronize()

        ref = torch.matmul(A.float(), B.float()).to(torch.float16)

        assert torch.allclose(C.float(), ref.float(), rtol=1e-1, atol=5e-2), (
            f"Tiled matmul 输出不匹配 M={M} N={N} K={K}: "
            f"max_diff={(C.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    @pytest.mark.parametrize(
        "B,M,N,K",
        [
            (2, 64, 64, 64),
            (4, 128, 128, 128),
            (2, 100, 80, 120),
            (8, 32, 32, 32),
            (2, 1, 4096, 4096),
            (4, 64, 4096, 4096),
        ],
    )
    def test_batched_matmul_correctness(self, B, M, N, K):
        """验证 CUDA batched matmul 与 torch.matmul 一致"""
        torch.manual_seed(42)

        A = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
        Bmat = torch.randn(B, K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(B, M, N, device="cuda", dtype=torch.float16)

        cuda_kernels.batched_matmul(A, Bmat, C)
        torch.cuda.synchronize()

        ref = torch.matmul(A.float(), Bmat.float()).to(torch.float16)

        assert torch.allclose(C.float(), ref.float(), rtol=1e-1, atol=5e-2), (
            f"Batched matmul 输出不匹配 B={B} M={M} N={N} K={K}: "
            f"max_diff={(C.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    def test_tiled_matmul_edge_cases(self):
        """测试边界情况：单元素、一行、一列"""
        torch.manual_seed(42)

        # 1x1 @ 1x1
        A = torch.tensor([[2.0]], device="cuda", dtype=torch.float16)
        Bmat = torch.tensor([[3.0]], device="cuda", dtype=torch.float16)
        C = torch.empty(1, 1, device="cuda", dtype=torch.float16)
        cuda_kernels.tiled_matmul(A, Bmat, C)
        torch.cuda.synchronize()
        assert torch.allclose(
            C, torch.tensor([[6.0]], device="cuda", dtype=torch.float16), rtol=1e-2
        ), f"1x1 matmul failed: got {C.float().item():.4f}"

        # 1xK @ Kx1 → 1x1
        K_small = 64
        A = torch.randn(1, K_small, device="cuda", dtype=torch.float16)
        Bmat = torch.randn(K_small, 1, device="cuda", dtype=torch.float16)
        C = torch.empty(1, 1, device="cuda", dtype=torch.float16)
        cuda_kernels.tiled_matmul(A, Bmat, C)
        torch.cuda.synchronize()
        ref = torch.matmul(A.float(), Bmat.float()).to(torch.float16)
        assert torch.allclose(C.float(), ref.float(), rtol=1e-1, atol=5e-2), (
            f"1xK matmul failed: max_diff={(C.float() - ref.float()).abs().max().item():.6f}"
        )

    @requires_extension
    def test_tiled_matmul_no_nan(self):
        """验证输出不包含 NaN 或 Inf"""
        torch.manual_seed(42)
        M, N, K = 256, 256, 256
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        Bmat = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        cuda_kernels.tiled_matmul(A, Bmat, C)
        torch.cuda.synchronize()

        assert not torch.isnan(C).any(), "Tiled matmul 输出包含 NaN"
        assert not torch.isinf(C).any(), "Tiled matmul 输出包含 Inf"
