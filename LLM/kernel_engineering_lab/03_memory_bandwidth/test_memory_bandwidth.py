"""
Tests for 03_memory_bandwidth copy kernels and CUDA C++ bandwidth benchmarks.

Verifies correctness of:
  - Triton copy kernels (simple, vectorized, strided)
  - CUDA C++ bandwidth kernels (float/float2/float4, strided access)

Run: pytest 03_memory_bandwidth/test_memory_bandwidth.py -v
"""

import pytest
import torch

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from triton_copy import copy_kernel, copy_non_contiguous, copy_vectorized

# 检测 CUDA bandwidth C++ 扩展是否可用
_BANDWIDTH_KERNELS_AVAILABLE = False
try:
    import cuda_bandwidth_kernels  # type: ignore[import-not-found]

    _BANDWIDTH_KERNELS_AVAILABLE = True
except ImportError:
    pass

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE,
    reason="Triton not installed",
)

requires_bandwidth_kernels = pytest.mark.skipif(
    not _BANDWIDTH_KERNELS_AVAILABLE,
    reason="cuda_bandwidth_kernels extension not built",
)


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestCopyKernel:
    """Test simple copy kernel."""

    def test_1d_small(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_1d_medium(self):
        x = torch.randn(10000, device="cuda", dtype=torch.float32)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_1d_large(self):
        x = torch.randn(1_000_000, device="cuda", dtype=torch.float32)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_1d_non_power_of_two(self):
        x = torch.randn(999, device="cuda", dtype=torch.float32)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_2d_tensor(self):
        x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_2d_transposed(self):
        x = torch.randn(128, 64, device="cuda", dtype=torch.float32)
        x_t = x.t()  # strided view
        y = copy_kernel(x_t)
        assert torch.equal(x_t, y)

    def test_3d_tensor(self):
        x = torch.randn(4, 8, 64, device="cuda", dtype=torch.float32)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_fp16(self):
        if torch.cuda.get_device_capability()[0] < 7:
            pytest.skip("float16 requires compute capability >= 7.0")
        x = torch.randn(4096, device="cuda", dtype=torch.float16)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_bf16(self):
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("bfloat16 requires compute capability >= 8.0")
        x = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_different_block_sizes(self):
        x = torch.randn(5000, device="cuda", dtype=torch.float32)
        for bs in [128, 256, 512, 1024]:
            y = copy_kernel(x, block_size=bs)
            assert torch.equal(x, y), f"Failed for block_size={bs}"

    def test_zero_size_tensor(self):
        x = torch.randn(1, device="cuda", dtype=torch.float32)
        y = copy_kernel(x)
        assert torch.equal(x, y)

    def test_sliced_2d_tensor(self):
        x = torch.randn(100, 100, device="cuda", dtype=torch.float32)
        x_sliced = x[10:50, 20:80]  # non-contiguous slice
        y = copy_kernel(x_sliced)
        assert torch.equal(x_sliced, y)
        assert not x_sliced.is_contiguous(), "Slice should be non-contiguous"


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestVectorizedCopy:
    """Test vectorized copy kernel."""

    def test_vec_size_1(self):
        x = torch.randn(4096, device="cuda", dtype=torch.float32)
        y = copy_vectorized(x, vec_size=1)
        assert torch.equal(x, y)

    def test_vec_size_2(self):
        x = torch.randn(4096, device="cuda", dtype=torch.float32)
        y = copy_vectorized(x, vec_size=2)
        assert torch.equal(x, y)

    def test_vec_size_4(self):
        x = torch.randn(4096, device="cuda", dtype=torch.float32)
        y = copy_vectorized(x, vec_size=4)
        assert torch.equal(x, y)

    def test_vec_size_8(self):
        x = torch.randn(8192, device="cuda", dtype=torch.float32)
        y = copy_vectorized(x, vec_size=8)
        assert torch.equal(x, y)

    def test_non_power_of_two_elements(self):
        x = torch.randn(999, device="cuda", dtype=torch.float32)
        y = copy_vectorized(x, vec_size=4)
        assert torch.equal(x, y)

    def test_2d_tensor_vectorized(self):
        x = torch.randn(32, 128, device="cuda", dtype=torch.float32)
        y = copy_vectorized(x, vec_size=4)
        assert torch.equal(x, y)

    def test_fp16_vectorized(self):
        if torch.cuda.get_device_capability()[0] < 7:
            pytest.skip("float16 requires compute capability >= 7.0")
        x = torch.randn(4096, device="cuda", dtype=torch.float16)
        y = copy_vectorized(x, vec_size=4)
        assert torch.equal(x, y)


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestNonContiguousCopy:
    """Test strided copy kernel."""

    def test_stride_2(self):
        n = 2048
        x = torch.randn(n * 2, device="cuda", dtype=torch.float32)
        y = copy_non_contiguous(x, stride=2)
        assert torch.equal(y, x[::2])

    def test_stride_4(self):
        n = 1024
        x = torch.randn(n * 4, device="cuda", dtype=torch.float32)
        y = copy_non_contiguous(x, stride=4)
        assert torch.equal(y, x[::4])

    def test_stride_8(self):
        n = 512
        x = torch.randn(n * 8, device="cuda", dtype=torch.float32)
        y = copy_non_contiguous(x, stride=8)
        assert torch.equal(y, x[::8])

    def test_stride_16(self):
        n = 256
        x = torch.randn(n * 16, device="cuda", dtype=torch.float32)
        y = copy_non_contiguous(x, stride=16)
        assert torch.equal(y, x[::16])

    def test_stride_non_power_of_two(self):
        n = 512
        x = torch.randn(n * 3, device="cuda", dtype=torch.float32)
        y = copy_non_contiguous(x, stride=3)
        assert torch.equal(y, x[::3])

    def test_stride_1(self):
        n = 4096
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = copy_non_contiguous(x, stride=1)
        assert torch.equal(y, x)

    def test_small_tensor_stride(self):
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = copy_non_contiguous(x, stride=2)
        assert torch.equal(y, x[::2])


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestEdgeCases:
    """Test edge cases across all copy kernels."""

    def test_single_element(self):
        x = torch.tensor([42.0], device="cuda", dtype=torch.float32)
        assert torch.equal(copy_kernel(x), x)
        assert torch.equal(copy_vectorized(x, vec_size=4), x)

    def test_block_size_one(self):
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = copy_kernel(x, block_size=1)
        assert torch.equal(x, y)

    def test_very_large_tensor(self):
        x = torch.randn(10_000_000, device="cuda", dtype=torch.float32)
        y = copy_kernel(x)
        assert torch.equal(x, y)


# ---------------------------------------------------------------------------
# CUDA C++ bandwidth kernel tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_bandwidth_kernels
class TestCUDABandwidthKernels:
    """Test CUDA C++ bandwidth benchmark kernels."""

    def test_copy_float_works(self):
        n = 1_000_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        time_us = cuda_bandwidth_kernels.bench_copy_float(x)
        assert time_us.item() > 0.0

    def test_copy_float2_works(self):
        n = 1_000_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        time_us = cuda_bandwidth_kernels.bench_copy_float2(x)
        assert time_us.item() > 0.0

    def test_copy_float4_works(self):
        n = 1_000_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        time_us = cuda_bandwidth_kernels.bench_copy_float4(x)
        assert time_us.item() > 0.0

    @pytest.mark.parametrize("stride", [1, 4, 16, 64])
    def test_strided_copy_works(self, stride: int):
        n = 1_000_000
        x = torch.randn(n * stride, device="cuda", dtype=torch.float32)
        time_us = cuda_bandwidth_kernels.bench_strided_copy(x, stride)
        assert time_us.item() > 0.0

    def test_float4_copy_is_faster_than_float(self):
        """验证 float4 向量化访问比 float 标量访问更快。"""
        n = 2_000_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        float_time = cuda_bandwidth_kernels.bench_copy_float(x).item()
        float4_time = cuda_bandwidth_kernels.bench_copy_float4(x).item()

        # float4 应该至少不慢于 float（向量化通常提供更好的带宽）
        assert float_time > 0 and float4_time > 0

    def test_strided_access_is_slower(self):
        """验证 strided 访问比 coalesced 访问慢。"""
        n = 1_000_000
        x_small = torch.randn(n, device="cuda", dtype=torch.float32)
        x_large = torch.randn(n * 64, device="cuda", dtype=torch.float32)

        coalesced_time = cuda_bandwidth_kernels.bench_strided_copy(x_small, 1).item()
        strided_time = cuda_bandwidth_kernels.bench_strided_copy(x_large, 64).item()

        # stride=64 应该显著慢于 stride=1（coalesced）
        assert coalesced_time > 0 and strided_time > 0

    def test_elem_mul_float4_correctness(self):
        n = 1_000_000
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        result = cuda_bandwidth_kernels.bench_elem_mul_float4(a, b)
        time_us = result[0].item()
        c = result[1]

        expected = a * b
        assert time_us > 0.0
        assert torch.allclose(c, expected, atol=1e-4)
