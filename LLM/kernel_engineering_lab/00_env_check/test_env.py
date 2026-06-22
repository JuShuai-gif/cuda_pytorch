"""
Tests for 00_env_check module.

Verifies that the GPU development environment is correctly configured.
Run with: pytest 00_env_check/test_env.py -v
"""

import pytest


def test_cuda_available():
    """Verify CUDA is available in PyTorch."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")
    assert torch.cuda.device_count() > 0


def test_create_tensor_on_gpu():
    """Verify PyTorch can allocate and operate on GPU tensors."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    x = torch.randn(100, 100, device="cuda")
    y = torch.randn(100, 100, device="cuda")
    z = x + y
    assert z.device.type == "cuda"
    assert z.shape == (100, 100)

    z2 = torch.matmul(x, y)
    torch.cuda.synchronize()
    assert z2.device.type == "cuda"
    assert z2.shape == (100, 100)


def test_triton_import():
    """Verify Triton can be imported."""
    try:
        import triton  # noqa: F401

        triton_available = True
    except ImportError:
        triton_available = False

    if not triton_available:
        pytest.skip("Triton is not installed on this machine")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    # Basic Triton kernel test to verify it actually works
    import triton
    import triton.language as tl

    @triton.jit
    def _identity_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, x, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _identity_kernel[grid](x, y, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()
    assert torch.allclose(x, y, atol=1e-6)


def test_gpu_count():
    """Verify GPU count > 0."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    assert torch.cuda.device_count() > 0


def test_compute_capability():
    """Verify compute capability is detectable."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        assert props.major >= 0
        assert props.minor >= 0
        assert props.name != ""
        assert props.total_mem > 0
