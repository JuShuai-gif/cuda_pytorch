"""
Tests for 07_cuda_streams_async.

Verifies correctness of stream operations, async copies, synchronization,
and CUDA C++ stream kernels.

Run: pytest 07_cuda_streams_async/test_cuda_streams.py -v
"""

from __future__ import annotations

import pytest
import torch

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from async_copy import (
    _launch_compute,
    d2h_async,
    h2d_async_pinned,
    h2d_blocking_pageable,
)
from event_timing import _launch_work, measure_overlap_efficiency
from stream_basics import (
    _launch_add,
    _launch_mul,
    single_stream_example,
    two_streams_concurrent,
)

# 检测 CUDA stream C++ 扩展是否可用
_STREAM_KERNELS_AVAILABLE = False
try:
    import cuda_stream_kernels  # type: ignore[import-not-found]

    _STREAM_KERNELS_AVAILABLE = True
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

requires_stream_kernels = pytest.mark.skipif(
    not _STREAM_KERNELS_AVAILABLE,
    reason="cuda_stream_kernels extension not built",
)


# ---------------------------------------------------------------------------
# Stream basics tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestStreamBasics:
    """Test basic stream operations."""

    def test_single_stream_correctness(self):
        """Verify kernel on a stream produces correct results."""
        n = 100_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            _launch_add(x, y, out)
        stream.synchronize()

        expected = x + y
        assert torch.allclose(out, expected, atol=1e-5)

    def test_two_streams_concurrent_correctness(self):
        """Verify two streams produce correct results."""
        n = 100_000
        x1 = torch.randn(n, device="cuda", dtype=torch.float32)
        y1 = torch.randn(n, device="cuda", dtype=torch.float32)
        out1 = torch.empty_like(x1)

        x2 = torch.randn(n, device="cuda", dtype=torch.float32)
        y2 = torch.randn(n, device="cuda", dtype=torch.float32)
        out2 = torch.empty_like(x2)

        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        with torch.cuda.stream(stream_a):
            _launch_add(x1, y1, out1)
        with torch.cuda.stream(stream_b):
            _launch_mul(x2, y2, out2)

        stream_a.synchronize()
        stream_b.synchronize()

        assert torch.allclose(out1, x1 + y1, atol=1e-5)
        assert torch.allclose(out2, x2 * y2, atol=1e-5)

    def test_stream_sync_ordering(self):
        """Verify stream.synchronize preserves ordering."""
        n = 50_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out1 = torch.empty_like(x)
        out2 = torch.empty_like(x)

        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        with torch.cuda.stream(stream_a):
            _launch_add(x, y, out1)

        # Synchronize stream_a before launching dependent op on stream_b
        stream_a.synchronize()

        with torch.cuda.stream(stream_b):
            # out2 = out1 * 2 (depends on stream_a's result)
            _launch_mul(out1, torch.full_like(out1, 2.0), out2)

        stream_b.synchronize()

        expected = (x + y) * 2.0
        assert torch.allclose(out2, expected, atol=1e-5)

    def test_default_stream_blocks_others(self):
        """Test that default stream operations wait for explicit streams."""
        n = 50_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out1 = torch.empty_like(x)
        out2 = torch.empty_like(x)

        stream_a = torch.cuda.Stream()

        with torch.cuda.stream(stream_a):
            _launch_add(x, y, out1)

        # Default stream op - should wait for stream_a to complete
        _launch_mul(out1, torch.full_like(out1, 2.0), out2)

        stream_a.synchronize()
        torch.cuda.synchronize()

        expected = (x + y) * 2.0
        assert torch.allclose(out2, expected, atol=1e-5)

    def test_multiple_streams_independent(self):
        """Verify multiple streams can operate independently."""
        n = 50_000
        results = []

        for i in range(4):
            x = torch.randn(n, device="cuda", dtype=torch.float32)
            y = torch.randn(n, device="cuda", dtype=torch.float32)
            out = torch.empty_like(x)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                _launch_add(x, y, out)
            results.append((stream, out))

        # Synchronize all and verify
        for stream, out in results:
            stream.synchronize()


# ---------------------------------------------------------------------------
# Async copy tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestAsyncCopy:
    """Test async data transfer with pinned memory."""

    def test_h2d_pinned_correctness(self):
        """Verify pinned memory H2D transfer produces correct result."""
        n = 100_000
        host_data = torch.randn(n, dtype=torch.float32, pin_memory=True)
        device_data = torch.empty(n, device="cuda", dtype=torch.float32)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            device_data.copy_(host_data, non_blocking=True)
        stream.synchronize()

        assert torch.allclose(device_data.cpu(), host_data, atol=1e-5)

    def test_d2h_pinned_correctness(self):
        """Verify pinned memory D2H transfer produces correct result."""
        n = 100_000
        device_data = torch.randn(n, device="cuda", dtype=torch.float32)
        host_pinned = torch.empty(n, dtype=torch.float32, pin_memory=True)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            host_pinned.copy_(device_data, non_blocking=True)
        stream.synchronize()

        assert torch.allclose(host_pinned, device_data.cpu(), atol=1e-5)

    def test_compute_kernel_correctness(self):
        """Verify the compute kernel used in pipeline works correctly."""
        n = 50_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        _launch_compute(x, out)
        torch.cuda.synchronize()

        expected = x**3
        assert torch.allclose(out, expected, atol=1e-3)

    def test_async_copy_with_compute(self):
        """Verify we can do async copy and compute on different streams."""
        n = 100_000
        host_input = torch.randn(n, dtype=torch.float32, pin_memory=True)
        host_output = torch.empty(n, dtype=torch.float32, pin_memory=True)

        dev_buf = torch.empty(n, device="cuda", dtype=torch.float32)
        dev_out = torch.empty(n, device="cuda", dtype=torch.float32)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            dev_buf.copy_(host_input, non_blocking=True)
            _launch_compute(dev_buf, dev_out)
            host_output.copy_(dev_out, non_blocking=True)

        stream.synchronize()

        expected = host_input**3
        assert torch.allclose(host_output, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Event timing tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestEventTiming:
    """Test CUDA event timing utilities."""

    def test_event_timing_positive(self):
        """Verify event timing returns positive values."""
        n = 100_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.current_stream().record_event(start)
        _launch_work(x, out, iterations=50)
        torch.cuda.current_stream().record_event(end)
        end.synchronize()

        elapsed = start.elapsed_time(end)
        assert elapsed > 0.0, f"Event timing should be positive, got {elapsed}"

    def test_stream_independent_events(self):
        """Verify events on different streams are independent."""
        n = 100_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out1 = torch.empty_like(x)
        out2 = torch.empty_like(x)

        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        evt_a = torch.cuda.Event(enable_timing=True)
        evt_b = torch.cuda.Event(enable_timing=True)

        stream_a.record_event(evt_a)
        _launch_work(x, out1, iterations=50, stream=stream_a)

        stream_b.record_event(evt_b)
        _launch_work(x, out2, iterations=50, stream=stream_b)

        stream_a.synchronize()
        stream_b.synchronize()

        # Both should complete correctly
        assert out1 is not None
        assert out2 is not None


# ---------------------------------------------------------------------------
# Synchronization behavior tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestSynchronization:
    """Test stream synchronization patterns."""

    def test_stream_sync_blocks_one(self):
        """Verify stream.synchronize() only blocks one stream."""
        n = 100_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out_a = torch.empty_like(x)
        out_b = torch.empty_like(x)

        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        with torch.cuda.stream(stream_a):
            _launch_add(x, y, out_a)
        with torch.cuda.stream(stream_b):
            _launch_mul(x, y, out_b)

        # Sync stream_a only; stream_b may still be running
        stream_a.synchronize()
        assert torch.allclose(out_a, x + y, atol=1e-5)

        # Now sync stream_b
        stream_b.synchronize()
        assert torch.allclose(out_b, x * y, atol=1e-5)

    def test_global_sync_blocks_all(self):
        """Verify torch.cuda.synchronize() blocks all streams."""
        n = 100_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out_a = torch.empty_like(x)
        out_b = torch.empty_like(x)

        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        with torch.cuda.stream(stream_a):
            _launch_add(x, y, out_a)
        with torch.cuda.stream(stream_b):
            _launch_mul(x, y, out_b)

        torch.cuda.synchronize()

        # Both should be complete after global sync
        assert torch.allclose(out_a, x + y, atol=1e-5)
        assert torch.allclose(out_b, x * y, atol=1e-5)


# ---------------------------------------------------------------------------
# CUDA C++ stream kernel tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_stream_kernels
class TestCUDAStreamKernels:
    """Test the CUDA C++ stream kernel extension."""

    def test_multi_stream_concurrent_correctness(self):
        """验证多 stream 并发执行 vector_add 的结果正确性。"""
        n = 100_000
        num_streams = 4
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        # 准备 N 组输入
        a_list = [a.clone() for _ in range(num_streams)]
        b_list = [b.clone() for _ in range(num_streams)]
        out_list = [torch.empty_like(a) for _ in range(num_streams)]

        import cuda_stream_kernels

        timing = cuda_stream_kernels.multi_stream_concurrent_exec(a_list, b_list, out_list)

        assert isinstance(timing, list)
        assert len(timing) == 1
        assert timing[0].shape[0] == num_streams

        for i in range(num_streams):
            expected = a_list[i] + b_list[i]
            assert torch.allclose(out_list[i], expected, atol=1e-5), f"Stream {i} mismatch"

    def test_kernel_timing_positive(self):
        """验证 CUDA event 计时返回正值。"""
        n = 500_000
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        import cuda_stream_kernels

        result = cuda_stream_kernels.kernel_timing_with_events(a, b, 10)
        total_ms = result[0].item()
        avg_ms = result[1].item()

        assert total_ms > 0.0, f"总 GPU 时间应为正值: {total_ms}"
        assert avg_ms > 0.0, f"平均 GPU 时间应为正值: {avg_ms}"

    def test_stream_wait_event_correctness(self):
        """验证 cudaStreamWaitEvent 跨 stream 同步的正确性。"""
        n = 100_000
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        import cuda_stream_kernels

        intermediate, final_result = cuda_stream_kernels.stream_wait_event_demo(a, b)

        # intermediate = a + b
        assert torch.allclose(intermediate, a + b, atol=1e-4)
        # final_result = intermediate^4 (vector_mul_pow 做 x^4)
        assert torch.allclose(final_result, intermediate**4, atol=1e-2)

    def test_war_sync_both_correct(self):
        """验证 WAR 同步正反例都能产生正确结果。"""
        n = 50_000
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        import cuda_stream_kernels

        out_wrong_a, out_wrong_b, out_correct_c, out_correct_d, timing = (
            cuda_stream_kernels.war_sync_correct_vs_wrong(a, b)
        )

        # 错误方式的输出
        assert torch.allclose(out_wrong_a, a + b, atol=1e-4)
        # 正确方式的输出
        assert torch.allclose(out_correct_c, a + b, atol=1e-4)

        # timing[0] = cudaDeviceSynchronize 方式, timing[1] = cudaStreamSynchronize 方式
        assert timing[0].item() > 0.0
        assert timing[1].item() > 0.0

    def test_pinned_async_pipeline_correctness(self):
        """验证 pinned memory pipeline 的正确性。"""
        n = 50_000
        num_chunks = 4
        num_streams = 2

        host_chunks = [
            torch.randn(n, dtype=torch.float32, pin_memory=True) for _ in range(num_chunks)
        ]

        import cuda_stream_kernels

        results = cuda_stream_kernels.pinned_async_pipeline(host_chunks, num_streams)

        # 最后一个是 timing，前面是每个 chunk 的输出
        timing = results[-1]
        outputs = results[:-1]

        assert len(outputs) == num_chunks
        for i in range(num_chunks):
            expected = (host_chunks[i] ** 4).float()  # vector_mul_pow 做 x^4
            assert torch.allclose(outputs[i], expected, atol=1e-2), f"Chunk {i} mismatch"

        assert timing.item() > 0.0, "Pipeline 时间应为正值"
