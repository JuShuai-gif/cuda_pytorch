import numpy as np
import pooling_module
import torch
import torch.nn.functional as F
import triton


def test_pooling_accuracy(
    batch_size=2, channels=16, height=32, width=32, kernel_size=2
):
    # Create a random input tensor
    input_np = np.random.rand(batch_size, channels, height, width).astype(np.float32)
    input_torch = torch.tensor(input_np)

    # Test max pooling
    print("Testing Max Pooling...")
    custom_max_output = pooling_module.pool_forward(input_np, kernel_size, "max")
    torch_max_output = F.max_pool2d(input_torch, kernel_size=kernel_size).numpy()
    max_diff = np.abs(custom_max_output - torch_max_output).max()
    print(f"Max Pooling - Maximum Difference: {max_diff}")

    # Test avg pooling
    print("\nTesting Average Pooling...")
    custom_avg_output = pooling_module.pool_forward(input_np, kernel_size, "avg")
    torch_avg_output = F.avg_pool2d(input_torch, kernel_size=kernel_size).numpy()
    avg_diff = np.abs(custom_avg_output - torch_avg_output).max()
    print(f"Average Pooling - Maximum Difference: {avg_diff}")

    return max_diff < 1e-5 and avg_diff < 1e-5


def benchmark_pooling(batch_size=8, channels=64, height=128, width=128, kernel_size=2):
    # Create tensors
    input_np = np.random.rand(batch_size, channels, height, width).astype(np.float32)
    input_torch = torch.tensor(input_np)

    # Benchmark custom max pooling
    custom_max_time = triton.testing.do_bench(
        lambda: pooling_module.pool_forward(input_np, kernel_size, "max")
    )

    # Benchmark torch max pooling
    torch_max_time = triton.testing.do_bench(
        lambda: F.max_pool2d(input_torch, kernel_size=kernel_size)
    )

    # Benchmark custom avg pooling
    custom_avg_time = triton.testing.do_bench(
        lambda: pooling_module.pool_forward(input_np, kernel_size, "avg")
    )

    # Benchmark torch avg pooling
    torch_avg_time = triton.testing.do_bench(
        lambda: F.avg_pool2d(input_torch, kernel_size=kernel_size)
    )

    print(f"\nInput size: [{batch_size}, {channels}, {height}, {width}]")
    print(
        f"Max Pooling - Custom: {custom_max_time:.3f}ms, PyTorch: {torch_max_time:.3f}ms"
    )
    print(
        f"Avg Pooling - Custom: {custom_avg_time:.3f}ms, PyTorch: {torch_avg_time:.3f}ms"
    )


if __name__ == "__main__":
    print("=== Testing Pooling Implementation ===")
    result = test_pooling_accuracy()
    print("\n✓ Tests passed!" if result else "\n✗ Tests failed!")

    print("\n=== Performance Benchmarks ===")
    benchmark_pooling(batch_size=2, channels=16, height=32, width=32)
    benchmark_pooling(batch_size=8, channels=64, height=128, width=128)
