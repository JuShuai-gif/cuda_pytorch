import subprocess
import sys

import numpy as np
import torch

subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
import conv_wrapper  # noqa: E402


def compare_with_pytorch():
    M = 2  # Number of output channels
    C = 3  # Number of input channels
    H_in = 28
    W_in = 28
    K = 3  # Kernel size

    # Expected output dimensions
    H_out = H_in - K + 1
    W_out = W_in - K + 1

    np.random.seed(42)

    x = torch.randn(1, C, H_in, W_in, requires_grad=True)
    w = torch.randn(M, C, K, K)

    # Forward pass in PyTorch
    y = torch.nn.functional.conv2d(x, w, padding=0)

    # Create a gradient for output
    dE_dY = torch.randn(1, M, H_out, W_out)

    y.backward(dE_dY)
    torch_dE_dX = x.grad.numpy()[0]  # Extract from batch dimension

    # Prepare inputs for our implementation
    numpy_dE_dY = dE_dY.numpy()[0].astype(np.float32)
    numpy_W = w.numpy().astype(np.float32)

    our_dE_dX = conv_wrapper.py_conv_backward_x_grad(numpy_dE_dY, numpy_W, H_in, W_in)

    print("PyTorch gradient shape:", torch_dE_dX.shape)
    print("Our gradient shape:", our_dE_dX.shape)

    abs_diff = np.abs(torch_dE_dX - our_dE_dX)
    max_diff = np.max(abs_diff)
    avg_diff = np.mean(abs_diff)

    print(f"Maximum absolute difference: {max_diff}")
    print(f"Average absolute difference: {avg_diff}")

    is_close = np.allclose(torch_dE_dX, our_dE_dX, rtol=1e-4, atol=1e-4)
    print(f"Results match: {is_close}")

    if not is_close:
        print("\nSample differences:")
        max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"At position {max_idx}:")
        print(f"  PyTorch value: {torch_dE_dX[max_idx]}")
        print(f"  Our value: {our_dE_dX[max_idx]}")

        for _ in range(3):
            c = np.random.randint(0, C)
            h = np.random.randint(0, H_in)
            w = np.random.randint(0, W_in)
            print(f"At position ({c},{h},{w}):")
            print(f"  PyTorch value: {torch_dE_dX[c,h,w]}")
            print(f"  Our value: {our_dE_dX[c,h,w]}")


def run_multiple_tests():
    print("\n=== Running multiple tests with different configurations ===")

    test_configs = [
        # (M, C, H_in, W_in, K)
        (2, 3, 5, 5, 3),  # Small square input
        (4, 3, 10, 10, 3),  # Medium square input
        (2, 1, 7, 5, 3),  # Non-square input
        (3, 2, 8, 8, 5),  # Larger kernel
        (1, 3, 6, 6, 2),  # Small kernel
    ]

    for config in test_configs:
        M, C, H_in, W_in, K = config
        print(f"\nTesting with M={M}, C={C}, H_in={H_in}, W_in={W_in}, K={K}")

        H_out = H_in - K + 1
        W_out = W_in - K + 1

        # Create random input and weights
        np.random.seed(42)

        # Create PyTorch tensors
        x = torch.randn(1, C, H_in, W_in, requires_grad=True)
        w = torch.randn(M, C, K, K)

        # Forward pass in PyTorch
        y = torch.nn.functional.conv2d(x, w, padding=0)

        # Create a gradient for output
        dE_dY = torch.randn(1, M, H_out, W_out)

        # Backward pass in PyTorch
        y.backward(dE_dY)
        torch_dE_dX = x.grad.numpy()[0]  # Extract from batch dimension

        # Prepare inputs for our implementation
        numpy_dE_dY = dE_dY.numpy()[0].astype(np.float32)  # Remove batch dimension
        numpy_W = w.numpy().astype(np.float32)

        # Run our C implementation via Cython
        our_dE_dX = conv_wrapper.py_conv_backward_x_grad(
            numpy_dE_dY, numpy_W, H_in, W_in
        )

        # Check if results are close
        is_close = np.allclose(torch_dE_dX, our_dE_dX, rtol=1e-4, atol=1e-4)
        max_diff = np.max(np.abs(torch_dE_dX - our_dE_dX))

        print(f"Results match: {is_close}, Max difference: {max_diff}")


if __name__ == "__main__":
    compare_with_pytorch()
    run_multiple_tests()
