import argparse
import random

import numpy as np
import torch
from cuda_wrappers.cublas_wrapper import cleanup_cublas
from cuda_wrappers.cudnn_wrapper import cleanup_cudnn
from examples.mnist_example import mnist_cnn_example
from examples.xor_example import xor_example

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main():
    parser = argparse.ArgumentParser(
        description="Run neural network examples with CUDA acceleration"
    )
    parser.add_argument("--xor", action="store_true", help="Run XOR example (default)")
    parser.add_argument("--mnist", action="store_true", help="Run MNIST CNN example")

    args = parser.parse_args()

    try:
        if args.mnist:
            print("Running MNIST CNN example with CUDA acceleration...")
            mnist_cnn_example()
            print("MNIST CNN example completed successfully")
        else:
            print("Running XOR example with CUDA acceleration...")
            xor_example()
            print("XOR example completed successfully")
    finally:
        # Clean up cuBLAS and cuDNN resources
        cleanup_cublas()
        cleanup_cudnn()
        print("Cleaned up cuBLAS and cuDNN resources")


if __name__ == "__main__":
    main()
