import math

import numpy as np
from cuda_wrappers.cublas_wrapper import cublas_matmul
from nn.layers.base import Layer


class Linear(Layer):
    def __init__(self, in_features, out_features, name=None):
        # Xavier/Glorot initialization
        stdv = 1.0 / math.sqrt(in_features)
        self.weight = np.random.uniform(
            -stdv, stdv, size=(out_features, in_features)
        ).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

        # Initialize gradients
        self.grad_weight = None
        self.grad_bias = None

        # Store input for backward pass
        self.input = None

        # Layer name for parameter identification
        self.name = name or id(self)

    def forward(self, x):
        # Save input for backward pass
        self.input = x

        # Compute output: y = x @ W^T + b using cuBLAS
        output = cublas_matmul(x, self.weight, transa=False, transb=True)

        # Add bias
        if self.bias is not None:
            output += self.bias

        return output

    def backward(self, grad_output):
        # Compute gradients for weights: dL/dW = grad_output.T @ input
        self.grad_weight = cublas_matmul(
            grad_output, self.input, transa=True, transb=False
        )

        # Compute gradients for bias: dL/db = sum(dL/dY)
        self.grad_bias = np.sum(grad_output, axis=0)

        # Compute gradient for input: dL/dX = dL/dY @ W
        grad_input = cublas_matmul(grad_output, self.weight, transa=False, transb=False)

        return grad_input

    def parameters(self):
        return [(self.weight, f"{self.name}_weight"), (self.bias, f"{self.name}_bias")]

    def get_gradients(self):
        return [
            (self.grad_weight, f"{self.name}_weight"),
            (self.grad_bias, f"{self.name}_bias"),
        ]
