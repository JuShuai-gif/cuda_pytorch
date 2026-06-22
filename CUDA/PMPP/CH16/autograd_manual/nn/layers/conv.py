import math
from ctypes import c_int

import numpy as np
from cuda_wrappers.cudnn_wrapper import conv2dcuda_lib
from nn.layers.base import Layer
from utils.conversion import np_to_c_float_p


class Conv2D(Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, name=None
    ):
        # Convert kernel_size and stride to tuples if they are integers
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Xavier/Glorot initialization for weights
        stdv = 1.0 / math.sqrt(in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = np.random.uniform(
            -stdv,
            stdv,
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
        ).astype(np.float32)

        # Initialize bias
        self.bias = np.zeros(out_channels, dtype=np.float32)

        # Initialize gradients
        self.grad_weight = None
        self.grad_bias = None

        # Store input and output dimensions for backward pass
        self.input = None
        self.input_shape = None
        self.output_shape = None

        # Layer name for parameter identification
        self.name = name or id(self)

    def forward(self, x):
        # Save input for backward pass
        self.input = x
        self.input_shape = x.shape

        batch_size, channels, height, width = x.shape

        # Calculate output dimensions
        out_height = (
            (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]
        ) + 1
        out_width = (
            (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]
        ) + 1

        # Allocate output tensor
        output = np.zeros(
            (batch_size, self.out_channels, out_height, out_width), dtype=np.float32
        )
        self.output_shape = output.shape

        # Convert to C pointers
        x_ptr = np_to_c_float_p(x)
        weight_ptr = np_to_c_float_p(self.weight)
        bias_ptr = np_to_c_float_p(self.bias)
        output_ptr = np_to_c_float_p(output)

        # Call cuDNN wrapper for convolution
        result = conv2dcuda_lib.conv2d_forward(
            x_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            c_int(batch_size),
            c_int(channels),
            c_int(height),
            c_int(width),
            c_int(self.out_channels),
            c_int(self.kernel_size[0]),
            c_int(self.kernel_size[1]),
            c_int(self.padding[0]),
            c_int(self.padding[1]),
            c_int(self.stride[0]),
            c_int(self.stride[1]),
        )

        if result != 0:
            raise Exception(f"conv2d_forward failed with code {result}")

        return output

    def backward(self, grad_output):
        batch_size, channels, height, width = self.input_shape

        # Initialize gradient tensors
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(self.input)

        # Convert to C pointers
        input_ptr = np_to_c_float_p(self.input)
        weight_ptr = np_to_c_float_p(self.weight)
        grad_output_ptr = np_to_c_float_p(grad_output)
        grad_input_ptr = np_to_c_float_p(grad_input)
        grad_weight_ptr = np_to_c_float_p(self.grad_weight)
        grad_bias_ptr = np_to_c_float_p(self.grad_bias)

        # Call cuDNN wrapper for backward pass
        result = conv2dcuda_lib.conv2d_backward(
            input_ptr,
            weight_ptr,
            grad_output_ptr,
            grad_input_ptr,
            grad_weight_ptr,
            grad_bias_ptr,
            c_int(batch_size),
            c_int(channels),
            c_int(height),
            c_int(width),
            c_int(self.out_channels),
            c_int(self.kernel_size[0]),
            c_int(self.kernel_size[1]),
            c_int(self.padding[0]),
            c_int(self.padding[1]),
            c_int(self.stride[0]),
            c_int(self.stride[1]),
        )

        if result != 0:
            raise Exception(f"conv2d_backward failed with code {result}")

        return grad_input

    def parameters(self):
        return [(self.weight, f"{self.name}_weight"), (self.bias, f"{self.name}_bias")]

    def get_gradients(self):
        return [
            (self.grad_weight, f"{self.name}_weight"),
            (self.grad_bias, f"{self.name}_bias"),
        ]
