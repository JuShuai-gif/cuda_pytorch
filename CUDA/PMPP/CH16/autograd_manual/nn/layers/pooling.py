from ctypes import c_int

import numpy as np
from cuda_wrappers.cudnn_wrapper import conv2dcuda_lib
from nn.layers.base import Layer
from utils.conversion import np_to_c_float_p, np_to_c_int_p


class MaxPooling2D(Layer):
    def __init__(self, kernel_size, stride=None, name=None):
        # Convert kernel_size to tuple if it's an integer
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # If stride is not specified, use kernel_size as stride
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Store for backward pass
        self.input = None
        self.max_indices = None
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
        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1

        # Initialize output tensor and indices for max locations
        output = np.zeros(
            (batch_size, channels, out_height, out_width), dtype=np.float32
        )
        self.max_indices = np.zeros(
            (batch_size, channels, out_height, out_width), dtype=np.int32
        )
        self.output_shape = output.shape

        # Convert to C pointers
        x_ptr = np_to_c_float_p(x)
        output_ptr = np_to_c_float_p(output)
        indices_ptr = np_to_c_int_p(self.max_indices)

        # Call cuDNN wrapper for max pooling
        result = conv2dcuda_lib.maxpool2d_forward(
            x_ptr,
            output_ptr,
            indices_ptr,
            c_int(batch_size),
            c_int(channels),
            c_int(height),
            c_int(width),
            c_int(self.kernel_size[0]),
            c_int(self.kernel_size[1]),
            c_int(self.stride[0]),
            c_int(self.stride[1]),
        )

        if result != 0:
            raise Exception(f"maxpool2d_forward failed with code {result}")

        return output

    def backward(self, grad_output):
        batch_size, channels, height, width = self.input_shape

        # Initialize gradient tensor
        grad_input = np.zeros((batch_size, channels, height, width), dtype=np.float32)

        # Convert to C pointers
        input_ptr = np_to_c_float_p(self.input)
        grad_output_ptr = np_to_c_float_p(grad_output)
        indices_ptr = np_to_c_int_p(self.max_indices)
        grad_input_ptr = np_to_c_float_p(grad_input)

        # Call cuDNN wrapper for max pooling backward
        result = conv2dcuda_lib.maxpool2d_backward(
            input_ptr,
            grad_output_ptr,
            indices_ptr,
            grad_input_ptr,
            c_int(batch_size),
            c_int(channels),
            c_int(height),
            c_int(width),
            c_int(self.kernel_size[0]),
            c_int(self.kernel_size[1]),
            c_int(self.stride[0]),
            c_int(self.stride[1]),
        )

        if result != 0:
            raise Exception(f"maxpool2d_backward failed with code {result}")

        return grad_input

    def parameters(self):
        # MaxPooling doesn't have learnable parameters
        return []

    def get_gradients(self):
        # No gradients to return
        return []
