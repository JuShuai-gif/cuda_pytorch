import ctypes
from ctypes import POINTER, c_float, c_int

# Load the cuDNN wrapper library
try:
    conv2dcuda_lib = ctypes.CDLL("./lib/conv2dcuda_wrapper.so")
    print("Successfully loaded conv2dcuda wrapper library")
except Exception as e:
    print(f"Error loading conv2dcuda library: {e}")
    raise

# Set up function signatures for conv2dcuda
conv2dcuda_lib.init_cudnn.restype = c_int
conv2dcuda_lib.cleanup_cudnn.restype = c_int

# Conv2D forward
conv2dcuda_lib.conv2d_forward.argtypes = [
    POINTER(c_float),  # input
    POINTER(c_float),  # weights
    POINTER(c_float),  # bias
    POINTER(c_float),  # output
    c_int,  # batch_size
    c_int,  # in_channels
    c_int,  # height
    c_int,  # width
    c_int,  # out_channels
    c_int,  # kernel_h
    c_int,  # kernel_w
    c_int,  # pad_h
    c_int,  # pad_w
    c_int,  # stride_h
    c_int,  # stride_w
]
conv2dcuda_lib.conv2d_forward.restype = c_int

# Conv2D backward
conv2dcuda_lib.conv2d_backward.argtypes = [
    POINTER(c_float),  # input
    POINTER(c_float),  # weights
    POINTER(c_float),  # d_output
    POINTER(c_float),  # d_input
    POINTER(c_float),  # d_weights
    POINTER(c_float),  # d_bias
    c_int,  # batch_size
    c_int,  # in_channels
    c_int,  # height
    c_int,  # width
    c_int,  # out_channels
    c_int,  # kernel_h
    c_int,  # kernel_w
    c_int,  # pad_h
    c_int,  # pad_w
    c_int,  # stride_h
    c_int,  # stride_w
]
conv2dcuda_lib.conv2d_backward.restype = c_int

# MaxPool2D forward
conv2dcuda_lib.maxpool2d_forward.argtypes = [
    POINTER(c_float),  # input
    POINTER(c_float),  # output
    POINTER(c_int),  # indices
    c_int,  # batch_size
    c_int,  # channels
    c_int,  # height
    c_int,  # width
    c_int,  # kernel_h
    c_int,  # kernel_w
    c_int,  # stride_h
    c_int,  # stride_w
]
conv2dcuda_lib.maxpool2d_forward.restype = c_int

# MaxPool2D backward
conv2dcuda_lib.maxpool2d_backward.argtypes = [
    POINTER(c_float),  # input
    POINTER(c_float),  # d_output
    POINTER(c_int),  # indices
    POINTER(c_float),  # d_input
    c_int,  # batch_size
    c_int,  # channels
    c_int,  # height
    c_int,  # width
    c_int,  # kernel_h
    c_int,  # kernel_w
    c_int,  # stride_h
    c_int,  # stride_w
]
conv2dcuda_lib.maxpool2d_backward.restype = c_int

# Initialize cuDNN
result_cudnn = conv2dcuda_lib.init_cudnn()
if result_cudnn != 0:
    raise Exception("Failed to initialize cuDNN")


def cleanup_cudnn():
    """Clean up cuDNN resources"""
    return conv2dcuda_lib.cleanup_cudnn()
