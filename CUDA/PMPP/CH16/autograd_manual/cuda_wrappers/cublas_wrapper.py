import ctypes
from ctypes import POINTER,c_float,c_int

import numpy as np
from utils.conversion import np_to_c_float_p

# Load the cuBLAS wrapper library
try:
    cublas_lib = ctypes.CDLL("./lib/libcublas_wrapper.so")
    print("Successfully loaded cuBLAS wrapper library")
except Exception as e:
    print(f"Error loading cuBLAS library: {e}")
    raise

# Set up function signatures for cuBLAS
cublas_lib.init_cublas.restype = c_int
cublas_lib.cleanup_cublas.restype = c_int
cublas_lib.sgemm_wrapper.argtypes = [
    POINTER(c_float),  # A
    POINTER(c_float),  # B
    POINTER(c_float),  # C
    c_int,  # m
    c_int,  # n
    c_int,  # k
    c_int,  # transa
    c_int,  # transb
]
cublas_lib.sgemm_wrapper.restype = c_int

# Initialize cuBLAS
result_cublas = cublas_lib.init_cublas()
if result_cublas != 0:
    raise Exception("Failed to initialize cuBLAS")


def cleanup_cublas():
    """Clean up cuBLAS resources"""
    return cublas_lib.cleanup_cublas()


# Direct matrix multiplication using cuBLAS
def cublas_matmul(a, b, transa=False, transb=False):
    # Ensure arrays are float32 and contiguous
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    # Get dimensions
    if not transa and not transb:
        # C(m,n) = A(m,k) @ B(k,n)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"
    elif transa and not transb:
        # C(m,n) = A(k,m)^T @ B(k,n)
        k, m = a.shape
        k2, n = b.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"
    elif not transa and transb:
        # C(m,n) = A(m,k) @ B(n,k)^T
        m, k = a.shape
        n, k2 = b.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"
    else:  # transa and transb
        # C(m,n) = A(k,m)^T @ B(n,k)^T
        k, m = a.shape
        n, k2 = b.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"

    # Allocate output array
    c = np.zeros((m, n), dtype=np.float32)

    # Get C pointers to arrays
    a_ptr = np_to_c_float_p(a)
    b_ptr = np_to_c_float_p(b)
    c_ptr = np_to_c_float_p(c)

    # Call C function
    result = cublas_lib.sgemm_wrapper(
        a_ptr,
        b_ptr,
        c_ptr,
        c_int(m),
        c_int(n),
        c_int(k),
        c_int(1 if transa else 0),
        c_int(1 if transb else 0),
    )

    if result != 0:
        raise Exception(f"cublas_matmul failed with code {result}")

    return c












