# cython: language_level=3
# distutils: language=c

import numpy as np

cimport numpy as np

np.import_array()  # Initialize NumPy C API

cdef extern:
    void convLayer_backward_x_grad(int M, int C, int H_in, int W_in, int K,
                                float* dE_dY, float* W, float* dE_dX)

def py_conv_backward_x_grad(np.ndarray[np.float32_t, ndim=3] dE_dY, 
                         np.ndarray[np.float32_t, ndim=4] W,
                         int H_in, int W_in):
    # Extract dimensions
    cdef int M = dE_dY.shape[0]
    cdef int C = W.shape[1]
    cdef int K = W.shape[2]
    
    # Create output array
    cdef np.ndarray[np.float32_t, ndim=3] dE_dX = np.zeros((C, H_in, W_in), dtype=np.float32)
    
    # Ensure arrays are contiguous in memory
    dE_dY = np.ascontiguousarray(dE_dY, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    dE_dX = np.ascontiguousarray(dE_dX, dtype=np.float32)
    
    # Call the C function
    convLayer_backward_x_grad(M, C, H_in, W_in, K,
                           <float*>dE_dY.data, <float*>W.data, <float*>dE_dX.data)
    
    return dE_dX