# pooling_cy.pyx - Cython interface to our C pooling function
import numpy as np

cimport numpy as np

np.import_array()  # Initialize NumPy C API

# Declare the C function
cdef extern from "pooling.c":
    void poolingLayer_forward(int M, int H, int W, int K, float* Y, float* S, const char* pooling_type)

# Python wrapper for the C function
def pool_forward(np.ndarray[np.float32_t, ndim=4] input_tensor, int kernel_size, str pool_type):
    """
    Apply pooling to the input tensor
    
    Parameters:
    -----------
    input_tensor : numpy.ndarray 
        Input tensor of shape (batch_size, channels, height, width)
    kernel_size : int
        Size of the pooling kernel (K x K)
    pool_type : str
        Type of pooling ('max' or 'avg')
    
    Returns:
    --------
    numpy.ndarray
        Output tensor after pooling
    """
    cdef int batch_size = input_tensor.shape[0]
    cdef int channels = input_tensor.shape[1]
    cdef int height = input_tensor.shape[2]
    cdef int width = input_tensor.shape[3]
    
    # Make sure the arrays are C-contiguous
    cdef np.ndarray[np.float32_t, ndim=4] input_c = np.ascontiguousarray(input_tensor, dtype=np.float32)
    
    # Create output array
    cdef np.ndarray[np.float32_t, ndim=4] output = np.zeros(
        (batch_size, channels, height // kernel_size, width // kernel_size), 
        dtype=np.float32
    )
    
    # Get pointer to the data
    cdef float* input_ptr
    cdef float* output_ptr
    cdef bytes pool_type_bytes = pool_type.encode('utf-8')
    
    # Process each sample in the batch
    for b in range(batch_size):
        # Get pointers to the data for this batch item
        input_ptr = <float*>np.PyArray_DATA(input_c[b])
        output_ptr = <float*>np.PyArray_DATA(output[b])
        
        poolingLayer_forward(
            channels, 
            height, 
            width, 
            kernel_size,
            input_ptr,
            output_ptr,
            pool_type_bytes
        )
        
    return output