from ctypes import POINTER, c_float, c_int

import numpy as np


# Helper function to convert numpy array to C float pointer
def np_to_c_float_p(np_array):
    # Make sure the array is contiguous and float32
    np_array = np.ascontiguousarray(np_array, dtype=np.float32)
    return np_array.ctypes.data_as(POINTER(c_float))


# Helper function to convert numpy array to C int pointer
def np_to_c_int_p(np_array):
    # Make sure the array is contiguous and int32
    np_array = np.ascontiguousarray(np_array, dtype=np.int32)
    return np_array.ctypes.data_as(POINTER(c_int))
