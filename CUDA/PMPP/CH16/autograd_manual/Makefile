# Simple Makefile for CUDA libraries

# Compiler
NVCC = nvcc

# Flags
CFLAGS = -Xcompiler -fPIC -shared

# Directories
SRC_DIR = src
LIB_DIR = lib

# Make sure the lib directory exists
$(shell mkdir -p $(LIB_DIR))

# Default target
all: $(LIB_DIR)/libcublas_wrapper.so $(LIB_DIR)/conv2dcuda_wrapper.so

# Compile cublas wrapper
$(LIB_DIR)/libcublas_wrapper.so: $(SRC_DIR)/cublas/cublas_wrapper.c
	$(NVCC) $(CFLAGS) $< -o $@ -lcublas

# Compile conv2d wrapper
$(LIB_DIR)/conv2dcuda_wrapper.so: $(SRC_DIR)/cudnn/conv2d.cu
	$(NVCC) $(CFLAGS) $< -o $@

# Clean target
clean:
	rm -f $(LIB_DIR)/*.so

.PHONY: all clean