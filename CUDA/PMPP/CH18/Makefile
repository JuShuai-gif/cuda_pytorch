NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_60 # Adjust the architecture as needed for your GPU

# Output executable
TARGET = energy_benchmark

all: $(TARGET)

# Direct compilation approach - no intermediate object files
$(TARGET): benchmark.cu cenergy.cu cenergy.h benchmark.h
	$(NVCC) $(CUDA_FLAGS) benchmark.cu cenergy.cu -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean