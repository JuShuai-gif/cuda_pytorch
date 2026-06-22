# CUDA BFS project Makefile
# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -g -O2 -std=c++11

# Include directory
INCLUDES = -I./include

# Source and object directories
SRC_DIR = src
OBJ_DIR = obj

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)

# Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Executable name
TARGET = bfs

# Phony targets
.PHONY: all clean dirs rebuild

# Default target
all: dirs $(TARGET)

# Create necessary directories
dirs:
	@mkdir -p $(OBJ_DIR)

# Build the executable
$(TARGET): $(OBJ_FILES)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Compile sources to objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean up build artifacts
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Rebuild everything
rebuild: clean all