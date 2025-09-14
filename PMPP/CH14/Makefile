# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -O2

# Get all .cu files and create corresponding target names without extension
SOURCES = $(wildcard *.cu)
TARGETS = $(SOURCES:.cu=)

# Default target
all: $(TARGETS)
	@echo "========================================="
	@echo "Build complete! Programs compiled:"
	@for target in $(TARGETS); do \
		echo "  - $$target"; \
	done
	@echo "========================================="
	@echo "Usage examples:"
	@echo "  ./coo            - Run COO SpMV"
	@echo "  ./csr            - Run CSR SpMV"
	@echo "  ./ell            - Run ELL SpMV"
	@echo "  ./jds            - Run JDS SpMV"
	@echo "  ./coo_to_csr     - Run COO to CSR conversion"
	@echo "  ./ell_coo_hybrid - Run Hybrid ELL-COO SpMV"
	@echo "========================================="

# Pattern rule for compilation
%: %.cu
	@echo "Compiling $< -> $@..."
	@$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "✓ Successfully compiled $@"

# Clean rule
clean:
	@echo "Cleaning up..."
	@rm -f $(TARGETS)
	@echo "✓ Removed all compiled programs"

# Help target
help:
	@echo "Available targets:"
	@echo "  make       - Build all programs"
	@echo "  make clean - Remove all compiled programs"
	@echo "  make help  - Show this help message"
	@echo "\nAvailable programs:"
	@for source in $(SOURCES); do \
		echo "  $$source"; \
	done

.PHONY: all clean help