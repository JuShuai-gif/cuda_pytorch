# Makefile for CUDA Stream / Event / Graph learning examples
# Target GPU: RTX 4070 (compute capability 8.9)

NVCC    := nvcc
ARCH    := -arch=sm_89
NVFLAGS := -O2 $(ARCH) -lineinfo

TARGETS := 01_stream_basics \
           02_event_basics \
           03_stream_event_sync \
           04_cuda_graph_basics \
           05_graph_mempool

.PHONY: all clean run
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# Build and run every example in order.
run: all
	@for t in $(TARGETS); do \
		echo "==================== $$t ===================="; \
		./$$t; echo; \
	done

clean:
	rm -f $(TARGETS)
