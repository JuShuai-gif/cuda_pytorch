# ============================================================
# kernel_engineering_lab - Industrial GPU Kernel Build System
# 借鉴: CUTLASS/PyTorch/Triton 的构建模式
# ============================================================

SHELL := /bin/bash
PROJECT_ROOT := $(shell pwd)
CONDA_ENV := lerobot_env
PYTHON := $(shell conda run -n $(CONDA_ENV) which python 2>/dev/null || which python3)
NVCC := $(shell which nvcc)
GPU_ARCH := $(shell $(PYTHON) -c "import torch; print('sm_' + str(torch.cuda.get_device_capability(0)[0]) + str(torch.cuda.get_device_capability(0)[1]))" 2>/dev/null || echo "sm_89")

# --- 检测 GPU 架构 ---
ARCH_FLAGS := -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89 \
              -gencode arch=compute_90,code=sm_90

# --- Python 路径发现 ---
PYTHONPATH := $(shell $(PYTHON) -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo "")
CUDA_BIN_PATH := /usr/local/cuda/bin
CONDA_BIN_PATH := $(shell $(PYTHON) -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "")/bin
PKG_PATH := $(shell $(PYTHON) -c 'import torch.utils.cpp_extension as ce; print(ce.include_paths()[0])' 2>/dev/null || echo "")

# --- Pytest 配置 ---
PYTEST := $(PYTHON) -m pytest
PYTEST_ARGS := -v --tb=short --color=yes -x

.PHONY: all build build-ext test test-all bench bench-all clean clean-all \
        env-check check-env profile-matmul profile-attention profile-ncu \
        lint typecheck help list-modules

# ============================================================
# 顶级目标
# ============================================================

all: build test

# ============================================================
# 构建目标
# ============================================================

# 构建所有 Python + CUDA 扩展
build: build-ext
	@echo "=== Build Complete ==="

# 构建 CUDA C++ extension
build-ext:
	@echo "=== Building CUDA Extensions ($(GPU_ARCH)) ==="
	cd $(PROJECT_ROOT)/01_cuda_basics && TORCH_CUDA_ARCH_LIST="$(shell echo $(GPU_ARCH) | sed 's/sm_//')" $(PYTHON) setup.py build_ext --inplace
	-cd $(PROJECT_ROOT)/03_memory_bandwidth && $(PYTHON) setup.py build_ext --inplace 2>/dev/null || true
	-cd $(PROJECT_ROOT)/07_cuda_streams_async && $(PYTHON) setup.py build_ext --inplace 2>/dev/null || true
	@echo "=== Build Complete ==="

# 构建 + 测试所有模块
test: build
	@echo "=== Running All Tests ==="
	export LD_LIBRARY_PATH="$(CUDA_BIN_PATH):$(CONDA_BIN_PATH)/lib:$$LD_LIBRARY_PATH" && \
	export PYTHONPATH="$(PROJECT_ROOT):$(PYTHONPATH)" && \
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/00_env_check/test_env.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/01_cuda_basics/test_cuda_basics.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/02_triton_basics/test_triton_basics.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/03_memory_bandwidth/test_mem_bw.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/04_operator_fusion/test_fusion.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/05_matmul_tiling/test_tiling.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/06_attention_flash_like/test_attention.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/07_cuda_streams_async/test_streams.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/08_memory_management/test_memory.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/09_graph_optimization/test_graph.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/10_kernel_codegen/test_codegen.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/11_autotune/test_autotune.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/12_inference_pipeline/test_pipeline.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/13_kernel_profile/test_profile.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/14_heterogeneous_scheduling/test_scheduler.py || true
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/15_numerical_precision/test_precision.py || true
	@echo "=== All Tests Complete ==="

# 仅运行 CUDA 扩展相关测试
test-ext: build-ext
	@echo "=== Running CUDA Extension Tests ==="
	$(PYTEST) $(PYTEST_ARGS) $(PROJECT_ROOT)/01_cuda_basics/test_cuda_basics.py

# 运行所有 benchmarks
bench: build
	@echo "=== Running All Benchmarks ==="
	$(PYTHON) $(PROJECT_ROOT)/01_cuda_basics/benchmark_cuda_basics.py
	@echo "=== Benchmarks Complete ==="

# 使用 benchmark framework 运行综合对比
bench-all: build
	@echo "=== Running Industrial Benchmark Suite ==="
	$(PYTHON) $(PROJECT_ROOT)/benchmarks/run_benchmark_suite.py

# ============================================================
# Profiling 目标
# ============================================================

# Nsight Compute profiling (需要 root 权限或 ncu 在 PATH 中)
profile-ncu-matmul:
	ncu --set full --csv -o $(PROJECT_ROOT)/profiles/ncu_matmul \
		$(PYTHON) $(PROJECT_ROOT)/01_cuda_basics/benchmark_cuda_basics.py

profile-ncu-attention:
	ncu --set full --csv -o $(PROJECT_ROOT)/profiles/ncu_attention \
		$(PYTHON) $(PROJECT_ROOT)/01_cuda_basics/benchmark_cuda_basics.py

# Nsight Systems profiling
profile-nsys:
	mkdir -p $(PROJECT_ROOT)/profiles
	nsys profile --trace=cuda,nvtx,osrt --output=$(PROJECT_ROOT)/profiles/nsys_report \
		$(PYTHON) $(PROJECT_ROOT)/01_cuda_basics/benchmark_cuda_basics.py

# PyTorch profiler
profile-torch:
	$(PYTHON) $(PROJECT_ROOT)/13_kernel_profile/profile_demo.py 2>/dev/null || \
	$(PYTHON) $(PROJECT_ROOT)/benchmarks/run_benchmark_suite.py --profile

# ============================================================
# 清理目标
# ============================================================

# 标准清理
clean:
	find $(PROJECT_ROOT) -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find $(PROJECT_ROOT) -type f -name '*.pyc' -delete 2>/dev/null || true
	find $(PROJECT_ROOT) -type f -name '*.pyo' -delete 2>/dev/null || true
	find $(PROJECT_ROOT) -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find $(PROJECT_ROOT) -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find $(PROJECT_ROOT) -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	find $(PROJECT_ROOT) -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned Python artifacts"

# 深度清理（包括编译产物）
clean-all: clean
	find $(PROJECT_ROOT) -type f -name '*.so' -delete 2>/dev/null || true
	find $(PROJECT_ROOT) -type f -name '*.o' -delete 2>/dev/null || true
	find $(PROJECT_ROOT) -type d -name 'build' -exec rm -rf {} + 2>/dev/null || true
	find $(PROJECT_ROOT) -type d -name 'dist' -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(PROJECT_ROOT)/profiles/* 2>/dev/null || true
	@echo "Full clean complete"

# ============================================================
# 环境检查
# ============================================================

env-check: check-env

check-env:
	@echo "=== Checking Environment ==="
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo "NVCC:   $$($(NVCC) --version 2>&1 | head -1 || echo 'not found')"
	@echo "GPU:    $$($(PYTHON) -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'not available')"
	@echo "CUDA:   $$($(PYTHON) -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not available')"
	@echo "Arch:   $(GPU_ARCH)"
	@echo ""
	-$(PYTHON) $(PROJECT_ROOT)/00_env_check/check_env.py 2>/dev/null || \
	 $(PYTHON) -c "
import torch, triton
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Triton: {triton.__version__}')
"

# ============================================================
# 代码质量
# ============================================================

lint:
	@echo "=== Linting Python ==="
	$(PYTHON) -m ruff check $(PROJECT_ROOT) --fix 2>/dev/null || \
	$(PYTHON) -m flake8 $(PROJECT_ROOT) 2>/dev/null || \
	echo "No linter found. Install ruff: pip install ruff"

typecheck:
	@echo "=== Type Checking ==="
	$(PYTHON) -m mypy $(PROJECT_ROOT)/benchmarks $(PROJECT_ROOT)/01_cuda_basics --ignore-missing-imports 2>/dev/null || \
	echo "No mypy found. Install: pip install mypy"

# ============================================================
# 文档
# ============================================================

list-modules:
	@echo "=== Kernel Engineering Lab Modules ==="
	@ls -d $(PROJECT_ROOT)/[0-9]* | sort | while read d; do \
		name=$$(basename $$d); \
		desc=$$(head -3 $$d/README.md 2>/dev/null | grep -v '^#' | head -1 2>/dev/null || echo ""); \
		printf "  %-35s %s\n" "$$name" "$$desc"; \
	done

# ============================================================
# 帮助
# ============================================================

help:
	@echo "kernel_engineering_lab - Industrial GPU Kernel Build System"
	@echo ""
	@echo "  构建:"
	@echo "    make build            构建所有 CUDA extensions"
	@echo "    make build-ext        构建 CUDA C++ extension"
	@echo ""
	@echo "  测试:"
	@echo "    make test             构建并运行所有测试"
	@echo "    make test-ext         构建并运行 CUDA 扩展测试"
	@echo ""
	@echo "  基准测试:"
	@echo "    make bench            运行标准 benchmarks"
	@echo "    make bench-all        运行工业级 benchmark suite"
	@echo ""
	@echo "  Profiling:"
	@echo "    make profile-ncu-matmul   Nsight Compute profiling (matmul)"
	@echo "    make profile-ncu-attn     Nsight Compute profiling (attention)"
	@echo "    make profile-nsys         Nsight Systems profiling"
	@echo "    make profile-torch        PyTorch profiler"
	@echo ""
	@echo "  清理:"
	@echo "    make clean            清理 Python artifacts"
	@echo "    make clean-all         深度清理（包括 .so 文件）"
	@echo ""
	@echo "  环境:"
	@echo "    make check-env         检查 CUDA/PyTorch/Triton 环境"
	@echo ""
	@echo "  代码质量:"
	@echo "    make lint              Ruff/Falke8 检查"
	@echo "    make typecheck         Mypy 类型检查"
	@echo ""
	@echo "  其他:"
	@echo "    make list-modules     列出所有模块"
	@echo "    make help             显示此帮助"
	@echo ""
	@echo "  当前 GPU: $(GPU_ARCH)"
