# Triton operator lab

PyTorch baseline vs Triton implementation for the classic operator ladder:
vector add, reduction, softmax, layernorm, rmsnorm, gemm.  Each operator has a
correctness check (triton vs torch reference) and a benchmark (CUDA-event
device time + synchronized wall time).

## Environment (critical)

Triton 3.6 bundles a `ptxas-blackwell` built from CUDA 12.9, which rejects
`sm_110a` (this machine's NVIDIA Thor).  The package sets
`TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.0/bin/ptxas` on import so Triton
uses the CUDA 13 ptxas instead.  If you run Triton outside this package, set it
manually:

```bash
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.0/bin/ptxas
```

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

# correctness
$PY -m unittest discover -s Work/src/kernel/triton/tests -v

# benchmark (fp32 and fp16)
$PY -m kernel.triton.benchmark --device cuda --dtype float32 --output /tmp/triton_fp32.json
$PY -m kernel.triton.benchmark --device cuda --dtype float16 --output /tmp/triton_fp16.json

# sweep BLOCK/num_warps for gemm
$PY -m kernel.triton.sweep --device cuda --dtype float16 --output /tmp/triton_sweep.json
```

## What the numbers teach

The results are not "Triton beats PyTorch".  They split cleanly into three
regimes (see `note/kernel/05_triton_kernel.md`):

1. memory-bound elementwise / reduction: Triton ties PyTorch (both bound by
   DRAM bandwidth, no headroom).
2. fusion wins (rmsnorm, softmax, layernorm in fp16): Triton is faster because
   it avoids materializing intermediate tensors that the eager reference does.
3. cuBLAS territory (gemm): Triton loses to the heavily-tuned cuBLAS, which is
   expected for a from-scratch tiled matmul.
