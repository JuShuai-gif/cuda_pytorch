# Operator fusion lab

Fused Triton kernels vs eager multi-kernel references, measuring kernel count,
memory traffic (analytical estimate), and latency.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/kernel/fusion/tests -v
$PY -m kernel.fusion.benchmark --device cuda --dtype float16 --output /tmp/fusion.json
```

## The honest result

Fusion always reduces kernel count and memory traffic, but the latency outcome
depends on whether the fused operator includes a GEMM:

- `residual_rmsnorm` (elementwise + reduction only): ~7x faster fused.
- `bias_relu` / `gemm_bias` / `dequant_gemm` (contain a GEMM): fused is
  *slower* because the hand-written Triton GEMM loses to cuBLAS.

See `note/kernel/06_operator_fusion.md` for why, and for the production-correct
way to fuse GEMMs (cuBLAS/CUTLASS epilogues, `torch.compile`).
