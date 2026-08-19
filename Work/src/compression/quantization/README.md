# Quantization fundamentals lab

Quantization formula, granularity (per-tensor/channel/token/group), numeric
format comparison (FP32/TF32/FP16/BF16), and weight-only INT8 PTQ.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/compression/quantization/tests -v
$PY -m compression.quantization.benchmark --device cuda --output /tmp/quant.json
```

## Headline results (Thor/sm_110)

- Granularity MSE on an outlier-column weight: per-channel is ~130x better
  than per-tensor; per-group can be *worse* than per-channel when an outlier
  column pollutes a whole group.
- GEMM (1024^3): tf32 3.2x, fp16 14.4x faster than fp32-ieee; bf16 has the
  same speed but ~7x the error of fp16 (7 mantissa bits).
- Weight-only INT8 PTQ: half the weight bytes, max diff ~0.01.

See `note/compression/01_quantization_fundamentals.md`.
