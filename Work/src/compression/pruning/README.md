# Pruning lab

FLOPs reduction vs real hardware speedup for unstructured / structured /
2:4 sparsity.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/compression/pruning/tests -v
$PY -m compression.pruning.benchmark --device cuda --output /tmp/prune.json
```

## The core lesson

- unstructured 99% sparsity -> FLOPs down to 1%, speedup ~1.0x (dense matmul
  does not skip zeros).
- structured 50% row pruning -> real ~1.3x speedup, but less than the 2x FLOPs
  reduction (smaller GEMMs are less efficient).
- 2:4 sparse needs hardware support; cuSPARSELt has no sm_110 kernel on this
  Thor, so it is Not Validated here.

See `note/compression/03_pruning.md`.
