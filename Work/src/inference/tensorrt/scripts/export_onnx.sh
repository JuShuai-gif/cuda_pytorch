#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Export the model to ONNX with PyTorch (flashrt env has torch).
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
PY=${TRT_PYTHON:-/home/guhaoran/miniconda3/envs/flashrt/bin/python}
OUTDIR=${1:-/tmp/trt_model}

"$PY" "$MODULE_DIR/python/export_onnx.py" --hidden 1024 --layers 4 --batch 1 --seq 16 --outdir "$OUTDIR"
