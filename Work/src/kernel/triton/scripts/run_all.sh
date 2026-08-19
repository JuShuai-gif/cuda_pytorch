#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Triton lab: correctness + benchmark + sweep.
# Uses the flashrt env (torch 2.11 + triton 3.6) and pins ptxas to CUDA 13.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../../.." && pwd -P)

PY=${TRITON_PYTHON:-/home/guhaoran/miniconda3/envs/flashrt/bin/python}
export PYTHONPATH="$REPO_ROOT/Work/src"
export TRITON_PTXAS_BLACKWELL_PATH=${TRITON_PTXAS_BLACKWELL_PATH:-/usr/local/cuda-13.0/bin/ptxas}

out_dir=${1:-/tmp/triton_lab}
mkdir -p "$out_dir"

echo "== correctness =="
"$PY" -m unittest discover -s "$SCRIPT_DIR/../tests" -v

echo "== benchmark fp32 =="
"$PY" -m kernel.triton.benchmark --device cuda --dtype float32 --output "$out_dir/fp32.json"

echo "== benchmark fp16 =="
"$PY" -m kernel.triton.benchmark --device cuda --dtype float16 --output "$out_dir/fp16.json"

echo "== sweep fp16 =="
"$PY" -m kernel.triton.sweep --device cuda --dtype float16 --output "$out_dir/sweep.json"
