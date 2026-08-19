#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Full TensorRT lab: export ONNX, build FP32 + FP16 engines, run + benchmark.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
BIN_DIR=${BIN_DIR:-"$MODULE_DIR/build/bin"}
OUTDIR=${1:-/tmp/trt_model}
BATCH=${2:-1}
SEQ=${3:-16}

mkdir -p "$OUTDIR"

echo "== export ONNX =="
bash "$SCRIPT_DIR/export_onnx.sh" "$OUTDIR"

echo "== build FP32 engine =="
"$BIN_DIR/build_engine" --onnx "$OUTDIR/model.onnx" --engine "$OUTDIR/fp32.engine" \
  --min-batch 1 --opt-batch 8 --max-batch 32 --min-seq 1 --opt-seq 16 --max-seq 64

echo "== build FP16 engine =="
"$BIN_DIR/build_engine" --onnx "$OUTDIR/model.onnx" --engine "$OUTDIR/fp16.engine" --fp16 \
  --min-batch 1 --opt-batch 8 --max-batch 32 --min-seq 1 --opt-seq 16 --max-seq 64

echo "== run FP32 =="
"$BIN_DIR/run_engine" --engine "$OUTDIR/fp32.engine" --input "$OUTDIR/input.bin" \
  --output-ref "$OUTDIR/output_ref.bin" --batch "$BATCH" --seq "$SEQ"

echo "== run FP16 =="
"$BIN_DIR/run_engine" --engine "$OUTDIR/fp16.engine" --input "$OUTDIR/input.bin" \
  --output-ref "$OUTDIR/output_ref.bin" --batch "$BATCH" --seq "$SEQ"
