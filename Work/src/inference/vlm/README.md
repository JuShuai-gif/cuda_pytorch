# VLM inference lab

Per-stage latency breakdown of a VLM pipeline (decode / preprocess / H2D /
vision encoder / connector / LLM).

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/inference/vlm/tests -v
$PY -m inference.vlm.benchmark --device cuda --output /tmp/vlm.json
```

## Headline (Thor/sm_110, 224x224 image)

vision encoder 37% + LLM 30% + CPU preprocess 26% + decode 6% + H2D 0.4%.
The CPU-side preprocess is a hidden ~33% that overlaps with GPU work in a
pipelined serving system.  See `note/inference/10_vlm_inference.md`.
