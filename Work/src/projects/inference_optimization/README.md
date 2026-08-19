# Inference optimization project

The optimization staircase (eager -> compile -> fp16 -> Triton -> CUDA Graph)
with a Before/After report generator.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m projects.inference_optimization.benchmark --device cuda --output /tmp/opt.json
$PY -m projects.inference_optimization.report --input /tmp/opt.json --output /tmp/report.md
```

## Headline

4.07x via fp16 (2.91x) + Triton RMSNorm (1.09x) + CUDA Graph (1.27x); TensorRT
FP16 reaches ~4.9x.  See `note/system_design/final_project_a_gpu_inference_optimization.md`.
