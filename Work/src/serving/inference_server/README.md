# Inference server lab

Minimal inference server (queue + worker + batching strategies) with
throughput/latency comparison.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/serving/inference_server/tests -v
$PY -m serving.inference_server.benchmark --device cuda --output /tmp/server.json
```

## Headline

dynamic batching wins (3130 req/s, p99 7.3ms) vs no-batch (439 req/s); pure
static batching stalls on the tail batch that never fills.  See
`note/serving/02_inference_server.md`.
