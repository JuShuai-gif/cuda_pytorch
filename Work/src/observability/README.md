# Observability lab

Three pillars (metrics / logs / traces) with request_id correlation across
Cloud -> Edge -> Robot.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/observability/tests -v
$PY -m observability.benchmark --output /tmp/observability.json
```

## Headline

p99 22ms vs p50 5ms; the slow request's trace shows the 20ms lives in
robot.model_infer, not cloud/edge.  See `note/observability/01_observability.md`.
