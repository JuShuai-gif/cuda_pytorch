# Autoscaling lab

Simulation of the three autoscaling metrics (CPU / queue / latency) under a
GPU-load spike.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/serving/autoscaling/tests -v
$PY -m serving.autoscaling.benchmark --output /tmp/autoscale.json
```

## Headline

CPU metric never scales (GPU saturated while CPU idle) -> 67000 dropped;
queue/latency scale correctly.  See `note/serving/04_autoscaling.md`.
