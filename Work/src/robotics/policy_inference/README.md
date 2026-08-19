# Robot policy inference lab

VLA policy control-loop statistics (jitter, deadline miss) and batch=1
naive-vs-CUDA-Graph.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/robotics/policy_inference/tests -v
$PY -m robotics.policy_inference.benchmark --device cuda --output /tmp/robot.json
```

## Headline

CPU jitter leaves the mean almost unchanged (7.17 -> 7.39ms) but pushes p99
8.7 -> 12.3ms and deadline-miss rate 0% -> 3%.  See
`note/robotics/01_vla_policy_inference.md`.
