# Robot runtime project

Naive vs optimized (double buffering + async H2D + CUDA Graph) runtime.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/projects/robot_runtime/tests -v
$PY -m projects.robot_runtime.benchmark --device cuda --output /tmp/robot_runtime.json
```

## Headline

single-frame latency ~unchanged (1.02x); continuous-stream throughput +45%
(258 -> 375 fps) via CPU/GPU overlap.  See the project B report.
