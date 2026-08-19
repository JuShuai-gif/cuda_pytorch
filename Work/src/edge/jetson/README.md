# Edge AI (Jetson) lab

Platform profile + sustained-load thermal/power observation on Jetson Thor.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m edge.jetson.benchmark --seconds 30 --output /tmp/edge.json
```

## Headline

idle: 44.6C / 24.8W; sustained GEMM load: 58.7 -> 74.6C, ~127W mean (135W
peak), CPU clock steady at 2601 MHz (MAXN, no throttling).  See
`note/edge/01_edge_ai_jetson.md`.
