# Realtime control lab

Second-order plant under PID control with variable latency: constant vs jittery.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/robotics/realtime/tests -v
$PY -m robotics.realtime.benchmark --output /tmp/realtime.json
```

## Headline

jittery 15ms-mean latency (p99 200ms) -> max tracking error 7x larger than
constant 10ms and never settles.  See `note/robotics/04_realtime.md`.
