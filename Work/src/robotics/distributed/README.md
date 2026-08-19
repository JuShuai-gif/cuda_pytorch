# Distributed fundamentals lab

Delivery semantics (at-most/at-least/exactly-once) and idempotency for robot
remote control.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/robotics/distributed/tests -v
$PY -m robotics.distributed.benchmark --output /tmp/distributed.json
```

## Headline

100 "move 1m" commands, 20% link loss: at-most-once -> 90m (lost), at-least-
once -> 114m (duplicated, dangerous), at-least-once+idempotent -> 100m (correct).
See `note/robotics/02_distributed_fundamentals.md`.
