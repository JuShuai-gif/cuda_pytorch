# Robot runtime lab

Sensor sync strategies + ROS-like primitives (topic/service/action/QoS).

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/robotics/runtime/tests -v
$PY -m robotics.runtime.benchmark --output /tmp/runtime.json
```

## Headline

latest sync: 200 cycles but ~21-33ms stale observations; exact sync: 19 cycles
but zero staleness.  See `note/robotics/03_robot_runtime.md`.
