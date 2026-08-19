# Cloud-edge architecture lab

Three-tier (Cloud / Edge Gateway / Robot) cooperation: task dispatch, model
rollout, data upload, fault recovery.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/cloud_edge/tests -v
$PY -m cloud_edge.benchmark --output /tmp/cloud_edge.json
```

## Headline

robot_1 goes offline -> its task is rescheduled to robot_2; model v2 rolls out
to all robots.  See `note/cloud_edge/01_architecture.md`.
