# Data loop lab

The robot data flywheel: runtime data -> failure mining -> training -> deploy.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/robotics/data_loop/tests -v
$PY -m robotics.data_loop.benchmark --output /tmp/data_loop.json
```

## Headline

4 flywheel rounds drop the fleet failure rate from 51.7% to 5.0%.  See
`note/robotics/05_data_loop.md`.
