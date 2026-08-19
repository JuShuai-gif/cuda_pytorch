# Canary release lab

Gray-release controller with monitoring and automatic rollback.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/serving/canary_release/tests -v
$PY -m serving.canary_release.benchmark --output /tmp/canary.json
```

## Headline

Accuracy regression caught at 1% traffic (9.0% vs 1.1% error -> rollback);
healthy version ramps to 100%.  See `note/serving/05_canary_release.md`.
