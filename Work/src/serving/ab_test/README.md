# A/B test lab

Model experiment decision: accuracy vs business metric (robot success rate).

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/serving/ab_test/tests -v
$PY -m serving.ab_test.benchmark --output /tmp/ab.json
```

## Headline

accuracy says ship the slow accurate model (95% vs 90%), but robot success
says ship the fast one (89% vs 0% - the slow model misses every deadline).
See `note/serving/06_ab_test.md`.
