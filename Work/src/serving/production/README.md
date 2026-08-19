# Production engineering lab

Production service surface: 12-factor config, structured logging, health check,
graceful shutdown, Dockerfile.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/serving/production/tests -v
$PY -m serving.production.benchmark --output /tmp/production.json
```

## Headline

config from env (secret redacted), JSON logs, health check, graceful shutdown.
See `note/serving/07_production_engineering.md`.
