# OTA lab

Model update flow with fault handling (corruption / disk full / load failure /
rollback).

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/cloud_edge/ota/tests -v
$PY -m cloud_edge.ota.benchmark --output /tmp/ota.json
```

## Headline

healthy -> v2; corrupted/disk-full/load-failure all stay on v1 (rollback).
See `note/cloud_edge/02_ota.md`.
