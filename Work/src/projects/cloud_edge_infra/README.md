# Final Project C lab

End-to-end cloud-edge infra: publish + OTA + task + metrics + fault recovery.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/projects/cloud_edge_infra/tests -v
$PY -m projects.cloud_edge_infra.benchmark --output /tmp/cloud_edge_infra.json
```

## Headline

OTA to 3 robots (100%), task dispatch with metrics/trace, fault injection +
recovery.  See the project C report.
