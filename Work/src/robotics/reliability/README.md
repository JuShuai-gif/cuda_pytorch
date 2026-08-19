# Reliability lab

Fault profiles (9 modes) + watchdog (crash restart, timeout fallback).

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/robotics/reliability/tests -v
$PY -m robotics.reliability.benchmark --output /tmp/reliability.json
```

## Headline

crash -> watchdog restart + safe_stop; hang -> timeout fallback.  See
`note/robotics/06_reliability.md` and the incident playbook.
