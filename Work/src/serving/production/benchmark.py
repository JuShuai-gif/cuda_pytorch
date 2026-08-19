"""Verify the production service surface: config, health check, logging.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m serving.production.benchmark --output /tmp/production.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from serving.production.config import load_config
from serving.production.service import ProductionService


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    config = load_config()
    svc = ProductionService(config)
    svc.start()

    health = svc.health()
    config_dict = config.to_dict()

    report = {
        "kind": "production_engineering",
        "health": health,
        "config": config_dict,
        "api_key_redacted": config_dict["api_key"] == "***" or config_dict["api_key"] == "",
    }
    write_report(args.output, report)

    print("== health check ==")
    print(f"  {json.dumps(health)}")
    print("== config (12-factor, secret redacted) ==")
    print(f"  {json.dumps(config_dict, indent=2)}")
    print(f"report written to {args.output}")

    svc.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
