"""Refuse-to-overwrite JSON report writing.

Benchmark evidence must not be silently clobbered by a rerun.  This helper
writes a report atomically and errors out if the destination already exists,
so a "before" number can never be mistaken for an "after" number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_report(path: str | Path, payload: Dict[str, Any], *, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing report {target}; pass overwrite=True to force"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
