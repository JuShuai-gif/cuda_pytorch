"""Large-bucket DDP baseline: correct, but intentionally overlap-unfriendly."""

from __future__ import annotations

from .options import DDPOptions, options_for_variant


def build_options(bucket_cap_mb: float | None = None) -> DDPOptions:
    return options_for_variant("baseline", bucket_cap_mb=bucket_cap_mb)
