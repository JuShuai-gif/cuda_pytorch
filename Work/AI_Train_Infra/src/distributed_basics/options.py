"""Explicit DDP configurations used by the baseline and candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DDPOptions:
    bucket_cap_mb: float
    gradient_as_bucket_view: bool
    static_graph: bool
    broadcast_buffers: bool = True
    find_unused_parameters: bool = False

    def validate(self) -> None:
        if self.bucket_cap_mb <= 0:
            raise ValueError("bucket_cap_mb must be > 0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def options_for_variant(
    variant: str,
    *,
    bucket_cap_mb: float | None = None,
) -> DDPOptions:
    """Return a teaching configuration without claiming it is universally faster.

    The baseline intentionally delays the first collective by putting this small
    model in one large bucket. The candidate allows earlier bucket readiness and
    removes the post-first-iteration gradient/bucket copy. Whether that wins is a
    topology- and shape-dependent measurement question.
    """

    if variant == "baseline":
        options = DDPOptions(
            bucket_cap_mb=256.0 if bucket_cap_mb is None else bucket_cap_mb,
            gradient_as_bucket_view=False,
            static_graph=False,
        )
    elif variant == "optimized":
        options = DDPOptions(
            bucket_cap_mb=1.0 if bucket_cap_mb is None else bucket_cap_mb,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    options.validate()
    return options
