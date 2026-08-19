"""OTA (over-the-air) model update flow.

The full update pipeline a robot runs to upgrade its model, and how each step
fails safely:

    registry -> download -> integrity check -> disk check -> install
             -> health check -> switch version  (rollback on any failure)

Faults are modeled explicitly:
  download interruption  - the artifact download fails; the robot retries
  corruption             - checksum mismatch; the artifact is rejected
  disk full              - install aborts, old model is kept
  load failure           - health check (a test inference) fails; rollback

A healthy update never leaves the robot without a working model: the current
version is only switched after every check passes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class ModelArtifact:
    version: str
    content: bytes
    checksum: str

    @staticmethod
    def make(version: str, content: bytes) -> "ModelArtifact":
        return ModelArtifact(version, content,
                             hashlib.sha256(content).hexdigest())


class ModelRegistry:
    def __init__(self, artifacts: list[ModelArtifact]):
        self.artifacts = {a.version: a for a in artifacts}

    def get(self, version: str) -> ModelArtifact | None:
        return self.artifacts.get(version)


class RobotOTA:
    def __init__(self, current_version: str, model_bytes: bytes, disk_capacity: int):
        self.current_version = current_version
        self.installed = {current_version: model_bytes}  # version -> bytes
        self.disk_capacity = disk_capacity
        self.log: list[str] = []

    def update(self, registry: ModelRegistry, target: str, *,
               download_fails: bool = False, corrupt: bool = False,
               disk_too_small: bool = False, load_fails: bool = False) -> str:
        artifact = registry.get(target)
        if artifact is None:
            return "unknown_version"

        # 1. Download (may be interrupted; the robot retries once).
        content = artifact.content
        if download_fails:
            self.log.append("download interrupted -> retry")
            # Retry succeeds (the second attempt works).
            content = artifact.content

        # 2. Integrity check (checksum).
        received = (b"corrupted" + content) if corrupt else content
        if hashlib.sha256(received).hexdigest() != artifact.checksum:
            self.log.append("checksum mismatch -> reject")
            return "corrupted"

        # 3. Disk check.
        if disk_too_small or len(received) > self.disk_capacity:
            self.log.append("disk full -> abort (keep old model)")
            return "disk_full"

        # 4. Install (write the new model to a staging slot).
        self.log.append("installed staged model")

        # 5. Health check: a test inference must succeed.
        if load_fails:
            self.log.append("health check failed -> rollback to " + self.current_version)
            return "health_check_failed"

        # 6. Switch version.
        self.installed[target] = received
        self.current_version = target
        self.log.append(f"switched to {target}")
        return "ok"
