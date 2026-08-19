"""Idempotent command executor for a robot.

The whole point of idempotency in robot control: a duplicated "move forward 1m"
command must not move the robot 2m.  An idempotent executor records the IDs of
commands it has already applied and ignores repeats, so at-least-once delivery
behaves like exactly-once.

The robot tracks its position; each command advances it by a fixed distance.
"""

from __future__ import annotations

from robotics.distributed.delivery import Command


class RobotExecutor:
    def __init__(self, idempotent: bool = True):
        self.idempotent = idempotent
        self.executed_ids: set[int] = set()
        self.position = 0.0
        self.applied = 0

    def apply(self, command: Command, step: float = 1.0) -> bool:
        """Apply one command; return True if it actually took effect."""
        if self.idempotent:
            if command.id in self.executed_ids:
                return False  # duplicate -> ignore
            self.executed_ids.add(command.id)
        self.position += step
        self.applied += 1
        return True
