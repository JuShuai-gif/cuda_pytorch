"""Correctness tests for delivery semantics and idempotency.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/robotics/distributed/tests -v
"""

from __future__ import annotations

import unittest

from robotics.distributed.delivery import Command, deliver_at_least_once, deliver_at_most_once
from robotics.distributed.idempotency import RobotExecutor


class TestDelivery(unittest.TestCase):
    def test_at_most_once_never_duplicates(self):
        cmds = [Command(i, "x") for i in range(100)]
        delivered = deliver_at_most_once(cmds, 0.2, seed=0)
        ids = [c.id for c in delivered]
        self.assertEqual(len(ids), len(set(ids)))  # no duplicates

    def test_at_least_once_can_duplicate(self):
        cmds = [Command(i, "x") for i in range(100)]
        delivered = deliver_at_least_once(cmds, 0.2, seed=0)
        ids = [c.id for c in delivered]
        self.assertGreater(len(ids), len(set(ids)))  # has duplicates


class TestIdempotency(unittest.TestCase):
    def test_idempotent_executor_ignores_duplicates(self):
        robot = RobotExecutor(idempotent=True)
        c = Command(1, "move")
        self.assertTrue(robot.apply(c))
        self.assertFalse(robot.apply(c))  # duplicate ignored
        self.assertEqual(robot.position, 1.0)

    def test_non_idempotent_applies_duplicates(self):
        robot = RobotExecutor(idempotent=False)
        c = Command(1, "move")
        robot.apply(c)
        robot.apply(c)  # duplicate applied again
        self.assertEqual(robot.position, 2.0)


if __name__ == "__main__":
    unittest.main()
