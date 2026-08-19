"""Message delivery semantics over an unreliable link.

The three delivery guarantees, and how a sender behaves under each:

  at-most-once   - send once, may be lost; never duplicated
  at-least-once  - resend until acknowledged; may be duplicated
  exactly-once   - the ideal; not achievable on an unreliable link alone,
                   approximated by "at-least-once + idempotent receiver"

This module simulates a lossy link and a sender that retries, producing the
*delivery counts* the receiver actually sees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Command:
    id: int
    payload: str


def deliver_at_most_once(commands: list[Command], loss_rate: float,
                         seed: int = 0) -> list[Command]:
    """Send each command once; a lost packet is gone (never duplicated)."""
    rng = random.Random(seed)
    delivered = []
    for c in commands:
        if rng.random() >= loss_rate:
            delivered.append(c)
    return delivered


def deliver_at_least_once(commands: list[Command], loss_rate: float,
                          seed: int = 0) -> list[Command]:
    """Resend each command until acknowledged; a retry may duplicate it."""
    rng = random.Random(seed)
    delivered = []
    for c in commands:
        # The receiver acks; if the ack is lost the sender resends, so the
        # receiver may see the command more than once.
        while True:
            delivered.append(c)          # the send attempt reaches the receiver
            if rng.random() >= loss_rate:  # ack got back -> stop resending
                break
            # ack lost -> resend (next loop iteration is a duplicate delivery)
    return delivered
