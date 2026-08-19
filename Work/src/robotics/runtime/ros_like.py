"""ROS-like primitives (topic / service / action / QoS), simplified.

A minimal, dependency-free model of the four ROS concepts so the semantics can
be understood without a ROS install:

  Topic    pub/sub; a publisher sends messages, subscribers get callbacks
  Service  request/response (one-shot RPC)
  Action   a long-running task with a goal, feedback, and a result (cancelable)
  QoS      reliability (reliable vs best-effort) and history depth

These are the abstractions a real robot runtime (ROS 2) is built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List


class QoS:
    def __init__(self, reliable: bool = True, depth: int = 10):
        self.reliable = reliable
        self.depth = depth


class Topic:
    """Pub/sub with a bounded history and QoS semantics."""

    def __init__(self, name: str, qos: QoS):
        self.name = name
        self.qos = qos
        self.history: List = []
        self.subscribers: List[Callable] = []

    def publish(self, msg):
        if len(self.history) >= self.qos.depth:
            self.history.pop(0)
        self.history.append(msg)
        for cb in self.subscribers:
            cb(msg)

    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)


class Service:
    """Request/response: one request, one response."""

    def __init__(self, name: str, handler: Callable):
        self.name = name
        self.handler = handler

    def call(self, request):
        return self.handler(request)


@dataclass
class ActionGoal:
    goal_id: str
    data: str


class Action:
    """A long-running task with feedback and a cancelable lifecycle."""

    def __init__(self, name: str, execute: Callable):
        self.name = name
        self.execute = execute
        self.feedback: List[str] = []
        self.active: dict[str, bool] = {}

    def send_goal(self, goal: ActionGoal):
        self.active[goal.goal_id] = True

        def feedback_cb(msg):
            self.feedback.append(f"{goal.goal_id}:{msg}")

        result = self.execute(goal, feedback_cb, self.active)
        self.active.pop(goal.goal_id, None)
        return result

    def cancel(self, goal_id: str):
        self.active[goal_id] = False
