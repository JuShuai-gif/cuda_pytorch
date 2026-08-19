"""Final Project C: robot cloud-edge infrastructure (end-to-end).

Integrates the components built across the stages into one working system:

    Cloud (model registry + task scheduler + data service + monitor + OTA)
      -> Edge Gateway (forward/aggregate/cache)
      -> Robot (execute tasks, upload data, apply model updates)

The demo walks the full lifecycle: publish a model, OTA it to the fleet, run a
task, observe metrics/traces, inject a fault, and recover.  It reuses the
cloud_edge, ota, and observability modules rather than reimplementing them.
"""

from __future__ import annotations

from cloud_edge.architecture import Cloud, EdgeGateway, Robot, Task
from cloud_edge.ota.ota import ModelArtifact, ModelRegistry, RobotOTA
from observability.observability import Metrics, StructuredLogger, Tracer


class CloudEdgeInfra:
    def __init__(self, n_robots: int = 3):
        self.cloud = Cloud()
        self.edge = EdgeGateway()
        self.robots = [Robot(f"robot_{i}") for i in range(n_robots)]
        for r in self.robots:
            self.edge.register(r)
        self.tracer = Tracer()
        self.metrics = Metrics()
        self.logger = StructuredLogger()

    def publish_and_ota(self, version: str, content: bytes):
        """Publish a model to the registry and OTA it to all robots."""
        artifact = ModelArtifact.make(version, content)
        registry = ModelRegistry([artifact])
        self.cloud.publish_model(version)
        for r in self.robots:
            ota = RobotOTA(r.model_version, b"old_weights", disk_capacity=1024)
            result = ota.update(registry, version)
            if result == "ok":
                r.apply_model(version)
            self.logger.log("deploy", "INFO", "ota_result",
                            robot_id=r.robot_id, version=version, result=result)
            self.metrics.record("ota_success", 1.0 if result == "ok" else 0.0)

    def run_task(self, task: Task):
        with self.tracer.span(task.task_id, "cloud.schedule"):
            self.cloud.schedule(task)
        with self.tracer.span(task.task_id, "edge.forward"):
            result = self.edge.forward(task)
        self.metrics.record("task_success", 1.0 if result == "ok" else 0.0)
        self.logger.log(task.task_id, "INFO" if result == "ok" else "WARN",
                        "task_result", result=result)
        return result

    def inject_offline(self, robot_id: str) -> str:
        robot = self.edge.robots[robot_id]
        robot.online = False
        t = Task(f"task_{robot_id}_x", robot_id, "navigate")
        result = self.run_task(t)
        if result == "offline":
            self.cloud.record_failure(robot_id, "offline")
        return result


def main(argv=None) -> int:
    import argparse
    from common.report import write_report
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    infra = CloudEdgeInfra(n_robots=3)

    # Lifecycle 1: publish v2 + OTA.
    infra.publish_and_ota("v2", b"model_v2_weights")

    # Lifecycle 2: run a task with tracing + metrics.
    infra.run_task(Task("t1", "robot_0", "pick_object"))

    # Lifecycle 3: fault injection + recovery.
    offline_result = infra.inject_offline("robot_1")

    report = {
        "kind": "cloud_edge_infra",
        "robot_versions": {r.robot_id: r.model_version for r in infra.robots},
        "task_success_rate": infra.metrics.summary("task_success")["mean"],
        "ota_success_rate": infra.metrics.summary("ota_success")["mean"],
        "failures": infra.cloud.failures,
        "offline_task_result": offline_result,
    }
    write_report(args.output, report)

    print("== robot model versions (after OTA) ==")
    for rid, v in report["robot_versions"].items():
        print(f"  {rid}: {v}")
    print(f"== metrics ==")
    print(f"  task_success_rate: {report['task_success_rate']:.0%}")
    print(f"  ota_success_rate: {report['ota_success_rate']:.0%}")
    print(f"== fault recovery ==")
    print(f"  failures: {report['failures']}")
    print(f"  offline task result: {report['offline_task_result']}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
