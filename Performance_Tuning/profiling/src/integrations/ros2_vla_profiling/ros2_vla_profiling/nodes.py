#!/usr/bin/env python3
"""ROS2目标机端到端实验节点。

Header.stamp记录capture时间，header.frame_id携带单调frame序号。ros2_tracing可自动
观察publish/take/callback/executor；应用日志补充callback age、inference与action延迟。
"""
import csv
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray


def qos(depth=2):
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class MockCamera(Node):
    def __init__(self):
        super().__init__("profiling_mock_camera")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.publisher = self.create_publisher(Image, "/profiling/camera", qos())
        self.frame = 0
        self.timer = self.create_timer(1.0 / self.get_parameter("fps").value, self.publish)

    def publish(self):
        width = self.get_parameter("width").value
        height = self.get_parameter("height").value
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.frame)
        msg.height, msg.width, msg.encoding, msg.step = height, width, "mono8", width
        msg.data = bytes([self.frame & 255]) * (width * height)
        self.publisher.publish(msg)
        self.frame += 1


class MockVLA(Node):
    def __init__(self):
        super().__init__("profiling_mock_vla")
        self.declare_parameter("work_ms", 5.0)
        self.declare_parameter("csv", "ros2_vla_samples.csv")
        self.subscription = self.create_subscription(Image, "/profiling/camera", self.callback, qos())
        self.action = self.create_publisher(Float64MultiArray, "/profiling/action", qos(10))
        self.rows = []

    def callback(self, msg):
        callback_begin_ns = self.get_clock().now().nanoseconds
        capture_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        queue_age_ms = (callback_begin_ns - capture_ns) / 1e6
        work_begin = time.perf_counter_ns()
        deadline = work_begin + int(self.get_parameter("work_ms").value * 1e6)
        checksum = 0
        # CPU busy work模拟preprocess/inference，避免sleep被误认为计算。
        while time.perf_counter_ns() < deadline:
            checksum = (checksum * 1664525 + 1013904223) & 0xFFFFFFFF
        work_ms = (time.perf_counter_ns() - work_begin) / 1e6
        action = Float64MultiArray(data=[float(msg.header.frame_id), float(checksum)])
        self.action.publish(action)
        e2e_ms = (self.get_clock().now().nanoseconds - capture_ns) / 1e6
        self.rows.append([msg.header.frame_id, queue_age_ms, work_ms, e2e_ms])
        if len(self.rows) % 100 == 0:
            self.flush()
            self.get_logger().info(
                f"frame={msg.header.frame_id} queue_age_ms={queue_age_ms:.3f} "
                f"work_ms={work_ms:.3f} e2e_ms={e2e_ms:.3f}")

    def flush(self):
        path = Path(self.get_parameter("csv").value)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "queue_age_ms", "work_ms", "e2e_ms"])
            writer.writerows(self.rows)

    def destroy_node(self):
        self.flush()
        return super().destroy_node()


def spin(cls):
    rclpy.init()
    node = cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def camera_main():
    spin(MockCamera)


def vla_main():
    spin(MockVLA)
