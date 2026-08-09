# ROS2 VLA Profiling目标包

```bash
mkdir -p /tmp/ros2-vla-ws/src
cp -r /home/ghr/code/cuda_pytorch/Performance_Tuning/profiling/src/integrations/ros2_vla_profiling /tmp/ros2-vla-ws/src/
cd /tmp/ros2-vla-ws
colcon build --symlink-install
source install/setup.bash
ros2 run ros2_vla_profiling mock_camera
ros2 run ros2_vla_profiling mock_vla
```

另一个终端采集：

```bash
ros2 topic hz /profiling/camera
ros2 topic bw /profiling/camera
ros2 trace -s profiling_vla
```

用参数制造queue与drop：

```bash
ros2 run ros2_vla_profiling mock_camera --ros-args -p fps:=50.0
ros2 run ros2_vla_profiling mock_vla --ros-args -p work_ms:=30.0
```

对比普通DDS、DDS Shared Memory和Loaned Message时，保持消息大小、QoS、频率、进程布局一致。Shared Memory Transport不自动等于Loaned Message或完全Zero-Copy；应结合trace、DDS统计与实际copy验证。
